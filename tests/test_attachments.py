"""Attaching files to a chat message with ``@path``."""

from __future__ import annotations

import pytest
from textual.widgets import Input

from bedrock import attachments
from bedrock.agentpack_app import AgentPackApp
from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.context import ContextStore
from bedrock.models import Conversation
from bedrock.storage import ConversationStore
from bedrock.ui.commands import parse_scientific_request
from bedrock.ui.widgets import SystemNotice, UserMessage

from tests.conftest import FakeBackend, make_services
from tests.test_agentpack_app import FakeAgentPackClient
from tests.workers import wait_for_app_workers


# ---- Finding references -----------------------------------------------------


def test_references_are_found_in_the_forms_people_type_and_paste():
    text = 'compare @a.csv, @"My Data.csv" and @My\\ Other.csv with @\'x y.txt\'.'
    assert attachments.find_references(text) == ["a.csv", "My Data.csv", "My Other.csv", "x y.txt"]


def test_an_at_sign_inside_a_word_is_not_a_reference():
    assert attachments.find_references("mail alice@example.com about it") == []


def test_resolve_ignores_at_words_that_are_not_files(tmp_path):
    assert attachments.resolve("thanks @alice", cwd=tmp_path) == []


def test_resolve_finds_relative_files_once(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    assert attachments.resolve("@notes.txt and again @notes.txt", cwd=tmp_path) == [tmp_path / "notes.txt"]


@pytest.mark.parametrize(
    "text, message",
    [
        ("summarise @missing.txt", "Could not find the file `missing.txt`"),
        ("summarise @sub", "is a folder"),
    ],
)
def test_resolve_reports_a_reference_that_was_meant_as_a_file(tmp_path, text, message):
    (tmp_path / "sub").mkdir()
    with pytest.raises(attachments.AttachmentError, match=message):
        attachments.resolve(text, cwd=tmp_path)


def test_resolve_limits_how_many_files_one_message_attaches(tmp_path):
    names = [f"f{i}.txt" for i in range(attachments.MAX_ATTACHMENTS + 1)]
    for name in names:
        (tmp_path / name).write_text("x")
    with pytest.raises(attachments.AttachmentError, match="at most"):
        attachments.resolve(" ".join(f"@{name}" for name in names), cwd=tmp_path)


# ---- Loading ----------------------------------------------------------------


def test_a_small_file_is_sent_whole(tmp_path):
    (tmp_path / "notes.txt").write_text("the pH was 6.8")
    [attached] = attachments.load([tmp_path / "notes.txt"], ContextStore(tmp_path / "ctx"))
    assert attached.mode == "whole"
    assert attached.text == "the pH was 6.8"
    assert attachments.summary([attached]) is None


def test_a_large_file_is_sent_as_its_digest_and_stored_for_search(tmp_path):
    rows = ["sample,ph,passed"] + [f"s{i},{6.0 + (i % 20) * 0.1:.1f},true" for i in range(5000)]
    (tmp_path / "assay.csv").write_text("\n".join(rows))
    store = ContextStore(tmp_path / "ctx")

    [attached] = attachments.load([tmp_path / "assay.csv"], store)

    assert attached.mode == "digest"
    assert len(attached.text) < attachments.MAX_INLINE_CHARS
    assert "sent as a digest" in attached.label
    assert "too large to send whole" in attachments.summary([attached])
    assert store.search("s4999", limit=1)


def test_a_binary_file_is_refused_in_plain_words(tmp_path):
    (tmp_path / "scan.pdf").write_bytes(b"%PDF-1.7\x00\x01\x02")
    with pytest.raises(attachments.AttachmentError, match="not a text file"):
        attachments.load([tmp_path / "scan.pdf"], ContextStore(tmp_path / "ctx"))


# ---- In the conversation ----------------------------------------------------


def test_attachments_go_to_the_model_and_survive_a_save_and_reload(tmp_path):
    (tmp_path / "notes.txt").write_text("the pH was 6.8")
    attached = attachments.load([tmp_path / "notes.txt"], ContextStore(tmp_path / "ctx"))
    conversation = Conversation()
    conversation.add_user("what was the pH in @notes.txt?", attached)

    [sent] = conversation.to_api()
    assert sent["content"].startswith("what was the pH in @notes.txt?")
    assert "[Attached file: notes.txt]\nthe pH was 6.8\n[End of notes.txt]" in sent["content"]

    reloaded = Conversation.from_dict(conversation.to_dict())
    assert reloaded.messages[0].content == "what was the pH in @notes.txt?"
    assert reloaded.to_api() == conversation.to_api()


def test_a_damaged_saved_attachment_is_skipped_not_fatal():
    data = Conversation().to_dict() | {
        "messages": [{"role": "user", "content": "hi", "attachments": [{"name": 3}, "junk"]}]
    }
    assert Conversation.from_dict(data).to_api() == [{"role": "user", "content": "hi"}]


def test_fold_accepts_an_at_path(tmp_path):
    command = parse_scientific_request("fold @candidate.fasta")
    assert command is not None and command.args == "candidate.fasta"


# ---- Both apps --------------------------------------------------------------


async def _submit(app, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await wait_for_app_workers(app)
    await pilot.pause()


async def test_chat_sends_an_attached_file_and_shows_only_what_was_typed(fake_backend: FakeBackend, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("the pH was 6.8")
    app = BedrockApp(
        settings=Settings(model="test-model", context_dir=str(tmp_path / "ctx")),
        backend=fake_backend,
        store=ConversationStore(tmp_path / "history"),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await _submit(app, pilot, f"what was the pH in @{notes}?")

        _, api_messages = fake_backend.requests[-1]
        assert "[Attached file: notes.txt]" in api_messages[-1]["content"]
        bubble = str(app.query_one(UserMessage).content)
        assert bubble.startswith(f"what was the pH in @{notes}?")
        assert "📎 notes.txt (14 bytes)" in bubble
        assert "the pH was 6.8" not in bubble

        # A follow-up still sends the file, because it is saved with the question.
        await _submit(app, pilot, "and is that acidic?")
        _, api_messages = fake_backend.requests[-1]
        assert any("the pH was 6.8" in message["content"] for message in api_messages)

        # Reopening the saved chat shows the same bubble.
        conversation_id = app.conversation.id
        await app.start_new_conversation()
        await app.open_conversation(conversation_id)
        await pilot.pause()
        assert "📎 notes.txt" in str(app.query(UserMessage).first().content)


async def test_chat_keeps_the_draft_when_an_attached_file_is_missing(fake_backend: FakeBackend, tmp_path):
    app = BedrockApp(settings=Settings(model="test-model"), backend=fake_backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        draft = f"summarise @{tmp_path / 'gone.txt'}"
        await _submit(app, pilot, draft)

        assert fake_backend.requests == []
        assert app.conversation.messages == []
        assert app.query_one("#prompt", Input).value == draft
        assert any("Could not find the file" in notice.source for notice in app.query(SystemNotice))


async def test_agent_mode_sends_an_attached_file(fake_client: FakeAgentPackClient, tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("the pH was 6.8")
    app = AgentPackApp(
        settings=Settings(user_id="alice", context_dir=str(tmp_path / "ctx")),
        client=fake_client,
        services=make_services(),
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt", Input).value = f"summarise @{notes}"
        await pilot.press("enter")
        for _ in range(20):
            if fake_client.sent_messages:
                break
            await pilot.pause(0.05)

        [(_, outgoing)] = fake_client.sent_messages
        assert outgoing.startswith(f"summarise @{notes}")
        assert "[Attached file: notes.txt]\nthe pH was 6.8" in outgoing
        assert "📎 notes.txt" in str(app.query_one(UserMessage).content)


@pytest.fixture
def fake_client() -> FakeAgentPackClient:
    return FakeAgentPackClient()
