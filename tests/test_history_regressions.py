"""Regression probes for concurrent history and demand-driven loading."""
import asyncio
import errno
import gc
import json
import multiprocessing
import weakref
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from bedrock.app import BedrockApp
from bedrock.config import Settings, load_settings
from bedrock.models import Conversation
from bedrock.storage import ConversationDeleted, ConversationStore, StorageConflict
from bedrock.titles import clean_title, generate_title
from bedrock.ui.widgets import AssistantMessage, ConversationList, SystemNotice, UserMessage
from tests.conftest import FakeBackend


def saved(store, text="question", *, title="Existing Title"):
    conversation = Conversation(model="test-model", generated_title=title)
    conversation.add_user(text)
    conversation.add_assistant("answer")
    store.save(conversation)
    return conversation


def test_other_windows_changes_survive_a_local_write(store):
    local = saved(store)
    removed = saved(store, "remove")
    assert len(store.list()) == 2
    other = ConversationStore(store.directory)
    added = saved(other, "external")
    other.delete(removed.id)
    local.add_user("local update")
    store.save(local)
    assert {row.id for row in store.list()} == {local.id, added.id}


def test_in_place_external_edits_refresh_metadata(store):
    conversation = saved(store)
    store.list()
    path = store.path_for(conversation.id)
    data = json.loads(path.read_text())
    data["generated_title"] = "An externally changed title"
    path.write_text(json.dumps(data))
    assert store.list()[0].title == data["generated_title"]


def test_loading_does_not_retain_transcripts(store):
    original = saved(store, "large content " * 10000)
    loaded = store.load(original.id)
    reference = weakref.ref(loaded)
    del loaded
    gc.collect()
    assert reference() is None
    assert not hasattr(store, "_snapshots")
    assert len(original.storage_revision) == 64


def test_forget_cannot_be_undone_by_a_stale_window(store):
    conversation = saved(store)
    other = ConversationStore(store.directory)
    stale = other.load(conversation.id)
    store.delete(conversation.id)
    stale.add_user("late update")
    with pytest.raises(ConversationDeleted):
        other.save(stale)
    assert other.list() == []
    assert not other.path_for(stale.id).exists()


def test_startup_cleans_both_temp_naming_schemes(store):
    store.directory.mkdir(parents=True)
    old = store.directory / "chat.json.tmp"
    new = store.directory / ".chat.ab12_cd3.tmp"
    unrelated = store.directory / "unrelated.tmp"
    for path in (old, new, unrelated):
        path.write_text("leftover")
    store.list()
    assert not old.exists() and not new.exists()
    assert unrelated.exists()


def test_unsupported_directory_chmod_does_not_hide_history(store, monkeypatch):
    conversation = saved(store)
    other = ConversationStore(store.directory)

    def unsupported(*args):
        raise OSError(errno.ENOTSUP, "no POSIX mode support")

    monkeypatch.setattr("os.chmod", unsupported)
    assert other.list()[0].id == conversation.id
    saved(other, "another chat")
    assert len(other.list()) == 2


def test_empty_history_environment_falls_back_to_config(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("save_history = false\n")
    assert not load_settings(path, {"BEDROCK_SAVE_HISTORY": ""}).save_history


def _competing_writer(directory, identifier, barrier, queue):
    store = ConversationStore(Path(directory))
    conversation = store.load(identifier)
    conversation.add_user(str(multiprocessing.current_process().pid))
    barrier.wait(timeout=10)
    try:
        store.save(conversation)
        queue.put("saved")
    except StorageConflict:
        queue.put("conflict")


def test_processes_serialize_revision_checks(store):
    conversation = saved(store)
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    queue = context.Queue()
    processes = [context.Process(target=_competing_writer, args=(str(store.directory), conversation.id, barrier, queue)) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert sorted(queue.get(timeout=15) for _ in processes) == ["conflict", "saved"]
    finally:
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join()
    assert all(process.exitcode == 0 for process in processes)
    assert len(store.load(conversation.id).messages) == 3


async def test_reasoning_in_both_streamed_requests_is_hidden():
    class Reasoner:
        calls = 0

        async def stream_chat(self, model, messages):
            self.calls += 1
            for piece in ["<thi", "nk>", "reasoning " * 400, "</thi", "nk>"]:
                yield piece
            yield "YES" if self.calls == 1 else "Bee Pollination"

    conversation = Conversation()
    conversation.add_user("bees")
    conversation.add_assistant("insects")
    assert await generate_title(Reasoner(), "reasoner", conversation) == "Bee Pollination"
    assert clean_title("<think>unfinished reasoning") is None


async def test_long_chat_can_reveal_every_earlier_message(store):
    conversation = Conversation(model="test-model", generated_title="Long Chat")
    for number in range(60):
        conversation.add_user(f"question {number}")
        conversation.add_assistant(f"answer {number}")
    store.save(conversation)
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.query(UserMessage)) + len(app.query(AssistantMessage)) == 50
        assert "70 hidden" in str(app.query_one("#earlier-messages", Button).label)
        await app.load_earlier_messages()
        await pilot.pause()
        await app.load_earlier_messages()
        await pilot.pause()
        assert not app.query("#earlier-messages")
        assert [str(widget.content) for widget in app.query(UserMessage)] == [f"question {n}" for n in range(60)]


async def test_newer_chat_blocks_send_and_keeps_typed_text(store):
    original = saved(store)
    path = store.path_for(original.id)
    data = json.loads(path.read_text()) | {"version": 2, "attachments": ["kept"]}
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.send_message("keep my draft")
        assert app.query_one(Input).value == "keep my draft"
        assert not backend.requests and not backend.title_requests
        assert any("read-only" in notice.source for notice in app.query(SystemNotice))
        assert path.read_bytes() == before


async def test_competing_app_saves_create_complete_separate_branches(store):
    conversation = saved(store)
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await pilot.pause()
        other = ConversationStore(store.directory)
        external = other.load(conversation.id)
        external.add_user("other question")
        external.add_assistant("other answer")
        other.save(external)
        await app.send_message("my question")
        await app.workers.wait_for_complete()
        assert app.conversation.id != conversation.id
        assert [m.content for m in other.load(conversation.id).messages][-2:] == ["other question", "other answer"]
        ours = other.load(app.conversation.id)
        assert ours.messages[-2].content == "my question"
        assert ours.messages[-1].role.value == "assistant"


async def test_unopened_history_is_never_loaded_for_titles(store, monkeypatch):
    for number in range(1000):
        saved(store, f"question {number}", title=None)
    loaded = []
    original = store.load

    def record(identifier, **kwargs):
        loaded.append(identifier)
        return original(identifier, **kwargs)

    monkeypatch.setattr(store, "load", record)
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert loaded == [app.conversation.id]
        assert len(backend.title_requests) == 1
        assert len(store.list()) == 1000


async def test_mouse_scroll_survives_unfocused_sidebar_refresh(store):
    for number in range(45):
        saved(store, str(number))
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test(size=(100, 25)) as pilot:
        await app.workers.wait_for_complete()
        await pilot.pause()
        sidebar = app.query_one(ConversationList)
        app.query_one(Input).focus()
        sidebar.scroll_to(y=20, animate=False, immediate=True)
        await pilot.pause()
        before = sidebar.scroll_y
        assert before > 0 and not sidebar.has_focus
        await app.refresh_sidebar().wait()
        await pilot.pause()
        assert sidebar.scroll_y == before
        saved(ConversationStore(store.directory), "external addition")
        await app.refresh_sidebar().wait()
        await pilot.pause()
        assert sidebar.scroll_y == before


async def test_resumed_notice_cannot_create_a_markdown_link(store):
    saved(store, title="[click](https://example.com)")
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test() as pilot:
        await pilot.pause()
        notice = next(notice for notice in app.query(SystemNotice) if "Resumed" in notice.source)
        assert "\\[click\\]" in notice.source


async def test_sidebar_timer_finds_other_windows_updates(store):
    saved(store)
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test() as pilot:
        await app.workers.wait_for_complete()
        other = ConversationStore(store.directory)
        added = saved(other, "new elsewhere")
        await asyncio.sleep(2.2)
        await pilot.pause()
        sidebar = app.query_one(ConversationList)
        assert added.id in {option.id for option in sidebar.options}
        other.delete(added.id)
        await asyncio.sleep(2.2)
        await pilot.pause()
        assert added.id not in {option.id for option in sidebar.options}


async def test_forget_during_stream_cannot_be_undone_by_cleanup(store):
    backend = FakeBackend(reply="streaming reply with several words")
    backend.fragment_delay = 0.1
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await app.send_message("question")
        identifier = app.conversation.id
        await asyncio.sleep(0.15)
        assert app._streaming
        await app.command_forget()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not store.path_for(identifier).exists()
        assert not store.list()
