"""`/digest` driven through the real Textual app, same as the other app tests."""

from textual.widgets import Input

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.ui.widgets import SystemNotice

from tests.conftest import FakeBackend

CSV = "\n".join(
    ["sample,ph,note"] + [f"S{i:04d},{6.0 + i % 20 * 0.1:.1f},routine" for i in range(200)]
)


def make_app(tmp_path, backend: FakeBackend) -> BedrockApp:
    return BedrockApp(
        settings=Settings(model="test-model", context_dir=str(tmp_path / "context")),
        backend=backend,
    )


def notices(app: BedrockApp) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def submit(app: BedrockApp, pilot, text: str) -> None:
    """Type a command and run it to completion.

    The text is set on the input rather than pressed key by key: these
    commands carry long temporary file paths, and eighty individual key events
    per test is most of the suite's runtime for no extra coverage. Submitting
    still goes through the real ``Input.Submitted`` handler.
    """
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await wait_for_app_workers(app)
    await pilot.pause()


async def test_digest_a_file_reports_both_savings(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")

        text = notices(app)
        assert "Compressed `assay.csv`" in text
        assert "smaller" in text and "cheaper to read" in text
        assert len(app.context_store.list()) == 1


async def test_digest_with_no_arguments_lists_what_is_stored(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        await submit(app, pilot, f"/digest")

        assert "Stored context" in notices(app)
        assert "assay.csv" in notices(app)


async def test_digest_search_returns_the_matching_content(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV.replace("S0100,6.0,routine", "S0100,9.9,CONTAMINATION found"))
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        await submit(app, pilot, f"/digest search contamination")

        assert "CONTAMINATION found" in notices(app)


async def test_digest_show_and_forget(tmp_path, fake_backend: FakeBackend):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path, fake_backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        artifact_id = app.context_store.list()[0].id

        await submit(app, pilot, f"/digest show {artifact_id}")
        assert "rows x 3 columns" in notices(app)

        await submit(app, pilot, f"/digest forget {artifact_id}")
        assert "Forgot `assay.csv`" in notices(app)
        assert app.context_store.list() == []


async def test_a_missing_file_is_a_plain_message_not_a_crash(tmp_path, fake_backend: FakeBackend):
    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest /no/such/file.csv")

        assert "No file at" in notices(app)
        assert app.context_store.list() == []


async def test_prose_gets_the_model_summary_on_top_of_the_computed_one(tmp_path):
    """The one case the model earns its keep: prose, where rules have little
    to say. The backend is a fake, so no server is involved."""
    path = tmp_path / "protocol.md"
    path.write_text("# Protocol\n\n" + "Samples rest at ambient temperature before loading. " * 20)
    backend = FakeBackend(reply="GIST: How to handle samples before an assay.\n- Rest at ambient first")
    app = make_app(tmp_path, backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")

        assert "How to handle samples before an assay." in notices(app)
        stored = app.context_store.list()[0]
        assert stored.digest.generated_by == "test-model"
        # Sent through the app's scheduler with the digest's limits.
        from bedrock.context.digest import DIGEST_REPLY_TOKENS
        from bedrock.inference import RequestOptions

        assert backend.request_options == [RequestOptions(max_tokens=DIGEST_REPLY_TOKENS, reasoning=False)]


async def test_structured_data_never_calls_the_model(tmp_path):
    """A table's computed schema is better than a summary of it, and asking a
    model anyway would spend time to make the digest worse."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    backend = FakeBackend(reply="GIST: nonsense\n- nonsense")
    app = make_app(tmp_path, backend)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")

        assert app.context_store.list()[0].digest.generated_by == "rules"
        assert backend.requests == []


# ---- Switching automatic context off in this mode ---------------------------
#
# This mode adds matching stored content to a message's one-turn system note
# (see test_app.py). The switch must stop that, here as in agent-pack mode.

QUESTION = "what was the ph in the assay results?"


async def stored_assay(app: BedrockApp, pilot, tmp_path) -> None:
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    await submit(app, pilot, f"/digest {path}")


def sent_system_text(backend: FakeBackend, index: int = -1) -> str:
    _, messages = backend.requests[index]
    return "\n".join(m["content"] for m in messages if m["role"] == "system")


def sent_question(backend: FakeBackend, index: int = -1) -> str:
    """The newest user message, where this turn's passages go (decision D3)."""
    _, messages = backend.requests[index]
    return messages[-1]["content"]


def passages_sent(backend: FakeBackend, index: int = -1) -> bool:
    """Passages were added to the question, and never to the system message."""
    assert "Relevant stored context" not in sent_system_text(backend, index)
    return "Relevant stored context" in sent_question(backend, index)


async def test_injection_notice_says_how_to_stop_it(tmp_path, fake_backend: FakeBackend):
    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        await stored_assay(app, pilot, tmp_path)
        await submit(app, pilot, QUESTION)

        assert passages_sent(fake_backend)
        added = [n for n in notices(app).split("\n\n") if "Added stored context from" in n]
        assert added and "`/digest auto off`" in added[0]


async def test_auto_context_setting_off_sends_nothing_extra(tmp_path, fake_backend: FakeBackend, monkeypatch):
    app = BedrockApp(
        settings=Settings(model="test-model", context_dir=str(tmp_path / "context"), auto_context=False),
        backend=fake_backend,
    )
    async with app.run_test() as pilot:
        await stored_assay(app, pilot, tmp_path)
        searched = []
        monkeypatch.setattr(app.context_store, "search", lambda *a, **k: searched.append(a) or [])
        await submit(app, pilot, QUESTION)

        _, messages = fake_backend.requests[-1]
        assert messages[-1] == {"role": "user", "content": QUESTION}
        assert all("Relevant stored context" not in m["content"] for m in messages)
        assert "Added stored context" not in notices(app)
        # Off means the store is not consulted at all.
        assert searched == []


async def test_digest_auto_off_then_on_changes_what_this_window_sends(tmp_path, fake_backend: FakeBackend):
    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        await stored_assay(app, pilot, tmp_path)

        await submit(app, pilot, "/digest auto off")
        assert app.auto_context.enabled is False
        last = str(app.query(SystemNotice)[-1].source)
        assert "now **off**" in last and "BEDROCK_AUTO_CONTEXT=0" in last

        await submit(app, pilot, QUESTION)
        assert not passages_sent(fake_backend)

        await submit(app, pilot, "/digest auto on")
        assert app.auto_context.enabled is True
        await submit(app, pilot, QUESTION)
        assert passages_sent(fake_backend)

        # Neither the switch nor the context was ever read as a file or saved.
        assert len(app.context_store.list()) == 1
        assert all("Relevant stored context" not in m.content for m in app.conversation.messages)


async def test_digest_auto_off_typed_while_a_prompt_waits_applies_to_that_prompt(tmp_path):
    """The switch is read when a prompt starts, not when it was typed."""
    from tests.test_queue_execution import SlowBackend, settle, type_prompt

    backend = SlowBackend()
    app = make_app(tmp_path, backend)
    second = "and what was the ph of sample S0003?"
    async with app.run_test() as pilot:
        await stored_assay(app, pilot, tmp_path)

        await type_prompt(app, pilot, QUESTION)
        await backend.started.wait()
        await type_prompt(app, pilot, second)
        # The first is being answered; the second waits its turn.
        assert [item.text for item in app.queue_snapshot().pending] == [second]

        # Commands run straight away, even while a prompt waits.
        await type_prompt(app, pilot, "/digest auto off")
        assert app.auto_context.enabled is False

        backend.release.set()
        await settle(app, pilot)

        assert len(backend.requests) == 2
        assert passages_sent(backend, 0)
        _, messages = backend.requests[1]
        assert messages[-1] == {"role": "user", "content": second}
        assert not passages_sent(backend, 1)


async def test_digest_auto_reports_the_setting_and_refuses_other_words(tmp_path, fake_backend: FakeBackend):
    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/digest auto")
        assert "is **on**" in str(app.query(SystemNotice)[-1].source)

        await submit(app, pilot, "/digest auto maybe")
        refusal = app.query(SystemNotice)[-1]
        assert refusal.has_class("error")
        assert "takes `on` or `off`" in str(refusal.source)
        assert app.auto_context.enabled is True

        # None of this was read as a file path to compress, or sent to the model.
        assert app.context_store.list() == []
        assert fake_backend.requests == []


async def test_help_texts_offer_the_switch_in_this_mode(tmp_path, fake_backend: FakeBackend):
    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/help")
        assert "/digest auto off" in str(app.query(SystemNotice)[-1].source)

        await submit(app, pilot, "/digest help")
        assert "`/digest auto off` stops it for this window" in str(app.query(SystemNotice)[-1].source)


async def test_digest_search_runs_off_the_ui_thread(tmp_path, fake_backend: FakeBackend, monkeypatch):
    import threading

    app = make_app(tmp_path, fake_backend)
    async with app.run_test() as pilot:
        threads = []
        real_search = app.context_store.search

        def search(*args, **kwargs):
            threads.append(threading.current_thread())
            return real_search(*args, **kwargs)

        monkeypatch.setattr(app.context_store, "search", search)
        await submit(app, pilot, "/digest search stability")

        assert threads and threads[0] is not threading.main_thread()
