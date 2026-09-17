"""'Stop & run next': cancel the running prompt and start the next one.

``PromptQueuePanel.SkipRequested`` is the shared contract with the queue
panel. These tests drive ``app.skip_running_prompt`` (and the message
handler that calls it) directly against the real app, the same way
``test_queue_execution.py`` and ``test_queue_recovery.py`` drive the rest of
the queue - fake backends, no live services.
"""

import asyncio
from types import SimpleNamespace

from textual.widgets import Input

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.storage import ConversationStore
from bedrock.ui.prompt_queue import PromptQueuePanel
from bedrock.ui.widgets import UserMessage

from tests.conftest import FakeBackend
from tests.test_queue_execution import SlowBackend, questions, settle, type_prompt


class PartialSlowBackend(FakeBackend):
    """Yields one fragment, then holds the rest open until released.

    Distinct from ``SlowBackend`` (which blocks before yielding anything):
    this is for asserting a partial reply survives being stopped.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.release = asyncio.Event()
        self.started = asyncio.Event()

    async def stream_chat(self, model, messages):
        from bedrock.titles import is_title_request, is_topic_check

        if is_topic_check(messages) or is_title_request(messages):
            async for fragment in super().stream_chat(model, messages):
                yield fragment
            return
        self.requests.append((model, list(messages)))
        yield "partial answer, "
        self.started.set()
        await self.release.wait()
        yield "the rest of it."


def make_app(store: ConversationStore, backend: FakeBackend) -> BedrockApp:
    return BedrockApp(settings=Settings(model="test-model"), backend=backend, store=store)


# ---- Streaming: partial replies are preserved, the prompt is not requeued --


async def test_skip_during_streaming_keeps_the_partial_reply_and_advances(store):
    backend = PartialSlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        snapshot = app.queue_snapshot()
        assert snapshot.can_skip is True

        # skip_running_prompt awaits its own cleanup, including starting the
        # next prompt; nothing more to wait for here.
        await app.skip_running_prompt()
        await pilot.pause()

        reply = next(m for m in app.conversation.messages if m.role.value == "assistant")
        assert "partial answer" in reply.content
        assert "(stopped)" in reply.content
        assert "the rest of it." not in reply.content

        # The stopped prompt is gone - not sitting at the front of the queue.
        assert "first" not in [m.text for m in app.queue_snapshot().pending]
        assert app.queue_snapshot().running.text == "second"

        backend.release.set()
        await settle(app, pilot)
    assert questions(backend) == ["first", "second"]


async def test_skip_with_nothing_waiting_does_nothing(store):
    """can_skip requires a waiting prompt; calling it anyway is a no-op."""
    backend = PartialSlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "only one")
        await backend.started.wait()

        assert app.queue_snapshot().can_skip is False
        await app.skip_running_prompt()
        await pilot.pause()

        # Still running - nothing was cancelled.
        assert app.queue_snapshot().running is not None
        assert questions(backend) == ["only one"]
        backend.release.set()
        await settle(app, pilot)


# ---- Preparation: stop retrying, but preserve the committed question -------


async def test_skip_during_recall_drops_the_prompt_and_advances(store, monkeypatch):
    backend = FakeBackend()
    app = make_app(store, backend)
    started, release = asyncio.Event(), asyncio.Event()

    async def recall(*args):
        started.set()
        await release.wait()
        return None

    monkeypatch.setattr(app, "recall_into", recall)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "do you remember when we discussed biology?"
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 2)
        await type_prompt(app, pilot, "second")

        assert app.queue_snapshot().can_skip is True
        await app.skip_running_prompt()
        release.set()  # let the cancelled recall coroutine unwind
        await wait_for_app_workers(app, cancelled_groups=frozenset({"queue", "chat"}))
        await settle(app, pilot)

    # The committed question remains visible and saved, but is not retried.
    texts = [m.content for m in app.conversation.messages]
    assert texts.count("do you remember when we discussed biology?") == 1
    assert [m.content for m in store.load(app.conversation.id).messages] == texts
    # Adjacent user turns are combined by Conversation.to_api; neither is lost.
    assert questions(backend) == ["do you remember when we discussed biology?\n\nsecond"]


async def test_esc_recall_retry_reuses_question_without_duplicate_bubbles(store, monkeypatch):
    backend = FakeBackend()
    app = make_app(store, backend)
    started, release = asyncio.Event(), asyncio.Event()

    async def recall(*args):
        started.set()
        await release.wait()

    monkeypatch.setattr(app, "recall_into", recall)
    text = "do you remember when we discussed biology?"
    async with app.run_test() as pilot:
        await type_prompt(app, pilot, text)
        for _ in range(3):
            await asyncio.wait_for(started.wait(), 2)
            app.action_cancel_stream()
            await wait_for_app_workers(app, cancelled_groups=frozenset({"queue", "chat"}))
            assert [str(w.content) for w in app.query(UserMessage)] == [text]
            assert [m.content for m in app.conversation.messages] == [text]
            assert [m.content for m in store.load(app.conversation.id).messages] == [text]
            started.clear()
            app.resume_queue()
        release.set()
        await settle(app, pilot)
        assert questions(backend) == [text]
        assert [str(w.content) for w in app.query(UserMessage)] == [text]
        assert not app._prepared_questions


async def test_skipped_recall_is_preserved_even_if_next_save_fails(store, monkeypatch):
    app = make_app(store, FakeBackend())
    started = asyncio.Event()

    async def recall(*args):
        started.set()
        await asyncio.Future()

    monkeypatch.setattr(app, "recall_into", recall)
    text = "do you remember when we discussed biology?"
    async with app.run_test() as pilot:
        await type_prompt(app, pilot, text)
        await asyncio.wait_for(started.wait(), 2)
        await type_prompt(app, pilot, "next")
        monkeypatch.setattr(app, "save_conversation", lambda *a, **kw: False)
        await app.skip_running_prompt()
        await settle(app, pilot)
        assert [str(w.content) for w in app.query(UserMessage)] == [text]
        assert [m.content for m in app.conversation.messages] == [text]
        assert [m.content for m in store.load(app.conversation.id).messages] == [text]
        assert app.query_one(Input).value == "next"


async def test_skip_stranded_item_uses_recovery_instead_of_discarding(store, monkeypatch):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        app.prompt_queue.add("stranded", app.conversation.id, "test-model")
        app.prompt_queue.start_next(app.conversation.id)
        app.prompt_queue.add("next", app.conversation.id, "test-model")
        notices = []
        monkeypatch.setattr(app, "notify", lambda text, **kwargs: notices.append(text))
        assert not app.queue_snapshot().can_skip
        await app.skip_running_prompt()
        assert [p.text for p in app.queue_snapshot().pending] == ["stranded", "next"]
        assert app.queue_snapshot().paused
        assert not app.prompt_queue.busy
        assert any("stopped unexpectedly" in text for text in notices)


async def test_skip_cleanup_failure_reports_and_does_not_advance(store, monkeypatch):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        app.prompt_queue.add("first", app.conversation.id, "test-model")
        app.prompt_queue.start_next(app.conversation.id)
        app.prompt_queue.add("next", app.conversation.id, "test-model")

        async def failed_wait():
            raise RuntimeError("cleanup save failed")

        monkeypatch.setattr(app, "_active_prompt_workers", lambda: [SimpleNamespace(wait=failed_wait)])
        notices = []
        monkeypatch.setattr(app, "notify", lambda text, **kwargs: notices.append(text))
        await app.skip_running_prompt()
        assert app.queue_snapshot().paused
        assert [p.text for p in app.queue_snapshot().pending] == ["next"]
        assert not app.backend.requests
        assert any("cleanup save failed" in text for text in notices)
        assert not app._skipping and not app._skip_no_requeue


async def test_failed_preparation_rolls_back_only_its_own_question(store, monkeypatch):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)

        def failed_save(*args, **kwargs):
            app.conversation.add_user("independent change")
            raise RuntimeError("save failed before committing")

        with monkeypatch.context() as patch:
            patch.setattr(app, "save_conversation", failed_save)
            await app.send_message("attempted question")
            await wait_for_app_workers(app)
        assert [m.content for m in app.conversation.messages] == ["independent change"]
        assert [p.text for p in app.queue_snapshot().pending] == ["attempted question"]


# ---- A paused queue: Skip explicitly resumes it ----------------------------


async def test_skip_resumes_an_explicitly_paused_queue(store):
    backend = PartialSlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        app.pause_queue()
        assert app.queue_snapshot().paused is True

        await app.skip_running_prompt()
        await pilot.pause()

        assert app.queue_snapshot().paused is False
        assert app.queue_snapshot().running.text == "second"
        backend.release.set()
        await settle(app, pilot)
    assert questions(backend) == ["first", "second"]


# ---- Guards: double clicks, natural-completion races -----------------------


async def test_double_click_only_skips_once(store):
    backend = SlowToCancelBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        await type_prompt(app, pilot, "third")

        first_skip = asyncio.ensure_future(app.skip_running_prompt())
        await pilot.pause()
        assert not first_skip.done()  # cleanup genuinely still in flight
        # A second click lands while the first is still cleaning up.
        await app.skip_running_prompt()
        backend.allow_cancel_to_finish.set()
        await first_skip
        await pilot.pause()

        # Exactly one prompt was skipped: "second" is now running, "third"
        # is still waiting - the second click did not also stop "second".
        assert app.queue_snapshot().running.text == "second"
        assert [item.text for item in app.queue_snapshot().pending] == ["third"]


class SequencedBackend(FakeBackend):
    """A fresh started/release pair per call, to catch a specific request."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.calls: list[tuple[asyncio.Event, asyncio.Event]] = []

    async def stream_chat(self, model, messages):
        from bedrock.titles import is_title_request, is_topic_check

        if is_topic_check(messages) or is_title_request(messages):
            async for fragment in super().stream_chat(model, messages):
                yield fragment
            return
        self.requests.append((model, list(messages)))
        started, release = asyncio.Event(), asyncio.Event()
        self.calls.append((started, release))
        started.set()
        await release.wait()
        yield self.reply


async def test_a_natural_completion_racing_the_click_is_not_disturbed(store):
    """The reply finishes on its own just before the cancellation lands."""
    backend = SequencedBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.calls[0][0].wait()
        await type_prompt(app, pilot, "second")

        backend.calls[0][1].set()  # "first" finishes; "second" starts on its own
        while len(backend.calls) < 2:
            await pilot.pause()

        # Nothing left to skip: "second" already took over on its own, with
        # nothing waiting behind it.
        assert app.queue_snapshot().can_skip is False
        await app.skip_running_prompt()
        await pilot.pause()
        assert app.prompt_queue.running is not None  # "second" was not touched

        backend.calls[1][1].set()
        await settle(app, pilot)
    assert questions(backend) == ["first", "second"]
    assert app.queue_snapshot().is_idle


# ---- Switching chats and /forget during cleanup ----------------------------


class SlowToCancelBackend(FakeBackend):
    """Never finishes on its own, and takes an extra held-open step to
    actually unwind once cancelled - a real window to act while a skip's
    cleanup is still in flight, rather than hoping to land inside it."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.started = asyncio.Event()
        self.allow_cancel_to_finish = asyncio.Event()

    async def stream_chat(self, model, messages):
        from bedrock.titles import is_title_request, is_topic_check

        if is_topic_check(messages) or is_title_request(messages):
            async for fragment in super().stream_chat(model, messages):
                yield fragment
            return
        self.requests.append((model, list(messages)))
        try:
            self.started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            await self.allow_cancel_to_finish.wait()
            raise
        yield self.reply  # pragma: no cover - never reached


async def test_switching_chats_during_cleanup_does_not_resume_the_old_one(store):
    backend = SlowToCancelBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        left_conversation = app.conversation.id

        skip = asyncio.ensure_future(app.skip_running_prompt())
        await pilot.pause()
        assert not skip.done()  # cleanup is genuinely still in flight

        await app.start_new_conversation()  # leave mid-cleanup
        backend.allow_cancel_to_finish.set()
        await skip
        await pilot.pause()

        # The chat left behind stays paused - nothing runs unwatched in it.
        # "first" was stopped and dropped; "second" is still waiting, untouched.
        assert app.prompt_queue.is_paused(left_conversation) is True
        assert [item.text for item in app.prompt_queue.pending_for(left_conversation)] == ["second"]


async def test_forget_during_cleanup_does_not_resurrect_the_stopped_prompt(store):
    backend = SlowToCancelBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        forgotten = app.conversation.id

        skip = asyncio.ensure_future(app.skip_running_prompt())
        await pilot.pause()
        assert not skip.done()

        await type_prompt(app, pilot, "/forget")
        backend.allow_cancel_to_finish.set()
        await skip
        await pilot.pause()

        assert forgotten not in app.prompt_queue.conversations()
        assert app.queue_snapshot().pending == ()


# ---- Ordinary Esc is unchanged ----------------------------------------------


async def test_escape_still_pauses_instead_of_advancing(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        app.action_cancel_stream()  # Esc, not Skip
        await wait_for_app_workers(app, cancelled_groups=frozenset({"chat"}))
        await pilot.pause()

        snapshot = app.queue_snapshot()
        assert snapshot.paused is True
        assert [item.text for item in snapshot.pending] == ["second"]
        assert questions(backend) == ["first"]
        backend.release.set()


# ---- Regressions: slow cleanup must not freeze the app, and a delayed ------
# ---- click must not land on a different prompt -----------------------------


async def test_stuck_cleanup_does_not_block_other_app_messages(store):
    """A SkipRequested handler that awaits cleanup inline blocks Textual's
    single App-level message queue for as long as cleanup takes - with no
    timeout, that is indefinite. Reproduced by holding cleanup open forever
    and confirming an unrelated message posted right after is still
    processed promptly, instead of queuing up behind the stuck handler."""
    backend = SlowToCancelBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        conversation_id = app.conversation.id

        running = app.queue_snapshot().running
        app.post_message(PromptQueuePanel.SkipRequested(running.id, conversation_id))
        # Give the handler a few ticks to actually start (and get stuck).
        for _ in range(5):
            await asyncio.sleep(0)

        # Posted right after, while cleanup is deliberately stuck forever.
        app.post_message(PromptQueuePanel.ClearRequested())

        async def cleared() -> None:
            while app.prompt_queue.pending_for(conversation_id):
                await asyncio.sleep(0.01)

        try:
            await asyncio.wait_for(cleared(), timeout=2.0)
        finally:
            backend.allow_cancel_to_finish.set()
            for _ in range(10):
                await asyncio.sleep(0)


async def test_delayed_click_does_not_stop_a_different_prompt(store):
    """The message a click posts is only handled once earlier messages in
    the app's queue have drained. If "first" finishes naturally and "second"
    starts before a delayed SkipRequested (still naming "first") is finally
    handled, it must be ignored rather than stopping "second" instead."""
    backend = SequencedBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.calls[0][0].wait()
        await type_prompt(app, pilot, "second")
        await type_prompt(app, pilot, "third")
        conversation_id = app.conversation.id

        # What the panel displayed - and what a click now would name - while
        # "first" was still running.
        first_id = app.queue_snapshot().running.id

        # "first" finishes naturally; "second" starts on its own before the
        # (simulated) delayed click is ever handled.
        backend.calls[0][1].set()
        while len(backend.calls) < 2:
            await pilot.pause()
        assert app.queue_snapshot().running.text == "second"

        # The delayed request still names "first".
        await app.skip_running_prompt(first_id, conversation_id)
        await pilot.pause()

        assert app.queue_snapshot().running.text == "second"  # untouched
        assert [item.text for item in app.queue_snapshot().pending] == ["third"]

        backend.calls[1][1].set()
        while len(backend.calls) < 3:
            await pilot.pause()
        backend.calls[2][1].set()
        await settle(app, pilot)
    assert questions(backend) == ["first", "second", "third"]


async def test_a_request_naming_a_different_conversation_is_ignored(store):
    """The identity check covers conversation_id too, not just item_id -
    a request queued behind a chat switch must not act on whatever this
    chat's queue happens to be running now."""
    backend = PartialSlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        running = app.queue_snapshot().running

        await app.skip_running_prompt(running.id, "not-the-open-conversation")
        await pilot.pause()

        assert app.queue_snapshot().running.text == "first"  # untouched
        backend.release.set()
        await settle(app, pilot)
    assert questions(backend) == ["first", "second"]
