"""Queued prompts running through the real app, one at a time.

``test_prompt_queue.py`` covers the queue on its own. These tests drive the
Textual app: what actually gets sent to the model, in what order, what each
queued prompt can see of the answers before it, and what stops the queue.
"""

import asyncio

import pytest
from textual.widgets import Input
from textual.worker import WorkerCancelled

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.inference import InferenceError
from bedrock.models import Conversation
from bedrock.storage import ConversationStore
from bedrock.ui.prompt_queue import PromptQueuePanel
from bedrock.ui.widgets import AssistantMessage, SystemNotice, UserMessage

from tests.conftest import FakeBackend


class SlowBackend(FakeBackend):
    """Holds each reply open until the test lets it finish."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.fail = False

    async def stream_chat(self, model, messages):
        from bedrock.titles import is_title_request, is_topic_check

        if is_topic_check(messages) or is_title_request(messages):
            async for fragment in super().stream_chat(model, messages):
                yield fragment
            return
        self.requests.append((model, list(messages)))
        self.started.set()
        await self.release.wait()
        if self.fail:
            raise InferenceError("Request failed: connection refused.")
        yield self.reply


def make_app(store: ConversationStore, backend: FakeBackend) -> BedrockApp:
    return BedrockApp(settings=Settings(model="test-model"), backend=backend, store=store)


async def type_prompt(app: BedrockApp, pilot, text: str) -> None:
    """Type and send without waiting: the point is to submit while busy."""
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause()


async def settle(app: BedrockApp, pilot, rounds: int = 6) -> None:
    """Let the queue drain: each finished reply may start the next prompt."""
    async def wait_for_prompt_workers(timeout: float = 10) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while workers := [worker for worker in app.workers if worker.group in {"queue", "chat"}]:
            tasks = [asyncio.create_task(worker.wait()) for worker in workers]
            try:
                _, pending = await asyncio.wait(
                    tasks, timeout=max(0, deadline - asyncio.get_running_loop().time())
                )
                if pending:
                    raise TimeoutError("Queue workers did not settle before the test deadline")
                for task in tasks:
                    try:
                        task.result()
                    except WorkerCancelled:
                        pass
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0)

    for _ in range(rounds):
        await wait_for_prompt_workers()
        await pilot.pause()
        if app.queue_snapshot().is_idle and not app._active_prompt_workers():
            return
    await wait_for_prompt_workers()


def questions(backend: FakeBackend) -> list[str]:
    """The last user message of each request the model received."""
    return [
        next(m["content"] for m in reversed(messages) if m["role"] == "user")
        for _, messages in backend.requests
    ]


# ---- Ordering and one at a time ---------------------------------------------


async def test_prompts_sent_while_busy_run_in_order(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        await type_prompt(app, pilot, "third")

        # Only the first is with the model; the others are waiting.
        assert questions(backend) == ["first"]
        snapshot = app.queue_snapshot()
        assert snapshot.running.text == "first"
        assert [item.text for item in snapshot.pending] == ["second", "third"]

        backend.release.set()
        await settle(app, pilot)

    assert questions(backend) == ["first", "second", "third"]
    assert [m.content for m in app.conversation.messages if m.role.value == "user"] == ["first", "second", "third"]
    assert app.queue_snapshot().is_idle


async def test_rapid_submissions_keep_their_order(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        for number in range(6):
            await type_prompt(app, pilot, f"prompt {number}")
        backend.release.set()
        await settle(app, pilot, rounds=12)
    assert questions(backend) == [f"prompt {number}" for number in range(6)]


async def test_a_queued_prompt_sees_the_answers_that_arrived_before_it(store):
    backend = SlowBackend(reply="the answer is 42.")
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "what is the answer?")
        await backend.started.wait()
        await type_prompt(app, pilot, "why?")
        backend.release.set()
        await settle(app, pilot)

    _, second_request = backend.requests[1]
    assert [m["role"] for m in second_request][-3:] == ["user", "assistant", "user"]
    assert "the answer is 42." in second_request[-2]["content"]


async def test_queue_items_hold_no_transcript(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        waiting = app.prompt_queue.pending_for(app.conversation.id)[0]
        assert (waiting.text, waiting.model) == ("second", "test-model")
        assert not hasattr(waiting, "messages") and not hasattr(waiting, "conversation")
        backend.release.set()
        await settle(app, pilot)


# ---- Failure, cancellation, deletion ----------------------------------------


async def test_a_failed_reply_pauses_the_queue_and_keeps_the_rest(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        backend.fail = True  # the reply to "first" fails
        backend.release.set()
        await settle(app, pilot)

        snapshot = app.queue_snapshot()
        assert snapshot.paused is True
        assert [item.text for item in snapshot.pending] == ["second"]
        assert questions(backend) == ["first"]
        assert any("connection refused" in n.source for n in app.query(SystemNotice))

        # The employee decides: resuming sends what is still waiting.
        backend.fail = False
        app.resume_queue()
        await settle(app, pilot)
        assert len(backend.requests) == 2
        # "first" got no answer, so the model sees both questions in one turn.
        assert "second" in questions(backend)[1]
        assert app.queue_snapshot().is_idle


async def test_cancelling_a_reply_pauses_the_queue(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        app.action_cancel_stream()  # Esc
        await wait_for_app_workers(app, cancelled_groups=frozenset({"chat"}))
        await pilot.pause()

        snapshot = app.queue_snapshot()
        assert snapshot.paused is True
        assert [item.text for item in snapshot.pending] == ["second"]
        assert questions(backend) == ["first"]
        backend.release.set()


async def test_forget_drops_pending_prompts_for_the_chat_and_its_copies(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        forgotten = app.conversation.id
        copy_id = "a-linked-copy"
        app.prompt_queue.add("queued against a copy", copy_id, "test-model")
        app.conversation.family_id = copy_id  # the copy and the open chat are one family

        await type_prompt(app, pilot, "/forget")
        await wait_for_app_workers(app, cancelled_groups=frozenset({"chat"}))
        await pilot.pause()

        assert app.prompt_queue.pending_for(forgotten) == ()
        assert app.prompt_queue.pending_for(copy_id) == ()
        backend.release.set()
        await settle(app, pilot)
        assert forgotten not in app.prompt_queue.conversations()
        assert copy_id not in app.prompt_queue.conversations()
    assert questions(backend) == ["first"]


async def test_a_prompt_for_a_chat_deleted_elsewhere_is_not_sent(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        # Another window deletes the chat while the second prompt waits.
        ConversationStore(store.directory).delete(app.conversation.id)
        backend.release.set()
        await settle(app, pilot)

        # Saving the finished answer fails, which pauses the queue: the waiting
        # prompt is neither sent nor lost.
        assert questions(backend) == ["first"]
        snapshot = app.queue_snapshot()
        assert snapshot.paused is True and [item.text for item in snapshot.pending] == ["second"]

        # Deletion is re-checked at dispatch, so resuming still refuses to send.
        app.resume_queue()
        await settle(app, pilot)
        assert questions(backend) == ["first"]
        assert app.query_one("#prompt", Input).value == "second"
        assert any("deleted in another window" in n.source for n in app.query(SystemNotice))


# ---- Switching conversations -------------------------------------------------


async def test_switching_chats_pauses_the_queue_and_returning_needs_a_resume(store):
    other = Conversation(model="test-model", generated_title="Other Chat")
    other.add_user("older question")
    other.add_assistant("older answer")
    store.save(other)

    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_for_app_workers(app)
        await type_prompt(app, pilot, "/new")
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        mine = app.conversation.id

        backend.release.set()
        await app.open_conversation(other.id)
        await settle(app, pilot)

        assert app.prompt_queue.is_paused(mine)
        assert [item.text for item in app.prompt_queue.pending_for(mine)] == ["second"]
        assert questions(backend) == ["first"]
        # The other chat's panel shows its own empty queue, not this one's.
        assert app.queue_snapshot().is_idle

        await app.open_conversation(mine)
        await settle(app, pilot)
        assert questions(backend) == ["first"]  # still waiting: coming back is not a resume

        app.resume_queue()
        await settle(app, pilot)
        assert questions(backend) == ["first", "second"]


# ---- The interface the panel uses --------------------------------------------


async def test_the_panel_interface_removes_pauses_resumes_and_clears(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        for text in ("second", "third", "fourth"):
            await type_prompt(app, pilot, text)

        second = app.queue_snapshot().pending[0]
        assert app.remove_queued_prompt(second.id) is True
        assert app.remove_queued_prompt(second.id) is False
        assert [item.text for item in app.queue_snapshot().pending] == ["third", "fourth"]

        app.pause_queue()
        assert app.queue_snapshot().paused is True

        app.clear_queue()
        snapshot = app.queue_snapshot()
        assert snapshot.pending == () and snapshot.running.text == "first"  # running is untouched

        app.resume_queue()
        backend.release.set()
        await settle(app, pilot)
    assert questions(backend) == ["first"]


async def test_the_panel_shows_the_queue_and_its_buttons_act_on_it(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        panel = app.query_one(PromptQueuePanel)
        heading = panel.query_one(".queue-heading").render()
        assert "1 waiting" in str(heading)

        panel.post_message(PromptQueuePanel.PauseRequested())
        await pilot.pause()
        assert app.queue_snapshot().paused is True

        panel.post_message(PromptQueuePanel.ClearRequested())
        await pilot.pause()
        assert app.queue_snapshot().pending == ()

        panel.post_message(PromptQueuePanel.ResumeRequested())
        await pilot.pause()
        assert app.queue_snapshot().paused is False

        backend.release.set()
        await settle(app, pilot)


# ---- Completion events -------------------------------------------------------


async def test_a_repeated_completion_never_starts_two_prompts(store):
    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")
        await type_prompt(app, pilot, "third")

        running = app.prompt_queue.running
        backend.release.set()
        await wait_for_app_workers(app)
        await pilot.pause()

        # A late duplicate completion for the first prompt arrives.
        app.finish_prompt(running, ok=True)
        app.finish_prompt(running, ok=True)
        await settle(app, pilot)

    assert questions(backend) == ["first", "second", "third"]
    assert questions(backend).count("second") == 1


async def test_an_unrelated_completion_does_not_advance_the_queue(store):
    from bedrock.prompt_queue import PromptItem

    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "second")

        stranger = PromptItem(id="not-ours", text="elsewhere", conversation_id=app.conversation.id, model="test-model")
        app.finish_prompt(stranger, ok=True)
        await pilot.pause()

        assert questions(backend) == ["first"]
        assert app.prompt_queue.busy
        backend.release.set()
        await settle(app, pilot)
    assert questions(backend) == ["first", "second"]


# ---- Context is built when the prompt starts, not when it is typed ----------


async def test_a_queued_prompt_recalls_past_chats_when_it_starts(store):
    past = Conversation(model="test-model", generated_title="Kyoto Trip Plan")
    past.add_user("Plan a three day trip to Kyoto")
    past.add_assistant("Day 1: Fushimi Inari. Day 2: Arashiyama.")
    store.save(past)

    backend = SlowBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await wait_for_app_workers(app)
        await type_prompt(app, pilot, "/new")
        await type_prompt(app, pilot, "first")
        await backend.started.wait()
        await type_prompt(app, pilot, "do you remember the kyoto trip we planned?")

        # Nothing is searched while it waits: recall happens at dispatch.
        assert app.conversation.recalled_context is None

        backend.release.set()
        await settle(app, pilot)

        assert "Arashiyama" in backend.requests[-1][1][-1]["content"]
        assert any("Kyoto Trip Plan" in n.source for n in app.query(SystemNotice))
