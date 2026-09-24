"""Titles are background work: they wait for the chat and give way to it.

These need both halves of the laptop work - the chat turn going through the
scheduler (``stream_reply``) and titles going through it as ``BACKGROUND`` -
which is why they were written at the join.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import PROFILES, Settings
from bedrock.titles import is_title_request, is_topic_check

from tests.conftest import FakeBackend


class RecordingBackend(FakeBackend):
    """Logs when each call starts and ends, and how many run at once.

    ``hold_first_topic_check`` keeps the first YES/NO request open until it is
    cancelled, so a test can send a message while a title is in progress.
    """

    def __init__(self, *, hold_first_topic_check: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.events: list[str] = []
        self.running = 0
        self.most_at_once = 0
        self.hold_first_topic_check = hold_first_topic_check
        self.topic_check_started = asyncio.Event()

    async def stream_chat(self, model, messages, options=None, on_usage=None):
        kind = "topic" if is_topic_check(messages) else "title" if is_title_request(messages) else "chat"
        self.events.append(f"{kind}-start")
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        finished = False
        try:
            if kind == "topic" and self.hold_first_topic_check:
                self.hold_first_topic_check = False
                self.topic_check_started.set()
                await asyncio.Event().wait()  # until the scheduler cancels it
            async for fragment in super().stream_chat(model, messages, options, on_usage):
                yield fragment
            finished = True
        finally:
            self.running -= 1
            self.events.append(f"{kind}-{'end' if finished else 'closed'}")


def make_app(backend: FakeBackend, store, **settings) -> BedrockApp:
    return BedrockApp(settings=Settings(model="test-model", **settings), backend=backend, store=store)


async def test_a_title_in_progress_is_stopped_by_a_new_message_and_retried_after_its_reply(store):
    backend = RecordingBackend(hold_first_topic_check=True)
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"bees please", "enter")
        await asyncio.wait_for(backend.topic_check_started.wait(), 5)

        await pilot.press(*"tell me more", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

        assert backend.events == [
            "chat-start", "chat-end",
            "topic-start", "topic-closed",   # the new message took the server
            "chat-start", "chat-end",
            "topic-start", "topic-end", "title-start", "title-end",  # tried again after the reply
        ]
        assert backend.most_at_once == 1
        assert app.conversation.id not in app._title_failures
        assert app.conversation.generated_title == "Re bees please"


async def test_a_title_waits_while_a_queued_prompts_reply_streams(store):
    """With the real quiet period, the gap between two queued prompts is not
    "idle": the title starts only after the second reply has finished."""
    backend = RecordingBackend(reply="one two three four five")
    backend.fragment_delay = 0.02
    backend.hold_reply = asyncio.Event()  # the first reply waits until both are typed
    laptop = replace(PROFILES["laptop"], idle_seconds=0.3)
    app = make_app(backend, store, profile=laptop)
    async with app.run_test() as pilot:
        await pilot.press(*"bees please", "enter")
        await pilot.press(*"and wasps", "enter")
        assert [item.text for item in app.queue_snapshot().pending] == ["and wasps"]
        backend.hold_reply.set()
        await wait_for_app_workers(app)
        await pilot.pause()

        assert backend.events == [
            "chat-start", "chat-end",
            "chat-start", "chat-end",
            "topic-start", "topic-end", "title-start", "title-end",
        ]
        assert backend.most_at_once == 1
        assert app.conversation.generated_title == "Re bees please"


async def test_titles_ask_the_model_that_answered_not_the_one_selected_since(store):
    """After `/model` with a reply already given, a title must not make the
    server load the newly selected model."""
    backend = RecordingBackend(topic_reply="NO")  # stays untitled, so it is asked again
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"hi", "enter")
        await wait_for_app_workers(app)
        await pilot.press(*"/model other-model", "enter")
        await pilot.pause()
        assert app.model_name == "other-model"

        app.conversation.title_checked_at = 0  # as if the chat had grown
        app.write_missing_titles()
        await wait_for_app_workers(app)
        await pilot.pause()

        assert [model for model, _ in backend.topic_checks] == ["test-model", "test-model"]
        assert "other-model" not in {model for model, _ in backend.requests}
