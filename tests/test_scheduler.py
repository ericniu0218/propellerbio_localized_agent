"""The scheduler: the one door every model call goes through.

Driven directly with ``FakeBackend``, no app and no server.
"""

from __future__ import annotations

import pytest

from bedrock.config import PROFILES
from bedrock.inference import Preempted, Priority, RequestOptions, Scheduler, Usage
from tests.conftest import FakeBackend

CHAT = [{"role": "user", "content": "hello there"}]


def scheduler_for(backend: FakeBackend, profile: str = "laptop") -> Scheduler:
    return Scheduler(backend, PROFILES[profile])


async def test_a_chat_call_is_forwarded_with_its_options():
    backend = FakeBackend(reply="one two")
    scheduler = scheduler_for(backend)
    options = RequestOptions(max_tokens=64)

    reply = await scheduler.complete("test-model", CHAT, priority=Priority.CHAT, options=options)

    assert reply == "one two "
    assert backend.requests == [("test-model", CHAT)]
    assert backend.request_options == [options]


async def test_chat_usage_is_kept_and_the_answering_model_is_remembered():
    backend = FakeBackend()
    backend.usage = Usage(prompt_tokens=12, completion_tokens=5, cached_prompt_tokens=8)
    scheduler = scheduler_for(backend)
    assert scheduler.last_usage is None and scheduler.loaded_model is None

    await scheduler.complete("test-model", CHAT, priority=Priority.CHAT)

    assert scheduler.last_usage == backend.usage
    assert scheduler.loaded_model == "test-model"


async def test_a_job_does_not_change_the_chat_usage_or_loaded_model():
    backend = FakeBackend()
    backend.usage = Usage(prompt_tokens=99)
    scheduler = scheduler_for(backend)

    await scheduler.complete("other-model", CHAT, priority=Priority.USER_JOB)

    assert scheduler.last_usage is None
    assert scheduler.loaded_model is None


async def test_background_without_a_model_uses_the_one_that_last_answered():
    backend = FakeBackend()
    scheduler = scheduler_for(backend)
    await scheduler.complete("other-model", CHAT, priority=Priority.CHAT)

    await scheduler.complete(None, [{"role": "user", "content": "hi"}], priority=Priority.BACKGROUND)

    assert [model for model, _ in backend.requests] == ["other-model", "other-model"]


async def test_background_before_any_chat_is_preempted_rather_than_loading_a_model():
    backend = FakeBackend()
    scheduler = scheduler_for(backend)

    with pytest.raises(Preempted):
        await scheduler.complete(None, CHAT, priority=Priority.BACKGROUND)
    assert backend.requests == []


async def test_only_background_calls_may_leave_the_model_out():
    scheduler = scheduler_for(FakeBackend())
    with pytest.raises(ValueError):
        await scheduler.complete(None, CHAT, priority=Priority.CHAT)


async def test_a_backend_with_the_old_two_argument_interface_still_works():
    class OldBackend:
        async def stream_chat(self, model, messages):
            yield "still "
            yield "works"

    scheduler = Scheduler(OldBackend(), PROFILES["laptop"])
    assert await scheduler.complete("m", CHAT, priority=Priority.USER_JOB, options=RequestOptions(max_tokens=5)) == "still works"


# ---------------------------------------------------------------------------
# Queueing, preemption and idle waiting
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
from dataclasses import replace  # noqa: E402


class GatedBackend:
    """Every call starts, says so, then waits until the test releases it.

    ``started`` lists call names in the order the server began them;
    ``closed`` lists those whose stream was closed before finishing.
    """

    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.closed: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}
        self.models: dict[str, str] = {}
        self.running = 0
        self.most_at_once = 0

    def gate(self, name: str) -> asyncio.Event:
        return self.gates.setdefault(name, asyncio.Event())

    async def stream_chat(self, model, messages, options=None, on_usage=None):
        name = messages[0]["content"]
        self.started.append(name)
        self.models[name] = model
        self.running += 1
        self.most_at_once = max(self.most_at_once, self.running)
        done = False
        try:
            yield f"{name}:"
            await self.gate(name).wait()
            yield "done"
            done = True
            self.finished.append(name)
        finally:
            self.running -= 1
            if not done:
                self.closed.append(name)


def ask(name: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": name}]


async def settle() -> None:
    """Let every task that can run, run: a hand-over takes several loop turns."""
    for _ in range(20):
        await asyncio.sleep(0)


def laptop(**changes):
    return replace(PROFILES["laptop"], **({"idle_seconds": 0.0} | changes))


async def test_one_call_at_a_time_on_a_laptop_and_most_urgent_goes_next():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    job = asyncio.create_task(scheduler.complete("m", ask("job-1"), priority=Priority.USER_JOB))
    await settle()
    later_job = asyncio.create_task(scheduler.complete("m", ask("job-2"), priority=Priority.USER_JOB))
    chat = asyncio.create_task(scheduler.complete("m", ask("chat"), priority=Priority.CHAT))
    await settle()
    assert backend.started == ["job-1"]  # the chat waits; a running job is not cancelled

    backend.gate("job-1").set()
    await settle()
    assert backend.started == ["job-1", "chat"]  # chat jumps ahead of the waiting job

    backend.gate("chat").set()
    backend.gate("job-2").set()
    assert await job == "job-1:done"
    assert await chat == "chat:done"
    assert await later_job == "job-2:done"
    assert backend.started == ["job-1", "chat", "job-2"]
    assert backend.most_at_once == 1
    assert not scheduler.busy


async def test_same_priority_is_first_come_first_served():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    tasks = [asyncio.create_task(scheduler.complete("m", ask(f"job-{i}"), priority=Priority.USER_JOB)) for i in range(3)]
    await settle()
    for i in range(3):
        backend.gate(f"job-{i}").set()
    await asyncio.gather(*tasks)
    assert backend.started == ["job-0", "job-1", "job-2"]


async def test_a_chat_preempts_running_background_work_and_closes_its_stream():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    await _answer_a_chat(scheduler, backend)

    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await settle()
    assert backend.started[-1] == "title"

    chat = asyncio.create_task(scheduler.complete("m", ask("chat-2"), priority=Priority.CHAT))
    with pytest.raises(Preempted):
        await title
    await settle()
    assert "title" in backend.closed  # the HTTP stream was closed, not left running
    assert backend.started[-1] == "chat-2"

    backend.gate("chat-2").set()
    assert await chat == "chat-2:done"
    assert backend.most_at_once == 1


async def test_background_waits_while_anything_else_runs_or_waits():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    await _answer_a_chat(scheduler, backend)
    job = asyncio.create_task(scheduler.complete("m", ask("job"), priority=Priority.USER_JOB))
    await settle()
    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await settle()
    assert "title" not in backend.started

    backend.gate("job").set()
    await job
    await settle()
    assert backend.started[-1] == "title"
    backend.gate("title").set()
    assert await title == "title:done"


async def test_background_waits_for_the_quiet_period_after_a_chat():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop(idle_seconds=0.2))
    await _answer_a_chat(scheduler, backend)
    backend.gate("title").set()

    loop = asyncio.get_running_loop()
    began = loop.time()
    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await asyncio.sleep(0.05)
    assert "title" not in backend.started
    assert await title == "title:done"
    assert loop.time() - began >= 0.19


async def test_a_new_chat_during_the_quiet_period_keeps_background_waiting():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop(idle_seconds=0.15))
    await _answer_a_chat(scheduler, backend)
    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await asyncio.sleep(0.1)
    backend.gate("chat-2").set()
    await scheduler.complete("m", ask("chat-2"), priority=Priority.CHAT)
    await asyncio.sleep(0.1)
    assert "title" not in backend.started  # the quiet period restarted with the new reply
    backend.gate("title").set()
    await title
    assert backend.started == ["chat", "chat-2", "title"]


async def test_a_user_job_keeps_background_waiting_in_the_gap_after_it():
    """A literature review is hundreds of ``USER_JOB`` calls with short gaps
    between them, and those gaps are not idle time. Counting only ``CHAT``
    meant a title slotted in between two review calls; on a laptop holding one
    model that unloads the review's model and loads the chat's, gigabytes each
    way, which is exactly what the idle rule exists to prevent."""
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop(idle_seconds=0.15))
    await _answer_a_chat(scheduler, backend)
    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await asyncio.sleep(0.1)

    backend.gate("review-step").set()
    await scheduler.complete("m", ask("review-step"), priority=Priority.USER_JOB)
    await asyncio.sleep(0.1)
    assert "title" not in backend.started  # the quiet period restarted with the review's call

    backend.gate("title").set()
    await title
    assert backend.started == ["chat", "review-step", "title"]


async def test_background_follows_the_model_that_last_answered_a_chat():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    await _answer_a_chat(scheduler, backend, model="first")
    backend.gate("chat-2").set()
    await scheduler.complete("second", ask("chat-2"), priority=Priority.CHAT)
    backend.gate("title").set()

    await scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND)

    assert backend.models["title"] == "second"
    assert scheduler.loaded_model == "second"


async def test_the_loaded_model_is_decided_when_background_work_starts_not_when_asked():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    await _answer_a_chat(scheduler, backend, model="first")
    chat = asyncio.create_task(scheduler.complete("second", ask("chat-2"), priority=Priority.CHAT))
    await settle()
    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await settle()
    backend.gate("chat-2").set()
    backend.gate("title").set()
    await chat
    await title
    assert backend.models["title"] == "second"


async def test_a_concurrent_server_runs_background_alongside_and_does_not_preempt():
    backend = GatedBackend()
    scheduler = Scheduler(backend, PROFILES["server"])
    await _answer_a_chat(scheduler, backend)
    title = asyncio.create_task(scheduler.complete(None, ask("title"), priority=Priority.BACKGROUND))
    await settle()
    chat = asyncio.create_task(scheduler.complete("m", ask("chat-2"), priority=Priority.CHAT))
    await settle()
    assert backend.running == 2
    backend.gate("title").set()
    backend.gate("chat-2").set()
    assert await title == "title:done"
    assert await chat == "chat-2:done"


async def test_stopping_early_closes_the_stream_and_frees_the_turn():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    stream = scheduler.stream("m", ask("job"), priority=Priority.USER_JOB)
    assert await stream.__anext__() == "job:"
    await stream.aclose()
    await settle()
    assert backend.closed == ["job"]
    assert not scheduler.busy


async def test_cancelling_the_caller_closes_the_stream_and_the_next_call_runs():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    job = asyncio.create_task(scheduler.complete("m", ask("job"), priority=Priority.USER_JOB))
    waiting = asyncio.create_task(scheduler.complete("m", ask("next"), priority=Priority.USER_JOB))
    await settle()
    job.cancel()
    with pytest.raises(asyncio.CancelledError):
        await job
    await settle()
    assert backend.closed == ["job"]
    assert backend.started == ["job", "next"]
    backend.gate("next").set()
    assert await waiting == "next:done"


async def test_a_caller_cancelled_while_waiting_leaves_the_line():
    backend = GatedBackend()
    scheduler = Scheduler(backend, laptop())
    job = asyncio.create_task(scheduler.complete("m", ask("job"), priority=Priority.USER_JOB))
    waiting = asyncio.create_task(scheduler.complete("m", ask("never"), priority=Priority.USER_JOB))
    await settle()
    waiting.cancel()
    await settle()
    backend.gate("job").set()
    await job
    await settle()
    assert backend.started == ["job"]
    assert not scheduler.busy


async def test_server_errors_reach_the_caller_and_free_the_turn():
    from bedrock.inference import InferenceError

    class Failing:
        async def stream_chat(self, model, messages, options=None, on_usage=None):
            raise InferenceError("Server error 500.")
            yield  # pragma: no cover

    scheduler = Scheduler(Failing(), laptop())
    with pytest.raises(InferenceError, match="500"):
        await scheduler.complete("m", ask("x"), priority=Priority.USER_JOB)
    assert not scheduler.busy


async def test_chat_usage_corrects_the_token_estimate():
    from bedrock.inference import tokens

    backend = FakeBackend(reply="ok")
    backend.usage = Usage(prompt_tokens=100)
    scheduler = Scheduler(backend, laptop())
    await scheduler.complete("m", [{"role": "user", "content": "x" * 300}], priority=Priority.CHAT)
    assert tokens.characters_per_token() < tokens.DEFAULT_CHARACTERS_PER_TOKEN


async def _answer_a_chat(scheduler: Scheduler, backend: GatedBackend, model: str = "m") -> None:
    backend.gate("chat").set()
    await scheduler.complete(model, ask("chat"), priority=Priority.CHAT)
