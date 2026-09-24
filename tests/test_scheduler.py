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
