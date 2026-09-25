"""The bridge from the pipeline's blocking code to bedrock's async scheduler.

Every test here drives a real `Scheduler`, because the parts worth proving are
exactly the parts a mock scheduler would not have: that the call goes out at
`USER_JOB` behind the chat, that being cancelled reads differently from failing,
and that calling from the wrong thread is refused rather than hanging forever.
"""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator, Callable

import pytest

from bedrock.config import PROFILES
from bedrock.inference import InferenceError, ModelInfo, Priority, RequestOptions, Scheduler, Usage
from bedrock.litreview.llm import LLMError, SchedulerClient
from bedrock.litreview.llm.scheduled import ModelServerUnavailable, ReviewCancelled


class ScriptedBackend:
    """A model server that answers with whatever it was told to, once.

    Records the model, messages and options of every call, so a test can assert
    on what really went over the seam.
    """

    def __init__(
        self,
        reply: str = "the answer",
        models: list[ModelInfo] | None = None,
        error: Exception | None = None,
        before_reply: Callable[[], None] | None = None,
    ) -> None:
        self.reply = reply
        self.models = models if models is not None else [ModelInfo(id="m")]
        self.error = error
        self.before_reply = before_reply
        self.calls: list[tuple[str, list[dict[str, str]], RequestOptions | None]] = []

    async def check_connection(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return list(self.models)

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        self.calls.append((model, list(messages), options))
        if self.before_reply is not None:
            self.before_reply()
        if self.error is not None:
            raise self.error
        yield self.reply

    async def close(self) -> None:
        pass


async def in_worker(client_call):
    """Run one blocking client call on a worker thread, as a review does."""
    return await asyncio.to_thread(client_call)


def make(backend, **kwargs) -> tuple[SchedulerClient, Scheduler]:
    scheduler = Scheduler(backend, PROFILES["workstation"])
    kwargs.setdefault("context_window", 12288)
    client = SchedulerClient(scheduler, asyncio.get_running_loop(), **kwargs)
    return client, scheduler


# ---------------------------------------------------------------------------
# What one call looks like on the wire
# ---------------------------------------------------------------------------


async def test_one_call_carries_the_prompt_the_seed_the_schema_and_a_user_job_priority():
    backend = ScriptedBackend(reply="<think>weighing it up</think>the answer")
    client, scheduler = make(backend, seed=7, model_options={"m": {"thinking": False}})

    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    reply = await in_worker(
        lambda: client.complete("grade this paper", model="m", schema=schema, max_tokens=512)
    )

    # The thinking trace is not the answer.
    assert reply == "the answer"

    [(model, messages, options)] = backend.calls
    assert model == "m"
    assert messages == [{"role": "user", "content": "grade this paper"}]
    assert options.temperature == 0.0
    assert options.seed == 7
    assert options.json_schema == schema
    assert options.max_tokens == 512
    # `thinking: false` in the model set becomes bedrock's reasoning switch.
    assert options.reasoning is False
    assert not scheduler.busy


async def test_a_model_set_that_says_nothing_about_thinking_leaves_the_server_alone():
    """MedGemma cannot turn its trace off, so asking is worse than stripping it."""
    backend = ScriptedBackend()
    client, _ = make(backend, model_options={"m": {}})
    await in_worker(lambda: client.complete("p", model="m"))
    assert backend.calls[0][2].reasoning is None


async def test_the_call_goes_in_behind_the_chat():
    """A review is a long job; the employee typing must never wait for it."""
    backend = ScriptedBackend()
    client, _ = make(backend)
    await in_worker(lambda: client.complete("p", model="m"))
    assert client._priority is Priority.USER_JOB


# ---------------------------------------------------------------------------
# The window check
# ---------------------------------------------------------------------------


async def test_a_prompt_that_cannot_fit_the_window_is_refused_before_anything_is_sent():
    """A server whose window is too small silently drops the start of the
    prompt, so the model answers a question it was never shown."""
    backend = ScriptedBackend()
    client, _ = make(backend, context_window=2048)

    with pytest.raises(LLMError, match="window is 2048"):
        await in_worker(lambda: client.complete("word " * 4000, model="m", max_tokens=1024))
    assert backend.calls == []


async def test_the_reply_limit_is_never_shrunk_to_make_something_fit():
    """max_tokens is part of the call cache key, and a short extraction reply
    is truncated JSON - a failure either way."""
    backend = ScriptedBackend()
    client, _ = make(backend, context_window=2048)
    with pytest.raises(LLMError, match="4096-token reply"):
        await in_worker(lambda: client.complete("short", model="m", max_tokens=4096))
    assert backend.calls == []


# ---------------------------------------------------------------------------
# Failure and cancellation read differently
# ---------------------------------------------------------------------------


async def test_an_ordinary_server_error_is_one_bad_reply_and_the_stage_carries_on():
    backend = ScriptedBackend(error=InferenceError("Server error 400: bad schema"))
    client, _ = make(backend)
    with pytest.raises(LLMError) as exc:
        await in_worker(lambda: client.complete("p", model="m"))
    assert not isinstance(exc.value, ModelServerUnavailable)
    assert "400" in str(exc.value)


async def test_an_unreachable_server_ends_the_run_rather_than_thinning_the_report():
    backend = ScriptedBackend(error=InferenceError("connection refused", unavailable=True))
    client, _ = make(backend)
    with pytest.raises(ModelServerUnavailable):
        await in_worker(lambda: client.complete("p", model="m"))


async def test_once_the_cancel_event_is_set_nothing_more_is_sent():
    cancel = threading.Event()
    backend = ScriptedBackend()
    client, _ = make(backend, cancel=cancel)

    await in_worker(lambda: client.complete("first", model="m"))
    cancel.set()
    with pytest.raises(ReviewCancelled):
        await in_worker(lambda: client.complete("second", model="m"))
    assert len(backend.calls) == 1


async def test_a_server_error_while_stopping_reads_as_stopped_not_as_broken():
    """Closing the app cancels in-flight work; that is not a server failure."""
    cancel = threading.Event()
    backend = ScriptedBackend(
        error=InferenceError("connection refused", unavailable=True),
        before_reply=cancel.set,
    )
    client, _ = make(backend, cancel=cancel)
    with pytest.raises(ReviewCancelled):
        await in_worker(lambda: client.complete("p", model="m"))


async def test_calling_from_the_loops_own_thread_is_refused_rather_than_deadlocking():
    client, _ = make(ScriptedBackend())
    with pytest.raises(RuntimeError, match="worker thread"):
        client.complete("p", model="m")  # no to_thread: this *is* the loop


# ---------------------------------------------------------------------------
# Which model wrote the report
# ---------------------------------------------------------------------------


async def test_version_of_pins_the_weights_when_the_server_reports_a_digest():
    backend = ScriptedBackend(models=[ModelInfo(id="medgemma1.5:4b", digest="abcdef0123456789aa")])
    client, _ = make(backend)
    assert await in_worker(lambda: client.version_of("medgemma1.5:4b")) == "medgemma1.5:4b@abcdef012345"


async def test_version_of_is_just_the_name_when_the_server_reports_no_digest():
    backend = ScriptedBackend(models=[ModelInfo(id="qwen3-8b-instruct")])
    client, _ = make(backend)
    assert await in_worker(lambda: client.version_of("qwen3-8b-instruct")) == "qwen3-8b-instruct"
    # ... and for a model the server does not list at all.
    assert await in_worker(lambda: client.version_of("mystery")) == "mystery"


async def test_a_server_that_cannot_list_models_still_lets_the_run_finish():
    """A version label is provenance, not a guarantee; it must not end a run."""

    class NoListing(ScriptedBackend):
        async def list_models(self):
            raise InferenceError("Could not list models", unavailable=True)

    client, _ = make(NoListing())
    assert await in_worker(lambda: client.version_of("m")) == "m"


async def test_model_version_follows_the_model_that_last_answered():
    backend = ScriptedBackend(models=[ModelInfo(id="m", digest="1234567890abcdef")])
    client, _ = make(backend)
    assert client.model_version == "scheduled"
    await in_worker(lambda: client.complete("p", model="m"))
    assert await in_worker(lambda: client.model_version) == "m@1234567890ab"
