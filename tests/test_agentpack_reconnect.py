"""AgentPackClient.events()'s reconnect behavior: backoff/jitter, sending
Last-Event-ID back to the server, distinguishing a retryable drop from a
permanent (4xx) failure, and never treating a disconnect as a reason to
stop listening.
"""

import json

import httpx
import pytest

from bedrock.agentpack.client import MAX_BACKOFF_S, AgentPackClient
from bedrock.inference import InferenceError


def sse_envelope(event_type: str, session_id: str, payload: dict, seq: int) -> str:
    body = {
        "event_id": f"e{seq}",
        "seq": seq,
        "type": event_type,
        "session_id": session_id,
        "timestamp": "now",
        "payload": payload,
    }
    return f"id: {seq}\nevent: {event_type}\ndata: {json.dumps(body)}\n\n"


class RecordingSleep:
    def __init__(self) -> None:
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


async def test_a_dropped_connection_is_retried_and_resumes_with_last_event_id():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            # First connection: two events, then the server just closes —
            # a drop, not a graceful "we're done" (real backends never
            # close this stream on their own).
            text = sse_envelope("agent.status", "sess_1", {"a": 1}, seq=1) + sse_envelope(
                "agent.status", "sess_1", {"a": 2}, seq=2
            )
            return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})
        # Second connection: must carry Last-Event-ID: 2.
        assert request.headers.get("last-event-id") == "2"
        text = sse_envelope("task.completed", "sess_1", {"result_ref": "r1"}, seq=3)
        return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})

    client = AgentPackClient(
        "http://fake", transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )
    stream = client.events("sess_1")

    first = await anext(stream)
    second = await anext(stream)
    assert [first.type, second.type] == ["agent.status", "agent.status"]

    reconnecting = await anext(stream)
    assert reconnecting.type == "client.reconnecting"
    assert reconnecting.payload["attempt"] == 1

    reconnected = await anext(stream)
    assert reconnected.type == "client.reconnected"

    third = await anext(stream)
    assert third.type == "task.completed"
    assert third.payload["result_ref"] == "r1"

    assert len(calls) == 2


async def test_first_connection_sends_no_last_event_id_header():
    seen_headers = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen_headers["last-event-id"] = request.headers.get("last-event-id")
        return httpx.Response(200, text="", headers={"content-type": "text/event-stream"})

    client = AgentPackClient(
        "http://fake", transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )
    stream = client.events("sess_1")
    await anext(stream)  # the client.reconnecting marker from the immediate clean close
    assert seen_headers["last-event-id"] is None


async def test_a_permanent_4xx_failure_is_raised_immediately_not_retried():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404, text="session not found")

    client = AgentPackClient(
        "http://fake", transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )
    with pytest.raises(InferenceError, match="Could not connect"):
        await anext(client.events("sess_1"))
    assert len(calls) == 1  # never retried


async def test_a_5xx_response_is_treated_as_retryable():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503, text="starting up")
        return httpx.Response(
            200,
            text=sse_envelope("agent.status", "sess_1", {"a": 1}, seq=1),
            headers={"content-type": "text/event-stream"},
        )

    client = AgentPackClient(
        "http://fake", transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )
    stream = client.events("sess_1")
    reconnecting = await anext(stream)
    assert reconnecting.type == "client.reconnecting"
    reconnected = await anext(stream)
    assert reconnected.type == "client.reconnected"
    event = await anext(stream)
    assert event.type == "agent.status"
    assert len(calls) == 2


async def test_a_connection_error_is_treated_as_retryable():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(
            200,
            text=sse_envelope("agent.status", "sess_1", {"a": 1}, seq=1),
            headers={"content-type": "text/event-stream"},
        )

    client = AgentPackClient(
        "http://fake", transport=httpx.MockTransport(handler), sleep=RecordingSleep()
    )
    stream = client.events("sess_1")
    reconnecting = await anext(stream)
    assert reconnecting.type == "client.reconnecting"
    await anext(stream)  # client.reconnected
    event = await anext(stream)
    assert event.type == "agent.status"
    assert len(calls) == 2


async def test_backoff_grows_exponentially_with_jitter_and_is_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    sleeper = RecordingSleep()
    client = AgentPackClient("http://fake", transport=httpx.MockTransport(handler), sleep=sleeper)
    stream = client.events("sess_1")

    # Each `reconnecting` event's own sleep only happens on the *next*
    # anext() call (it comes after the yield in the generator) — request
    # one more than the 6 sleeps we want recorded.
    for _ in range(7):
        event = await anext(stream)
        assert event.type == "client.reconnecting"

    assert len(sleeper.delays) == 6
    # Jitter means we can't assert exact values, only the envelope: each
    # delay is within [half, full] of the un-jittered exponential value,
    # and the sequence is non-decreasing once capped.
    expected_ceiling = [1.0, 2.0, 4.0, 8.0, 16.0, MAX_BACKOFF_S]
    for delay, ceiling in zip(sleeper.delays, expected_ceiling):
        assert ceiling / 2 <= delay <= ceiling
    assert sleeper.delays[-1] <= MAX_BACKOFF_S


async def test_reconnect_attempt_counter_resets_after_a_successful_reconnect():
    """Attempt 1 fails, attempt 2 succeeds and yields one event, the
    connection then closes cleanly (another drop) — the *next* reconnect
    attempt must be counted as attempt 1 again, not attempt 3, since the
    successful reconnect in between should have reset the counter."""
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ConnectError("down", request=request)
        # Calls 2+ each succeed with exactly one event, then close —
        # each success is immediately followed by another drop.
        return httpx.Response(
            200,
            text=sse_envelope("agent.status", "sess_1", {"n": len(calls)}, seq=len(calls)),
            headers={"content-type": "text/event-stream"},
        )

    sleeper = RecordingSleep()
    client = AgentPackClient("http://fake", transport=httpx.MockTransport(handler), sleep=sleeper)
    stream = client.events("sess_1")

    first_reconnecting = await anext(stream)
    assert first_reconnecting.type == "client.reconnecting"
    assert first_reconnecting.payload["attempt"] == 1

    assert (await anext(stream)).type == "client.reconnected"
    assert (await anext(stream)).type == "agent.status"

    second_reconnecting = await anext(stream)  # clean close -> drop -> new attempt
    assert second_reconnecting.type == "client.reconnecting"
    assert second_reconnecting.payload["attempt"] == 1  # reset, not 2
