"""SSE reconnection: drives app.api.sse.sse_event_stream directly against a
real EventBus (no HTTP transport involved — see that module's docstring
for why). Covers the exact scenarios from the reconnection spec: a drop
mid-run with no gaps/duplicates on reconnect, reconnect after completion,
and reconnect with an expired Last-Event-ID.
"""

import asyncio
import json

import pytest
from starlette.requests import Request

from app.api.sse import _parse_last_event_id, sse_event_stream
from app.events.bus import EventBus
from app.schemas.enums import EventType
from app.schemas.events import AgentStatusPayload, TaskAssignedPayload, TaskCompletedPayload

SESSION = "sess-1"


async def _next(gen, timeout: float = 1.0) -> dict:
    return await asyncio.wait_for(gen.__anext__(), timeout=timeout)


async def _assigned(bus: EventBus, task_id: str = "t1") -> None:
    await bus.publish_typed(
        SESSION, EventType.TASK_ASSIGNED, TaskAssignedPayload(task_id=task_id, assigned_to="runner", title="x")
    )


async def _completed(bus: EventBus, task_id: str = "t1") -> None:
    await bus.publish_typed(
        SESSION, EventType.TASK_COMPLETED, TaskCompletedPayload(task_id=task_id, title="x", result_ref="r1")
    )


async def test_drop_mid_run_then_reconnect_delivers_everything_once_in_order():
    bus = EventBus()

    await _assigned(bus)  # seq 1, before the client ever connects

    gen_a = sse_event_stream(bus, SESSION, None)
    received: list[int] = []

    item = await _next(gen_a)
    assert int(item["id"]) == 1
    received.append(1)

    # Job keeps producing while the client is connected.
    await bus.publish_typed(
        SESSION, EventType.AGENT_STATUS, AgentStatusPayload(agent_id="a1", role="runner", status="running_tool")
    )  # seq 2
    item = await _next(gen_a)
    assert int(item["id"]) == 2
    received.append(2)

    # Simulate the connection dropping — the client stops reading and the
    # generator is torn down, same as a real disconnect.
    await gen_a.aclose()

    # The job keeps running while nobody is listening (must not be
    # cancelled by the disconnect — that's enforced at the API-route level
    # via asyncio.create_task in routes_session.py, not by this generator,
    # but the event bus/history side must also just keep accumulating).
    await bus.publish_typed(
        SESSION, EventType.AGENT_STATUS, AgentStatusPayload(agent_id="a1", role="runner", status="idle")
    )  # seq 3
    await _completed(bus)  # seq 4

    # Reconnect with Last-Event-ID: 2 (the last one the client actually saw).
    gen_b = sse_event_stream(bus, SESSION, last_event_id=2)

    item = await _next(gen_b)
    assert int(item["id"]) == 3
    received.append(3)
    item = await _next(gen_b)
    assert int(item["id"]) == 4
    received.append(4)
    assert item["event"] == "task.completed"

    # Live continuation still works after a reconnect.
    await bus.publish_typed(
        SESSION, EventType.TASK_ASSIGNED, TaskAssignedPayload(task_id="t2", assigned_to="runner", title="y")
    )  # seq 5
    item = await _next(gen_b)
    assert int(item["id"]) == 5
    received.append(5)

    assert received == [1, 2, 3, 4, 5]  # no gaps, no duplicates, in order

    await gen_a.aclose()
    await gen_b.aclose()


async def test_reconnect_after_completion_gets_final_state_not_a_hang():
    bus = EventBus()
    await _assigned(bus)  # seq 1
    await _completed(bus)  # seq 2 — the run is already over before anyone connects

    # A client that saw the assignment but disconnected before the
    # completion reconnects — it must learn the run finished, not sit
    # waiting on the live queue for an event that already happened.
    gen = sse_event_stream(bus, SESSION, last_event_id=1)
    item = await _next(gen)
    assert item["event"] == "task.completed"
    assert int(item["id"]) == 2

    payload = json.loads(item["data"])
    assert payload["payload"]["result_ref"] == "r1"

    await gen.aclose()


async def test_fresh_connection_after_completion_also_gets_the_final_state():
    bus = EventBus()
    await _assigned(bus)
    await _completed(bus)

    gen = sse_event_stream(bus, SESSION, None)
    first = await _next(gen)
    second = await _next(gen)
    assert [first["event"], second["event"]] == ["task.assigned", "task.completed"]
    await gen.aclose()


async def test_reconnect_with_expired_last_event_id_gets_an_explicit_resync():
    bus = EventBus(history_maxlen=3)
    for i in range(10):
        await _assigned(bus, task_id=f"t{i}")
    # Client last saw seq=1 (the very first task.assigned), long since
    # evicted from the size-3 buffer — this must not silently skip to
    # "whatever's left" without telling the client a gap happened.
    gen = sse_event_stream(bus, SESSION, last_event_id=1)

    resync_item = await _next(gen)
    assert resync_item["event"] == "session.resync"
    assert "id" not in resync_item  # never part of the ordered sequence
    envelope = json.loads(resync_item["data"])
    assert envelope["session_id"] == SESSION
    resync_payload = envelope["payload"]
    assert resync_payload["reason"] == "gap"
    assert len(resync_payload["events"]) == 3
    assert [e["seq"] for e in resync_payload["events"]] == [8, 9, 10]

    # The buffered window is then delivered as normal, individually
    # id'd events, so the client can still process them through its
    # regular event handling after resetting its local state.
    item = await _next(gen)
    assert int(item["id"]) == 8
    item = await _next(gen)
    assert int(item["id"]) == 9
    item = await _next(gen)
    assert int(item["id"]) == 10

    await gen.aclose()


async def test_resync_includes_terminal_state_even_if_evicted_from_the_buffer():
    bus = EventBus(history_maxlen=2)
    await _completed(bus)  # seq 1 — will be evicted below
    for i in range(5):
        await _assigned(bus, task_id=f"t{i}")  # seq 2..6, evicts seq 1 out of the buffer

    gen = sse_event_stream(bus, SESSION, last_event_id=0)
    resync_item = await _next(gen)
    resync_payload = json.loads(resync_item["data"])["payload"]
    assert resync_payload["reason"] == "gap"
    assert resync_payload["terminal"] is not None
    assert resync_payload["terminal"]["type"] == "task.completed"
    await gen.aclose()


async def test_first_event_of_a_connection_carries_a_retry_hint():
    bus = EventBus()
    await _assigned(bus)
    gen = sse_event_stream(bus, SESSION, None)
    item = await _next(gen)
    assert item.get("retry") == 2000
    await gen.aclose()


async def test_disconnecting_does_not_prevent_the_bus_from_continuing_to_track_history():
    """The generator's own teardown must not, itself, disrupt anything
    else about the bus — a disconnect is purely this one subscriber going
    away, never a signal to stop tracking a session's events."""
    bus = EventBus()
    gen = sse_event_stream(bus, SESSION, None)
    await _assigned(bus)
    await _next(gen)
    await gen.aclose()

    await _completed(bus)
    assert bus.terminal_for(SESSION) is not None
    assert len(bus.history(SESSION)) == 2


def _request_with_header(value: str | None) -> Request:
    headers = [(b"last-event-id", value.encode())] if value is not None else []
    return Request({"type": "http", "headers": headers})


def test_parse_last_event_id_reads_the_header():
    assert _parse_last_event_id(_request_with_header("42")) == 42


def test_parse_last_event_id_is_none_when_absent():
    assert _parse_last_event_id(_request_with_header(None)) is None


def test_parse_last_event_id_is_none_when_not_a_number():
    assert _parse_last_event_id(_request_with_header("not-a-number")) is None


# No HTTP-transport test of the live route (neither TestClient nor a raw
# httpx.AsyncClient against ASGITransport): both reliably hang connecting
# to `/sessions/{id}/events`. Isolated with a throwaway script — a plain
# FastAPI StreamingResponse streams fine through ASGITransport, but
# sse_starlette.EventSourceResponse specifically does not resolve, with or
# without something already available to send. This looks like a real
# incompatibility between sse-starlette's ping-task internals and how
# these test transports drive the ASGI app, not a bug in `stream_events`
# — uvicorn serves this endpoint fine in practice (see manual verification
# in the Phase 1 summary). `stream_events` itself is a 5-line function
# (parse the header, call sse_event_stream, wrap in EventSourceResponse);
# everything it delegates to is fully covered above and in
# test_event_bus.py. If sse-starlette fixes this upstream, or this route
# ever needs a real HTTP-level test, retry with an ASGITransport-based
# async client first.
