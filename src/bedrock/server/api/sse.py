"""SSE endpoint for the frontend's live event stream — see
docs/server/ARCHITECTURE.md section 7 for the full event list, and
src/bedrock/server/events/bus.py for the reconnection/replay contract this implements.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from sse_starlette.sse import EventSourceResponse

from bedrock.server.events import get_event_bus
from bedrock.server.events.bus import EventBus
from bedrock.server.schemas.enums import EventType
from bedrock.server.schemas.events import Event, SessionResyncPayload

router = APIRouter(tags=["events"])

# Milliseconds an EventSource-compliant client should wait before
# reconnecting on an unexpected drop, absent our own backoff logic on top
# (see bedrock-private-ai's AgentPackClient, which does its own
# exponential-backoff-with-jitter and ignores this for anything past its
# first retry — this is a floor/hint for any other client, e.g. a browser
# using the native EventSource API against src/bedrock/server/static/index.html).
CLIENT_RETRY_MS = 2000

# How often to send a keep-alive comment. Chosen so proxies/load balancers
# with typical 30-60s idle timeouts never see a silent connection — matters
# most for a session left open during a long real-mode Director review,
# not for AlphaFold/PyMOL runs specifically, which don't use this stream
# (they're local subprocesses on the client, not backend tasks).
HEARTBEAT_SECONDS = 15


def _parse_last_event_id(request: Request) -> int | None:
    raw = request.headers.get("last-event-id")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


async def sse_event_stream(bus: EventBus, session_id: str, last_event_id: int | None) -> AsyncIterator[dict]:
    """The actual stream, decoupled from FastAPI's Request/Response so it
    can be driven directly in tests (see tests/test_sse_reconnect.py) —
    a real ASGI transport isn't needed to exercise replay/resync/live-event
    logic, only to prove the route itself is wired up correctly.
    """
    # subscribe() and replay() are both plain sync calls with no `await`
    # between them, so they execute atomically with respect to publish() —
    # no event can land in `queue` after the replay snapshot is taken but
    # before the live loop starts reading it, which would otherwise
    # deliver that event twice. Doing this here, as the generator's first
    # step, matters: by the time this generator actually starts running,
    # the caller has already moved on (returned a response, or otherwise
    # yielded control), so subscribing any earlier would reopen exactly
    # that window.
    queue = bus.subscribe(session_id)
    first = True
    try:
        replay = bus.replay(session_id, last_event_id)
        if replay.resync:
            resync_payload = SessionResyncPayload(
                reason=replay.reason or "gap",
                events=replay.events,
                terminal=replay.terminal,
            )
            # Wrapped in the same Event envelope every other event uses
            # (session_id/payload included) even though this one is
            # synthesized here rather than coming from EventBus.publish —
            # a client's generic envelope parsing shouldn't need a special
            # case for exactly one event type. No `id:` on the wire for
            # this frame: it's explicitly not part of the ordered
            # sequence, so it must never become a client's Last-Event-ID.
            envelope = Event(
                type=EventType.SESSION_RESYNC, session_id=session_id, payload=resync_payload.model_dump()
            )
            resync_event = {"event": envelope.type.value, "data": envelope.model_dump_json()}
            if first:
                resync_event["retry"] = CLIENT_RETRY_MS
                first = False
            yield resync_event

        for event in replay.events:
            item = {"event": event.type.value, "data": event.model_dump_json(), "id": str(event.seq)}
            if first:
                item["retry"] = CLIENT_RETRY_MS
                first = False
            yield item

        while True:
            event = await queue.get()
            item = {"event": event.type.value, "data": event.model_dump_json(), "id": str(event.seq)}
            if first:
                item["retry"] = CLIENT_RETRY_MS
                first = False
            yield item
    except asyncio.CancelledError:
        pass
    finally:
        bus.unsubscribe(session_id, queue)


@router.get("/sessions/{session_id}/events")
async def stream_events(session_id: str, request: Request):
    bus = get_event_bus()
    last_event_id = _parse_last_event_id(request)
    return EventSourceResponse(
        sse_event_stream(bus, session_id, last_event_id), ping=HEARTBEAT_SECONDS
    )
