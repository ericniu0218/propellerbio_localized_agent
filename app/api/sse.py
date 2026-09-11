"""SSE endpoint for the frontend's live event stream — see
docs/ARCHITECTURE.md section 7 for the full event list."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.events import get_event_bus

router = APIRouter(tags=["events"])


@router.get("/sessions/{session_id}/events")
async def stream_events(session_id: str):
    bus = get_event_bus()
    queue = bus.subscribe(session_id)

    async def event_generator():
        try:
            # Replay history first so a late-connecting frontend catches up.
            for event in bus.history(session_id):
                yield {"event": event.type.value, "data": event.model_dump_json()}
            while True:
                event = await queue.get()
                yield {"event": event.type.value, "data": event.model_dump_json()}
        except asyncio.CancelledError:
            pass
        finally:
            bus.unsubscribe(session_id, queue)

    return EventSourceResponse(event_generator())
