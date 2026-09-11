"""In-process async pub/sub feeding the SSE stream.

Single-process by default (fine for one on-prem server); swap for a Redis
pub/sub-backed implementation behind the same interface if the deployment
grows multi-worker. Every published event is also handed to the audit sink,
since audit-relevant events (tool calls, reviews, approvals) should be
logged as they happen, not reconstructed after the fact.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Awaitable, Callable
from functools import lru_cache

from pydantic import BaseModel

from app.schemas.enums import EventType
from app.schemas.events import Event

AuditSink = Callable[[Event], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Event]]] = defaultdict(list)
        self._audit_sinks: list[AuditSink] = []
        self._history: dict[str, list[Event]] = defaultdict(list)

    def register_audit_sink(self, sink: AuditSink) -> None:
        if sink not in self._audit_sinks:
            self._audit_sinks.append(sink)

    async def publish_typed(self, session_id: str, event_type: EventType, payload: BaseModel) -> None:
        await self.publish(Event(type=event_type, session_id=session_id, payload=payload.model_dump()))

    async def publish(self, event: Event) -> None:
        self._history[event.session_id].append(event)
        for queue in list(self._subscribers[event.session_id]):
            await queue.put(event)
        for sink in self._audit_sinks:
            await sink(event)

    def subscribe(self, session_id: str) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue()
        self._subscribers[session_id].append(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue[Event]) -> None:
        if queue in self._subscribers[session_id]:
            self._subscribers[session_id].remove(queue)

    def history(self, session_id: str) -> list[Event]:
        return list(self._history[session_id])


@lru_cache
def get_event_bus() -> EventBus:
    return EventBus()
