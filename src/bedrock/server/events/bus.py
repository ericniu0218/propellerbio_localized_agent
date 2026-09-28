"""In-process async pub/sub feeding the SSE stream.

Single-process by default (fine for one on-prem server); swap for a Redis
pub/sub-backed implementation behind the same interface if the deployment
grows multi-worker. Every published event is also handed to the audit sink,
since audit-relevant events (tool calls, reviews, approvals) should be
logged as they happen, not reconstructed after the fact.

Known limitation, not fixed here (see NEXT_STEPS.md §5): all of this —
history, sequence counters, terminal state — is in-memory. A server
restart loses it, same as it always has; a client reconnecting after a
restart gets `reason="no_history"` and starts fresh rather than hanging.
Only durable job storage (explicitly out of scope for this task) fixes
that for real.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import lru_cache

from pydantic import BaseModel

from bedrock.server.schemas.enums import EventType
from bedrock.server.schemas.events import Event

AuditSink = Callable[[Event], Awaitable[None]]

# Bounded so long-running sessions (an AlphaFold-adjacent chat left open for
# hours, say) don't grow this without limit. A reconnect gap wider than this
# many events triggers an explicit resync rather than silently replaying a
# partial/misleading window.
EVENT_HISTORY_MAXLEN = 500

_TERMINAL_EVENT_TYPES = (EventType.TASK_COMPLETED, EventType.TASK_FAILED)


@dataclass
class ReplayResult:
    events: list[Event]
    resync: bool
    reason: str | None = None  # set when resync is True
    terminal: Event | None = field(default=None)


class EventBus:
    def __init__(self, history_maxlen: int = EVENT_HISTORY_MAXLEN) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[Event]]] = defaultdict(list)
        self._audit_sinks: list[AuditSink] = []
        self._history: dict[str, deque[Event]] = defaultdict(lambda: deque(maxlen=history_maxlen))
        self._next_seq: dict[str, int] = defaultdict(lambda: 1)
        self._terminal: dict[str, Event] = {}

    def register_audit_sink(self, sink: AuditSink) -> None:
        if sink not in self._audit_sinks:
            self._audit_sinks.append(sink)

    async def publish_typed(self, session_id: str, event_type: EventType, payload: BaseModel) -> None:
        await self.publish(Event(type=event_type, session_id=session_id, payload=payload.model_dump()))

    async def publish(self, event: Event) -> None:
        event.seq = self._next_seq[event.session_id]
        self._next_seq[event.session_id] += 1
        self._history[event.session_id].append(event)
        if event.type in _TERMINAL_EVENT_TYPES:
            self._terminal[event.session_id] = event
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

    def terminal_for(self, session_id: str) -> Event | None:
        return self._terminal.get(session_id)

    def replay(self, session_id: str, last_seq: int | None) -> ReplayResult:
        """What to send a (re)connecting client. `last_seq` is the caller's
        Last-Event-ID, already parsed to int (None if absent/invalid).

        - No last_seq: fresh connection: send the whole current buffer.
        - last_seq within what we still hold: send exactly what's new.
        - last_seq older than the buffer's oldest surviving event (or the
          buffer is empty but we've clearly issued events before, or
          last_seq looks like it's from a session we have no record of):
          resync — the caller must not assume the returned events are a
          complete history, just the most recent window we still have.
        """
        # Read-only lookups: `.get` on purpose, never `[...]` — this method
        # must not fabricate history/seq-counter entries for a session
        # it's never actually published to (a defaultdict's `[...]` would).
        buffer = self._history.get(session_id, ())
        terminal = self._terminal.get(session_id)

        if last_seq is None:
            return ReplayResult(events=list(buffer), resync=False)

        if not buffer:
            highest_known = self._next_seq.get(session_id, 1) - 1
            if last_seq == highest_known:
                # Caught up exactly — including the legitimate "brand new
                # session, nothing published, client correctly says -1/0"
                # case, not just a coincidental match.
                return ReplayResult(events=[], resync=False)
            if highest_known == 0:
                # We have never issued anything for this session, yet the
                # caller claims to have seen event `last_seq` — most likely
                # a server restart wiped in-memory state (NEXT_STEPS.md §5
                # limitation), not a client bug.
                return ReplayResult(events=[], resync=True, reason="no_history", terminal=terminal)
            if last_seq > highest_known:
                # Claims to have seen an event newer than any we've issued.
                return ReplayResult(
                    events=[], resync=True, reason="unknown_last_event_id", terminal=terminal
                )
            # Real events existed but none survived the bounded buffer.
            return ReplayResult(events=[], resync=True, reason="gap", terminal=terminal)

        oldest_seq = buffer[0].seq
        if last_seq >= oldest_seq - 1:
            return ReplayResult(events=[e for e in buffer if e.seq > last_seq], resync=False)

        return ReplayResult(events=list(buffer), resync=True, reason="gap", terminal=terminal)


@lru_cache
def get_event_bus() -> EventBus:
    return EventBus()
