"""EventBus's replay/resync contract — the core logic behind SSE
reconnection (NEXT_STEPS.md §5 in the frontend repo). Pure unit tests, no
FastAPI/SSE involved; tests/test_sse_reconnect.py covers the wire format.
"""

from bedrock.server.events.bus import EventBus
from bedrock.server.schemas.enums import EventType
from bedrock.server.schemas.events import Event

SESSION = "sess-1"


def _event(event_type: EventType = EventType.AGENT_STATUS) -> Event:
    return Event(type=event_type, session_id=SESSION, payload={})


async def test_seq_is_monotonic_and_starts_at_one():
    bus = EventBus()
    first = _event()
    second = _event()
    await bus.publish(first)
    await bus.publish(second)
    assert first.seq == 1
    assert second.seq == 2


async def test_seq_is_independent_per_session():
    bus = EventBus()
    a = Event(type=EventType.AGENT_STATUS, session_id="a", payload={})
    b = Event(type=EventType.AGENT_STATUS, session_id="b", payload={})
    await bus.publish(a)
    await bus.publish(b)
    assert a.seq == 1
    assert b.seq == 1


async def test_fresh_connection_with_no_last_event_id_gets_everything_buffered():
    bus = EventBus()
    for _ in range(3):
        await bus.publish(_event())
    result = bus.replay(SESSION, last_seq=None)
    assert not result.resync
    assert [e.seq for e in result.events] == [1, 2, 3]


async def test_reconnect_within_the_buffer_gets_only_whats_new():
    bus = EventBus()
    for _ in range(5):
        await bus.publish(_event())
    result = bus.replay(SESSION, last_seq=2)
    assert not result.resync
    assert [e.seq for e in result.events] == [3, 4, 5]


async def test_reconnect_already_caught_up_gets_nothing_and_is_not_a_gap():
    bus = EventBus()
    for _ in range(3):
        await bus.publish(_event())
    result = bus.replay(SESSION, last_seq=3)
    assert not result.resync
    assert result.events == []


async def test_reconnect_past_the_evicted_window_triggers_resync():
    bus = EventBus(history_maxlen=3)
    for _ in range(10):  # far more than the buffer holds
        await bus.publish(_event())
    # Client last saw seq=1, but the buffer only holds the most recent 3 (8,9,10).
    result = bus.replay(SESSION, last_seq=1)
    assert result.resync
    assert result.reason == "gap"
    assert [e.seq for e in result.events] == [8, 9, 10]


async def test_reconnect_with_no_history_at_all_is_a_resync_not_a_hang():
    bus = EventBus()
    # Nothing ever published for this session — e.g. after a server restart.
    result = bus.replay("never-seen-session", last_seq=42)
    assert result.resync
    assert result.reason == "no_history"
    assert result.events == []


async def test_first_ever_connection_to_a_brand_new_session_is_not_a_resync():
    bus = EventBus()
    # No Last-Event-ID at all (fresh client) against a session with no
    # events yet — this is normal, not an error state.
    result = bus.replay("brand-new-session", last_seq=None)
    assert not result.resync
    assert result.events == []


async def test_terminal_event_survives_buffer_eviction():
    bus = EventBus(history_maxlen=2)
    await bus.publish(_event())
    await bus.publish(Event(type=EventType.TASK_COMPLETED, session_id=SESSION, payload={"x": 1}))
    # Evict the completed event out of the bounded buffer entirely.
    await bus.publish(_event())
    await bus.publish(_event())
    assert bus.terminal_for(SESSION) is not None
    assert bus.terminal_for(SESSION).type == EventType.TASK_COMPLETED
    assert bus.terminal_for(SESSION).payload == {"x": 1}


async def test_resync_carries_the_terminal_event_even_when_evicted():
    bus = EventBus(history_maxlen=2)
    await bus.publish(Event(type=EventType.TASK_COMPLETED, session_id=SESSION, payload={"done": True}))
    for _ in range(5):
        await bus.publish(_event())
    result = bus.replay(SESSION, last_seq=0)
    assert result.resync
    assert result.terminal is not None
    assert result.terminal.payload == {"done": True}


async def test_reconnect_that_is_not_a_gap_still_reports_no_terminal_field_used():
    bus = EventBus()
    await bus.publish(_event())
    result = bus.replay(SESSION, last_seq=0)
    assert not result.resync
    assert result.terminal is None  # only populated on the resync path


async def test_failed_is_also_tracked_as_terminal():
    bus = EventBus()
    await bus.publish(Event(type=EventType.TASK_FAILED, session_id=SESSION, payload={}))
    assert bus.terminal_for(SESSION) is not None
    assert bus.terminal_for(SESSION).type == EventType.TASK_FAILED
