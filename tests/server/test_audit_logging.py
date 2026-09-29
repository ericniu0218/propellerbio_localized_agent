"""Every prompt, agent decision, tool call, and output must be logged to the
append-only audit store — docs/server/ARCHITECTURE.md section 8. The audit sink
(bedrock/server/compliance/audit.py) is wired to the event bus at API startup; these
tests wire it directly so they don't need a running FastAPI app.
"""

from bedrock.server.compliance.audit import audit_event_sink
from bedrock.server.db import get_database
from bedrock.server.events import get_event_bus
from bedrock.server.orchestrator.loop import handle_message
from bedrock.server.schemas.enums import EventType


async def test_every_event_type_produces_an_audit_row(session_id):
    bus = get_event_bus()
    bus.register_audit_sink(audit_event_sink)

    db = get_database()
    # Mirrors what the real API does (src/bedrock/server/api/routes_session.py): look up
    # the session's actual user rather than passing an arbitrary one, since
    # the audit sink attributes every row to the session's registered user.
    user_id = db.get_session_user(session_id)
    task_id = await handle_message(session_id, user_id, "What is the boiling point of water?")

    entries = db.get_audit_log(session_id)
    assert len(entries) > 0
    assert all(e.session_id == session_id for e in entries)
    assert all(e.user_id == user_id for e in entries)

    actions = {e.action for e in entries}
    assert EventType.TASK_COMPLETED.value in actions
    assert EventType.AGENT_STATUS.value in actions

    completed_entry = next(e for e in entries if e.action == EventType.TASK_COMPLETED.value)
    assert completed_entry.task_id == task_id

    # Streamed tokens are not a row each: every model reply is one agent.output
    # row carrying the whole text (bedrock.server.compliance.audit).
    assert EventType.AGENT_TOKEN.value not in actions
    assert "agent.output" in actions


async def test_audit_log_has_no_update_or_delete_methods_on_database():
    # Structural guard for the append-only requirement: the audit table
    # should only ever be written via write_audit (INSERT) and read via
    # get_audit_log (SELECT) — no update/delete path should exist for it.
    db = get_database()
    audit_related = [name for name in dir(db) if "audit" in name.lower()]
    assert set(audit_related) == {"write_audit", "get_audit_log"}
