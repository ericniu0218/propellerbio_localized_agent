"""Audit sink: every event published on the event bus that is
compliance-relevant gets written to the append-only audit_log table before
(or as) it reaches subscribers. Registered on the EventBus in app.main at
startup. See docs/ARCHITECTURE.md section 8."""

from __future__ import annotations

from bedrock.server.db import get_database
from bedrock.server.schemas.enums import AgentRole, EventType
from bedrock.server.schemas.events import Event

# Every event type is audit-relevant per the spec ("every prompt, agent
# decision, tool call, approval/rejection, and output is logged"), so we log
# all of them rather than trying to guess which ones matter.
_AUDITED_TYPES = set(EventType)


def _agent_from_payload(payload: dict) -> AgentRole | None:
    role = payload.get("role") or payload.get("reviewed_by") or payload.get("created_by")
    if role in (r.value for r in AgentRole):
        return AgentRole(role)
    return None


async def audit_event_sink(event: Event) -> None:
    if event.type not in _AUDITED_TYPES:
        return
    db = get_database()
    user_id = db.get_session_user(event.session_id)
    task_id = event.payload.get("task_id")
    agent = _agent_from_payload(event.payload)
    model_name = event.payload.get("model_name")
    db.write_audit(
        session_id=event.session_id,
        user_id=user_id,
        action=event.type.value,
        detail=event.payload,
        task_id=task_id,
        agent=agent,
        model_name=model_name,
    )
