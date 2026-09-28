from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.compliance.approval_gates import get_approval_gate
from app.events import get_event_bus
from app.schemas.api import ApprovalRequest
from app.schemas.enums import EventType
from app.schemas.events import ApprovalResolvedPayload
from app.task_store import get_task_store

router = APIRouter(tags=["approval"])


@router.post("/approvals")
async def resolve_approval(req: ApprovalRequest) -> dict:
    """Record a human approval, then release the task.

    `approved_by` used to be accepted and thrown away: the API asked who was
    signing and kept no trace of the answer, which is worse than not asking,
    because the endpoint looks like it records a signer. The decision is now
    written to the audit log first and the task released second — if the record
    cannot be written, the approval does not take effect. An unrecorded
    approval on a regulatory or clinical task is exactly what the audit trail
    exists to make impossible.
    """
    gate = get_approval_gate()
    if not gate.is_pending(req.task_id):
        raise HTTPException(status_code=409, detail="no pending approval for this task")

    session_id = get_task_store().get_task_session_id(req.task_id)
    if session_id is None:
        raise HTTPException(status_code=404, detail="unknown task")

    # Published rather than written straight to the database so the existing
    # audit sink persists it and the frontend sees the outcome on the event
    # stream, the same path every other decision takes.
    await get_event_bus().publish_typed(
        session_id,
        EventType.APPROVAL_RESOLVED,
        ApprovalResolvedPayload(
            task_id=req.task_id,
            decision=req.decision.value,
            approved_by=req.approved_by,
            comment=req.comment,
            recorded_at=datetime.now(timezone.utc).isoformat(),
        ),
    )

    gate.resolve(req.task_id, req.decision)
    return {"task_id": req.task_id, "decision": req.decision.value}
