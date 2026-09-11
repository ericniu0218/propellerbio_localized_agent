from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.compliance.approval_gates import get_approval_gate
from app.schemas.api import ApprovalRequest

router = APIRouter(tags=["approval"])


@router.post("/approvals")
async def resolve_approval(req: ApprovalRequest) -> dict:
    gate = get_approval_gate()
    if not gate.is_pending(req.task_id):
        raise HTTPException(status_code=409, detail="no pending approval for this task")
    gate.resolve(req.task_id, req.decision)
    return {"task_id": req.task_id, "decision": req.decision.value}
