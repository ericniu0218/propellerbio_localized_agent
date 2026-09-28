from __future__ import annotations

from fastapi import APIRouter, HTTPException

from bedrock.server.compliance import approvals
from bedrock.server.schemas.api import ApprovalRequest

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
    try:
        await approvals.resolve_approval(
            req.task_id,
            req.decision,
            signer=req.approved_by,
            # A name in the request body, not a signed-in identity: this API has
            # no sign-in yet (the terminal app cannot sign in).
            signer_verified=False,
            comment=req.comment,
        )
    except approvals.NotPending:
        raise HTTPException(status_code=409, detail="no pending approval for this task")
    except approvals.UnknownTask:
        raise HTTPException(status_code=404, detail="unknown task")
    return {"task_id": req.task_id, "decision": req.decision.value}
