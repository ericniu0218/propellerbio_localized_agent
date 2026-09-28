"""One way to decide a human approval, whichever screen it came from.

There were two. ``POST /approvals`` (the terminal app, the legacy page) wrote
the signature record - who approved, what, when - to the audit log first and
released the task second, so an approval that could not be recorded did not take
effect. The workspace's approve button released the gate directly: no record
naming the signer, on exactly the regulatory and clinical answers the record
exists for. Both now call ``resolve_approval``.

``signer_verified`` says whether the signer is a signed-in identity or only a
name the client supplied. It is recorded rather than enforced here: the terminal
app cannot sign in yet (docs/prompts/AGENT_PACK_INTEGRATION.md section 9.1).
"""

from __future__ import annotations

from datetime import datetime, timezone

from bedrock.server.compliance.approval_gates import get_approval_gate
from bedrock.server.events import get_event_bus
from bedrock.server.schemas.enums import ApprovalDecision, EventType
from bedrock.server.schemas.events import ApprovalResolvedPayload
from bedrock.server.task_store import get_task_store


class NotPending(Exception):
    """Nothing is waiting for a decision on this task."""


class UnknownTask(Exception):
    """No task has this id."""


async def resolve_approval(
    task_id: str,
    decision: ApprovalDecision,
    *,
    signer: str,
    signer_verified: bool,
    comment: str | None = None,
) -> None:
    gate = get_approval_gate()
    if not gate.is_pending(task_id):
        raise NotPending(task_id)
    session_id = get_task_store().get_task_session_id(task_id)
    if session_id is None:
        raise UnknownTask(task_id)

    # Published rather than written straight to the database, so the audit sink
    # persists it and clients see the outcome on the event stream - and before
    # the gate is released: an approval whose record fails does not take effect.
    await get_event_bus().publish_typed(
        session_id,
        EventType.APPROVAL_RESOLVED,
        ApprovalResolvedPayload(
            task_id=task_id,
            decision=decision.value,
            approved_by=signer,
            comment=comment,
            recorded_at=datetime.now(timezone.utc).isoformat(),
            signer_verified=signer_verified,
        ),
    )
    gate.resolve(task_id, decision)
