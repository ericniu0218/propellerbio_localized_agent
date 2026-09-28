"""The human approval must leave a record naming who approved it.

`approved_by` was accepted by the endpoint and discarded, so a regulatory or
clinical task could be released with no trace of who released it — while the
API's shape implied a signer was being recorded.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient

from bedrock.server.compliance.approval_gates import get_approval_gate
from bedrock.server.compliance.audit import audit_event_sink
from bedrock.server.events import get_event_bus
from bedrock.server.db import get_database
from bedrock.server.main import app
from bedrock.server.schemas.enums import ApprovalDecision, EventType
from bedrock.server.task_store import get_task_store


async def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _pending_task(session_id: str) -> str:
    """A task sitting on a human approval gate.

    The audit sink is wired here rather than by FastAPI's startup hook, which
    ASGITransport does not run — the same approach tests/test_audit_logging.py
    takes.
    """
    get_event_bus().register_audit_sink(audit_event_sink)
    from bedrock.server.schemas.enums import AgentRole, TaskComplexity, TaskDomain, TaskStatus
    from bedrock.server.schemas.tasks import TaskNode, TaskPlan

    # A fresh id per test: the task store maps task_id -> session_id globally,
    # so a shared id makes the second test's audit rows land in the first
    # test's session.
    task_id = f"task_{uuid4().hex[:10]}"
    node = TaskNode(
        task_id=task_id,
        assigned_to=AgentRole.SPECIALIST,
        title="Draft the regulatory submission section",
        description="high-stakes",
        domain=TaskDomain.REGULATORY,
        complexity=TaskComplexity.HARD,
        status=TaskStatus.WAITING_FOR_APPROVAL,
    )
    get_task_store().save_plan(TaskPlan(
        session_id=session_id,
        root_request="Draft the regulatory submission section",
        root_task_id=node.task_id,
        nodes=[node],
    ))
    get_approval_gate().open(node.task_id)
    return node.task_id


async def test_the_approver_is_written_to_the_audit_log(session_id):
    task_id = await _pending_task(session_id)

    async with await _client() as client:
        response = await client.post(
            "/approvals",
            json={
                "task_id": task_id,
                "decision": "approve",
                "approved_by": "dr.chen@example.com",
                "comment": "Reviewed against SOP-042.",
            },
        )
    assert response.status_code == 200

    entries = get_database().get_audit_log(session_id)
    signed = [e for e in entries if e.action == EventType.APPROVAL_RESOLVED.value]
    assert signed, "the approval left no audit entry"

    detail = signed[-1].detail
    assert detail["approved_by"] == "dr.chen@example.com"
    assert detail["decision"] == "approve"
    assert detail["comment"] == "Reviewed against SOP-042."
    assert detail["recorded_at"], "a signature needs a timestamp"


async def test_a_rejection_is_recorded_too(session_id):
    task_id = await _pending_task(session_id)

    async with await _client() as client:
        response = await client.post(
            "/approvals",
            json={"task_id": task_id, "decision": "reject", "approved_by": "qa@example.com"},
        )
    assert response.status_code == 200

    entries = get_database().get_audit_log(session_id)
    signed = [e for e in entries if e.action == EventType.APPROVAL_RESOLVED.value]
    assert signed[-1].detail["decision"] == "reject"
    assert signed[-1].detail["approved_by"] == "qa@example.com"


async def test_the_gate_still_releases_the_task(session_id):
    """Recording must not break the thing it is recording."""
    task_id = await _pending_task(session_id)
    gate = get_approval_gate()
    future = gate._pending[task_id]

    async with await _client() as client:
        await client.post(
            "/approvals",
            json={"task_id": task_id, "decision": "approve", "approved_by": "someone@example.com"},
        )

    assert future.done()
    assert future.result() is ApprovalDecision.APPROVE


async def test_an_unknown_task_is_refused(session_id):
    async with await _client() as client:
        response = await client.post(
            "/approvals",
            json={"task_id": "nope", "decision": "approve", "approved_by": "x@example.com"},
        )
    assert response.status_code == 409  # nothing pending, so nothing to sign
