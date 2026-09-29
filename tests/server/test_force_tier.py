"""force_tier lets a caller pick which tier handles a request, but must
never let a high-stakes request skip the Director — see
docs/server/ARCHITECTURE.md section 2 and src/bedrock/server/orchestrator/loop.py.
"""

import asyncio

from bedrock.server.compliance.approval_gates import get_approval_gate
from bedrock.server.events import get_event_bus
from bedrock.server.orchestrator.loop import handle_message
from bedrock.server.schemas.enums import AgentRole, ApprovalDecision, EventType, TaskStatus
from bedrock.server.task_store.store import get_task_store


async def _auto_approve(session_id: str) -> None:
    """A high-stakes request blocks on a human decision — stand in for the
    human so the test doesn't hang forever waiting for one."""
    bus = get_event_bus()
    gate = get_approval_gate()
    queue = bus.subscribe(session_id)
    while True:
        event = await queue.get()
        if event.type == EventType.APPROVAL_REQUIRED:
            gate.resolve(event.payload["task_id"], ApprovalDecision.APPROVE)
            return


async def test_force_director_runs_the_hard_path_even_for_a_plain_question(session_id):
    task_id = await handle_message(
        session_id, "test-user", "What is the boiling point of water?", force_tier="director"
    )
    task = get_task_store().get_task(task_id)
    # The hard path's synthesis step is assigned to the Director.
    assert task.assigned_to == AgentRole.DIRECTOR
    assert task.status == TaskStatus.COMPLETED


async def test_force_runner_skips_specialist_even_for_a_normal_looking_task(session_id):
    task_id = await handle_message(
        session_id,
        "test-user",
        "Summarize what our internal cell culture SOP says about mycoplasma testing",
        force_tier="runner",
    )
    task = get_task_store().get_task(task_id)
    assert task.assigned_to == AgentRole.RUNNER
    assert task.status == TaskStatus.COMPLETED


async def test_force_tier_cannot_bypass_the_high_stakes_safety_net(session_id):
    # Even asking for the cheapest tier, a high-stakes request must still
    # route through the Director — which means it also hits the human
    # approval gate, so a fake human has to clear it for the test to finish.
    approver = asyncio.create_task(_auto_approve(session_id))
    _, task_id = await asyncio.gather(
        approver,
        handle_message(
            session_id,
            "test-user",
            "This is a final report for regulatory submission on our safety findings.",
            force_tier="runner",
        ),
    )
    task = get_task_store().get_task(task_id)
    assert task.assigned_to == AgentRole.DIRECTOR
