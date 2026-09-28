"""The main orchestration loop: routes a message per app.orchestrator.router,
runs the appropriate agent(s), drives the Director's reject/retry/escalate
loop, and gates high-stakes final output on human approval. Every step
publishes events on the EventBus (which also feeds the audit log) — see
docs/ARCHITECTURE.md sections 2, 5, and 8.

Runs as a background asyncio task per message (scheduled by the API layer)
so REST calls return immediately and the frontend follows progress over SSE.
"""

from __future__ import annotations

from app.agents.director import Director
from app.agents.runner import Runner
from app.agents.specialist import Specialist
from app.compliance.approval_gates import get_approval_gate
from app.db import get_database
from app.events import get_event_bus
from app.orchestrator.router import effective_complexity, is_high_stakes
from app.schemas.enums import (
    AgentRole,
    AgentStatus,
    ApprovalDecision,
    EventType,
    TaskComplexity,
    TaskStatus,
)
from app.schemas.events import (
    ApprovalRequiredPayload,
    ReviewDecisionPayload,
    TaskAssignedPayload,
    TaskCompletedPayload,
    TaskFailedPayload,
    TaskPlannedPayload,
)
from app.schemas.tasks import TaskNode
from app.task_store.store import get_task_store


_FORCE_TIER_COMPLEXITY = {
    "runner": TaskComplexity.SIMPLE,
    "specialist": TaskComplexity.NORMAL,
    "director": TaskComplexity.HARD,
}


async def handle_message(
    session_id: str,
    user_id: str,
    message: str,
    root_task_id: str | None = None,
    force_tier: str | None = None,
) -> str:
    """`root_task_id` can be supplied by the caller so it's known
    synchronously, before this (potentially long-running, gate-blocking)
    coroutine is scheduled as a background task — see
    `app/api/routes_session.py::send_message`, which does exactly that
    rather than racing a fire-and-forget task with `asyncio.sleep(0)`.

    `force_tier` ("runner" | "specialist" | "director") lets a caller pick
    which tier handles the request instead of letting the Runner's
    classifier decide — e.g. the frontend's tier picker / "Create Agent"
    button. It never overrides the high-stakes safety net below: a
    regulatory/clinical/safety request is always routed through the
    Director regardless of what was requested, since that check exists
    specifically so a UI control can't be used to skip it.
    """
    bus = get_event_bus()
    db = get_database()
    store = get_task_store()

    db.add_history_turn(session_id, "user", message)
    root_task_id = root_task_id or store.new_task_id()

    runner = Runner(session_id=session_id)
    await runner.emit_started(root_task_id, "Classify request")
    classification = await runner.classify(message, task_id=root_task_id)
    high_stakes = is_high_stakes(classification.domain, message)
    if high_stakes:
        complexity = TaskComplexity.HARD
    elif force_tier in _FORCE_TIER_COMPLEXITY:
        complexity = _FORCE_TIER_COMPLEXITY[force_tier]
    else:
        complexity = effective_complexity(classification.complexity, high_stakes)
    await runner.emit_status(AgentStatus.DONE, root_task_id, "Classify request")

    if complexity == TaskComplexity.SIMPLE and not high_stakes:
        return await _handle_simple(session_id, user_id, message, root_task_id, classification, runner)

    if complexity == TaskComplexity.NORMAL and not high_stakes:
        return await _handle_normal(session_id, user_id, message, root_task_id, classification, runner)

    return await _handle_hard(session_id, user_id, message, root_task_id, classification, high_stakes, runner)


async def _handle_simple(session_id, user_id, message, task_id, classification, runner: Runner) -> str:
    bus = get_event_bus()
    db = get_database()
    store = get_task_store()

    node = TaskNode(
        task_id=task_id,
        title="Answer directly",
        description=message,
        assigned_to=AgentRole.RUNNER,
        complexity=classification.complexity,
        domain=classification.domain,
        status=TaskStatus.IN_PROGRESS,
    )
    store.update_task(session_id, node)
    await bus.publish_typed(session_id, EventType.TASK_ASSIGNED, TaskAssignedPayload(
        task_id=task_id, assigned_to=AgentRole.RUNNER, title=node.title
    ))

    answer = await runner.call_model(runner.build_prompt(message), task_id=task_id)
    formatted = await runner.format_output(answer, [], task_id=task_id)
    result_ref = store.store_result(session_id, formatted)

    node.status = TaskStatus.COMPLETED
    node.result_ref = result_ref
    store.update_task(session_id, node)
    db.add_history_turn(session_id, "runner", formatted)

    await runner.emit_status(AgentStatus.DONE, task_id, node.title)
    await bus.publish_typed(session_id, EventType.TASK_COMPLETED, TaskCompletedPayload(
        task_id=task_id, title=node.title, result_ref=result_ref, citations=[]
    ))
    return task_id


async def _handle_normal(session_id, user_id, message, task_id, classification, runner: Runner) -> str:
    bus = get_event_bus()
    db = get_database()
    store = get_task_store()

    node = TaskNode(
        task_id=task_id,
        title="Specialist task",
        description=message,
        assigned_to=AgentRole.SPECIALIST,
        complexity=classification.complexity,
        domain=classification.domain,
        status=TaskStatus.IN_PROGRESS,
    )
    store.update_task(session_id, node)
    await bus.publish_typed(session_id, EventType.TASK_ASSIGNED, TaskAssignedPayload(
        task_id=task_id, assigned_to=AgentRole.SPECIALIST, title=node.title
    ))

    specialist = Specialist(session_id=session_id, user_id=user_id)
    await specialist.emit_started(task_id, node.title)
    content, citations = await specialist.execute(node)
    await specialist.emit_status(AgentStatus.DONE, task_id, node.title)

    formatted = await runner.format_output(content, citations, task_id=task_id)
    result_ref = store.store_result(session_id, formatted)

    node.status = TaskStatus.COMPLETED
    node.result_ref = result_ref
    node.citations = citations
    store.update_task(session_id, node)
    db.add_history_turn(session_id, "specialist", formatted)

    await bus.publish_typed(session_id, EventType.TASK_COMPLETED, TaskCompletedPayload(
        task_id=task_id, title=node.title, result_ref=result_ref, citations=citations
    ))
    return task_id


async def _handle_hard(session_id, user_id, message, task_id, classification, high_stakes, runner: Runner) -> str:
    bus = get_event_bus()
    db = get_database()
    store = get_task_store()
    gate = get_approval_gate()

    director = Director(session_id=session_id)
    await director.emit_started(task_id, "Plan task")
    plan = await director.plan(session_id, message, classification.domain, high_stakes)
    store.save_plan(plan)
    await bus.publish_typed(session_id, EventType.TASK_PLANNED, TaskPlannedPayload(plan=plan))
    await director.emit_status(AgentStatus.DONE, task_id, "Plan task")

    research_node, synth_node = plan.nodes[0], plan.nodes[1]
    specialist = Specialist(session_id=session_id, user_id=user_id)

    content, citations, blocked = await _run_with_review_loop(
        session_id, research_node, specialist, director
    )
    if blocked:
        return research_node.task_id

    upstream_content, upstream_citations = content, citations

    await bus.publish_typed(session_id, EventType.TASK_ASSIGNED, TaskAssignedPayload(
        task_id=synth_node.task_id, assigned_to=AgentRole.DIRECTOR, title=synth_node.title
    ))
    await director.emit_started(synth_node.task_id, synth_node.title)
    final_text = await director.synthesize(synth_node, upstream_content, upstream_citations)
    await director.emit_status(AgentStatus.DONE, synth_node.task_id, synth_node.title)

    outcome = director.review(synth_node, final_text, upstream_citations)
    if not outcome.approved:
        synth_node.status = TaskStatus.BLOCKED_ON_HUMAN
        synth_node.rejection_reason = outcome.reason
        store.update_task(session_id, synth_node)
        await bus.publish_typed(session_id, EventType.TASK_FAILED, TaskFailedPayload(
            task_id=synth_node.task_id, title=synth_node.title, error=outcome.reason or "review failed",
            blocked_on_human=True,
        ))
        return synth_node.task_id

    # Human approval gate for high-stakes final output — see
    # docs/ARCHITECTURE.md section 8.
    if high_stakes:
        synth_node.status = TaskStatus.WAITING_FOR_APPROVAL
        store.update_task(session_id, synth_node)
        # Open the gate's future BEFORE publishing the event: a subscriber
        # (e.g. the API's approve endpoint, or a fast test) may resolve it
        # the instant it sees the event, and resolve() is a no-op if no
        # future is open yet — opening first closes that race.
        gate.open(synth_node.task_id)
        await bus.publish_typed(session_id, EventType.APPROVAL_REQUIRED, ApprovalRequiredPayload(
            task_id=synth_node.task_id,
            title=synth_node.title,
            reason="High-stakes output requires human approval before completion.",
            domain=synth_node.domain,
            preview=final_text[:500],
        ))
        decision = await gate.wait_for_decision(synth_node.task_id)
        if decision == ApprovalDecision.REJECT:
            synth_node.status = TaskStatus.FAILED
            store.update_task(session_id, synth_node)
            await bus.publish_typed(session_id, EventType.TASK_FAILED, TaskFailedPayload(
                task_id=synth_node.task_id, title=synth_node.title,
                error="Rejected by human approver.", blocked_on_human=False,
            ))
            return synth_node.task_id

    formatted = await runner.format_output(final_text, upstream_citations, task_id=synth_node.task_id)
    result_ref = store.store_result(session_id, formatted)
    synth_node.status = TaskStatus.COMPLETED
    synth_node.result_ref = result_ref
    synth_node.citations = upstream_citations
    store.update_task(session_id, synth_node)
    db.add_history_turn(session_id, "director", formatted)

    await bus.publish_typed(session_id, EventType.TASK_COMPLETED, TaskCompletedPayload(
        task_id=synth_node.task_id, title=synth_node.title, result_ref=result_ref,
        citations=upstream_citations,
    ))
    return synth_node.task_id


async def _run_with_review_loop(
    session_id: str, node: TaskNode, specialist: Specialist, director: Director
) -> tuple[str, list[str], bool]:
    """Runs a Specialist step, then the Director review/retry loop (max 2
    retries per docs/ARCHITECTURE.md). Returns (content, citations, blocked)
    where blocked=True means it was escalated to a human and the caller
    should stop."""
    bus = get_event_bus()
    store = get_task_store()

    await bus.publish_typed(session_id, EventType.TASK_ASSIGNED, TaskAssignedPayload(
        task_id=node.task_id, assigned_to=AgentRole.SPECIALIST, title=node.title
    ))

    working_node = node
    while True:
        await specialist.emit_started(working_node.task_id, working_node.title)
        content, citations = await specialist.execute(working_node)
        await specialist.emit_status(AgentStatus.DONE, working_node.task_id, working_node.title)

        outcome = director.review(working_node, content, citations)
        if outcome.approved:
            await bus.publish_typed(session_id, EventType.REVIEW_APPROVED, ReviewDecisionPayload(
                task_id=working_node.task_id, reviewed_by=AgentRole.DIRECTOR, approved=True,
                retry_count=working_node.retry_count,
            ))
            working_node.status = TaskStatus.COMPLETED
            working_node.citations = citations
            store.update_task(session_id, working_node)
            return content, citations, False

        await bus.publish_typed(session_id, EventType.REVIEW_REJECTED, ReviewDecisionPayload(
            task_id=working_node.task_id, reviewed_by=AgentRole.DIRECTOR, approved=False,
            reason=outcome.reason, fix_instruction=outcome.fix_instruction,
            retry_count=working_node.retry_count,
        ))

        if working_node.retry_count >= working_node.max_retries:
            working_node.status = TaskStatus.BLOCKED_ON_HUMAN
            working_node.rejection_reason = outcome.reason
            store.update_task(session_id, working_node)
            await bus.publish_typed(session_id, EventType.TASK_FAILED, TaskFailedPayload(
                task_id=working_node.task_id, title=working_node.title,
                error=outcome.reason or "review failed", blocked_on_human=True,
            ))
            return content, citations, True

        working_node = working_node.model_copy(update={
            "retry_count": working_node.retry_count + 1,
            "rejection_reason": outcome.reason,
            "fix_instruction": outcome.fix_instruction,
        })
        store.update_task(session_id, working_node)
