"""Director reject/retry/escalation loop.

Runner classification always falls back to a generic result against the
mock model (it doesn't emit real JSON), so these tests exercise the
Director/Specialist review loop directly with a deterministic setup rather
than going through the classifier: a citation-required domain
(internal_docs) plus a query that matches nothing in the local RAG corpus
guarantees the Specialist comes back with zero citations every time, which
the Director must reject — per docs/server/ARCHITECTURE.md section 8's
hallucination guardrail — up to `max_retries` times before escalating to a
human (docs/server/ARCHITECTURE.md section 2: "Max 2 retries per task, then
escalate to the human in the chat").
"""

from bedrock.server.agents.director import Director
from bedrock.server.agents.specialist import Specialist
from bedrock.server.events import get_event_bus
from bedrock.server.orchestrator.loop import _run_with_review_loop
from bedrock.server.schemas.enums import AgentRole, EventType, TaskComplexity, TaskDomain, TaskStatus
from bedrock.server.schemas.tasks import TaskNode
from bedrock.server.task_store.store import get_task_store


def _make_node(task_id: str) -> TaskNode:
    return TaskNode(
        task_id=task_id,
        title="Research something not in the corpus",
        # Deliberately zero real English words, so it cannot share a token
        # with any corpus paragraph — the keyword-overlap index requires at
        # least one shared token, and real words (even stopwords like "the"
        # or "that") show up somewhere in the sample corpus.
        description="zqxvbn qwzxcvb fjklopq vbnmzxq tyuiopq asdfghq",
        assigned_to=AgentRole.SPECIALIST,
        complexity=TaskComplexity.HARD,
        domain=TaskDomain.INTERNAL_DOCS,  # citation-required domain
    )


async def test_missing_citation_escalates_to_human_after_max_retries(session_id):
    node = _make_node(get_task_store().new_task_id())
    get_task_store().update_task(session_id, node)

    director = Director(session_id=session_id)
    specialist = Specialist(session_id=session_id, user_id="test-user")

    content, citations, blocked = await _run_with_review_loop(session_id, node, specialist, director)

    assert blocked is True
    assert citations == []

    final = get_task_store().get_task(node.task_id)
    assert final.status == TaskStatus.BLOCKED_ON_HUMAN
    assert final.retry_count == final.max_retries == 2
    assert "citation" in (final.rejection_reason or "").lower()

    events = get_event_bus().history(session_id)
    rejected = [e for e in events if e.type == EventType.REVIEW_REJECTED]
    failed = [e for e in events if e.type == EventType.TASK_FAILED]
    assert len(rejected) == 3  # initial attempt + 2 retries, all rejected
    assert len(failed) == 1
    assert failed[0].payload["blocked_on_human"] is True


async def test_task_with_citations_available_is_approved_without_retry(session_id):
    node = TaskNode(
        task_id=get_task_store().new_task_id(),
        title="Research mycoplasma testing",
        description="mycoplasma contamination cell culture testing",
        assigned_to=AgentRole.SPECIALIST,
        complexity=TaskComplexity.HARD,
        domain=TaskDomain.INTERNAL_DOCS,
    )
    get_task_store().update_task(session_id, node)

    director = Director(session_id=session_id)
    specialist = Specialist(session_id=session_id, user_id="test-user")

    content, citations, blocked = await _run_with_review_loop(session_id, node, specialist, director)

    assert blocked is False
    assert citations, "expected the RAG hit on the SOP corpus to produce a citation"

    final = get_task_store().get_task(node.task_id)
    assert final.status == TaskStatus.COMPLETED
    assert final.retry_count == 0

    events = get_event_bus().history(session_id)
    approved = [e for e in events if e.type == EventType.REVIEW_APPROVED]
    assert len(approved) == 1
