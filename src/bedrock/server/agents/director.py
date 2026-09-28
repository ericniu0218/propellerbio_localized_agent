"""Director: the DeepSeek reasoning model. Plans hard/high-stakes requests
into a small task graph, reviews Specialist output, and handles the hardest
reasoning itself (final synthesis, conflicting-evidence analysis,
regulatory/safety judgment calls). Called rarely and thinks a lot when it
runs — see docs/ARCHITECTURE.md sections 2 and 3.

Never does grunt work: no formatting, no summarizing tool output, no simple
lookups. Those stay with the Runner/Specialist.
"""

from __future__ import annotations

from bedrock.server.schemas.enums import AgentRole, TaskComplexity, TaskDomain, TaskStatus
from bedrock.server.schemas.tasks import TaskNode, TaskPlan
from bedrock.server.agents.base import BaseAgent

_STATIC_PREFIX = (
    "You are the Director, the senior reasoning agent in a biotech AI "
    "system. You only handle hard, multi-step, or high-stakes work: "
    "planning task graphs, reviewing Specialist output for correctness and "
    "citation coverage, judging conflicting evidence and regulatory/safety "
    "risk, and writing final syntheses. You never do formatting, "
    "summarization of raw tool output, or simple lookups — that is the "
    "Runner's and Specialist's job. When you reject a step, always give a "
    "specific, actionable reason and a concrete fix instruction."
)

# Citations are mandatory in these domains — the Director rejects on their
# absence without needing a model call, per the hallucination-guardrail
# requirement in docs/ARCHITECTURE.md section 8.
CITATION_REQUIRED_DOMAINS = {
    TaskDomain.LITERATURE,
    TaskDomain.INTERNAL_DOCS,
    TaskDomain.REGULATORY,
    TaskDomain.CLINICAL,
    TaskDomain.MEETINGS,
}


class ReviewOutcome:
    def __init__(self, approved: bool, reason: str | None = None, fix_instruction: str | None = None) -> None:
        self.approved = approved
        self.reason = reason
        self.fix_instruction = fix_instruction


class Director(BaseAgent):
    role = AgentRole.DIRECTOR
    static_prompt_prefix = _STATIC_PREFIX

    async def plan(self, session_id: str, request: str, domain: TaskDomain, is_high_stakes: bool) -> TaskPlan:
        """Decomposes a hard/high-stakes request into a small dependency
        graph. Kept to a research-then-synthesize shape by default — a real
        deployment would let the Director's model output an arbitrary
        graph; this default is deterministic enough to test while still
        routing through a genuine Director review step for the final node.
        """
        from bedrock.server.task_store.store import TaskStore

        research_id = TaskStore.new_task_id()
        synth_id = TaskStore.new_task_id()

        nodes = [
            TaskNode(
                task_id=research_id,
                title="Research and draft",
                description=request,
                assigned_to=AgentRole.SPECIALIST,
                depends_on=[],
                complexity=TaskComplexity.HARD,
                domain=domain,
                is_high_stakes=is_high_stakes,
            ),
            TaskNode(
                task_id=synth_id,
                title="Final synthesis and review",
                description=f"Synthesize a final answer for: {request}",
                assigned_to=AgentRole.DIRECTOR,
                depends_on=[research_id],
                complexity=TaskComplexity.HARD,
                domain=domain,
                is_high_stakes=is_high_stakes,
            ),
        ]
        return TaskPlan(session_id=session_id, root_request=request, nodes=nodes)

    def review(self, task: TaskNode, content: str, citations: list[str]) -> ReviewOutcome:
        """Deterministic hallucination/citation guardrail first (no model
        call needed for a hard rule); qualitative judgment would be a model
        call in real mode for anything that passes the hard rule but is
        still high-stakes. Kept synchronous+deterministic here for testable,
        low-latency retry-loop behavior; a real deployment can upgrade the
        "high-stakes qualitative" branch to call self.call_model.
        """
        if task.domain in CITATION_REQUIRED_DOMAINS and not citations:
            return ReviewOutcome(
                approved=False,
                reason="No citation attached to a claim in a citation-required domain.",
                fix_instruction="Re-run with rag_search/literature_search and cite the specific doc_id#section.",
            )
        if not content.strip():
            return ReviewOutcome(
                approved=False,
                reason="Empty output.",
                fix_instruction="Produce a substantive answer to the task description.",
            )
        return ReviewOutcome(approved=True)

    async def synthesize(
        self, task: TaskNode, upstream_content: str, upstream_citations: list[str], extra: str = ""
    ) -> str:
        # ``extra`` is the conversation so far; the material sent with the
        # message already reached the research step, and arrives here digested.
        context = (
            f"Synthesize a final answer for: {task.description}\n\n"
            f"Upstream research/draft:\n{upstream_content}\n\n"
            f"Citations available: {', '.join(upstream_citations) or '(none)'}"
        )
        if extra:
            context += f"\n\n{extra}"
        return await self.call_model(self.build_prompt(context), task_id=task.task_id, thinking=True)
