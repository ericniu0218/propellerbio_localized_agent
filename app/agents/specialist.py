"""Specialist: the Qwen workhorse. Does literature/internal-doc research,
bioinformatics/chemistry/data-analysis via tools, and drafts documents. Uses
thinking mode only when the Director marks a task "hard" — see
docs/ARCHITECTURE.md sections 2 and 6.
"""

from __future__ import annotations

from uuid import uuid4

from app.schemas.enums import AgentRole, EventType, TaskDomain
from app.schemas.events import ToolCalledPayload, ToolResultPayload
from app.schemas.tasks import TaskNode
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry, get_tool_registry
from app.agents.base import BaseAgent

_STATIC_PREFIX = (
    "You are the Specialist, the primary workhorse agent in a biotech AI "
    "system. You research, analyze, and draft using the tool results you "
    "are given. Every scientific claim must be backed by a citation or a "
    "tool result — never state a claim without one. Be precise and terse; "
    "you are not writing marketing copy."
)

# Which tool this Specialist reaches for by domain. A real implementation
# would let the model choose freely among tool descriptions; this
# deterministic mapping keeps mock-mode and tests reproducible while still
# exercising the real tool-call/tool-result event path end to end.
_DOMAIN_TOOL: dict[TaskDomain, str | None] = {
    TaskDomain.LITERATURE: "literature_search",
    TaskDomain.INTERNAL_DOCS: "rag_search",
    TaskDomain.BIOINFORMATICS: None,  # tool args are task-specific; see execute()
    TaskDomain.CHEMISTRY: None,
    TaskDomain.DATA_ANALYSIS: None,
    TaskDomain.REGULATORY: "rag_search",
    TaskDomain.CLINICAL: "rag_search",
    TaskDomain.LAB_DATA: None,
    TaskDomain.GENERAL: None,
}


class Specialist(BaseAgent):
    role = AgentRole.SPECIALIST
    static_prompt_prefix = _STATIC_PREFIX

    def __init__(self, *args, tool_registry: ToolRegistry | None = None, user_id: str = "unknown", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tools = tool_registry or get_tool_registry()
        self._user_id = user_id

    async def execute(self, task: TaskNode) -> tuple[str, list[str]]:
        """Runs the task's research/tool step (if its domain implies one),
        then drafts the final content. Returns (content, citations)."""
        tool_result = await self._maybe_call_tool(task)
        content = await self._draft(task, tool_result)
        citations = tool_result.citations if tool_result else []
        return content, citations

    async def _maybe_call_tool(self, task: TaskNode) -> ToolResult | None:
        tool_name = _DOMAIN_TOOL.get(task.domain)
        if tool_name is None:
            return None

        call_id = uuid4().hex[:8]
        args = {"query": task.description}
        if tool_name == "rag_search":
            args["user_id"] = self._user_id

        await self.emit(
            EventType.TOOL_CALLED,
            ToolCalledPayload(
                agent_id=self.agent_id,
                role=self.role,
                task_id=task.task_id,
                tool_name=tool_name,
                call_id=call_id,
                arguments=args,
            ).model_dump(),
        )
        tool = self._tools.get(tool_name)
        result = await tool.run(tool.parse_args(args))
        await self.emit(
            EventType.TOOL_RESULT,
            ToolResultPayload(
                agent_id=self.agent_id,
                role=self.role,
                task_id=task.task_id,
                tool_name=tool_name,
                call_id=call_id,
                summary=result.summary,
                result_ref="",  # orchestrator fills in the task-store ref
                citations=result.citations,
            ).model_dump(),
        )
        return result

    async def _draft(self, task: TaskNode, tool_result: ToolResult | None) -> str:
        context = f"Task: {task.title}\n{task.description}"
        if task.fix_instruction:
            context += f"\n\nThe Director rejected a prior attempt. Fix instruction: {task.fix_instruction}"
        if tool_result and tool_result.content:
            context += f"\n\nTool result ({', '.join(tool_result.citations)}):\n{tool_result.content}"
        thinking = task.complexity.value == "hard"
        return await self.call_model(
            self.build_prompt(context), task_id=task.task_id, thinking=thinking
        )
