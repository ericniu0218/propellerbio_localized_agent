"""Specialist: the Qwen workhorse. Does literature/internal-doc research,
bioinformatics/chemistry/data-analysis via tools, and drafts documents. Uses
thinking mode only when the Director marks a task "hard" — see
docs/ARCHITECTURE.md sections 2 and 6.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from bedrock.server.schemas.enums import AgentRole, EventType, TaskDomain
from bedrock.server.schemas.events import ToolCalledPayload, ToolResultPayload
from bedrock.server.schemas.tasks import TaskNode
from bedrock.server.tools.base import ToolResult
from bedrock.server.tools.registry import ToolRegistry, get_tool_registry
from bedrock.server.agents.base import BaseAgent

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
    TaskDomain.MEETINGS: "meeting_transcripts",
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
        args = await self._build_tool_args(tool_name, task)

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

    async def _build_tool_args(self, tool_name: str, task: TaskNode) -> dict:
        if tool_name == "rag_search":
            return {"query": task.description, "user_id": self._user_id}
        if tool_name == "meeting_transcripts":
            # The tool ranks excerpts against the question, so it needs the
            # question itself as well as the date range to fetch.
            from_time, to_time = await self._extract_date_range(task)
            return {
                "query": task.description,
                "from_time": from_time,
                "to_time": to_time,
                "user_id": self._user_id,
            }
        return {"query": task.description}

    async def _extract_date_range(self, task: TaskNode) -> tuple[str, str]:
        """Resolves a relative time phrase in the task ("last Tuesday
        afternoon") into a concrete ISO 8601 [from, to] range, via the same
        model this agent already drafts with — see
        app/models/mock_fixtures.py::_fake_date_range for the mock-mode
        stand-in. Falls back to the last 7 days if the model's output
        doesn't parse, mirroring Runner.classify's fail-safe-toward-a-
        bounded-default philosophy rather than silently searching nothing.
        """
        reference = datetime.now(timezone.utc)
        prompt = (
            f"The current date and time is {reference.isoformat()}. Extract "
            "the date/time range this request refers to. Respond with "
            'JSON: {"from": "<ISO 8601>", "to": "<ISO 8601>"}.\n\n'
            f"Request: {task.description}"
        )
        raw = await self.call_model(self.build_prompt(prompt), task_id=task.task_id)
        try:
            data = json.loads(_extract_json(raw))
            return data["from"], data["to"]
        except (KeyError, ValueError, json.JSONDecodeError):
            return (reference - timedelta(days=7)).isoformat(), reference.isoformat()

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


def _extract_json(raw: str) -> str:
    """Same tolerant extraction as app/agents/runner.py::_extract_json —
    duplicated rather than imported to keep the two agents' modules
    independent; it's 4 lines, not worth coupling over."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise json.JSONDecodeError("no JSON object found", raw, 0)
    return raw[start : end + 1]
