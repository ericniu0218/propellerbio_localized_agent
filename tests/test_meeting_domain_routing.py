"""End-to-end check that a meetings-domain task actually reaches the
meeting_transcripts tool through the real Specialist/Director flow —
not just that the tool works in isolation (tests/test_meeting_transcripts_tool.py
already covers that). Registering a tool is not the same as it being
reachable; this is what catches the difference.
"""

from app.agents.director import Director
from app.agents.specialist import Specialist
from app.events import get_event_bus
from app.orchestrator.loop import _run_with_review_loop, handle_message
from app.schemas.enums import AgentRole, EventType, TaskComplexity, TaskDomain, TaskStatus
from app.task_store.store import get_task_store
from app.tools.base import Tool, ToolResult
from pydantic import BaseModel


class _RecordingArgs(BaseModel):
    from_time: str
    to_time: str
    user_id: str


class _RecordingMeetingTool(Tool):
    name = "meeting_transcripts"
    description = "test double"
    args_model = _RecordingArgs

    def __init__(self) -> None:
        self.calls: list[_RecordingArgs] = []

    async def run(self, args: _RecordingArgs) -> ToolResult:
        self.calls.append(args)
        return ToolResult(
            summary="Found 1 meeting transcript.",
            content="[meeting:m1] Weekly sync: we shipped the thing.",
            citations=["meeting:m1"],
        )


class _FakeRegistry:
    def __init__(self, tool: Tool) -> None:
        self._tool = tool

    def get(self, name: str) -> Tool:
        assert name == "meeting_transcripts"
        return self._tool


async def test_meetings_domain_task_reaches_the_tool_with_a_resolved_date_range(session_id):
    tool = _RecordingMeetingTool()
    node = get_task_store().new_task_id()
    from app.schemas.tasks import TaskNode

    task = TaskNode(
        task_id=node,
        title="What happened in yesterday's meeting",
        description="What happened in yesterday's standup meeting?",
        assigned_to=AgentRole.SPECIALIST,
        complexity=TaskComplexity.NORMAL,
        domain=TaskDomain.MEETINGS,
    )
    get_task_store().update_task(session_id, task)

    director = Director(session_id=session_id)
    specialist = Specialist(session_id=session_id, user_id="alice", tool_registry=_FakeRegistry(tool))

    content, citations, blocked = await _run_with_review_loop(session_id, task, specialist, director)

    assert blocked is False
    assert citations == ["meeting:m1"]
    assert len(tool.calls) == 1
    call = tool.calls[0]
    assert call.user_id == "alice"
    # The mock date-range fixture resolves "yesterday" to a real ISO range,
    # not a placeholder — this is the actual proof the extraction step ran.
    assert call.from_time < call.to_time

    final = get_task_store().get_task(task.task_id)
    assert final.status == TaskStatus.COMPLETED


async def test_meetings_domain_without_a_citation_is_rejected_like_other_citation_domains(session_id):
    """Meeting answers must cite which meeting, same hallucination
    guardrail as literature/internal_docs/regulatory/clinical."""

    class _NoCitationTool(_RecordingMeetingTool):
        async def run(self, args: _RecordingArgs) -> ToolResult:
            self.calls.append(args)
            return ToolResult(summary="No meeting transcripts found.", content="", citations=[])

    tool = _NoCitationTool()
    from app.schemas.tasks import TaskNode

    task = TaskNode(
        task_id=get_task_store().new_task_id(),
        title="What happened last week",
        description="What happened in last week's meetings?",
        assigned_to=AgentRole.SPECIALIST,
        complexity=TaskComplexity.NORMAL,
        domain=TaskDomain.MEETINGS,
    )
    get_task_store().update_task(session_id, task)

    director = Director(session_id=session_id)
    specialist = Specialist(session_id=session_id, user_id="alice", tool_registry=_FakeRegistry(tool))

    _, citations, blocked = await _run_with_review_loop(session_id, task, specialist, director)

    assert blocked is True
    assert citations == []


async def test_a_raw_chat_message_reaches_the_real_tool_through_handle_message(session_id):
    """Goes through the actual public entrypoint (app.api.routes_session
    calls exactly this) with the real tool registry — no stub. With
    DATACENTER_BASE_URL unset in the test environment, the real tool
    correctly reports itself unconfigured rather than attempting a
    network call (see test_meeting_transcripts_tool.py), which is what
    this test actually verifies reaches all the way through: mock
    classification -> meetings domain -> date extraction -> real tool.

    Note: mock classification gives "meetings" TaskComplexity.NORMAL (see
    app/models/mock_fixtures.py::_fake_classification), which routes
    through _handle_normal — the same tier literature_search/rag_search
    get by default. That tier does not run the Director's citation
    guardrail (only _handle_hard does), so a citation-required domain
    classified as merely "normal" can complete without one. That gap
    predates this change and isn't specific to meetings — see
    docs/ARCHITECTURE.md open question 8.
    """
    await handle_message(
        session_id=session_id,
        user_id="alice",
        message="What happened in yesterday's standup meeting?",
    )

    events = get_event_bus().history(session_id)
    tool_calls = [e for e in events if e.type == EventType.TOOL_CALLED]
    assert any(e.payload["tool_name"] == "meeting_transcripts" for e in tool_calls)

    tool_call = next(e for e in tool_calls if e.payload["tool_name"] == "meeting_transcripts")
    args = tool_call.payload["arguments"]
    assert args["user_id"] == "alice"
    assert args["from_time"] < args["to_time"]
