from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from app.schemas.agents import AgentCard
from app.schemas.enums import AgentRole, AgentStatus, EventType, TaskDomain
from app.schemas.tasks import TaskPlan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionStartedPayload(BaseModel):
    session_id: str
    user_id: str


class TaskPlannedPayload(BaseModel):
    plan: TaskPlan


class TaskAssignedPayload(BaseModel):
    task_id: str
    assigned_to: AgentRole
    title: str


class AgentStartedPayload(BaseModel):
    agent: AgentCard


class AgentStatusPayload(BaseModel):
    agent_id: str
    role: AgentRole
    status: AgentStatus
    task_id: str | None = None
    task_title: str | None = None
    tokens_used: int = 0
    elapsed_ms: int = 0


class AgentTokenPayload(BaseModel):
    """One streamed text chunk from an agent's model call."""

    agent_id: str
    role: AgentRole
    task_id: str | None = None
    text: str
    done: bool = False


class ToolCalledPayload(BaseModel):
    agent_id: str
    role: AgentRole
    task_id: str | None = None
    tool_name: str
    call_id: str
    arguments: dict


class ToolResultPayload(BaseModel):
    agent_id: str
    role: AgentRole
    task_id: str | None = None
    tool_name: str
    call_id: str
    summary: str
    result_ref: str
    citations: list[str] = Field(default_factory=list)


class ReviewDecisionPayload(BaseModel):
    task_id: str
    reviewed_by: AgentRole
    approved: bool
    reason: str | None = None
    fix_instruction: str | None = None
    retry_count: int = 0


class ApprovalRequiredPayload(BaseModel):
    task_id: str
    title: str
    reason: str
    domain: TaskDomain
    preview: str


class TaskCompletedPayload(BaseModel):
    task_id: str
    title: str
    result_ref: str
    citations: list[str] = Field(default_factory=list)


class TaskFailedPayload(BaseModel):
    task_id: str
    title: str
    error: str
    blocked_on_human: bool = False


class UsageUpdatedPayload(BaseModel):
    session_id: str
    agent_id: str
    role: AgentRole
    task_id: str | None = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens_for_task: int
    budget_for_task: int


class Event(BaseModel):
    """Envelope for every event on the SSE stream. `type` discriminates the
    shape of `payload` — see EventType and the *_Payload models above.
    Kept as a single generic envelope (rather than a tagged union at the
    wire level) so the frontend can deserialize the envelope first and the
    payload against a lookup table, and so contracts/types.ts stays simple."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    type: EventType
    session_id: str
    timestamp: str = Field(default_factory=_now)
    payload: dict
