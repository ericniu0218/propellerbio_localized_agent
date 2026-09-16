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


class SessionResyncPayload(BaseModel):
    """Sent as a session.resync event when a reconnecting client's
    Last-Event-ID is older than the server's bounded event buffer — see
    app/events/bus.py. `events` is the buffer's current contents (the most
    recent events we still have; there is a gap before them the client
    cannot recover) and is empty when `reason` is `"no_history"` (a session
    the server has never seen any events for, e.g. after a server
    restart — the known in-memory-buffer limitation, see NEXT_STEPS.md §5).
    `terminal` — the session's most recent task.completed/task.failed
    event, if any, kept outside the bounded buffer specifically so a
    client can learn a run finished even if that event itself was evicted
    from `events` — lets the client resolve a pending answer immediately
    instead of waiting forever for an event that will never arrive again.
    """

    reason: str  # "gap" | "unknown_last_event_id" | "no_history"
    events: list["Event"] = Field(default_factory=list)
    terminal: "Event | None" = None


class Event(BaseModel):
    """Envelope for every event on the SSE stream. `type` discriminates the
    shape of `payload` — see EventType and the *_Payload models above.
    Kept as a single generic envelope (rather than a tagged union at the
    wire level) so the frontend can deserialize the envelope first and the
    payload against a lookup table, and so contracts/types.ts stays simple.

    `seq` is a monotonically increasing counter assigned per session by
    EventBus.publish (starting at 1) — separate from `event_id` (a random
    identity used elsewhere, e.g. audit logging). `seq` is what becomes the
    SSE wire `id:` field and what a reconnecting client sends back as
    Last-Event-ID; it exists specifically so "give me everything after N"
    is a simple integer comparison rather than needing an ordered log of
    opaque UUIDs.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    seq: int = 0
    type: EventType
    session_id: str
    timestamp: str = Field(default_factory=_now)
    payload: dict


SessionResyncPayload.model_rebuild()
