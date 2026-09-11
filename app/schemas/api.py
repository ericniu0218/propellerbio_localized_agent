from pydantic import BaseModel, Field

from app.schemas.agents import AgentCard
from app.schemas.enums import AgentRole, ApprovalDecision, TaskStatus


class StartSessionRequest(BaseModel):
    user_id: str


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str


class SendMessageRequest(BaseModel):
    session_id: str
    message: str
    # "runner" | "specialist" | "director" | None (None = let the Runner's
    # classifier decide, the default). Never bypasses the high-stakes
    # safety net — see app/orchestrator/loop.py::handle_message.
    force_tier: str | None = None


class SendMessageResponse(BaseModel):
    session_id: str
    task_id: str
    accepted: bool = True


class CancelTaskRequest(BaseModel):
    task_id: str


class ApprovalRequest(BaseModel):
    task_id: str
    decision: ApprovalDecision
    approved_by: str
    comment: str | None = None


class HistoryTurn(BaseModel):
    turn_id: str
    role: str            # "user" | agent role value
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list[HistoryTurn]


class AuditLogEntry(BaseModel):
    entry_id: str
    session_id: str
    task_id: str | None = None
    user_id: str
    agent: AgentRole | None = None
    model_name: str | None = None
    action: str
    detail: dict
    created_at: str


class AuditLogResponse(BaseModel):
    session_id: str
    entries: list[AuditLogEntry]


class AgentPanelResponse(BaseModel):
    session_id: str
    agents: list[AgentCard]


class AgentTierInfo(BaseModel):
    """One entry in the frontend's tier picker — see GET /agent-tiers.
    `description` is deliberately plain-language (docs/UX.md), even though
    this endpoint also exposes the underlying model name for anyone who
    wants it."""

    key: str            # "runner" | "specialist" | "director"
    name: str            # friendly tier name, e.g. "Deep Analysis"
    model_name: str
    description: str


class AgentTiersResponse(BaseModel):
    tiers: list[AgentTierInfo]


class ResultResponse(BaseModel):
    """Fetches the actual content behind a `result_ref` — agents pass refs
    to each other internally (docs/ARCHITECTURE.md section 6); a client
    (e.g. the terminal app) needs this to show the finished answer."""

    result_ref: str
    content: str
    content_type: str = "text/plain"
