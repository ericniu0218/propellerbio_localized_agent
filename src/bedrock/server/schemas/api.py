from typing import Literal

from pydantic import BaseModel, Field

from bedrock.server.schemas.agents import AgentCard
from bedrock.server.schemas.enums import AgentRole, ApprovalDecision, TaskStatus


class StartSessionRequest(BaseModel):
    user_id: str


class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str


class ContextBlock(BaseModel):
    """Material sent with a message to answer from: an attached file, a stored
    passage, a remembered note. Kept apart from ``message`` so the Runner
    classifies the question rather than reading every attachment, and fenced
    as data, not instructions, when an agent drafts from it."""

    id: str
    title: str
    text: str
    kind: Literal["attachment", "passage", "memory", "artifact", "other"] = "other"


class SendMessageRequest(BaseModel):
    session_id: str
    message: str
    # Optional and additive: a client that sends none behaves exactly as before.
    context: list[ContextBlock] = Field(default_factory=list)
    # "runner" | "specialist" | "director" | None (None = let the Runner's
    # classifier decide, the default). Never bypasses the high-stakes
    # safety net — see src/bedrock/server/orchestrator/loop.py::handle_message.
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
    `description` is deliberately plain-language (docs/server/UX.md), even though
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
    to each other internally (docs/server/ARCHITECTURE.md section 6); a client
    (e.g. the terminal app) needs this to show the finished answer."""

    result_ref: str
    content: str
    content_type: str = "text/plain"
