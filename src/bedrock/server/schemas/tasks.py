from pydantic import BaseModel, Field

from bedrock.server.schemas.enums import AgentRole, TaskComplexity, TaskDomain, TaskStatus


class TaskNode(BaseModel):
    """One node in the Director's task graph."""

    task_id: str
    title: str
    description: str
    assigned_to: AgentRole
    depends_on: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    complexity: TaskComplexity = TaskComplexity.NORMAL
    domain: TaskDomain = TaskDomain.GENERAL
    is_high_stakes: bool = False
    retry_count: int = 0
    max_retries: int = 2
    result_ref: str | None = None       # reference into the task store, not full content
    rejection_reason: str | None = None
    # Kept separate from `description` (rather than appended to it) so a
    # retry's fix instruction guides drafting without polluting the query
    # text handed to search tools — see docs/ARCHITECTURE.md section 2.
    fix_instruction: str | None = None
    citations: list[str] = Field(default_factory=list)


class TaskPlan(BaseModel):
    """The Director's decomposition of a request into a small dependency graph."""

    session_id: str
    root_request: str
    nodes: list[TaskNode]
    created_by: AgentRole = AgentRole.DIRECTOR
