from pydantic import BaseModel

from bedrock.server.schemas.enums import AgentRole, AgentStatus


class AgentCard(BaseModel):
    """Everything the frontend's right-hand 'running agents' panel needs
    to render one agent card."""

    agent_id: str
    name: str
    role: AgentRole
    model_name: str
    status: AgentStatus
    current_task_id: str | None = None
    current_task_title: str | None = None
    tokens_used: int = 0
    elapsed_ms: int = 0
