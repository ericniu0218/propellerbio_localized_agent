"""Plain data structures for talking to the biotech agent-pack backend.

Deliberately independent of the backend's own Pydantic models — this is a
separate Python process/repo. `AgentEvent.type`/`payload` mirror the
backend's `contracts/types.ts` `Event` envelope closely enough that adding
a new event type there rarely requires a change here (payload is kept as a
plain dict; callers read the keys they need).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentEvent:
    """One event off the backend's SSE stream (`GET /sessions/{id}/events`)."""

    type: str
    session_id: str
    payload: dict


@dataclass
class AgentCardState:
    """What the agent panel needs for one running/finished agent, for the
    current turn. Mirrors the backend's `AgentCard` shape (agent_id, role,
    status, current_task_title) plus a friendly line computed client-side —
    see `bedrock.agentpack.friendly`."""

    agent_id: str
    role: str
    status: str
    task_title: str | None = None

    def friendly_line(self) -> str:
        from bedrock.agentpack.friendly import friendly_agent_line

        return friendly_agent_line(self.role, self.status, self.task_title)
