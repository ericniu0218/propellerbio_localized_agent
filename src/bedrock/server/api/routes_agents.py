"""Snapshot endpoint for the right-hand 'running agents' panel — the SSE
stream is the primary source of truth, but a fresh page load wants an
immediate snapshot too. Derived from the event history rather than a
separate store, since agent.status/agent.started events already carry
everything an AgentCard needs.
"""

from __future__ import annotations

from fastapi import APIRouter

from bedrock.server.events import get_event_bus
from bedrock.server.schemas.agents import AgentCard
from bedrock.server.schemas.api import AgentPanelResponse
from bedrock.server.schemas.enums import EventType

router = APIRouter(tags=["agents"])


@router.get("/sessions/{session_id}/agents", response_model=AgentPanelResponse)
async def get_agent_panel(session_id: str) -> AgentPanelResponse:
    bus = get_event_bus()
    cards: dict[str, AgentCard] = {}
    for event in bus.history(session_id):
        if event.type == EventType.AGENT_STARTED:
            card = AgentCard.model_validate(event.payload["agent"])
            cards[card.agent_id] = card
        elif event.type == EventType.AGENT_STATUS:
            p = event.payload
            if p["agent_id"] in cards:
                card = cards[p["agent_id"]]
                cards[p["agent_id"]] = card.model_copy(update={
                    "status": p["status"],
                    "current_task_id": p.get("task_id"),
                    "current_task_title": p.get("task_title"),
                    "tokens_used": p.get("tokens_used", card.tokens_used),
                    "elapsed_ms": p.get("elapsed_ms", card.elapsed_ms),
                })
    return AgentPanelResponse(session_id=session_id, agents=list(cards.values()))
