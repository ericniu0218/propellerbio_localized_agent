"""Lets a client (e.g. the web frontend's tier picker) show the three
agent tiers with a plain-language description and the underlying model
name — see docs/UX.md section 1 for why the default view otherwise hides
model names; this endpoint exists specifically for a UI that wants to show
them deliberately (a tier picker, an admin/technical view).

Descriptions live here rather than in config/models.yaml because they're
UI copy, not serving config — but the model name itself is read from that
file, so the two can never drift apart.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import load_models_config
from app.schemas.api import AgentTierInfo, AgentTiersResponse

router = APIRouter(tags=["agent-tiers"])

_DESCRIPTIONS = {
    "runner": (
        "Fast and cheap. Best for simple questions, quick lookups, and "
        "formatting — answers directly, no planning step."
    ),
    "specialist": (
        "The main workhorse. Does the real work — research, data analysis, "
        "and drafting — using documents and tools."
    ),
    "director": (
        "Deep analysis. Plans complex or multi-step work, hands pieces to "
        "the workhorse, then checks the results before anything is final."
    ),
}

_NAMES = {
    "runner": "Quick Answer",
    "specialist": "Standard Task",
    "director": "Deep Analysis",
}


@router.get("/agent-tiers", response_model=AgentTiersResponse)
async def get_agent_tiers() -> AgentTiersResponse:
    models = load_models_config()
    tiers = [
        AgentTierInfo(
            key=key,
            name=_NAMES[key],
            model_name=models.for_role(key).model_name,
            description=_DESCRIPTIONS[key],
        )
        for key in ("runner", "specialist", "director")
    ]
    return AgentTiersResponse(tiers=tiers)
