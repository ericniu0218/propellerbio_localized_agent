from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import (
    routes_agent_tiers,
    routes_agents,
    routes_approval,
    routes_auth,
    routes_history,
    routes_results,
    routes_session,
    sse,
)
from app.compliance.airgap_check import AirgapViolation, run_airgap_check
from app.compliance.audit import audit_event_sink
from app.config import get_settings
from app.events import get_event_bus

logger = logging.getLogger("agent_pack")


def create_app() -> FastAPI:
    app = FastAPI(title="Biotech Agent Pack", version="0.1.0")

    app.include_router(routes_session.router)
    app.include_router(routes_auth.router)
    app.include_router(routes_approval.router)
    app.include_router(routes_history.router)
    app.include_router(routes_agents.router)
    app.include_router(routes_agent_tiers.router)
    app.include_router(routes_results.router)
    app.include_router(sse.router)

    @app.on_event("startup")
    async def startup() -> None:
        settings = get_settings()
        if settings.airgap_check_enabled and settings.mode == "real":
            try:
                run_airgap_check()
            except AirgapViolation as exc:
                logger.critical(str(exc))
                raise
        get_event_bus().register_audit_sink(audit_event_sink)
        logger.info("Agent pack started in %s mode.", settings.mode)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "mode": get_settings().mode}

    static_dir = Path(__file__).resolve().parent / "static"

    @app.get("/")
    async def index() -> FileResponse:
        # A minimal reference web client (not the primary product surface —
        # see the terminal client in ../frontend — but a real, no-install
        # page anyone can open at this server's address in a browser).
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
