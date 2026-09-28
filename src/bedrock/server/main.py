from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from bedrock.server.api import (
    routes_agent_tiers,
    routes_agents,
    routes_approval,
    routes_auth,
    routes_history,
    routes_results,
    routes_session,
    sse,
)
from bedrock.server.compliance.airgap_check import AirgapViolation, run_airgap_check
from bedrock.server.compliance.audit import audit_event_sink
from bedrock.server.config import get_settings
from bedrock.server.events import get_event_bus
from bedrock.server.meetings import routes as routes_meetings
from bedrock.server.workspace import routes as routes_workspace

logger = logging.getLogger("agent_pack")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Anything a restart interrupted. Transcription runs for tens of
    # minutes, so a deploy in the middle of one used to lose it silently.
    try:
        resumed = routes_meetings.resume_interrupted()
        if resumed:
            logger.info("Resumed %d interrupted meeting recording(s)", len(resumed))
    except Exception:
        logger.exception("Could not resume interrupted recordings")

    # Retention runs here rather than on a timer: a background thread that
    # silently deletes company records should not start by importing a
    # module. One pass per start is predictable and auditable.
    try:
        swept = routes_meetings.run_retention_sweep()
        if swept["audioDeleted"]:
            logger.info(
                "Retention: deleted %d recording(s), freed %d MB",
                swept["audioDeleted"], swept["bytesFreed"] // 1_000_000,
            )
    except Exception:
        logger.exception("Retention sweep failed")

    settings = get_settings()
    if settings.airgap_check_enabled and settings.mode == "real":
        try:
            run_airgap_check()
        except AirgapViolation as exc:
            logger.critical(str(exc))
            raise
    get_event_bus().register_audit_sink(audit_event_sink)
    logger.info("Agent pack started in %s mode.", settings.mode)
    yield
    from bedrock.server.models.hub import get_model_hub

    await get_model_hub().aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="Biotech Agent Pack", version="0.1.0", lifespan=lifespan)

    app.include_router(routes_session.router)
    app.include_router(routes_auth.router)
    app.include_router(routes_approval.router)
    app.include_router(routes_history.router)
    app.include_router(routes_agents.router)
    app.include_router(routes_agent_tiers.router)
    app.include_router(routes_results.router)
    app.include_router(sse.router)
    app.include_router(routes_workspace.router)
    # After the workspace router: this one reuses its `identity` dependency and
    # its store, and sits under the same /workspace/api prefix.
    app.include_router(routes_meetings.router)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "mode": get_settings().mode}

    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        # The scientist-facing workspace: three-column chat/projects/memory/
        # meetings UI backed by /workspace/api, wrapping the existing
        # orchestrator rather than replacing it (see WORKSPACE_ARCHITECTURE.md).
        return FileResponse(static_dir / "workspace" / "index.html")

    @app.get("/legacy")
    async def legacy_index() -> FileResponse:
        # The original minimal reference web client — kept, not deleted,
        # per the redesign brief's instruction to wrap rather than remove
        # working functionality.
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()
