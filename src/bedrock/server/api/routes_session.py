from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.db import get_database
from app.events import get_event_bus
from app.orchestrator.loop import handle_message
from app.orchestrator.session import start_session
from app.schemas.api import (
    CancelTaskRequest,
    SendMessageRequest,
    SendMessageResponse,
    SessionResponse,
    StartSessionRequest,
)
from app.schemas.enums import EventType
from app.schemas.events import TaskFailedPayload
from app.task_store.store import get_task_store

router = APIRouter(tags=["session"])


@router.post("/sessions", response_model=SessionResponse)
async def create_session(req: StartSessionRequest) -> SessionResponse:
    session_id = await start_session(req.user_id)
    return SessionResponse(
        session_id=session_id, user_id=req.user_id, created_at=datetime.now(timezone.utc).isoformat()
    )


@router.post("/messages", response_model=SendMessageResponse)
async def send_message(req: SendMessageRequest) -> SendMessageResponse:
    # Fire-and-forget: the orchestration loop runs in the background and
    # streams progress over SSE (GET /sessions/{id}/events). The task id is
    # generated *here*, synchronously, so the response is always correct —
    # a high-stakes request can block on a human-approval gate for a long
    # time before its first task ever "completes," so waiting on the
    # background run (or racing it with a bare `asyncio.sleep(0)`) to learn
    # the id is not an option. Note this id is the *root* (classification)
    # task; a multi-step plan's eventual final step gets its own id, which
    # a client learns from the `task.planned`/`task.completed` events.
    user_id = get_database().get_session_user(req.session_id)
    root_task_id = get_task_store().new_task_id()
    asyncio.create_task(
        handle_message(req.session_id, user_id, req.message, root_task_id, req.force_tier)
    )
    return SendMessageResponse(session_id=req.session_id, task_id=root_task_id, accepted=True)


@router.post("/tasks/cancel")
async def cancel_task(req: CancelTaskRequest) -> dict:
    store = get_task_store()
    task = store.get_task(req.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="task not found")
    session_id = store.get_task_session_id(req.task_id)
    from app.schemas.enums import TaskStatus

    task.status = TaskStatus.CANCELLED
    store.update_task(session_id, task)
    # Previously silent on the event stream: a client waiting on this task
    # (or reconnecting later, per app/events/bus.py's terminal tracking)
    # would otherwise never learn it was cancelled rather than still running.
    await get_event_bus().publish_typed(
        session_id,
        EventType.TASK_FAILED,
        TaskFailedPayload(
            task_id=req.task_id, title=task.title, error="Cancelled by user.", blocked_on_human=False
        ),
    )
    return {"task_id": req.task_id, "status": "cancelled"}
