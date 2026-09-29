from __future__ import annotations

from fastapi import APIRouter

from bedrock.server.db import get_database
from bedrock.server.schemas.api import AuditLogResponse, HistoryResponse

router = APIRouter(tags=["history"])


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str) -> HistoryResponse:
    turns = get_database().get_history(session_id)
    return HistoryResponse(session_id=session_id, turns=turns)


@router.get("/sessions/{session_id}/audit-log", response_model=AuditLogResponse)
async def get_audit_log(session_id: str) -> AuditLogResponse:
    entries = get_database().get_audit_log(session_id)
    return AuditLogResponse(session_id=session_id, entries=entries)
