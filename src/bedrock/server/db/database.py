"""Thin facade over SQLAlchemy sessions used by the task store, audit sink,
and API routers. Kept synchronous and short-lived-session for simplicity;
swap for async SQLAlchemy + asyncpg if request volume ever makes this a
bottleneck (flagged in docs/ARCHITECTURE.md as a known simplification)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from functools import lru_cache
from uuid import uuid4

from app.db.base import SessionLocal, init_db
from app.db.models import (
    AuditLogRow,
    HistoryTurnRow,
    ResultCacheRow,
    ResultRow,
    SessionRow,
    TaskRow,
    TokenUsageRow,
)
from app.schemas.api import AuditLogEntry, HistoryTurn
from app.schemas.enums import AgentRole
from app.schemas.tasks import TaskNode


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self) -> None:
        init_db()

    # -- sessions ----------------------------------------------------------
    def create_session(self, session_id: str, user_id: str) -> None:
        with SessionLocal() as db:
            db.add(SessionRow(session_id=session_id, user_id=user_id))
            db.commit()

    def get_session_user(self, session_id: str) -> str:
        with SessionLocal() as db:
            row = db.get(SessionRow, session_id)
            return row.user_id if row else "unknown"

    # -- history -------------------------------------------------------------
    def add_history_turn(self, session_id: str, role: str, content: str) -> HistoryTurn:
        turn = HistoryTurn(
            turn_id=str(uuid4()), role=role, content=content, created_at=_now_iso()
        )
        with SessionLocal() as db:
            db.add(
                HistoryTurnRow(
                    turn_id=turn.turn_id,
                    session_id=session_id,
                    role=role,
                    content=content,
                )
            )
            db.commit()
        return turn

    def get_history(self, session_id: str) -> list[HistoryTurn]:
        with SessionLocal() as db:
            rows = (
                db.query(HistoryTurnRow)
                .filter(HistoryTurnRow.session_id == session_id)
                .order_by(HistoryTurnRow.created_at)
                .all()
            )
            return [
                HistoryTurn(
                    turn_id=r.turn_id,
                    role=r.role,
                    content=r.content,
                    created_at=r.created_at.isoformat(),
                )
                for r in rows
            ]

    # -- tasks ---------------------------------------------------------------
    def upsert_task(self, session_id: str, node: TaskNode) -> None:
        with SessionLocal() as db:
            row = db.get(TaskRow, node.task_id)
            if row is None:
                row = TaskRow(task_id=node.task_id, session_id=session_id)
            row.title = node.title
            row.status = node.status.value
            row.data = node.model_dump(mode="json")
            db.add(row)
            db.commit()

    def get_task(self, task_id: str) -> TaskNode | None:
        with SessionLocal() as db:
            row = db.get(TaskRow, task_id)
            return TaskNode.model_validate(row.data) if row else None

    def get_task_session_id(self, task_id: str) -> str | None:
        with SessionLocal() as db:
            row = db.get(TaskRow, task_id)
            return row.session_id if row else None

    def list_tasks(self, session_id: str) -> list[TaskNode]:
        with SessionLocal() as db:
            rows = db.query(TaskRow).filter(TaskRow.session_id == session_id).all()
            return [TaskNode.model_validate(r.data) for r in rows]

    # -- results (reference-passing store) -----------------------------------
    def store_result(self, session_id: str, content: str, content_type: str = "text/plain") -> str:
        result_ref = f"res_{uuid4().hex[:12]}"
        with SessionLocal() as db:
            db.add(
                ResultRow(
                    result_ref=result_ref,
                    session_id=session_id,
                    content=content,
                    content_type=content_type,
                )
            )
            db.commit()
        return result_ref

    def get_result(self, result_ref: str) -> str | None:
        with SessionLocal() as db:
            row = db.get(ResultRow, result_ref)
            return row.content if row else None

    # -- result cache ---------------------------------------------------------
    @staticmethod
    def cache_key(query: str, doc_set: list[str]) -> str:
        payload = query + "|" + ",".join(sorted(doc_set))
        return hashlib.sha256(payload.encode()).hexdigest()

    def cache_get(self, key: str) -> str | None:
        with SessionLocal() as db:
            row = db.get(ResultCacheRow, key)
            return row.result_ref if row else None

    def cache_set(self, key: str, result_ref: str) -> None:
        with SessionLocal() as db:
            db.merge(ResultCacheRow(cache_key=key, result_ref=result_ref))
            db.commit()

    # -- audit log (append-only) ---------------------------------------------
    def write_audit(
        self,
        *,
        session_id: str,
        user_id: str,
        action: str,
        detail: dict,
        task_id: str | None = None,
        agent: AgentRole | None = None,
        model_name: str | None = None,
    ) -> AuditLogEntry:
        entry = AuditLogEntry(
            entry_id=str(uuid4()),
            session_id=session_id,
            task_id=task_id,
            user_id=user_id,
            agent=agent,
            model_name=model_name,
            action=action,
            detail=detail,
            created_at=_now_iso(),
        )
        with SessionLocal() as db:
            db.add(
                AuditLogRow(
                    entry_id=entry.entry_id,
                    session_id=session_id,
                    task_id=task_id,
                    user_id=user_id,
                    agent=agent.value if agent else None,
                    model_name=model_name,
                    action=action,
                    detail=detail,
                )
            )
            db.commit()
        return entry

    def get_audit_log(self, session_id: str) -> list[AuditLogEntry]:
        with SessionLocal() as db:
            rows = (
                db.query(AuditLogRow)
                .filter(AuditLogRow.session_id == session_id)
                .order_by(AuditLogRow.created_at)
                .all()
            )
            return [
                AuditLogEntry(
                    entry_id=r.entry_id,
                    session_id=r.session_id,
                    task_id=r.task_id,
                    user_id=r.user_id,
                    agent=AgentRole(r.agent) if r.agent else None,
                    model_name=r.model_name,
                    action=r.action,
                    detail=r.detail,
                    created_at=r.created_at.isoformat(),
                )
                for r in rows
            ]

    # -- token usage -----------------------------------------------------------
    def record_usage(
        self,
        *,
        session_id: str,
        task_id: str | None,
        agent_id: str,
        role: AgentRole,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        with SessionLocal() as db:
            db.add(
                TokenUsageRow(
                    session_id=session_id,
                    task_id=task_id,
                    agent_id=agent_id,
                    role=role.value,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
            db.commit()

    def total_tokens_for_task(self, task_id: str) -> int:
        with SessionLocal() as db:
            rows = db.query(TokenUsageRow).filter(TokenUsageRow.task_id == task_id).all()
            return sum(r.prompt_tokens + r.completion_tokens for r in rows)


@lru_cache
def get_database() -> Database:
    return Database()
