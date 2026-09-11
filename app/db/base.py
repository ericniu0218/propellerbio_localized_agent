"""SQLAlchemy engine/session setup.

Defaults to a local SQLite file so tests and mock mode need no external
services. In real deployment, set DATABASE_URL to a Postgres DSN — the
audit_log table's INSERT/SELECT-only grant (see docs/ARCHITECTURE.md
section 8) should be applied at the Postgres role level, which SQLite has
no equivalent for; that's expected to be a real-mode-only hardening step.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def make_engine():
    url = get_settings().database_url
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


_engine = make_engine()
SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=_engine)
