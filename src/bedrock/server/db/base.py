"""SQLAlchemy engine/session setup.

Defaults to a SQLite file in the server's data directory so tests and mock mode
need no external services. In real deployment, set DATABASE_URL to a Postgres
DSN - the audit_log table's INSERT/SELECT-only grant (see
docs/server/ARCHITECTURE.md section 8) should be applied at the Postgres role
level, which SQLite has no equivalent for; that's expected to be a
real-mode-only hardening step.

The engine is made on first use, not at import: which database a process talks
to is a setting, and settings are read when first asked for (see
``bedrock.server.config``). ``dispose_engine`` lets tests start again.
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from bedrock.server.config import get_settings, private_dir


class Base(DeclarativeBase):
    pass


_engine = None
_sessions = None


def make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    if url.startswith("sqlite:///") and url != "sqlite:///:memory:":
        private_dir(Path(url[len("sqlite:///"):]).expanduser().parent)
    return create_engine(url, connect_args=connect_args)


def get_engine():
    global _engine, _sessions
    if _engine is None:
        _engine = make_engine(get_settings().database_url)
        _sessions = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def SessionLocal():  # noqa: N802 - kept under its old name; callers use `with SessionLocal() as db`
    get_engine()
    return _sessions()


def init_db() -> None:
    Base.metadata.create_all(bind=get_engine())


def dispose_engine() -> None:
    global _engine, _sessions
    if _engine is not None:
        _engine.dispose()
    _engine = _sessions = None
