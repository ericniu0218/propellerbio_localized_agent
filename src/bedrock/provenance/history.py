"""Durable local action history for compliance-facing employee workflows.

The history records what this client attempted and what it later learned about
the outcome. It is intentionally metadata-only and local to the configured OS
account: it does not authenticate the configured actor and is not tamper-proof.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_HISTORY_DIR = Path.home() / ".local" / "share" / "bedrock" / "history"


class HistoryError(RuntimeError):
    """The local action history could not validate or complete an operation."""


class HistoryStore:
    """Append-only metadata journal with replay-safe intent/outcome slots.

    Every method opens its own SQLite connection. That keeps the store safe to
    call from separate CLI processes and from ``asyncio.to_thread`` workers.
    """

    ACTIONS = frozenset({"fold", "digest.ingest", "digest.forget", "approval", "run.tag"})
    STATES = frozenset({"intent", "succeeded", "failed", "cancelled", "unknown"})
    DETAIL_FIELDS = frozenset(
        {
            "artifact_id",
            "run_id",
            "project_id",
            "session_id",
            "task_id",
            "decision",
            "error_code",
            "manifest_sha256",
            "source_sha256",
            "source_kind",
            "source_bytes",
            "actor_source",
        }
    )

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root).expanduser() if root else DEFAULT_HISTORY_DIR
        self._prepare_storage()
        self._migrate()

    @property
    def database_path(self) -> Path:
        return self.root / "history.sqlite3"

    def append(
        self,
        operation_id: str,
        *,
        actor_id: str,
        action: str,
        state: str,
        subject_id: str | None = None,
        details: dict | None = None,
    ) -> int:
        body = self._event_body(
            operation_id,
            actor_id=actor_id,
            action=action,
            state=state,
            subject_id=subject_id,
            details=details,
        )
        slot = "intent" if state == "intent" else "outcome"
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN IMMEDIATE")
                existing = db.execute(
                    "SELECT sequence, body FROM events WHERE operation_id = ? AND slot = ?",
                    (operation_id, slot),
                ).fetchone()
                if existing is not None:
                    if existing["body"] != body:
                        raise HistoryError(
                            "Operation ID was reused with different event content."
                        )
                    db.commit()
                    return int(existing["sequence"])

                if slot == "outcome":
                    intent = db.execute(
                        "SELECT body FROM events WHERE operation_id = ? AND slot = 'intent'",
                        (operation_id,),
                    ).fetchone()
                    prior = json.loads(intent["body"]) if intent is not None else {}
                    decoded = json.loads(body)
                    if (
                        prior.get("actor_id") != decoded["actor_id"]
                        or prior.get("action") != decoded["action"]
                    ):
                        raise HistoryError("Outcome requires the matching committed intent.")

                cursor = db.execute(
                    """INSERT INTO events
                    (operation_id, slot, recorded_at, body)
                    VALUES (?, ?, ?, ?)""",
                    (operation_id, slot, _now(), body),
                )
                db.commit()
                return int(cursor.lastrowid)
        except HistoryError:
            raise
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            raise HistoryError("Could not write the local action history.") from exc

    def events(self, *, after: int = 0, limit: int = 50) -> list[dict]:
        if type(after) is not int or after < 0:
            raise HistoryError("History cursor must be a non-negative number.")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise HistoryError("History page size must be between 1 and 100.")
        try:
            with closing(self._connect()) as db:
                rows = db.execute(
                    """SELECT sequence, operation_id, recorded_at, body
                    FROM events WHERE sequence > ? ORDER BY sequence LIMIT ?""",
                    (after, limit),
                ).fetchall()
            return [self._decode_row(row) for row in rows]
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            raise HistoryError("Could not read the local action history.") from exc

    def export(self, destination: Path) -> int:
        """Write a consistent, private JSONL snapshot without overwriting files."""
        target = destination.expanduser()
        try:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            file_descriptor = os.open(target, flags, 0o600)
        except FileExistsError as exc:
            raise HistoryError(f"Export destination already exists: {target}") from exc
        except OSError as exc:
            raise HistoryError("Could not create the history export.") from exc

        count = 0
        try:
            with closing(self._connect()) as db:
                db.execute("BEGIN")
                cursor = db.execute(
                    """SELECT sequence, operation_id, recorded_at, body
                    FROM events ORDER BY sequence"""
                )
                with os.fdopen(file_descriptor, "w", encoding="utf-8") as output:
                    file_descriptor = -1
                    for row in cursor:
                        output.write(
                            json.dumps(
                                self._decode_row(row),
                                sort_keys=True,
                                separators=(",", ":"),
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        count += 1
                    output.flush()
                    os.fsync(output.fileno())
                db.commit()
            return count
        except (OSError, sqlite3.Error, ValueError, json.JSONDecodeError) as exc:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
            raise HistoryError("Could not export the local action history.") from exc

    def latest_project_assignments(self) -> dict[str, str]:
        """Return the latest successful project tag for each run.

        This is an internal projection for ``RunCatalog``. The events remain the
        source of truth and no mutable second tag store is introduced.
        """
        try:
            with closing(self._connect()) as db:
                rows = db.execute(
                    "SELECT sequence, body FROM events WHERE slot = 'outcome' ORDER BY sequence"
                ).fetchall()
            assignments: dict[str, str] = {}
            for row in rows:
                body = json.loads(row["body"])
                details = body.get("details", {})
                if (
                    body.get("action") == "run.tag"
                    and body.get("state") == "succeeded"
                    and isinstance(body.get("subject_id"), str)
                    and isinstance(details.get("project_id"), str)
                ):
                    assignments[body["subject_id"]] = details["project_id"]
            return assignments
        except (OSError, sqlite3.Error, json.JSONDecodeError) as exc:
            raise HistoryError("Could not read run project assignments.") from exc

    def _prepare_storage(self) -> None:
        try:
            if self.root.is_symlink():
                raise HistoryError("History directory must not be a symlink.")
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root.chmod(0o700)
            path = self.database_path
            if path.is_symlink():
                raise HistoryError("History database must not be a symlink.")
            flags = os.O_CREAT | os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            os.close(descriptor)
            path.chmod(0o600)
        except HistoryError:
            raise
        except OSError as exc:
            raise HistoryError("Could not prepare the private history directory.") from exc

    def _migrate(self) -> None:
        try:
            with closing(self._connect()) as db, db:
                db.execute(
                    "CREATE TABLE IF NOT EXISTS history_schema (version INTEGER NOT NULL)"
                )
                row = db.execute("SELECT version FROM history_schema").fetchone()
                if row is None:
                    db.execute("INSERT INTO history_schema(version) VALUES (?)", (SCHEMA_VERSION,))
                elif int(row["version"]) != SCHEMA_VERSION:
                    raise HistoryError(
                        f"Unsupported history schema version: {row['version']}"
                    )
                db.execute(
                    """CREATE TABLE IF NOT EXISTS events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        operation_id TEXT NOT NULL,
                        slot TEXT NOT NULL CHECK(slot IN ('intent', 'outcome')),
                        recorded_at TEXT NOT NULL,
                        body TEXT NOT NULL,
                        UNIQUE(operation_id, slot)
                    )"""
                )
        except HistoryError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise HistoryError("Could not initialize the local action history.") from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _event_body(
        self,
        operation_id: str,
        *,
        actor_id: str,
        action: str,
        state: str,
        subject_id: str | None,
        details: dict | None,
    ) -> str:
        _bounded_text(operation_id, "Operation ID", 256)
        _bounded_text(actor_id, "Actor ID", 256)
        if action not in self.ACTIONS:
            raise HistoryError("Unsupported history action.")
        if state not in self.STATES:
            raise HistoryError("Unsupported history state.")
        if subject_id is not None:
            _bounded_text(subject_id, "Subject ID", 256)

        metadata: dict[str, Any] = dict(details or {})
        metadata.setdefault("actor_source", "config")
        unexpected = metadata.keys() - self.DETAIL_FIELDS
        if unexpected:
            raise HistoryError("Only allowlisted metadata may enter local history.")
        for key, value in metadata.items():
            if type(value) not in (str, int, bool, type(None)):
                raise HistoryError("History metadata must contain scalar values.")
            if isinstance(value, str) and len(value) > 512:
                raise HistoryError(f"History metadata field {key} is too long.")
        if metadata.get("decision") not in (None, "approve", "reject"):
            raise HistoryError("Invalid approval decision.")

        try:
            body = json.dumps(
                {
                    "actor_id": actor_id,
                    "action": action,
                    "state": state,
                    "subject_id": subject_id,
                    "details": metadata,
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise HistoryError("History event contains invalid metadata.") from exc
        if len(body.encode("utf-8")) > 8192:
            raise HistoryError("History event exceeds 8 KiB.")
        return body

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "sequence": int(row["sequence"]),
            "operation_id": row["operation_id"],
            "recorded_at": row["recorded_at"],
            **json.loads(row["body"]),
        }


def _bounded_text(value: str, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise HistoryError(f"{label} must contain between 1 and {maximum} characters.")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")
