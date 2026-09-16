"""SQLite-backed shared workspaces for the native Bedrock application.

This is deliberately a local persistence module, not a browser API.  A
configured Bedrock installation points every participant at the same private
vault and supplies a stable ``user_id``.  The module is the only code allowed
to touch workspace tables, so authorization and the invariants around
idempotency, lineage, secrets, and events cannot be bypassed accidentally by
the TUI or by a trusted local integration.

SQLite is opened with a regular rollback journal instead of WAL.  That is a
better default for a vault an operator may place on a correctly configured
shared filesystem; writes are short and use ``BEGIN IMMEDIATE`` so invite
redemption and artifact publication remain atomic across processes.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

from bedrock.workspaces.models import (
    CORPUS_KIND,
    FOLD_KIND,
    Artifact,
    ArtifactInput,
    ArtifactSummary,
    IntegrationKey,
    Invite,
    Member,
    Page,
    PublishResult,
    Role,
    Scope,
    Workspace,
    WorkspaceEvent,
)
from bedrock.workspaces.validation import (
    WorkspaceError,
    canonical_json,
    corpus_documents,
    require_integer,
    require_string,
    secret,
    sha256,
    validate_artifact_input,
)

DEFAULT_WORKSPACE_DIR = Path.home() / ".local" / "share" / "bedrock" / "workspaces"
_KEY_PATTERN = re.compile(r"^bwk_[a-f0-9]{64}$")
_META_COLUMNS = """id, workspace_id, kind, schema_version, title, summary,
provenance, parent_ids, created_by, author_name, created, checksum"""


@dataclass(frozen=True)
class UserPrincipal:
    id: str
    name: str
    type: Literal["user"] = "user"


@dataclass(frozen=True)
class IntegrationPrincipal:
    id: str
    name: str
    workspace_id: str
    owner_id: str
    scope: Scope
    type: Literal["integration"] = "integration"


Principal = UserPrincipal | IntegrationPrincipal
Permission = Literal["read", "write", "manage"]


class WorkspaceStore:
    """One authorization-scoped view of one private workspace vault."""

    def __init__(self, root: Path | str | None, actor: Principal) -> None:
        self.root = Path(root).expanduser() if root else DEFAULT_WORKSPACE_DIR
        self.actor = actor
        self._lock = threading.RLock()
        self._connection = self._open()
        try:
            self._migrate()
        except Exception:
            self._connection.close()
            raise

    @classmethod
    def for_user(
        cls,
        root: Path | str | None,
        user_id: str,
        display_name: str | None = None,
    ) -> "WorkspaceStore":
        stable_id = require_string(user_id, "User ID", 160)
        name = require_string(display_name or stable_id, "Display name", 160)
        return cls(root, UserPrincipal(stable_id, name))

    @classmethod
    def for_integration_key(cls, root: Path | str | None, token: str) -> "WorkspaceStore":
        """Open a scoped view for a trusted local producer or consumer."""
        bootstrap = cls.for_user(root, "__integration_lookup__", "Integration lookup")
        try:
            actor = bootstrap.integration_principal(token)
        finally:
            bootstrap.close()
        return cls(root, actor)

    @property
    def database_path(self) -> Path:
        return self.root / "workspaces.sqlite3"

    @property
    def actor_id(self) -> str:
        return f"{self.actor.type}:{self.actor.id}"

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    # ---- Workspaces and membership --------------------------------------

    def list_workspaces(self) -> tuple[Workspace, ...]:
        actor = self._user()
        with self._lock:
            rows = self._connection.execute(
                """SELECT w.id, w.name, w.created, m.role
                FROM workspaces w
                JOIN workspace_members m ON w.id = m.workspace_id
                WHERE m.user_id = ? ORDER BY w.created, w.id""",
                (actor.id,),
            ).fetchall()
        return tuple(
            Workspace(row["id"], row["name"], row["role"], row["created"]) for row in rows
        )

    def resolve_workspace(self, reference: str) -> Workspace | None:
        ref = reference.strip()
        if not ref:
            return None
        matches = [
            item
            for item in self.list_workspaces()
            if item.id == ref or item.id.startswith(ref) or item.name.casefold() == ref.casefold()
        ]
        return matches[0] if len(matches) == 1 else None

    def create_workspace(self, name: str) -> Workspace:
        actor = self._user()
        title = require_string(name, "Workspace name", 80)
        workspace_id = str(uuid.uuid4())
        created = _now()
        with self._transaction() as db:
            db.execute(
                "INSERT INTO workspaces (id, name, created_by, created) VALUES (?, ?, ?, ?)",
                (workspace_id, title, actor.id, created),
            )
            db.execute(
                """INSERT INTO workspace_members
                (workspace_id, user_id, display_name, role, joined)
                VALUES (?, ?, ?, 'owner', ?)""",
                (workspace_id, actor.id, actor.name, created),
            )
            self._event(db, workspace_id, "workspace.created", workspace_id, created)
        return Workspace(workspace_id, title, "owner", created)

    def authorize(self, workspace_id: str, permission: Permission = "read") -> Role:
        with self._lock:
            return self._authorize(self._connection, workspace_id, permission)

    def members(self, workspace_id: str) -> tuple[Member, ...]:
        with self._lock:
            self._authorize(self._connection, workspace_id)
            rows = self._connection.execute(
                """SELECT user_id, display_name, role, joined
                FROM workspace_members WHERE workspace_id = ?
                ORDER BY joined, user_id""",
                (workspace_id,),
            ).fetchall()
        return tuple(
            Member(row["user_id"], row["display_name"], row["role"], row["joined"])
            for row in rows
        )

    def change_member(self, workspace_id: str, user_id: str, role: Role | None) -> None:
        with self._transaction() as db:
            self._authorize(db, workspace_id, "manage")
            target = db.execute(
                "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            ).fetchone()
            if target is None:
                raise WorkspaceError(404, "Member not found.")
            if target["role"] == "owner":
                raise WorkspaceError(409, "The workspace owner cannot be removed or demoted.")
            if role is not None and role not in {"editor", "viewer"}:
                raise WorkspaceError(400, "Choose editor or viewer.")
            if role is None:
                db.execute(
                    "DELETE FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                    (workspace_id, user_id),
                )
                event_type = "member.removed"
            else:
                db.execute(
                    "UPDATE workspace_members SET role = ? WHERE workspace_id = ? AND user_id = ?",
                    (role, workspace_id, user_id),
                )
                event_type = "member.role_changed"
            self._event(db, workspace_id, event_type, user_id)

    # ---- Invitations -----------------------------------------------------

    def create_invite(self, workspace_id: str, role: Role, *, days: int = 7) -> Invite:
        actor = self._user()
        if role not in {"editor", "viewer"}:
            raise WorkspaceError(400, "Choose editor or viewer.")
        lifetime = require_integer(days, "expiry days", 1, 30)
        token = secret("bwi_")
        invite_id = str(uuid.uuid4())
        created = _now()
        expires = created + lifetime * 86_400_000
        with self._transaction() as db:
            self._authorize(db, workspace_id, "manage")
            db.execute(
                """INSERT INTO workspace_invites
                (id, workspace_id, token_hash, role, created_by, expires, created)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (invite_id, workspace_id, sha256(token), role, actor.id, expires, created),
            )
            self._event(db, workspace_id, "invite.created", invite_id, created)
        return Invite(invite_id, token, role, expires)

    def list_invites(self, workspace_id: str) -> tuple[dict, ...]:
        with self._lock:
            self._authorize(self._connection, workspace_id, "manage")
            rows = self._connection.execute(
                """SELECT id, role, expires, used_by, revoked, created
                FROM workspace_invites WHERE workspace_id = ?
                ORDER BY created DESC LIMIT 100""",
                (workspace_id,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def join(self, token: str) -> str:
        actor = self._user()
        token_hash = sha256(require_string(token, "Invite token", 100))
        now = _now()
        with self._transaction() as db:
            invite = db.execute(
                """SELECT id, workspace_id, role, used_by
                FROM workspace_invites
                WHERE token_hash = ? AND revoked IS NULL AND expires > ?
                AND (used_by IS NULL OR used_by = ?)""",
                (token_hash, now, actor.id),
            ).fetchone()
            if invite is None:
                raise WorkspaceError(404, "Invite is invalid, expired, revoked, or already used.")

            existing = db.execute(
                "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (invite["workspace_id"], actor.id),
            ).fetchone()
            if invite["used_by"] == actor.id:
                if existing is None:
                    raise WorkspaceError(409, "This invitation was already used.")
                return invite["workspace_id"]

            claimed = db.execute(
                """UPDATE workspace_invites SET used_by = ?
                WHERE id = ? AND used_by IS NULL AND revoked IS NULL AND expires > ?""",
                (actor.id, invite["id"], now),
            )
            if claimed.rowcount != 1:
                raise WorkspaceError(409, "This invitation was used or revoked.")
            inserted = db.execute(
                """INSERT INTO workspace_members
                (workspace_id, user_id, display_name, role, joined)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workspace_id, user_id) DO NOTHING""",
                (invite["workspace_id"], actor.id, actor.name, invite["role"], now),
            )
            if inserted.rowcount:
                self._event(db, invite["workspace_id"], "member.joined", actor.id, now)
            return invite["workspace_id"]

    def revoke_invite(self, workspace_id: str, invite_id: str) -> None:
        with self._transaction() as db:
            self._authorize(db, workspace_id, "manage")
            changed = db.execute(
                """UPDATE workspace_invites SET revoked = COALESCE(revoked, ?)
                WHERE workspace_id = ? AND id = ?""",
                (_now(), workspace_id, invite_id),
            )
            if changed.rowcount != 1:
                raise WorkspaceError(404, "Invite not found.")
            self._event(db, workspace_id, "invite.revoked", invite_id)

    # ---- Integration keys ------------------------------------------------

    def create_integration_key(
        self,
        workspace_id: str,
        name: str,
        scope: Scope,
        *,
        days: int = 30,
    ) -> IntegrationKey:
        actor = self._user()
        title = require_string(name, "Integration name", 80)
        if scope not in {"read", "write"}:
            raise WorkspaceError(400, "Choose read or write scope.")
        lifetime = require_integer(days, "expiry days", 1, 90)
        token = secret("bwk_")
        key_id = str(uuid.uuid4())
        created = _now()
        expires = created + lifetime * 86_400_000
        with self._transaction() as db:
            self._authorize(db, workspace_id, "manage")
            db.execute(
                """INSERT INTO workspace_tokens
                (id, workspace_id, name, token_hash, scope, created_by, created, expires)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (key_id, workspace_id, title, sha256(token), scope, actor.id, created, expires),
            )
            self._event(db, workspace_id, "integration.created", key_id, created)
        return IntegrationKey(key_id, title, token, scope, expires)

    def list_integration_keys(self, workspace_id: str) -> tuple[dict, ...]:
        with self._lock:
            self._authorize(self._connection, workspace_id, "manage")
            rows = self._connection.execute(
                """SELECT id, name, scope, created, expires, revoked
                FROM workspace_tokens WHERE workspace_id = ?
                ORDER BY created DESC LIMIT 100""",
                (workspace_id,),
            ).fetchall()
        return tuple(dict(row) for row in rows)

    def revoke_integration_key(self, workspace_id: str, key_id: str) -> None:
        with self._transaction() as db:
            self._authorize(db, workspace_id, "manage")
            changed = db.execute(
                """UPDATE workspace_tokens SET revoked = COALESCE(revoked, ?)
                WHERE workspace_id = ? AND id = ?""",
                (_now(), workspace_id, key_id),
            )
            if changed.rowcount != 1:
                raise WorkspaceError(404, "Integration key not found.")
            self._event(db, workspace_id, "integration.revoked", key_id)

    def integration_principal(self, token: str) -> IntegrationPrincipal:
        if not _KEY_PATTERN.fullmatch(token):
            raise WorkspaceError(401, "Invalid integration key.")
        with self._lock:
            row = self._connection.execute(
                """SELECT id, name, workspace_id, created_by, scope
                FROM workspace_tokens
                WHERE token_hash = ? AND revoked IS NULL AND expires > ?""",
                (sha256(token), _now()),
            ).fetchone()
        if row is None:
            raise WorkspaceError(401, "Integration key expired, revoked, or invalid.")
        return IntegrationPrincipal(
            row["id"], row["name"], row["workspace_id"], row["created_by"], row["scope"]
        )

    # ---- Immutable artifacts --------------------------------------------

    def list_artifacts(
        self,
        workspace_id: str,
        *,
        kind: str = "",
        query: str = "",
        limit: int = 30,
        cursor: str | None = None,
    ) -> Page[ArtifactSummary]:
        size = require_integer(limit, "limit", 1, 100)
        kind = require_string(kind, "Kind", 100, empty=True)
        query = require_string(query, "Search", 160, empty=True)
        before_created = 9_223_372_036_854_775_807
        before_id = "\uffff"
        if cursor:
            try:
                marker = json.loads(cursor)
                before_created = require_integer(marker["created"], "cursor", 0, before_created)
                before_id = require_string(marker["id"], "Cursor ID")
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                raise WorkspaceError(400, "Invalid cursor.") from exc

        with self._lock:
            self._authorize(self._connection, workspace_id)
            rows = self._connection.execute(
                f"""SELECT {_META_COLUMNS} FROM workspace_artifacts
                WHERE workspace_id = ?
                AND (? = '' OR kind = ?)
                AND (? = '' OR instr(lower(title || ' ' || summary), lower(?)) > 0)
                AND (created < ? OR (created = ? AND id < ?))
                ORDER BY created DESC, id DESC LIMIT ?""",
                (
                    workspace_id,
                    kind,
                    kind,
                    query,
                    query,
                    before_created,
                    before_created,
                    before_id,
                    size + 1,
                ),
            ).fetchall()
        items = tuple(_summary(row) for row in rows[:size])
        next_cursor = None
        if len(rows) > size and items:
            last = items[-1]
            next_cursor = canonical_json({"created": last.created, "id": last.id})
        return Page(items, next_cursor=next_cursor)

    def resolve_artifact(self, workspace_id: str, reference: str) -> Artifact | None:
        ref = reference.strip()
        if not ref:
            return None
        with self._lock:
            self._authorize(self._connection, workspace_id)
            exact = self._connection.execute(
                "SELECT * FROM workspace_artifacts WHERE workspace_id = ? AND id = ?",
                (workspace_id, ref),
            ).fetchone()
            if exact is not None:
                return _artifact(exact)
            rows = self._connection.execute(
                """SELECT * FROM workspace_artifacts
                WHERE workspace_id = ?
                AND (substr(id, 1, length(?)) = ? OR lower(title) = lower(?))
                ORDER BY created DESC, id DESC LIMIT 2""",
                (workspace_id, ref, ref, ref),
            ).fetchall()
        return _artifact(rows[0]) if len(rows) == 1 else None

    def resolve_invite_id(self, workspace_id: str, reference: str) -> str | None:
        return self._resolve_managed_id(
            workspace_id, "workspace_invites", reference, "Invite"
        )

    def resolve_integration_key_id(self, workspace_id: str, reference: str) -> str | None:
        return self._resolve_managed_id(
            workspace_id, "workspace_tokens", reference, "Integration key"
        )

    def get_artifact(self, workspace_id: str, artifact_id: str) -> Artifact:
        with self._lock:
            self._authorize(self._connection, workspace_id)
            row = self._connection.execute(
                "SELECT * FROM workspace_artifacts WHERE workspace_id = ? AND id = ?",
                (workspace_id, artifact_id),
            ).fetchone()
        if row is None:
            raise WorkspaceError(404, "Shared result not found.")
        return _artifact(row)

    def publish(
        self,
        workspace_id: str,
        value: ArtifactInput | dict,
        request_key: str,
    ) -> PublishResult:
        artifact_input = validate_artifact_input(value)
        key = require_string(request_key, "Idempotency key", 128)
        request_hash = sha256(canonical_json(artifact_input.to_dict()))
        with self._transaction() as db:
            self._authorize(db, workspace_id, "write")
            existing = db.execute(
                """SELECT * FROM workspace_artifacts
                WHERE workspace_id = ? AND created_by = ? AND request_key = ?""",
                (workspace_id, self.actor_id, key),
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request_hash:
                    raise WorkspaceError(409, "This idempotency key was used for different content.")
                return PublishResult(_artifact(existing), True)

            parents = []
            for parent_id in artifact_input.parent_ids:
                row = db.execute(
                    "SELECT * FROM workspace_artifacts WHERE workspace_id = ? AND id = ?",
                    (workspace_id, parent_id),
                ).fetchone()
                if row is None:
                    raise WorkspaceError(404, "Shared result not found.")
                parents.append(_artifact(row))
            self._validate_lineage(artifact_input, parents)

            artifact_id = str(uuid.uuid4())
            created = _now()
            content = canonical_json(artifact_input.content)
            checksum = sha256(content)
            db.execute(
                """INSERT INTO workspace_artifacts
                (id, workspace_id, kind, schema_version, title, summary, content,
                 provenance, parent_ids, created_by, author_name, created, checksum,
                 request_key, request_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact_id,
                    workspace_id,
                    artifact_input.kind,
                    artifact_input.schema_version,
                    artifact_input.title,
                    artifact_input.summary,
                    content,
                    canonical_json(artifact_input.provenance),
                    canonical_json(list(artifact_input.parent_ids)),
                    self.actor_id,
                    self.actor.name,
                    created,
                    checksum,
                    key,
                    request_hash,
                ),
            )
            self._event(db, workspace_id, "artifact.created", artifact_id, created)
            row = db.execute("SELECT * FROM workspace_artifacts WHERE id = ?", (artifact_id,)).fetchone()
            assert row is not None
            return PublishResult(_artifact(row), False)

    def events(
        self,
        workspace_id: str,
        *,
        after: int = 0,
        limit: int = 50,
    ) -> Page[WorkspaceEvent]:
        cursor = require_integer(after, "event cursor", 0, 9_223_372_036_854_775_807)
        size = require_integer(limit, "limit", 1, 100)
        with self._lock:
            self._authorize(self._connection, workspace_id)
            rows = self._connection.execute(
                """SELECT sequence, type, actor_id, subject_id, created
                FROM workspace_events WHERE workspace_id = ? AND sequence > ?
                ORDER BY sequence ASC LIMIT ?""",
                (workspace_id, cursor, size + 1),
            ).fetchall()
        items = tuple(
            WorkspaceEvent(
                row["sequence"], row["type"], row["actor_id"], row["subject_id"], row["created"]
            )
            for row in rows[:size]
        )
        return Page(
            items,
            next_after=items[-1].sequence if items else cursor,
            has_more=len(rows) > size,
        )

    # ---- Internals -------------------------------------------------------

    def _open(self) -> sqlite3.Connection:
        existed = self.root.exists()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not existed:
            os.chmod(self.root, 0o700)
        new_database = not self.database_path.exists()
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA busy_timeout = 10000")
        if new_database:
            os.chmod(self.database_path, 0o600)
        return connection

    def _migrate(self) -> None:
        migration_dir = Path(__file__).with_name("migrations")
        with self._transaction() as db:
            db.execute(
                """CREATE TABLE IF NOT EXISTS workspace_migrations
                (name TEXT PRIMARY KEY NOT NULL, applied INTEGER NOT NULL)"""
            )
            applied = {
                row["name"] for row in db.execute("SELECT name FROM workspace_migrations").fetchall()
            }
            for path in sorted(migration_dir.glob("*.sql")):
                if path.name in applied:
                    continue
                # executescript commits implicitly, so split this intentionally
                # simple migration format into individual statements here.
                for statement in path.read_text(encoding="utf-8").split(";"):
                    if statement.strip():
                        db.execute(statement)
                db.execute(
                    "INSERT INTO workspace_migrations (name, applied) VALUES (?, ?)",
                    (path.name, _now()),
                )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _user(self) -> UserPrincipal:
        if not isinstance(self.actor, UserPrincipal):
            raise WorkspaceError(403, "A workspace owner is required for this action.")
        return self.actor

    def _authorize(
        self,
        db: sqlite3.Connection,
        workspace_id: str,
        permission: Permission = "read",
    ) -> Role:
        actor = self.actor
        if isinstance(actor, IntegrationPrincipal):
            if (
                actor.workspace_id != workspace_id
                or permission == "manage"
                or (permission == "write" and actor.scope != "write")
            ):
                raise WorkspaceError(403, "This integration does not have that workspace permission.")
            member_id = actor.owner_id
        else:
            member_id = actor.id
        row = db.execute(
            "SELECT role FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
            (workspace_id, member_id),
        ).fetchone()
        if row is None:
            raise WorkspaceError(404, "Workspace not found.")
        role: Role = row["role"]
        if (permission == "manage" and role != "owner") or (
            permission == "write" and role == "viewer"
        ):
            raise WorkspaceError(403, "Your workspace role does not allow this action.")
        return role

    def _event(
        self,
        db: sqlite3.Connection,
        workspace_id: str,
        event_type: str,
        subject_id: str,
        created: int | None = None,
    ) -> None:
        db.execute(
            """INSERT INTO workspace_events
            (workspace_id, type, actor_id, subject_id, created) VALUES (?, ?, ?, ?, ?)""",
            (workspace_id, event_type, self.actor_id, subject_id, created or _now()),
        )

    def _resolve_managed_id(
        self,
        workspace_id: str,
        table: Literal["workspace_invites", "workspace_tokens"],
        reference: str,
        label: str,
    ) -> str | None:
        ref = reference.strip()
        if not ref:
            return None
        with self._lock:
            self._authorize(self._connection, workspace_id, "manage")
            rows = self._connection.execute(
                f"""SELECT id FROM {table} WHERE workspace_id = ?
                AND substr(id, 1, length(?)) = ? ORDER BY created DESC LIMIT 2""",
                (workspace_id, ref, ref),
            ).fetchall()
        if len(rows) > 1:
            raise WorkspaceError(409, f"{label} prefix `{reference}` is ambiguous.")
        return rows[0]["id"] if rows else None

    @staticmethod
    def _validate_lineage(value: ArtifactInput, parents: list[Artifact]) -> None:
        if value.kind != FOLD_KIND:
            return
        parent = parents[0] if parents else None
        if parent is None or parent.kind != CORPUS_KIND:
            raise WorkspaceError(400, "A fold parent must be a corpus.")
        document_ids = {item["id"] for item in corpus_documents(parent.content.get("documents"))}
        groups = value.content.get("groups", [])
        assert isinstance(groups, list)
        for group in groups:
            assert isinstance(group, dict)
            for citation in group.get("documentIds", []):
                if citation not in document_ids:
                    raise WorkspaceError(
                        400, "Fold citations must reference documents in the parent corpus."
                    )


def _summary(row: sqlite3.Row) -> ArtifactSummary:
    return ArtifactSummary(
        id=row["id"],
        workspace_id=row["workspace_id"],
        kind=row["kind"],
        schema_version=row["schema_version"],
        title=row["title"],
        summary=row["summary"],
        provenance=json.loads(row["provenance"]),
        parent_ids=tuple(json.loads(row["parent_ids"])),
        created_by=row["created_by"],
        author_name=row["author_name"],
        created=row["created"],
        checksum=row["checksum"],
    )


def _artifact(row: sqlite3.Row) -> Artifact:
    summary = _summary(row)
    return Artifact(**summary.__dict__, content=json.loads(row["content"]))


def _now() -> int:
    return int(time.time() * 1000)
