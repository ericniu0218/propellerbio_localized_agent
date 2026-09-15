"""Dependency-free data structures for Bedrock shared workspaces.

The web reference called these records "artifacts".  We keep that word and
the same wire-shaped fields because it is the useful part of the design: a
workspace owns immutable results, while a user or integration is recorded as
their author.  Nothing here knows about Textual, SQLite, or a model server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, TypeAlias, TypeVar

JsonScalar: TypeAlias = None | bool | int | float | str
Json: TypeAlias = JsonScalar | list["Json"] | dict[str, "Json"]
Role: TypeAlias = Literal["owner", "editor", "viewer"]
Scope: TypeAlias = Literal["read", "write"]

CORPUS_KIND = "brief.corpus"
FOLD_KIND = "brief.fold"
MAX_REQUEST_BYTES = 256 * 1024


@dataclass(frozen=True)
class Workspace:
    id: str
    name: str
    role: Role
    created: int


@dataclass(frozen=True)
class Member:
    user_id: str
    display_name: str
    role: Role
    joined: int


@dataclass(frozen=True)
class ArtifactInput:
    kind: str
    schema_version: int
    title: str
    summary: str
    content: dict[str, Json]
    provenance: dict[str, Json]
    parent_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Json]:
        """Return the stable public shape used for validation and hashing."""
        return {
            "kind": self.kind,
            "schemaVersion": self.schema_version,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "provenance": self.provenance,
            "parentIds": list(self.parent_ids),
        }


@dataclass(frozen=True)
class ArtifactSummary:
    id: str
    workspace_id: str
    kind: str
    schema_version: int
    title: str
    summary: str
    provenance: dict[str, Json]
    parent_ids: tuple[str, ...]
    created_by: str
    author_name: str
    created: int
    checksum: str


@dataclass(frozen=True)
class Artifact(ArtifactSummary):
    content: dict[str, Json]

    def to_dict(self) -> dict[str, Json]:
        return {
            "id": self.id,
            "workspaceId": self.workspace_id,
            "kind": self.kind,
            "schemaVersion": self.schema_version,
            "title": self.title,
            "summary": self.summary,
            "content": self.content,
            "provenance": self.provenance,
            "parentIds": list(self.parent_ids),
            "createdBy": self.created_by,
            "authorName": self.author_name,
            "created": self.created,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class WorkspaceEvent:
    sequence: int
    type: str
    actor_id: str
    subject_id: str
    created: int


T = TypeVar("T")


@dataclass(frozen=True)
class Page(Generic[T]):
    items: tuple[T, ...]
    next_cursor: str | None = None
    next_after: int | None = None
    has_more: bool = False


@dataclass(frozen=True)
class Invite:
    id: str
    token: str
    role: Role
    expires: int


@dataclass(frozen=True)
class IntegrationKey:
    id: str
    name: str
    token: str
    scope: Scope
    expires: int


@dataclass(frozen=True)
class PublishResult:
    artifact: Artifact
    replayed: bool
