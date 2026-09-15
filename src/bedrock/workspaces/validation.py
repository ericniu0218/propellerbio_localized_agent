"""Validation and canonical JSON for immutable workspace artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
import secrets
from collections.abc import Mapping

from bedrock.workspaces.models import (
    CORPUS_KIND,
    FOLD_KIND,
    MAX_REQUEST_BYTES,
    ArtifactInput,
    Json,
)

_KIND = re.compile(r"^[a-z][a-z0-9_-]*(\.[a-z][a-z0-9_-]*)+$")


class WorkspaceError(ValueError):
    """A safe, user-facing workspace error with an HTTP-like status.

    The status makes the local module straightforward to expose through a
    private service later without coupling this client to a web framework.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def require_string(
    value: object,
    label: str,
    maximum: int = 160,
    *,
    empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise WorkspaceError(400, f"{label} must be text.")
    result = value.strip()
    minimum = 0 if empty else 1
    if len(result) < minimum or len(value) > maximum:
        raise WorkspaceError(400, f"{label} must be {minimum}–{maximum} characters.")
    return result


def require_integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise WorkspaceError(400, f"Invalid {label}.")
    return value


def require_object(value: object, label: str = "Value") -> dict[str, Json]:
    if not isinstance(value, Mapping):
        raise WorkspaceError(400, f"{label} must be a JSON object.")
    result: dict[str, Json] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise WorkspaceError(400, "JSON object keys must be text.")
        _validate_json(item)
        result[key] = item  # type: ignore[assignment]
    return result


def canonical_json(value: object) -> str:
    """The one JSON representation used by checksums and idempotency."""
    _validate_json(value)
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise WorkspaceError(400, "Value must be valid JSON.") from exc


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def secret(prefix: str) -> str:
    return prefix + secrets.token_hex(32)


def corpus_documents(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise WorkspaceError(400, "A corpus needs 1–200 documents.")
    if not 1 <= len(value) <= 200:
        raise WorkspaceError(400, "A corpus needs 1–200 documents.")
    ids: set[str] = set()
    documents = []
    for item in value:
        data = require_object(item, "Document")
        document_id = require_string(data.get("id"), "Document ID")
        if document_id in ids:
            raise WorkspaceError(400, "Document IDs must be unique.")
        ids.add(document_id)
        documents.append(
            {
                "id": document_id,
                "title": require_string(data.get("title"), "Document title", 240),
                "text": require_string(data.get("text"), "Document text", 16_000),
                "source": require_string(data.get("source"), "Source", 240),
            }
        )
    return documents


def validate_artifact_input(value: ArtifactInput | Mapping[str, object]) -> ArtifactInput:
    if isinstance(value, ArtifactInput):
        data: Mapping[str, object] = value.to_dict()
    elif isinstance(value, Mapping):
        data = value
    else:
        raise WorkspaceError(400, "Artifact must be a JSON object.")

    kind = require_string(data.get("kind"), "Kind", 100)
    if not _KIND.fullmatch(kind):
        raise WorkspaceError(400, "Use a namespaced kind, such as acme.analysis.")
    schema_version = require_integer(
        data.get("schemaVersion", data.get("schema_version")), "schemaVersion", 1, 10_000
    )
    raw_parents = data.get("parentIds", data.get("parent_ids", []))
    if not isinstance(raw_parents, list):
        raise WorkspaceError(400, "Provide up to 20 parent IDs.")
    if len(raw_parents) > 20:
        raise WorkspaceError(400, "Provide up to 20 parent IDs.")
    parents: list[str] = []
    for parent in raw_parents:
        item = require_string(parent, "Parent ID")
        if item not in parents:
            parents.append(item)

    result = ArtifactInput(
        kind=kind,
        schema_version=schema_version,
        title=require_string(data.get("title"), "Title"),
        summary=require_string(data.get("summary"), "Summary", 2_000, empty=True),
        content=require_object(data.get("content"), "Content"),
        provenance=require_object(data.get("provenance"), "Provenance"),
        parent_ids=tuple(parents),
    )

    if kind.startswith("brief.") and kind not in {CORPUS_KIND, FOLD_KIND}:
        raise WorkspaceError(400, "The brief namespace is reserved. Use your application namespace.")
    if kind in {CORPUS_KIND, FOLD_KIND} and schema_version != 1:
        raise WorkspaceError(400, "Unsupported built-in schema version.")
    if kind == CORPUS_KIND:
        corpus_documents(result.content.get("documents"))
        require_string(result.content.get("method"), "Compression method")
        require_integer(result.content.get("originalBytes"), "originalBytes", 0, 1_000_000_000)
        require_integer(
            result.content.get("storedTextBytes"), "storedTextBytes", 0, 1_000_000_000
        )
    if kind == FOLD_KIND:
        require_string(result.content.get("method"), "Fold method")
        groups = result.content.get("groups")
        if (
            len(result.parent_ids) != 1
            or not isinstance(groups, list)
            or len(groups) > 200
        ):
            raise WorkspaceError(400, "A fold needs one corpus parent and up to 200 groups.")
        for value in groups:
            group = require_object(value, "Fold group")
            require_string(group.get("source"), "Group source", 240)
            require_string(group.get("text"), "Group text", 16_000)
            citations = group.get("documentIds")
            if (
                not isinstance(citations, list)
                or not 1 <= len(citations) <= 200
            ):
                raise WorkspaceError(400, "A group needs document citations.")
            for citation in citations:
                require_string(citation, "Citation ID")

    if len(canonical_json(result.to_dict()).encode("utf-8")) > MAX_REQUEST_BYTES:
        raise WorkspaceError(413, "Result exceeds the 256 KiB workspace limit.")
    return result


def _validate_json(value: object, depth: int = 0) -> None:
    if depth > 20:
        raise WorkspaceError(400, "JSON nesting exceeds 20 levels.")
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise WorkspaceError(400, "JSON numbers must be finite.")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json(item, depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise WorkspaceError(400, "JSON object keys must be text.")
            _validate_json(item, depth + 1)
        return
    raise WorkspaceError(400, "Value must be valid JSON.")
