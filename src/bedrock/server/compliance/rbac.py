"""Role-based access control for document search tools. Agents inherit the
requesting user's permissions — never broader — per docs/server/ARCHITECTURE.md
section 8.

No company identity provider was specified (see ARCHITECTURE.md open
question 5), so this ships a pluggable interface plus a simple in-memory
grant table for dev/mock mode. Swap `PermissionSource` for a real OIDC/LDAP
group lookup in real mode without touching callers.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class PermissionSource(Protocol):
    def groups_for_user(self, user_id: str) -> set[str]: ...


class StaticPermissionSource:
    """Dev/mock default: everyone is in "general"; extend via
    `grant(user_id, group)` in tests or a config file in real deployments."""

    def __init__(self) -> None:
        self._grants: dict[str, set[str]] = {}

    def grant(self, user_id: str, group: str) -> None:
        self._grants.setdefault(user_id, {"general"}).add(group)

    def groups_for_user(self, user_id: str) -> set[str]:
        return self._grants.get(user_id, {"general"})


class RBACFilter:
    def __init__(self, source: PermissionSource) -> None:
        self._source = source

    def groups_for(self, user_id: str) -> set[str]:
        """The groups this user belongs to.

        Exposed because not every caller has documents shaped like RAG hits.
        `meeting_transcripts` filters `MeetingTranscript` objects that come off
        the datacenter contract, and has to send the groups to the connector as
        part of the signed request, so it needs the set itself rather than a
        filtered list.
        """
        return self._source.groups_for_user(user_id)

    def filter_documents(self, user_id: str, documents: list[dict]) -> list[dict]:
        """`documents` are RAG search hits, each expected to carry an
        `access_groups` field (list[str]); a doc with no `access_groups`
        is treated as "general" (visible to everyone)."""
        user_groups = self._source.groups_for_user(user_id)
        visible = []
        for doc in documents:
            doc_groups = set(doc.get("access_groups", ["general"]))
            if doc_groups & user_groups:
                visible.append(doc)
        return visible


@lru_cache
def get_rbac_filter() -> RBACFilter:
    return RBACFilter(StaticPermissionSource())
