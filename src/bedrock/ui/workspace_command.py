"""Native `/workspace` workflow shared by both Bedrock application modes.

The Gmail reference used a multi-panel web route.  Bedrock is intentionally a
one-input terminal application, so the same product flow is expressed as a
small command vocabulary.  The important safety interaction is kept: publish
first renders the exact digest corpus and bounded extract, and only
`/workspace confirm` persists it.  Complete source chunks remain private.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from bedrock.context import ContextStore
from bedrock.workspaces import (
    Artifact,
    ArtifactInput,
    Workspace,
    WorkspaceError,
    WorkspaceStore,
    canonical_json,
    corpus_from_context,
    fold_corpus,
)
from bedrock.workspaces.validation import sha256

Notify = Callable[[str, str], Awaitable[None]]
SelectionChanged = Callable[[str | None], None]

WORKSPACE_HELP = """\
**`/workspace` — private team results**

| Command | What it does |
|---|---|
| `/workspace` | List your workspaces |
| `/workspace create <name>` | Create and select a workspace |
| `/workspace use <id or name>` | Select a workspace |
| `/workspace join <invite>` | Join and select a workspace |
| `/workspace items [search]` | List shared results |
| `/workspace show <result>` | Read one shared result |
| `/workspace publish <digest...> [--title "Title"]` | Preview local digests for sharing |
| `/workspace confirm` | Publish the current preview |
| `/workspace cancel` | Discard the current preview |
| `/workspace fold <result>` | Group a corpus by source |
| `/workspace export <result> <path>` | Export one immutable result as JSON |
| `/workspace members` | List members and roles |
| `/workspace invite <editor|viewer>` | Create a single-use 7-day invitation |
| `/workspace member <user> <editor|viewer|remove>` | Change workspace membership |
| `/workspace token <read|write> <name> [days]` | Create an integration key |
| `/workspace invites` / `/workspace tokens` | List credentials without secrets |
| `/workspace revoke <invite|token> <id>` | Revoke access |
| `/workspace events [after]` | Read the resumable activity feed |

`/team` is a shorter alias. Published results are immutable; publish a new
result with lineage when something changes. Integration and invitation
secrets are shown once and stored only as hashes.
"""


@dataclass(frozen=True)
class PendingPublication:
    workspace_id: str
    workspace_name: str
    value: ArtifactInput
    request_key: str


class WorkspaceCommandSession:
    def __init__(
        self,
        store: WorkspaceStore,
        context_store: ContextStore,
        *,
        selection_changed: SelectionChanged | None = None,
    ) -> None:
        self.store = store
        self.context_store = context_store
        self.active_workspace_id: str | None = None
        self.pending: PendingPublication | None = None
        self._selection_changed = selection_changed or (lambda _name: None)

    def close(self) -> None:
        self.store.close()

    async def run(self, args: str, notify: Notify) -> None:
        try:
            await self._run(args.strip(), notify)
        except WorkspaceError as exc:
            await notify(str(exc), "error")
        except (OSError, ValueError) as exc:
            await notify(f"Could not complete the workspace action: {exc}", "error")

    async def _run(self, args: str, notify: Notify) -> None:
        verb, _, rest = args.partition(" ")
        verb, rest = verb.casefold(), rest.strip()

        if not args or verb in {"list", "spaces"}:
            await notify(self._workspace_listing(), "info")
        elif verb in {"help", "?"}:
            await notify(WORKSPACE_HELP, "info")
        elif verb == "create":
            workspace = self.store.create_workspace(rest)
            self._select(workspace)
            await notify(
                f"Created and selected **{workspace.name}** (`{_short(workspace.id)}`).", "info"
            )
        elif verb == "use":
            workspace = self.store.resolve_workspace(rest)
            if workspace is None:
                raise WorkspaceError(404, f"No workspace uniquely matches `{rest}`.")
            self._select(workspace)
            await notify(f"Selected **{workspace.name}** ({workspace.role}).", "info")
        elif verb == "join":
            workspace_id = self.store.join(rest)
            workspace = self.store.resolve_workspace(workspace_id)
            assert workspace is not None
            self._select(workspace)
            await notify(f"Joined and selected **{workspace.name}** as {workspace.role}.", "info")
        elif verb in {"items", "results"}:
            workspace = self._active()
            await notify(self._artifact_listing(workspace, rest), "info")
        elif verb == "show":
            workspace = self._active()
            artifact = self._artifact(workspace, rest)
            await notify(_format_artifact(artifact), "info")
        elif verb == "publish":
            workspace = self._active(write=True)
            references, title = _publish_arguments(rest)
            value = corpus_from_context(self.context_store, references, title=title)
            request_key = "digest:" + sha256(canonical_json(value.to_dict()))
            self.pending = PendingPublication(workspace.id, workspace.name, value, request_key)
            await notify(_format_preview(workspace, value), "info")
        elif verb == "confirm":
            if self.pending is None:
                raise WorkspaceError(409, "There is no publication preview to confirm.")
            pending = self.pending
            active = self._active(write=True)
            if active.id != pending.workspace_id:
                raise WorkspaceError(
                    409, "The selected workspace changed. Preview the result again before publishing."
                )
            result = self.store.publish(pending.workspace_id, pending.value, pending.request_key)
            self.pending = None
            replay = " (already published)" if result.replayed else ""
            await notify(
                f"Published **{result.artifact.title}** as `{_short(result.artifact.id)}`{replay}.",
                "info",
            )
        elif verb == "cancel":
            self.pending = None
            await notify("Discarded the publication preview.", "info")
        elif verb == "fold":
            workspace = self._active(write=True)
            parent = self._artifact(workspace, rest)
            value = fold_corpus(parent)
            key = "fold:" + sha256(canonical_json(value.to_dict()))
            result = self.store.publish(workspace.id, value, key)
            replay = " (already created)" if result.replayed else ""
            await notify(
                f"Created **{result.artifact.title}** as `{_short(result.artifact.id)}`{replay}.",
                "info",
            )
        elif verb == "export":
            workspace = self._active()
            reference, path = _two_arguments(rest, "Use `/workspace export <result> <path>`.")
            artifact = self._artifact(workspace, reference)
            destination = Path(path).expanduser()
            if destination.exists():
                raise WorkspaceError(409, f"`{destination}` already exists; choose a new path.")
            if not destination.parent.is_dir():
                raise WorkspaceError(404, f"Folder `{destination.parent}` does not exist.")
            destination.write_text(
                json.dumps(artifact.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            await notify(f"Exported **{artifact.title}** to `{destination}`.", "info")
        elif verb == "members":
            workspace = self._active()
            members = self.store.members(workspace.id)
            lines = [f"**{workspace.name} members**", ""]
            lines.extend(
                f"- **{member.display_name}** — {member.role} (`{member.user_id}`)"
                for member in members
            )
            await notify("\n".join(lines), "info")
        elif verb == "member":
            workspace = self._active(manage=True)
            user_id, role = _two_arguments(
                rest, "Use `/workspace member <user> <editor|viewer|remove>`."
            )
            if role not in {"editor", "viewer", "remove"}:
                raise WorkspaceError(400, "Choose editor, viewer, or remove.")
            self.store.change_member(
                workspace.id, user_id, None if role == "remove" else role  # type: ignore[arg-type]
            )
            await notify(
                f"{'Removed' if role == 'remove' else 'Updated'} `{user_id}`.", "info"
            )
        elif verb == "invite":
            workspace = self._active(manage=True)
            if rest not in {"editor", "viewer"}:
                raise WorkspaceError(400, "Choose editor or viewer.")
            invite = self.store.create_invite(workspace.id, rest)  # type: ignore[arg-type]
            await notify(
                "\n".join(
                    [
                        f"**Single-use {invite.role} invitation**",
                        "",
                        f"`{invite.token}`",
                        "",
                        f"Expires {_date(invite.expires)}. Copy it now; the secret is not stored.",
                    ]
                ),
                "info",
            )
        elif verb == "invites":
            workspace = self._active(manage=True)
            rows = self.store.list_invites(workspace.id)
            lines = [f"**{workspace.name} invitations**", ""]
            if not rows:
                lines.append("No invitations.")
            for row in rows:
                state = "revoked" if row["revoked"] else "used" if row["used_by"] else "open"
                lines.append(
                    f"- `{_short(row['id'])}` — {row['role']}, {state}, expires {_date(row['expires'])}"
                )
            await notify("\n".join(lines), "info")
        elif verb == "token":
            workspace = self._active(manage=True)
            scope, name, days = _token_arguments(rest)
            key = self.store.create_integration_key(workspace.id, name, scope, days=days)
            await notify(
                "\n".join(
                    [
                        f"**{key.scope} integration key for {key.name}**",
                        "",
                        f"`{key.token}`",
                        "",
                        f"Expires {_date(key.expires)}. Copy it now; the secret is not stored.",
                    ]
                ),
                "info",
            )
        elif verb == "tokens":
            workspace = self._active(manage=True)
            rows = self.store.list_integration_keys(workspace.id)
            lines = [f"**{workspace.name} integration keys**", ""]
            if not rows:
                lines.append("No integration keys.")
            for row in rows:
                state = "revoked" if row["revoked"] else "active"
                lines.append(
                    f"- `{_short(row['id'])}` — **{row['name']}**, {row['scope']}, {state}, expires {_date(row['expires'])}"
                )
            await notify("\n".join(lines), "info")
        elif verb == "revoke":
            workspace = self._active(manage=True)
            kind, reference = _two_arguments(
                rest, "Use `/workspace revoke <invite|token> <id>`."
            )
            if kind == "invite":
                invite_id = self.store.resolve_invite_id(workspace.id, reference)
                if invite_id is None:
                    raise WorkspaceError(404, f"No invite matches `{reference}`.")
                self.store.revoke_invite(workspace.id, invite_id)
            elif kind == "token":
                key_id = self.store.resolve_integration_key_id(workspace.id, reference)
                if key_id is None:
                    raise WorkspaceError(404, f"No integration key matches `{reference}`.")
                self.store.revoke_integration_key(workspace.id, key_id)
            else:
                raise WorkspaceError(400, "Choose invite or token.")
            await notify(f"Revoked {kind} `{reference}`.", "info")
        elif verb == "events":
            workspace = self._active()
            try:
                after = int(rest) if rest else 0
            except ValueError as exc:
                raise WorkspaceError(400, "Event cursor must be a number.") from exc
            page = self.store.events(workspace.id, after=after)
            lines = [f"**{workspace.name} activity after {after}**", ""]
            if not page.items:
                lines.append("No newer activity.")
            lines.extend(
                f"- `{event.sequence}` {event.type} — `{_short(event.subject_id)}` by {event.actor_id}"
                for event in page.items
            )
            if page.has_more:
                lines.append(f"\nMore is available: `/workspace events {page.next_after}`")
            await notify("\n".join(lines), "info")
        else:
            raise WorkspaceError(
                400, f"Unknown workspace action `{verb}`. Type `/workspace help` for a list."
            )

    def _workspace_listing(self) -> str:
        workspaces = self.store.list_workspaces()
        if not workspaces:
            return "No team workspaces yet. Create one with `/workspace create <name>` or join with an invitation."
        lines = ["**Your workspaces**", ""]
        for workspace in workspaces:
            marker = "→" if workspace.id == self.active_workspace_id else " "
            lines.append(
                f"- {marker} **{workspace.name}** — {workspace.role} (`{_short(workspace.id)}`)"
            )
        if self.active_workspace_id is None:
            lines.append("\nSelect one with `/workspace use <id or name>`.")
        return "\n".join(lines)

    def _artifact_listing(self, workspace: Workspace, query: str) -> str:
        page = self.store.list_artifacts(workspace.id, query=query)
        lines = [f"**{workspace.name} results**", ""]
        if not page.items:
            lines.append("No matching shared results.")
        for item in page.items:
            lines.append(
                f"- `{_short(item.id)}` **{item.title}** — {item.kind}, by {item.author_name}\n  {item.summary}"
            )
        return "\n".join(lines)

    def _active(self, *, write: bool = False, manage: bool = False) -> Workspace:
        workspaces = self.store.list_workspaces()
        if self.active_workspace_id is None and len(workspaces) == 1:
            self._select(workspaces[0])
        workspace = next(
            (item for item in workspaces if item.id == self.active_workspace_id), None
        )
        if workspace is None:
            raise WorkspaceError(409, "Select a workspace first with `/workspace use <id or name>`.")
        permission = "manage" if manage else "write" if write else "read"
        self.store.authorize(workspace.id, permission)  # catches role changes since selection
        return workspace

    def _artifact(self, workspace: Workspace, reference: str) -> Artifact:
        artifact = self.store.resolve_artifact(workspace.id, reference)
        if artifact is None:
            raise WorkspaceError(404, f"No shared result uniquely matches `{reference}`.")
        return artifact

    def _select(self, workspace: Workspace) -> None:
        if self.active_workspace_id != workspace.id:
            self.pending = None
        self.active_workspace_id = workspace.id
        self._selection_changed(workspace.name)


def _publish_arguments(value: str) -> tuple[list[str], str | None]:
    try:
        words = shlex.split(value)
    except ValueError as exc:
        raise WorkspaceError(400, f"Could not read publish arguments: {exc}") from exc
    references: list[str] = []
    title = None
    index = 0
    while index < len(words):
        word = words[index]
        if word == "--title":
            if title is not None or index + 1 >= len(words):
                raise WorkspaceError(400, "Use one `--title \"Title\"` option.")
            title = words[index + 1]
            index += 2
        elif word.startswith("--title="):
            if title is not None:
                raise WorkspaceError(400, "Use one `--title` option.")
            title = word.partition("=")[2]
            index += 1
        elif word.startswith("--"):
            raise WorkspaceError(400, f"Unknown publish option `{word}`.")
        else:
            references.append(word)
            index += 1
    if not references:
        raise WorkspaceError(
            400, "Choose a stored digest: `/workspace publish <digest> [--title \"Title\"]`."
        )
    return references, title


def _token_arguments(value: str) -> tuple[str, str, int]:
    try:
        words = shlex.split(value)
    except ValueError as exc:
        raise WorkspaceError(400, f"Could not read token arguments: {exc}") from exc
    if len(words) < 2 or words[0] not in {"read", "write"}:
        raise WorkspaceError(400, "Use `/workspace token <read|write> <name> [days]`.")
    days = 30
    if len(words) >= 3 and words[-1].isdigit():
        days = int(words.pop())
    return words[0], " ".join(words[1:]), days


def _two_arguments(value: str, usage: str) -> tuple[str, str]:
    try:
        words = shlex.split(value)
    except ValueError as exc:
        raise WorkspaceError(400, f"Could not read arguments: {exc}") from exc
    if len(words) != 2:
        raise WorkspaceError(400, usage)
    return words[0], words[1]


def _format_preview(workspace: Workspace, value: ArtifactInput) -> str:
    documents = value.content["documents"]
    assert isinstance(documents, list)
    lines = [
        f"**Review before publishing to {workspace.name}**",
        "",
        f"**{value.title}** — {value.summary}",
        "",
    ]
    for document in documents:
        assert isinstance(document, dict)
        text = str(document["text"])
        lines.extend([f"**{document['title']}**", "", text, ""])
    lines.extend(
        [
            "Only the text above is shared; full source chunks stay in your private context store.",
            "Run `/workspace confirm` to publish this exact preview or `/workspace cancel` to discard it.",
        ]
    )
    return "\n".join(lines)


def _format_artifact(artifact: Artifact) -> str:
    lines = [
        f"**{artifact.title}**",
        "",
        f"{artifact.summary}",
        "",
        f"- ID: `{artifact.id}`",
        f"- Kind: `{artifact.kind}` v{artifact.schema_version}",
        f"- Author: {artifact.author_name} (`{artifact.created_by}`)",
        f"- Checksum: `{artifact.checksum}`",
    ]
    if artifact.parent_ids:
        lines.append("- Parents: " + ", ".join(f"`{item}`" for item in artifact.parent_ids))
    lines.append("")
    if artifact.kind == "brief.corpus":
        documents = artifact.content.get("documents", [])
        assert isinstance(documents, list)
        for document in documents:
            assert isinstance(document, dict)
            lines.extend(
                [
                    f"### {document.get('title', 'Untitled')}",
                    "",
                    str(document.get("text", "")),
                    "",
                    f"Source: {document.get('source', 'unknown')}",
                    "",
                ]
            )
    elif artifact.kind == "brief.fold":
        groups = artifact.content.get("groups", [])
        assert isinstance(groups, list)
        for group in groups:
            assert isinstance(group, dict)
            lines.extend(
                [
                    f"### {group.get('source', 'Source')}",
                    "",
                    str(group.get("text", "")),
                    "",
                    "Cites: " + ", ".join(f"`{item}`" for item in group.get("documentIds", [])),
                    "",
                ]
            )
    else:
        lines.extend(["```json", json.dumps(artifact.content, ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines)


def _short(value: object) -> str:
    return str(value)[:8]


def _date(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
