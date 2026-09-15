"""The terminal workflow previews before it mutates shared state."""

from __future__ import annotations

from bedrock.context import ContextStore, ingest
from bedrock.context.adapters import Source
from bedrock.ui.workspace_command import WorkspaceCommandSession
from bedrock.workspaces import UserPrincipal, WorkspaceStore


async def test_preview_confirm_read_and_export_flow(tmp_path):
    context = ContextStore(tmp_path / "context")
    local = ingest(
        Source.from_text(
            "Protocol heading\n\nKeep samples at four degrees.",
            name="protocol.md",
            suffix=".md",
        ),
        store=context,
    )
    store = WorkspaceStore(tmp_path / "vault", UserPrincipal("alice", "Alice"))
    session = WorkspaceCommandSession(store, context)
    notices: list[tuple[str, str]] = []

    async def notify(text: str, kind: str = "info") -> None:
        notices.append((text, kind))

    try:
        await session.run("create Research", notify)
        workspace = store.list_workspaces()[0]
        await session.run(f'publish {local.id[:8]} --title "Reviewed protocol"', notify)
        assert store.list_artifacts(workspace.id).items == ()
        preview = notices[-1][0]
        assert "Review before publishing" in preview
        assert "Keep samples at four degrees" in preview
        assert "full source chunks stay" in preview

        await session.run("confirm", notify)
        listed = store.list_artifacts(workspace.id).items
        assert len(listed) == 1
        await session.run(f"show {listed[0].id[:8]}", notify)
        assert "Reviewed protocol" in notices[-1][0]

        destination = tmp_path / "shared-result.json"
        await session.run(f"export {listed[0].id[:8]} {destination}", notify)
        exported = destination.read_text(encoding="utf-8")
        assert '"workspaceId"' in exported
        assert '"checksum"' in exported
    finally:
        session.close()


async def test_switching_workspaces_discards_pending_preview(tmp_path):
    context = ContextStore(tmp_path / "context")
    local = ingest(Source.from_text("Private source", name="one.md"), store=context)
    store = WorkspaceStore(tmp_path / "vault", UserPrincipal("alice", "Alice"))
    session = WorkspaceCommandSession(store, context)
    notices: list[tuple[str, str]] = []

    async def notify(text: str, kind: str = "info") -> None:
        notices.append((text, kind))

    try:
        await session.run("create First", notify)
        await session.run(f"publish {local.id}", notify)
        assert session.pending is not None
        await session.run("create Second", notify)
        assert session.pending is None
        await session.run("confirm", notify)
        assert notices[-1][1] == "error"
        assert "no publication preview" in notices[-1][0].lower()
    finally:
        session.close()
