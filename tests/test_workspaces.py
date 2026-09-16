"""Completion checks for the native shared-workspace port."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from bedrock.workspaces import (
    ArtifactInput,
    UserPrincipal,
    WorkspaceError,
    WorkspaceStore,
    canonical_json,
    fold_corpus,
)
from bedrock.workspaces.models import CORPUS_KIND
from bedrock.workspaces.validation import sha256


def corpus(title: str = "Research corpus") -> ArtifactInput:
    documents = [
        {
            "id": "doc-1",
            "title": "Launch evidence",
            "source": "Lab notebook",
            "text": "First passage with launch evidence.",
        },
        {
            "id": "doc-2",
            "title": "Supporting evidence",
            "source": "Lab notebook",
            "text": "Second passage with supporting evidence.",
        },
    ]
    return ArtifactInput(
        kind=CORPUS_KIND,
        schema_version=1,
        title=title,
        summary="Two reviewed passages.",
        content={
            "documents": documents,
            "method": "test-v1",
            "originalBytes": 80,
            "storedTextBytes": 80,
        },
        provenance={"processor": "test-v1"},
    )


@pytest.fixture
def team(tmp_path):
    root = tmp_path / "workspace-vault"
    owner = WorkspaceStore(root, UserPrincipal("owner", "Owner"))
    workspace = owner.create_workspace("Launch team")
    editor = WorkspaceStore(root, UserPrincipal("editor", "Editor"))
    viewer = WorkspaceStore(root, UserPrincipal("viewer", "Viewer"))
    editor.join(owner.create_invite(workspace.id, "editor").token)
    viewer.join(owner.create_invite(workspace.id, "viewer").token)
    yield root, owner, editor, viewer, workspace
    owner.close()
    editor.close()
    viewer.close()


def test_two_users_share_one_immutable_corpus_and_derived_fold(team):
    _, owner, editor, viewer, workspace = team
    saved = owner.publish(workspace.id, corpus(), "capture-1").artifact

    assert editor.get_artifact(workspace.id, saved.id) == saved
    folded = editor.publish(workspace.id, fold_corpus(saved), "fold-1").artifact
    assert folded.created_by == "user:editor"
    assert folded.parent_ids == (saved.id,)
    teammate_copy = viewer.get_artifact(workspace.id, folded.id)
    assert teammate_copy.checksum == folded.checksum
    assert teammate_copy.checksum == sha256(canonical_json(teammate_copy.content))


def test_roles_and_workspace_bound_lineage_are_enforced(team):
    root, owner, editor, viewer, workspace = team
    item = owner.publish(workspace.id, corpus(), "one").artifact
    outsider = WorkspaceStore(root, UserPrincipal("outsider", "Outsider"))
    try:
        with pytest.raises(WorkspaceError, match="Workspace not found") as hidden:
            outsider.get_artifact(workspace.id, item.id)
        assert hidden.value.status == 404
        with pytest.raises(WorkspaceError) as read_only:
            viewer.publish(workspace.id, corpus(), "blocked")
        assert read_only.value.status == 403
        with pytest.raises(WorkspaceError) as not_owner:
            editor.create_invite(workspace.id, "viewer")
        assert not_owner.value.status == 403

        other = outsider.create_workspace("Other")
        cross_workspace = ArtifactInput(**{**corpus().__dict__, "parent_ids": (item.id,)})
        with pytest.raises(WorkspaceError) as wrong_parent:
            outsider.publish(other.id, cross_workspace, "parent")
        assert wrong_parent.value.status == 404
        with pytest.raises(WorkspaceError) as owner_guard:
            owner.change_member(workspace.id, "owner", None)
        assert owner_guard.value.status == 409
    finally:
        outsider.close()


def test_idempotency_canonicalizes_content_and_does_not_duplicate_events(team):
    _, owner, _, _, workspace = team
    value = corpus()
    first = owner.publish(workspace.id, value, "stable-key")
    reordered = ArtifactInput(
        **{
            **value.__dict__,
            "content": dict(reversed(tuple(value.content.items()))),
        }
    )
    replay = owner.publish(workspace.id, reordered, "stable-key")
    assert replay.replayed is True
    assert replay.artifact.id == first.artifact.id

    changed = ArtifactInput(**{**value.__dict__, "title": "Changed"})
    with pytest.raises(WorkspaceError) as mismatch:
        owner.publish(workspace.id, changed, "stable-key")
    assert mismatch.value.status == 409
    created = [event for event in owner.events(workspace.id).items if event.type == "artifact.created"]
    assert len(created) == 1


def test_concurrent_idempotent_publication_creates_one_result_and_event(team):
    root, owner, _, _, workspace = team
    second_process = WorkspaceStore(root, UserPrincipal("owner", "Owner"))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(store.publish, workspace.id, corpus(), "concurrent")
                for store in (owner, second_process)
            ]
            results = [future.result() for future in futures]
        assert results[0].artifact.id == results[1].artifact.id
        assert sum(result.replayed for result in results) == 1
        events = [
            event
            for event in owner.events(workspace.id).items
            if event.type == "artifact.created" and event.subject_id == results[0].artifact.id
        ]
        assert len(events) == 1
    finally:
        second_process.close()


def test_artifact_and_event_rollback_together(team):
    root, owner, _, _, workspace = team
    with sqlite3.connect(root / "workspaces.sqlite3") as db:
        db.execute(
            """CREATE TRIGGER reject_artifact_event
            BEFORE INSERT ON workspace_events
            WHEN NEW.type = 'artifact.created'
            BEGIN SELECT RAISE(ABORT, 'forced event failure'); END"""
        )
    with pytest.raises(sqlite3.IntegrityError):
        owner.publish(workspace.id, corpus(), "rollback")
    assert owner.list_artifacts(workspace.id).items == ()


def test_invites_are_single_use_replayable_by_winner_and_cannot_restore_member(team):
    root, owner, _, _, workspace = team
    invitation = owner.create_invite(workspace.id, "viewer")
    first = WorkspaceStore(root, UserPrincipal("first", "First"))
    second = WorkspaceStore(root, UserPrincipal("second", "Second"))
    try:
        assert first.join(invitation.token) == workspace.id
        assert first.join(invitation.token) == workspace.id
        with pytest.raises(WorkspaceError) as already_used:
            second.join(invitation.token)
        assert already_used.value.status == 404

        owner.change_member(workspace.id, "first", None)
        with pytest.raises(WorkspaceError) as removed:
            first.join(invitation.token)
        assert removed.value.status == 409
    finally:
        first.close()
        second.close()


def test_integration_keys_are_hashed_scoped_bound_and_revocable(team):
    root, owner, _, _, workspace = team
    read_key = owner.create_integration_key(workspace.id, "Reader", "read")
    with sqlite3.connect(root / "workspaces.sqlite3") as db:
        stored_hash = db.execute(
            "SELECT token_hash FROM workspace_tokens WHERE id = ?", (read_key.id,)
        ).fetchone()[0]
    assert read_key.token not in stored_hash
    assert len(stored_hash) == 64

    reader = WorkspaceStore.for_integration_key(root, read_key.token)
    try:
        assert reader.list_artifacts(workspace.id).items == ()
        with pytest.raises(WorkspaceError) as denied:
            reader.publish(workspace.id, corpus(), "blocked")
        assert denied.value.status == 403
        other = owner.create_workspace("Elsewhere")
        with pytest.raises(WorkspaceError) as bounded:
            reader.list_artifacts(other.id)
        assert bounded.value.status == 403
    finally:
        reader.close()

    write_key = owner.create_integration_key(workspace.id, "Writer", "write")
    writer = WorkspaceStore.for_integration_key(root, write_key.token)
    try:
        assert writer.publish(workspace.id, corpus(), "writer-result").artifact.created_by == (
            f"integration:{write_key.id}"
        )
    finally:
        writer.close()

    owner.revoke_integration_key(workspace.id, read_key.id)
    with pytest.raises(WorkspaceError) as revoked:
        WorkspaceStore.for_integration_key(root, read_key.token)
    assert revoked.value.status == 401


def test_lists_are_metadata_only_paginated_and_search_is_literal(team):
    _, owner, _, _, workspace = team
    for index in range(4):
        owner.publish(workspace.id, corpus(f"Research {index}"), f"page-{index}")
    first = owner.list_artifacts(workspace.id, limit=2)
    second = owner.list_artifacts(workspace.id, limit=2, cursor=first.next_cursor)
    assert len({item.id for item in first.items + second.items}) == 4
    assert first.next_cursor is not None
    assert not hasattr(first.items[0], "content")
    assert owner.list_artifacts(workspace.id, query="%").items == ()
    feed = owner.events(workspace.id, limit=2)
    resumed = owner.events(workspace.id, after=feed.next_after or 0)
    assert all(event.sequence > (feed.next_after or 0) for event in resumed.items)


def test_invalid_builtin_versions_citations_and_large_values_are_rejected(team):
    _, owner, _, _, workspace = team
    value = corpus()
    bad_version = ArtifactInput(**{**value.__dict__, "schema_version": 2})
    with pytest.raises(WorkspaceError) as version:
        owner.publish(workspace.id, bad_version, "version")
    assert version.value.status == 400

    parent = owner.publish(workspace.id, value, "parent").artifact
    fold = fold_corpus(parent)
    bad_fold = ArtifactInput(
        **{
            **fold.__dict__,
            "content": {
                "method": "test",
                "groups": [
                    {"source": "test", "text": "Made up", "documentIds": ["missing"]}
                ],
            },
        }
    )
    with pytest.raises(WorkspaceError) as citations:
        owner.publish(workspace.id, bad_fold, "citations")
    assert citations.value.status == 400

    huge = ArtifactInput(
        kind="research.summary",
        schema_version=1,
        title="Large",
        summary="",
        content={"text": "x" * 270_000},
        provenance={},
    )
    with pytest.raises(WorkspaceError) as size:
        owner.publish(workspace.id, huge, "large")
    assert size.value.status == 413
