"""E1-E4: read-only windows, lock contention, external edits and deletion names."""
import os
from types import SimpleNamespace

import pytest

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.models import Conversation, new_conversation_id
from bedrock.storage import ConversationDeleted, ConversationStore, HISTORY_RESCAN_SECONDS
from textual.widgets import Input
from tests.conftest import FakeBackend
from tests.test_history_followup import _no_flock
from tests.test_history_regressions import saved


@pytest.mark.parametrize("action", ["quit", "new", "switch"])
async def test_leaving_a_read_only_window_does_not_fork_remote_updates(store, action):
    target = saved(store, "other chat")
    original = saved(store)
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test():
        await wait_for_app_workers(app)
        other = ConversationStore(store.directory)
        remote = other.load(original.id)
        remote.add_user("new question in the writing window")
        remote.add_assistant("new answer")
        other.save(remote)
        remote_bytes = other.path_for(original.id).read_bytes()
        if action == "quit":
            await app.action_quit()
        elif action == "new":
            await app.start_new_conversation()
        else:
            await app.open_conversation(target.id)
            assert other.latest().id == target.id  # Opening still records use.
        assert {s.id for s in other.list()} == {original.id, target.id}
        assert other.path_for(original.id).read_bytes() == remote_bytes


async def test_open_racing_remote_update_does_not_fork(store, monkeypatch):
    original = saved(store)
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test():
        await wait_for_app_workers(app)
        await app.start_new_conversation()
        real_load = store.load

        def load_then_update(identifier, **kwargs):
            loaded = real_load(identifier, **kwargs)
            other = ConversationStore(store.directory)
            remote = other.load(identifier)
            remote.add_user("remote edit during open")
            other.save(remote)
            return loaded

        monkeypatch.setattr(store, "load", load_then_update)
        await app.open_conversation(original.id)
        assert app.conversation.id == original.id
        assert len(store.list()) == 1
        assert real_load(original.id).messages[-1].content == "remote edit during open"


async def test_changed_message_content_is_still_saved(store):
    original = saved(store)
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test():
        await wait_for_app_workers(app)
        app.conversation.messages[-1].content = "completed streamed answer"
        assert app.save_conversation()
        assert store.load(original.id).messages[-1].content == "completed streamed answer"


async def test_stale_fallback_lock_fails_without_sleep_and_preserves_draft(store, monkeypatch):
    original = saved(store)
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test():
        await wait_for_app_workers(app)
        lock = store.directory / ".history-writer"
        lock.mkdir()
        monkeypatch.setattr("bedrock.storage.fcntl.flock", _no_flock)

        def unexpected_sleep(seconds):
            pytest.fail("A UI save must not sleep waiting for a stale lock")

        monkeypatch.setattr("bedrock.storage.time.sleep", unexpected_sleep)
        for _ in range(2):
            await app.send_message("keep my draft")
            await wait_for_app_workers(app)
            assert app.query_one(Input).value == "keep my draft"
            assert not backend.requests
            assert len(app.conversation.messages) == 2
            assert lock.is_dir()
        lock.rmdir()  # Simulate the operator recovering the crashed writer.
        await app.send_message("keep my draft")
        await wait_for_app_workers(app)
        assert len(backend.requests) == 1
        assert len(store.load(original.id).messages) == 4


def test_periodic_poll_detects_in_place_edit_with_preserved_mtime(store, monkeypatch):
    now = [100.0]
    monkeypatch.setattr("bedrock.storage.time", SimpleNamespace(monotonic=lambda: now[0]))
    original = saved(store, title="Old title")
    store.list()
    signature = store._directory_signature()
    path = store.path_for(original.id)
    before = path.stat()
    contents = path.read_text().replace("Old title", "New title")
    path.write_text(contents)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert store._directory_signature() == signature
    assert not store.directory_changed()
    now[0] += HISTORY_RESCAN_SECONDS
    assert store.directory_changed()
    assert store.list()[0].title == "New title"
    assert not store.directory_changed()


def test_new_chat_deletion_names_hide_start_time_and_block_related_copies(store):
    original = saved(store)
    stale = store.load(original.id)
    assert store.delete(original.id)
    markers = list(store.directory.glob("*.deleted"))
    assert markers and all(marker.read_bytes() == b"" for marker in markers)
    assert all(len(marker.stem) == 32 and "-" not in marker.stem for marker in markers)
    stale.id = new_conversation_id()
    stale.storage_revision = None
    with pytest.raises(ConversationDeleted):
        store.save(stale)


def test_legacy_deletion_markers_still_block_synced_files_and_copies(store):
    original = Conversation(id="20260915-120000-abcdef")
    original.add_user("legacy chat")
    store.save(original)
    stale = store.load(original.id)
    legacy_marker = store.path_for(original.id).with_suffix(".deleted")
    legacy_marker.touch()
    # Simulate a sync tool restoring the deleted JSON beside its old marker.
    assert store.list() == []
    with pytest.raises(ConversationDeleted):
        store.load(original.id)
    stale.id = new_conversation_id()
    stale.storage_revision = None
    with pytest.raises(ConversationDeleted):
        store.save(stale)
