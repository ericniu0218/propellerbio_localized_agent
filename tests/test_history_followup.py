"""D1-D6: metadata merges, family deletion, storage fallbacks and idle I/O."""
import errno
import json
import multiprocessing
import os
import subprocess
import sys
from datetime import datetime, timezone
from uuid import UUID

import pytest
from textual.widgets import Input

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.models import Conversation, new_conversation_id
from bedrock.storage import ConversationDeleted, ConversationStore, StorageConflict
from bedrock.ui.widgets import SystemNotice
from tests.conftest import FakeBackend
from tests.test_history_regressions import saved


def test_bad_history_setting_exits_with_helpful_error_without_traceback(tmp_path):
    env = os.environ | {"BEDROCK_SAVE_HISTORY": "maybe", "BEDROCK_HISTORY_DIR": str(tmp_path)}
    result = subprocess.run([sys.executable, "-m", "bedrock"], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "BEDROCK_SAVE_HISTORY" in result.stderr and "'maybe'" in result.stderr
    assert "Traceback" not in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("metadata", ["title", "checked", "touch"])
def test_remote_metadata_updates_merge_with_new_messages(store, metadata):
    original = saved(store, title=None)
    other = ConversationStore(store.directory)
    stale = other.load(original.id)
    if metadata == "title":
        original.generated_title = "Salary Discussion"
    elif metadata == "checked":
        original.title_checked_at = 2
    store.save(original, touch=metadata == "touch")
    stale.add_user("my follow up")
    other.save(stale)
    restored = store.load(original.id)
    assert len(store.list()) == 1
    assert restored.messages[-1].content == "my follow up"
    assert restored.generated_title == original.generated_title
    assert restored.title_checked_at == original.title_checked_at


def test_local_title_is_saved_alongside_remote_touch(store):
    original = saved(store, title=None)
    other = ConversationStore(store.directory)
    stale = other.load(original.id)
    store.save(original)
    touched_at = original.updated_at
    stale.generated_title = "A New Title"
    other.save(stale, touch=False)
    restored = store.load(original.id)
    assert restored.generated_title == "A New Title"
    assert restored.updated_at == touched_at


async def test_background_title_in_another_window_does_not_fork_chat(store):
    original = saved(store, title=None)
    backend = FakeBackend(topic_reply="NO")
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        other = ConversationStore(store.directory)
        remote = other.load(original.id)
        remote.generated_title = "Salary Discussion"
        other.save(remote, touch=False)
        await app.send_message("next question")
        await wait_for_app_workers(app)
        assert app.conversation.id == original.id
        assert app.conversation.generated_title == "Salary Discussion"
        assert len(other.list()) == 1


async def test_forget_removes_all_related_copies_but_not_same_title_chats(store):
    original = saved(store, "My salary is 123456", title="Salary")
    unrelated = saved(store, "an unrelated topic", title="Salary")
    store.save(original)  # reopen the original
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test() as pilot:
        await pilot.pause()
        other = ConversationStore(store.directory)
        remote = other.load(original.id)
        remote.add_user("question in the other window")
        remote.add_assistant("its answer")
        other.save(remote)
        await app.send_message("question in my window")
        await wait_for_app_workers(app)
        copy_id = app.conversation.id
        assert copy_id != original.id
        assert other.load(copy_id).family_id == remote.family_id
        pending_copy = other.load(copy_id)
        await app.command_forget()
        await wait_for_app_workers(app)
        assert {row.id for row in other.list()} == {unrelated.id}
        assert not other.path_for(original.id).exists()
        assert not other.path_for(copy_id).exists()
        # Even a new competing copy prepared before /forget must be rejected.
        pending_copy.id = new_conversation_id()
        pending_copy.storage_revision = None
        with pytest.raises(ConversationDeleted):
            other.save(pending_copy)


def test_legacy_unlinked_copies_are_forgotten_together(store):
    original = saved(store, "My salary is 123456")
    data = json.loads(store.path_for(original.id).read_text())
    data.pop("family_id")
    store.path_for(original.id).write_text(json.dumps(data))
    copy_id = new_conversation_id()
    data["id"] = copy_id
    data["messages"].append({"role": "user", "content": "copy follow-up"})
    store.path_for(copy_id).write_text(json.dumps(data))
    assert store.delete(copy_id)
    assert not list(store.directory.glob("*.json"))


def test_legacy_family_can_be_forgotten_after_original_file_disappears(store):
    original = saved(store, "salary")
    path = store.path_for(original.id)
    data = json.loads(path.read_text())
    data.pop("family_id")
    path.write_text(json.dumps(data))
    opened = store.load(original.id)
    copy_id = new_conversation_id()
    data["id"] = copy_id
    store.path_for(copy_id).write_text(json.dumps(data))
    path.unlink()
    assert store.delete(opened.id, family_id=opened.family_id)
    assert not list(store.directory.glob("*.json"))


async def test_external_forget_blocks_requests_and_retains_the_draft(store):
    original = saved(store)
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        before = len(app.conversation.messages)
        ConversationStore(store.directory).delete(original.id)
        await app.send_message("please keep this draft")
        assert not backend.requests
        assert len(app.conversation.messages) == before
        assert app.query_one(Input).value == "please keep this draft"
        assert any("deleted" in notice.source and "nothing was sent" in notice.source for notice in app.query(SystemNotice))
        await app.start_new_conversation()
        await app.send_message("a new question")
        await wait_for_app_workers(app)
        assert len(backend.requests) == 1
        assert store.latest().id != original.id


async def test_delete_between_preflight_and_save_still_blocks_the_request(store, monkeypatch):
    original = saved(store)
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model"), backend, store)
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        real_save = store.save

        def raced_save(conversation, touch=True, **kwargs):
            ConversationStore(store.directory).delete(original.id)
            return real_save(conversation, touch=touch, **kwargs)

        monkeypatch.setattr(store, "save", raced_save)
        await app.send_message("my draft")
        assert not backend.requests
        assert app.query_one(Input).value == "my draft"


def test_ids_keep_full_uuid_entropy_and_remain_compatible(monkeypatch):
    frozen = datetime(2026, 9, 15, tzinfo=timezone.utc)
    monkeypatch.setattr("bedrock.models.conversation.utc_now", lambda: frozen)
    values = iter([UUID("12345600-0000-4000-8000-000000000001"), UUID("12345600-0000-4000-8000-000000000002")])
    monkeypatch.setattr("bedrock.models.conversation.uuid.uuid4", lambda: next(values))
    first, second = new_conversation_id(), new_conversation_id()
    assert first != second
    assert len(first.split("-")[-1]) == 32
    assert len(first) == 32  # No chat-start timestamp leaks through deletion markers.


def _no_flock(*args):
    raise OSError(errno.ENOTSUP, "File locking unsupported")


def test_storage_without_flock_can_save_load_and_forget(store, monkeypatch):
    monkeypatch.setattr("bedrock.storage.fcntl.flock", _no_flock)
    original = saved(store)
    assert store.load(original.id).messages[0].content == "question"
    assert not (store.directory / ".history-writer").exists()
    assert store.delete(original.id)
    assert not store.list()


def _fallback_writer(directory, identifier, barrier, queue):
    import bedrock.storage as storage

    storage.fcntl.flock = _no_flock
    store = storage.ConversationStore(directory)
    conversation = store.load(identifier)
    conversation.add_user(str(os.getpid()))
    barrier.wait(timeout=10)
    try:
        store.save(conversation)
        queue.put("saved")
    except StorageConflict:
        queue.put("conflict")


def test_fallback_lock_serializes_processes(store):
    original = saved(store)
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(2), context.Queue()
    processes = [context.Process(target=_fallback_writer, args=(str(store.directory), original.id, barrier, queue)) for _ in range(2)]
    for process in processes:
        process.start()
    try:
        assert sorted(queue.get(timeout=15) for _ in processes) == ["conflict", "saved"]
    finally:
        for process in processes:
            process.join(timeout=15)
            if process.is_alive():
                process.terminate()
                process.join()
    assert all(process.exitcode == 0 for process in processes)


async def test_idle_polling_does_not_scan_chat_files(store, monkeypatch):
    saved(store)
    app = BedrockApp(Settings(model="test-model"), FakeBackend(), store)
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await app.refresh_sidebar().wait()

        def unexpected_scan(*args, **kwargs):
            pytest.fail("Idle polling must not scan individual chat files")

        monkeypatch.setattr(store, "list", unexpected_scan)
        monkeypatch.setattr(store, "_fingerprint", unexpected_scan)
        for _ in range(10):
            await app.poll_history().wait()


def test_local_save_does_not_hide_remote_changes_from_polling(store):
    original = saved(store)
    store.list()
    other = ConversationStore(store.directory)
    added = saved(other, "elsewhere")
    original.add_user("local change")
    store.save(original)
    assert store.directory_changed()
    assert {row.id for row in store.list()} == {original.id, added.id}
    assert not store.directory_changed()
