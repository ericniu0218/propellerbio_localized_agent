"""Stress-report regressions: preserve drafts, recover startup, bound queues."""
import asyncio
import json

import pytest
from textual.widgets import Input

from bedrock.inference import InferenceError
from bedrock.models import Conversation
from bedrock.models.conversation import SCHEMA_VERSION
from bedrock.prompt_queue import MAX_QUEUE_ITEMS, MAX_PROMPT_CHARS, MAX_QUEUE_CHARS, PromptQueue, QueueFull
from bedrock.storage import StorageError
from bedrock.ui.widgets import SystemNotice
from tests.conftest import FakeBackend
from tests.test_agentpack_app import FakeAgentPackClient, make_app as agent_app
from tests.test_queue_execution import SlowBackend, make_app, settle, questions
from tests.workers import wait_for_app_workers


@pytest.mark.parametrize("mode", ["chat", "agent"])
async def test_unexpected_startup_error_preserves_item_and_allows_retry(store, monkeypatch, mode):
    backend = FakeBackend() if mode == "chat" else FakeAgentPackClient()
    app = make_app(store, backend) if mode == "chat" else agent_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        dispatch = app.dispatch_prompt

        async def fail(item):
            raise RuntimeError("startup failed")

        monkeypatch.setattr(app, "dispatch_prompt", fail)
        send = app.send_message if mode == "chat" else app.queue_prompt
        await send("keep this prompt")
        await pilot.pause()
        assert not app.prompt_queue.busy
        assert app.queue_snapshot().paused
        assert [item.text for item in app.queue_snapshot().pending] == ["keep this prompt"]
        assert any("start the prompt" in notice.source for notice in app.query(SystemNotice))
        monkeypatch.setattr(app, "dispatch_prompt", dispatch)
        app.resume_queue()
        await pilot.pause()
        if mode == "chat":
            await settle(app, pilot)
            assert questions(backend) == ["keep this prompt"]
        else:
            assert backend.sent_messages == [("sess_fake", "keep this prompt")]


async def test_unexpected_save_error_does_not_duplicate_question_on_retry(store, monkeypatch):
    backend = FakeBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        save = app.save_conversation

        def fail(*args, **kwargs):
            raise RuntimeError("unexpected save failure")

        monkeypatch.setattr(app, "save_conversation", fail)
        await app.send_message("keep once")
        await wait_for_app_workers(app)
        assert app.conversation.messages == []
        monkeypatch.setattr(app, "save_conversation", save)
        app.resume_queue()
        await settle(app, pilot)
        assert questions(backend) == ["keep once"]
        assert [m.content for m in app.conversation.messages if m.role.value == "user"] == ["keep once"]


async def test_dispatch_for_closed_chat_is_preserved_at_front(store):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        old = app.conversation.id
        first = app.prompt_queue.add("keep first", old, "test-model")
        app.prompt_queue.add("keep second", old, "test-model")
        app.prompt_queue.start_next(old)
        app.conversation = Conversation()
        await app.dispatch_prompt(first)
        assert not app.prompt_queue.busy
        assert app.prompt_queue.is_paused(old)
        assert [p.text for p in app.prompt_queue.pending_for(old)] == ["keep first", "keep second"]


async def test_agent_send_failure_keeps_prompt_at_front(monkeypatch):
    fake_client = FakeAgentPackClient()
    app = agent_app(fake_client)
    async with app.run_test() as pilot:
        await pilot.pause()
        send = fake_client.send_message

        async def fail(*args):
            raise InferenceError("connection refused")

        monkeypatch.setattr(fake_client, "send_message", fail)
        await app.queue_prompt("unsent draft")
        assert app.queue_snapshot().paused and not app.prompt_queue.busy
        assert [p.text for p in app.queue_snapshot().pending] == ["unsent draft"]
        monkeypatch.setattr(fake_client, "send_message", send)
        app.resume_queue()
        await pilot.pause()
        assert fake_client.sent_messages == [("sess_fake", "unsent draft")]


@pytest.mark.parametrize("mode", ["chat", "agent"])
async def test_stranded_item_recovers_but_preparation_is_not_interrupted(store, mode):
    app = make_app(store, FakeBackend()) if mode == "chat" else agent_app(FakeAgentPackClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        key = app.conversation.id if mode == "chat" else app.queue_key
        item = app.prompt_queue.add("stranded", key, None)
        app.prompt_queue.start_next(key)
        app._preparing_prompts.add(item.id)
        app.recover_queue()
        assert app.prompt_queue.running == item
        app._preparing_prompts.clear()
        app.recover_queue()
        app.recover_queue()
        assert not app.prompt_queue.busy and app.queue_snapshot().paused
        assert [p.id for p in app.queue_snapshot().pending] == [item.id]


async def test_typing_remains_responsive_during_recall(store, monkeypatch):
    app = make_app(store, FakeBackend())
    started, release = asyncio.Event(), asyncio.Event()

    async def recall(*args):
        started.set()
        await release.wait()
        return None

    monkeypatch.setattr(app, "recall_into", recall)
    async with app.run_test() as pilot:
        app.query_one(Input).value = "do you remember when we discussed biology?"
        try:
            await asyncio.wait_for(pilot.press("enter"), 2)
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.wait_for(pilot.press(*"next draft"), 2)
            assert app.query_one(Input).value == "next draft"
            app.recover_queue()
            assert app.prompt_queue.busy
        finally:
            release.set()
        await settle(app, pilot)


@pytest.mark.parametrize("mode", ["chat", "agent"])
async def test_quit_warns_and_requires_second_request(store, mode, monkeypatch):
    backend = FakeBackend() if mode == "chat" else FakeAgentPackClient()
    app = make_app(store, backend) if mode == "chat" else agent_app(backend)
    notices = []
    async with app.run_test() as pilot:
        await pilot.pause()
        monkeypatch.setattr(app, "notify", lambda text, **kwargs: notices.append(text))
        key = app.conversation.id if mode == "chat" else app.queue_key
        app.prompt_queue.add("first waiting", key, None)
        app.prompt_queue.add("second waiting", key, None)
        app.prompt_queue.pause(key)
        await app.action_quit()
        assert not backend.closed
        assert app.prompt_queue.pending_count == 2
        assert "2 waiting" in notices[-1] and "confirm" in notices[-1]
        await app.action_quit()
        assert backend.closed


async def test_agent_clear_and_chat_new_explain_waiting_prompts(store):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        old = app.conversation.id
        app.prompt_queue.add("waiting", old, None)
        await app.start_new_conversation()
        assert app.prompt_queue.is_paused(old)
        assert any("1 waiting" in n.source and "paused" in n.source for n in app.query(SystemNotice))
    app = agent_app(FakeAgentPackClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.prompt_queue.add("waiting", app.queue_key, None)
        await app.action_clear_chat()
        assert not app.prompt_queue.pending_count
        assert any("Cleared the chat and 1 waiting" in n.source for n in app.query(SystemNotice))


async def test_model_change_explains_frozen_model_and_read_only_keeps_prompt(store):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        app.prompt_queue.add("waiting", app.conversation.id, "test-model")
        await app.command_model("other-model")
        assert any("already waiting" in n.source and r"test\-model" in n.source for n in app.query(SystemNotice))
        app.conversation.source_version = SCHEMA_VERSION + 1
        await app.advance_queue()
        assert not app.prompt_queue.busy
        assert [p.text for p in app.queue_snapshot().pending] == ["waiting"]
        assert any("read-only" in n.source for n in app.query(SystemNotice))


def test_newer_disk_schema_is_not_overwritten_or_forked(store):
    chat = Conversation()
    chat.add_user("original")
    path = store.save(chat)
    data = json.loads(path.read_text())
    data["version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    chat.add_user("later")
    with pytest.raises(StorageError, match="read-only"):
        store.save(chat)
    assert path.read_bytes() == before
    assert len(list(store.directory.glob("*.json"))) == 1


def test_bounds_and_retry_reservation_apply_across_conversations():
    queue = PromptQueue()
    first = queue.add("first", "a", None)
    queue.start_next("a")
    for n in range(MAX_QUEUE_ITEMS - 1):
        queue.add("waiting", str(n), None)
    with pytest.raises(QueueFull):
        queue.add("overflow", "other", None)
    assert queue.requeue_front(first)
    assert queue.pending_count == MAX_QUEUE_ITEMS
    other = PromptQueue()
    with pytest.raises(QueueFull):
        other.add("x" * (MAX_PROMPT_CHARS + 1), "a", None)
    for _ in range(MAX_QUEUE_CHARS // MAX_PROMPT_CHARS):
        other.add("x" * MAX_PROMPT_CHARS, "a", None)
    with pytest.raises(QueueFull):
        other.add("x", "a", None)


def test_clear_and_empty_queues_do_not_keep_pause_or_metadata():
    queue = PromptQueue()
    queue.add("old", "a", None)
    queue.pause("a")
    queue.clear("a")
    queue.add("new", "a", None)
    item = queue.start_next("a")
    assert item.text == "new"
    queue.finish(item.id)
    assert queue.conversations() == set()
    assert not queue._pending


def test_forget_rejects_late_requeue_and_leaves_no_pause_marker():
    queue = PromptQueue()
    item = queue.add("forgotten", "a", None)
    queue.start_next("a")
    queue.pause("a")
    queue.drop_conversations(["a"])
    assert not queue.requeue_front(item)
    assert not queue.finish(item.id)
    assert queue.conversations() == set()


async def test_notifications_are_coalesced_and_overflow_preserves_draft(store, monkeypatch):
    app = make_app(store, FakeBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        snapshots = []
        original = app._emit_queue_changed

        def emitted():
            snapshots.append(app.queue_snapshot())
            original()

        monkeypatch.setattr(app, "_emit_queue_changed", emitted)
        for n in range(MAX_QUEUE_ITEMS):
            app.prompt_queue.add(str(n), app.conversation.id, None)
            app.queue_changed()
        app.prompt_queue.pause(app.conversation.id)
        await pilot.pause()
        assert len(snapshots) == 1 and len(snapshots[0].pending) == MAX_QUEUE_ITEMS
        await app.send_message("keep overflow")
        assert app.query_one(Input).value == "keep overflow"
        assert app.prompt_queue.pending_count == MAX_QUEUE_ITEMS


@pytest.mark.parametrize("paused", [False, True])
async def test_switch_during_recall_releases_open_chat_queue(store, monkeypatch, paused):
    backend = FakeBackend()
    app = make_app(store, backend)
    started, release = asyncio.Event(), asyncio.Event()

    async def recall(*args):
        started.set()
        await release.wait()
        return None

    monkeypatch.setattr(app, "recall_into", recall)
    async with app.run_test() as pilot:
        await app.send_message("do you remember when we discussed biology?")
        try:
            await asyncio.wait_for(started.wait(), 2)
            old_id = app.conversation.id
            await app.start_new_conversation()
            await app.send_message("question in the open chat")
            if paused:
                app.pause_queue()
            await pilot.pause()
            assert not backend.requests
            assert len(app.queue_snapshot().pending) == 1
        finally:
            release.set()
        await settle(app, pilot)
        assert not app.prompt_queue.busy
        assert app.prompt_queue.is_paused(old_id)
        if paused:
            assert not backend.requests
            assert len(app.queue_snapshot().pending) == 1
        else:
            assert questions(backend) == ["question in the open chat"]
            assert app.queue_snapshot().is_idle


async def test_success_in_old_chat_releases_open_chat_queue(store):
    backend = FakeBackend()
    app = make_app(store, backend)
    async with app.run_test() as pilot:
        old_id = app.conversation.id
        item = app.prompt_queue.add("old request", old_id, "test-model")
        app.prompt_queue.start_next(old_id)
        await app.start_new_conversation()
        await app.send_message("open request")
        app.finish_prompt(item, ok=True)
        await settle(app, pilot)
        assert questions(backend) == ["open request"]


@pytest.mark.parametrize("mode", ["chat", "agent"])
async def test_queue_lifecycle_notifications_share_a_display_interval(store, monkeypatch, mode):
    app = make_app(store, FakeBackend()) if mode == "chat" else agent_app(FakeAgentPackClient())
    async with app.run_test() as pilot:
        await pilot.pause()
        timers = []
        monkeypatch.setattr(app, "set_timer", lambda delay, callback: timers.append((delay, callback)))
        key = app.conversation.id if mode == "chat" else app.queue_key
        item = app.prompt_queue.add("quick request", key, "test-model")
        app.queue_changed()
        await asyncio.sleep(0)
        app.prompt_queue.start_next(key)
        app.queue_changed()
        await asyncio.sleep(0)
        app.prompt_queue.finish(item.id)
        app.queue_changed()
        assert len(timers) == 1
        delay, emit = timers.pop()
        assert 0 < delay <= 1 / 60
        snapshots = []
        with monkeypatch.context() as capture:
            capture.setattr(app, "post_message", snapshots.append)
            emit()
        assert len(snapshots) == 1
        assert snapshots[0].snapshot.is_idle
        assert not app._queue_update_scheduled
