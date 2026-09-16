"""The queue component on its own: ordering, pausing, and one run at a time.

These tests use no app and no disk. The queue's job is to be boring and
predictable; the app tests in ``test_queue_execution.py`` cover what happens
when real requests run through it.
"""

import pytest

from bedrock.prompt_queue import PromptQueue, QueuedPrompt, QueueSnapshot


@pytest.fixture
def queue() -> PromptQueue:
    return PromptQueue()


def add_three(queue: PromptQueue, conversation_id: str = "chat-1") -> list[str]:
    return [queue.add(text, conversation_id, "test-model").id for text in ("first", "second", "third")]


def test_prompts_run_in_the_order_they_were_typed(queue):
    add_three(queue)
    started = []
    for _ in range(3):
        item = queue.start_next("chat-1")
        started.append(item.text)
        queue.finish(item.id)
    assert started == ["first", "second", "third"]
    assert queue.start_next("chat-1") is None


def test_only_one_prompt_runs_at_a_time(queue):
    add_three(queue)
    running = queue.start_next("chat-1")
    assert running.text == "first" and queue.busy
    assert queue.start_next("chat-1") is None  # still busy
    queue.finish(running.id)
    assert not queue.busy
    assert queue.start_next("chat-1").text == "second"


def test_every_item_has_its_own_id(queue):
    ids = add_three(queue) + add_three(queue, "chat-2")
    assert len(set(ids)) == 6


def test_items_carry_their_conversation_and_model(queue):
    item = queue.add("hello", "chat-1", "llama3.2")
    assert (item.text, item.conversation_id, item.model) == ("hello", "chat-1", "llama3.2")


# ---- Snapshots --------------------------------------------------------------


def test_snapshot_shows_this_conversations_pending_running_and_paused(queue):
    add_three(queue)
    running = queue.start_next("chat-1")
    queue.pause("chat-1")
    snapshot = queue.snapshot("chat-1")
    assert [p.text for p in snapshot.pending] == ["second", "third"]
    assert snapshot.running == QueuedPrompt(running.id, "first", "test-model")
    assert snapshot.paused is True
    assert len(snapshot) == 2 and not snapshot.is_idle


def test_snapshot_hides_another_conversations_work(queue):
    queue.add("mine", "chat-1", "test-model")
    queue.add("theirs", "chat-2", "test-model")
    running = queue.start_next("chat-2")
    snapshot = queue.snapshot("chat-1")
    assert snapshot.running is None and [p.text for p in snapshot.pending] == ["mine"]
    assert queue.snapshot("chat-2").running.text == running.text


def test_an_unknown_conversation_has_an_empty_snapshot(queue):
    assert queue.snapshot("never-used") == QueueSnapshot(conversation_id="never-used")
    assert queue.snapshot("never-used").is_idle


def test_snapshots_cannot_be_used_to_change_the_queue(queue):
    add_three(queue)
    snapshot = queue.snapshot("chat-1")
    with pytest.raises((AttributeError, TypeError)):
        snapshot.pending[0].text = "rewritten"
    with pytest.raises((AttributeError, TypeError)):
        snapshot.running = snapshot.pending[0]
    assert isinstance(snapshot.pending, tuple)
    # And it is a copy: later changes do not rewrite an old snapshot.
    queue.clear("chat-1")
    assert len(snapshot.pending) == 3


# ---- Pausing, removing, clearing --------------------------------------------


def test_pausing_stops_the_next_prompt_starting(queue):
    add_three(queue)
    assert queue.pause("chat-1") is True
    assert queue.pause("chat-1") is False  # already paused: nothing changed
    assert queue.start_next("chat-1") is None
    assert queue.resume("chat-1") is True
    assert queue.resume("chat-1") is False
    assert queue.start_next("chat-1").text == "first"


def test_pausing_one_conversation_leaves_another_alone(queue):
    queue.add("mine", "chat-1", "test-model")
    queue.add("theirs", "chat-2", "test-model")
    queue.pause("chat-1")
    assert queue.start_next("chat-1") is None
    assert queue.start_next("chat-2").text == "theirs"


def test_removing_a_waiting_prompt_keeps_the_rest_in_order(queue):
    first, second, third = add_three(queue)
    assert queue.remove(second).text == "second"
    assert queue.remove(second) is None  # gone, and removing it again is harmless
    assert [p.text for p in queue.snapshot("chat-1").pending] == ["first", "third"]


def test_a_running_prompt_cannot_be_removed(queue):
    add_three(queue)
    running = queue.start_next("chat-1")
    assert queue.remove(running.id) is None
    assert queue.snapshot("chat-1").running.id == running.id


def test_clearing_drops_waiting_prompts_but_not_the_running_one(queue):
    add_three(queue)
    running = queue.start_next("chat-1")
    dropped = queue.clear("chat-1")
    assert [item.text for item in dropped] == ["second", "third"]
    snapshot = queue.snapshot("chat-1")
    assert snapshot.pending == () and snapshot.running.id == running.id
    assert queue.busy


def test_dropping_deleted_conversations_takes_their_pending_and_pause(queue):
    add_three(queue, "chat-1")
    queue.add("copy of the same chat", "copy-1", "test-model")
    queue.add("unrelated", "chat-2", "test-model")
    queue.pause("chat-1")
    dropped = queue.drop_conversations(["chat-1", "copy-1"])
    assert len(dropped) == 4
    assert queue.snapshot("chat-1") == QueueSnapshot(conversation_id="chat-1")
    assert queue.is_paused("chat-1") is False
    assert [p.text for p in queue.snapshot("chat-2").pending] == ["unrelated"]


# ---- Finishing --------------------------------------------------------------


def test_only_the_running_prompt_can_finish_the_run(queue):
    first, second, _ = add_three(queue)
    running = queue.start_next("chat-1")
    assert queue.finish(second) is False  # a waiting prompt is not the running one
    assert queue.busy
    assert queue.finish(running.id) is True


def test_a_repeated_completion_never_releases_the_next_prompt(queue):
    add_three(queue)
    first = queue.start_next("chat-1")
    assert queue.finish(first.id) is True
    second = queue.start_next("chat-1")
    # A duplicate or delayed completion for the first prompt arrives late.
    assert queue.finish(first.id) is False
    assert queue.busy and queue.running.id == second.id


def test_an_unrelated_completion_is_ignored(queue):
    add_three(queue)
    running = queue.start_next("chat-1")
    assert queue.finish("some-other-request") is False
    assert queue.running.id == running.id


def test_starting_directly_marks_the_queue_busy(queue):
    item = queue.add("straight through", "chat-1", "test-model")
    queue.remove(item.id)
    assert queue.start(item) is True
    assert queue.busy and queue.snapshot("chat-1").running.id == item.id
    assert queue.start(item) is False  # something is already running
    assert queue.finish(item.id) is True


def test_conversations_lists_everything_the_queue_knows(queue):
    queue.add("a", "chat-1", None)
    running = queue.add("b", "chat-2", None)
    queue.start_next("chat-2")
    queue.pause("chat-3")
    assert queue.conversations() == {"chat-1", "chat-2", "chat-3"}
    queue.finish(running.id)
    queue.clear("chat-1")
    queue.resume("chat-3")
    assert queue.conversations() == set()
