"""The storage layer on its own: real files in a temporary directory."""

import json
import stat

import pytest

from bedrock.models import Conversation
from bedrock.storage import ConversationStore, StorageConflict, StorageError


def a_conversation(question: str = "hello", answer: str = "hi there") -> Conversation:
    conversation = Conversation(model="test-model")
    conversation.add_user(question)
    conversation.add_assistant(answer)
    return conversation


def test_saved_conversation_comes_back_after_a_restart(store):
    original = a_conversation("what is our VPN policy?", "Ask IT.")
    store.save(original)

    # A brand new store object, as if the app had been closed and reopened.
    reopened = ConversationStore(store.directory)
    restored = reopened.load(original.id)

    assert restored.id == original.id
    assert restored.model == "test-model"
    assert [m.content for m in restored.messages] == ["what is our VPN policy?", "Ask IT."]


def test_saving_twice_updates_one_file_rather_than_adding_another(store):
    conversation = a_conversation()
    store.save(conversation)
    conversation.add_user("a follow-up")
    store.save(conversation)

    assert len(list(store.directory.glob("*.json"))) == 1
    assert len(store.load(conversation.id).messages) == 3


def test_latest_returns_the_most_recently_updated(store):
    first = a_conversation("first question")
    second = a_conversation("second question")
    store.save(first)
    store.save(second)
    store.save(first)  # touching the older one makes it the newest again

    assert store.latest().id == first.id


def test_latest_is_none_when_nothing_has_been_saved(store):
    assert store.latest() is None
    assert store.list() == []


def test_list_summarises_without_loading_the_whole_history(store):
    store.save(a_conversation("older"))
    newer = a_conversation("newer question here")
    store.save(newer)

    summaries = store.list()

    assert [s.excerpt for s in summaries] == ["newer question here", "older"]
    assert [s.title for s in summaries] == ["Untitled", "Untitled"]
    assert summaries[0].id == newer.id
    assert summaries[0].message_count == 2
    assert summaries[0].model == "test-model"
    assert store.list(limit=1) == summaries[:1]


def test_system_prompt_is_supplied_on_load_not_read_from_the_file(store):
    conversation = a_conversation()
    conversation.system_prompt = "Prompt at save time."
    store.save(conversation)

    restored = store.load(conversation.id, system_prompt="Prompt at load time.")
    assert restored.system_prompt == "Prompt at load time."


def test_delete_removes_the_file(store):
    conversation = a_conversation()
    store.save(conversation)

    assert store.delete(conversation.id) is True
    assert store.list() == []
    assert store.delete(conversation.id) is False  # already gone, not an error


def test_one_corrupt_file_does_not_hide_the_others(store):
    good = a_conversation("readable")
    store.save(good)
    (store.directory / "broken.json").write_text("{ this is not json")

    assert [s.id for s in store.list()] == [good.id]
    with pytest.raises(StorageError):
        store.load("broken")


def test_loading_something_that_was_never_saved_raises_storage_error(store):
    with pytest.raises(StorageError):
        store.load("20260101-000000-abcdef")


def test_files_are_readable_only_by_their_owner(store):
    path = store.save(a_conversation())

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700


def test_a_conversation_id_cannot_escape_the_directory(store):
    # Invalid ids are refused rather than normalised into a colliding filename.
    with pytest.raises(StorageError):
        store.path_for("../../etc/passwd")
    with pytest.raises(StorageError):
        store.path_for("abc/../def")
    with pytest.raises(StorageError):
        store.path_for("../..")


def test_saved_file_is_readable_plain_json(store):
    path = store.save(a_conversation("hello", "hi"))
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["version"] == 1
    assert data["excerpt"] == "hello"
    assert data["title"] == "Untitled"
    assert data["title_checked_at"] == 0
    assert [m["role"] for m in data["messages"]] == ["user", "assistant"]


def test_a_failed_save_leaves_no_half_written_file(store, monkeypatch):
    conversation = a_conversation()
    store.save(conversation)

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("os.replace", explode)
    with pytest.raises(StorageError):
        store.save(conversation)

    # The previous version is still intact and no stray .tmp file is left behind.
    assert len(store.load(conversation.id).messages) == 2
    assert list(store.directory.glob("*.tmp")) == []


def test_malformed_valid_json_is_skipped_without_hiding_good_chats(store):
    good = a_conversation("readable")
    store.save(good)
    (store.directory / "malformed.json").write_text(json.dumps({"id": "malformed", "messages": 5}))
    assert [summary.id for summary in store.list()] == [good.id]
    with pytest.raises(StorageError):
        store.load("malformed")


def test_message_with_non_text_content_is_skipped(store):
    data = a_conversation().to_dict()
    data["messages"].append({"role": "assistant", "content": None})
    store.directory.mkdir(parents=True)
    store.path_for(data["id"]).write_text(json.dumps(data))
    assert len(store.load(data["id"]).messages) == 2


def test_naive_saved_timestamp_is_assumed_to_be_utc(store):
    data = a_conversation().to_dict()
    data["updated_at"] = "2026-09-14T10:00:00"
    store.directory.mkdir(parents=True)
    store.path_for(data["id"]).write_text(json.dumps(data))
    assert store.list()[0].updated_at.utcoffset() is not None


def test_a_copied_file_with_a_duplicate_internal_id_is_ignored(store):
    conversation = a_conversation()
    original = store.save(conversation)
    (store.directory / f"{conversation.id}-copy.json").write_bytes(original.read_bytes())
    assert [summary.id for summary in store.list()] == [conversation.id]


def test_lone_surrogate_is_saved_as_an_ascii_json_escape(store):
    conversation = a_conversation(answer="half: \ud83d")
    path = store.save(conversation)
    assert "\\ud83d" in path.read_text("utf-8")
    assert store.load(conversation.id).messages[-1].content == "half: \ud83d"


def test_two_loaded_copies_detect_conflict_without_interleaving_turns(store):
    conversation = a_conversation("shared", "answer")
    store.save(conversation)
    left_store = ConversationStore(store.directory)
    right_store = ConversationStore(store.directory)
    left = left_store.load(conversation.id)
    right = right_store.load(conversation.id)
    left.add_user("from left")
    right.add_user("from right")
    left_store.save(left)
    with pytest.raises(StorageConflict):
        right_store.save(right)
    assert store.load(conversation.id).messages[-1].content == "from left"
    assert right.messages[-1].content == "from right"


def test_newer_schema_is_read_only_and_unknown_fields_survive(store):
    conversation = a_conversation()
    data = conversation.to_dict() | {"version": 2, "attachments": ["chat.bin"]}
    data["messages"][0]["attachments"] = ["message.bin"]
    store.directory.mkdir(parents=True)
    store.path_for(conversation.id).write_text(json.dumps(data))
    loaded = store.load(conversation.id)
    assert loaded.to_dict()["attachments"] == ["chat.bin"]
    assert loaded.to_dict()["messages"][0]["attachments"] == ["message.bin"]
    with pytest.raises(StorageError, match="newer app version"):
        store.save(loaded)
