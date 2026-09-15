"""The data model on its own: no UI, no filesystem, no server."""

import json
from uuid import UUID

from bedrock.models import Conversation, Message, Role, new_conversation_id


def test_to_api_sends_only_role_and_content():
    conversation = Conversation(system_prompt="Be brief.")
    conversation.add_user("hello")
    conversation.add_assistant("hi")

    assert conversation.to_api() == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_roundtrip_preserves_everything_worth_keeping():
    original = Conversation(model="test-model")
    original.add_user("what is a vector database?")
    original.add_assistant("A store for embeddings.")

    restored = Conversation.from_dict(original.to_dict())

    assert restored.id == original.id
    assert restored.model == "test-model"
    assert restored.created_at == original.created_at
    assert restored.updated_at == original.updated_at
    assert [(m.role, m.content) for m in restored.messages] == [
        (Role.USER, "what is a vector database?"),
        (Role.ASSISTANT, "A store for embeddings."),
    ]
    assert restored.messages[0].timestamp == original.messages[0].timestamp


def test_saved_form_is_plain_json():
    conversation = Conversation()
    conversation.add_user("hello")
    # No exception means every value is JSON-serialisable.
    assert json.loads(json.dumps(conversation.to_dict()))["messages"][0]["content"] == "hello"


def test_system_prompt_is_not_saved_but_comes_from_config_on_load():
    conversation = Conversation(system_prompt="Old prompt.")
    conversation.add_user("hello")

    assert "system_prompt" not in conversation.to_dict()

    restored = Conversation.from_dict(conversation.to_dict(), system_prompt="New prompt.")
    assert restored.system_prompt == "New prompt."
    assert restored.to_api()[0] == {"role": "system", "content": "New prompt."}


def test_a_conversation_is_untitled_until_the_model_names_it():
    conversation = Conversation()
    conversation.add_user("  How do   I reset my password?\n")
    assert conversation.title == "Untitled"  # no invented name

    conversation.generated_title = "Password Reset"
    assert conversation.title == "Password Reset"


def test_excerpt_is_the_first_question():
    conversation = Conversation()
    assert conversation.excerpt == "Empty conversation"

    conversation.add_user("  How do   I reset my password?\n")
    conversation.add_user("second question")
    assert conversation.excerpt == "How do I reset my password?"


def test_long_excerpts_are_shortened():
    conversation = Conversation()
    conversation.add_user("word " * 100)
    assert len(conversation.excerpt) <= 60
    assert conversation.excerpt.endswith("…")


def test_one_broken_message_does_not_lose_the_conversation():
    data = {
        "id": "abc",
        "messages": [
            {"role": "user", "content": "kept"},
            {"role": "wizard", "content": "unknown role, skipped"},
            "not even a dict",
            {"role": "assistant", "content": "also kept"},
        ],
    }
    restored = Conversation.from_dict(data)
    assert [m.content for m in restored.messages] == ["kept", "also kept"]


def test_unreadable_timestamps_fall_back_instead_of_raising():
    restored = Conversation.from_dict(
        {"id": "abc", "created_at": "not a date", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert restored.created_at is not None


def test_ids_are_unique_random_uuids_without_timestamps():
    ids = [new_conversation_id() for _ in range(50)]
    assert len(set(ids)) == 50

    assert all(len(identifier) == 32 and UUID(identifier).version == 4 for identifier in ids)


def test_message_to_api_omits_local_bookkeeping():
    assert Message(Role.USER, "hi").to_api() == {"role": "user", "content": "hi"}


def test_to_api_merges_consecutive_roles_for_strict_chat_templates():
    conversation = Conversation()
    conversation.add_user("first")
    conversation.add_user("second")
    assert conversation.to_api() == [{"role": "user", "content": "first\n\nsecond"}]
