"""Summary titles: building the request, cleaning the reply, and storing it."""

import asyncio

import pytest

from bedrock.inference import InferenceError
from bedrock.models import Conversation
from bedrock.titles import (
    MAX_TITLE_CHARACTERS,
    NO_TOPIC,
    TITLE_INSTRUCTION,
    clean_title,
    generate_title,
    is_title_request,
    means_yes,
    title_request,
    wants_title,
)


def test_reasoning_block_before_yes_is_ignored():
    assert means_yes("<think>Long private reasoning.</think>\nYES")


def test_title_control_characters_are_removed():
    assert clean_title("Bees\x1b]0;pwned\x07 Facts") == "Bees]0;pwned Facts"

from tests.conftest import FakeBackend


def chat(*turns: str, system_prompt: str | None = None) -> Conversation:
    conversation = Conversation(system_prompt=system_prompt)
    for index, text in enumerate(turns):
        (conversation.add_user if index % 2 == 0 else conversation.add_assistant)(text)
    return conversation


# ---- What gets sent -------------------------------------------------------


def test_request_carries_the_opening_and_the_latest_messages():
    turns = [f"message {n}" for n in range(1, 11)]
    messages = title_request(chat(*turns))

    assert is_title_request(messages)
    assert messages[0] == {"role": "system", "content": TITLE_INSTRUCTION}
    body = messages[1]["content"]
    for kept in ["message 1", "message 2", "message 7", "message 8", "message 9", "message 10"]:
        assert f": {kept}\n" in body + "\n", kept
    for skipped in ["message 3", "message 4", "message 5", "message 6"]:
        assert f": {skipped}\n" not in body + "\n", skipped


def test_a_topic_raised_after_small_talk_reaches_the_model():
    conversation = chat("hi", "Hello!", "how are you", "Fine, thanks.", "ok", "Great.", "I need to fix a Python import error", "Let's look.")
    assert "Python import error" in title_request(conversation)[1]["content"]


def test_the_instruction_tells_the_model_how_to_say_there_is_no_topic():
    assert NO_TOPIC in TITLE_INSTRUCTION


def test_request_trims_long_pasted_text():
    conversation = chat("x" * 5000, "ok")
    assert len(title_request(conversation)[1]["content"]) < 1500


def test_request_leaves_out_the_configured_system_prompt():
    conversation = chat("hello", "hi", system_prompt="COMPANY CHAT RULES")
    assert "COMPANY CHAT RULES" not in str(title_request(conversation))


def test_chat_requests_are_not_mistaken_for_title_requests():
    assert not is_title_request(chat("hello", "hi").to_api())
    assert not is_title_request([])


# ---- Cleaning up what comes back ------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Bee Pollination Facts", "Bee Pollination Facts"),
        ('"Bee Pollination Facts"', "Bee Pollination Facts"),
        ("**Bee Pollination Facts**", "Bee Pollination Facts"),
        ("Title: Bee Pollination Facts", "Bee Pollination Facts"),
        ("Bee Pollination Facts.", "Bee Pollination Facts"),
        ("Here is a short title: Bee Facts", "Bee Facts"),
        ("Here's a title:\n\nBee Facts", "Bee Facts"),
        ("Sure! Title: Bee Facts", "Bee Facts"),
        ("  Bee   Facts  \nsecond line ignored", "Bee Facts"),
        ("游戏翻译解释", "游戏翻译解释"),
        ("《游戏翻译解释》", "游戏翻译解释"),
        ("Time Management", "Time Management"),  # "time" alone is not vague
        ("Conversation Skills for Nurses", "Conversation Skills for Nurses"),
        ("Getting Started with Docker", "Getting Started with Docker"),
        ("Help Desk Setup", "Help Desk Setup"),
        ("Question About Taxes", "Question About Taxes"),
    ],
)
def test_clean_title_accepts_what_small_models_actually_send(raw, expected):
    assert clean_title(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   \n  ",
        '""',
        "I cannot determine a good title because the conversation does not have enough context yet.",
        # The model's way of saying "no concrete topic yet":
        "NONE",
        "none.",
        "**NONE**",
        '"NONE"',
        "Title: NONE",
        # ...and the ways small models say it without following instructions:
        "Unclear Conversation Topic",
        "Basic Conversation Topic",
        "Getting Started",
        "Let's Get Started",
        "Getting Started with Conversations",
        "Greetings",
        "New Conversation",
        # Real llama3.2 titles for chats with no topic, from testing:
        "Fresh Conversation Begins",
        "STARTING A NEW CONVERSATION",
        "INITIATING NEW CONVERSATION HERE",
        "GETTING HELP WITH SOMETHING",
        "Greeting and Introduction",
        "Initial Conversation Start",
    ],
)
def test_clean_title_rejects_things_that_are_not_titles(raw):
    assert clean_title(raw) is None


def test_clean_title_shortens_long_titles():
    title = clean_title("Supercalifragilisticexpialidocious Antidisestablishmentarianism")
    assert len(title) <= MAX_TITLE_CHARACTERS and title.endswith("…")


# ---- When a title is wanted -----------------------------------------------


def test_a_title_is_wanted_once_there_is_a_question_and_an_answer():
    assert not wants_title(Conversation())
    assert not wants_title(chat("only a question"))
    assert not wants_title(chat("question", "   "))  # a reply that never arrived
    conversation = chat("question", "answer")
    assert wants_title(conversation)
    conversation.generated_title = "Already Titled"
    assert not wants_title(conversation)


def test_after_finding_no_topic_the_model_is_only_asked_again_once_there_is_more():
    conversation = chat("hi", "Hello!")
    conversation.title_checked_at = 2  # the model saw both messages and said NONE
    assert not wants_title(conversation)

    conversation.add_user("I need help with my taxes")
    assert wants_title(conversation)


# ---- Talking to the model -------------------------------------------------


async def test_generate_title_asks_the_backend_and_cleans_the_answer():
    backend = FakeBackend(title_reply='"Bee Facts."')
    title = await generate_title(backend, "test-model", chat("bees", "Bees are insects."))

    assert title == "Bee Facts"
    assert backend.title_requests[0][0] == "test-model"
    assert backend.requests == []  # not counted as a chat turn


async def test_generate_title_stops_reading_a_rambling_model():
    closed = asyncio.Event()

    class Rambler:
        async def stream_chat(self, model, messages):
            try:
                while True:
                    yield "word "
            finally:
                closed.set()

    title = await generate_title(Rambler(), "m", chat("q", "a"))
    assert title is None  # an essay is not a title
    assert closed.is_set()  # and the stream was closed, not abandoned


async def test_generate_title_lets_server_errors_through():
    backend = FakeBackend()
    backend.fail_titles = True
    with pytest.raises(InferenceError):
        await generate_title(backend, "test-model", chat("q", "a"))


# ---- Stored with the conversation ------------------------------------------


def test_generated_title_is_saved_and_preferred_over_the_excerpt():
    conversation = chat("can you help me with something about bees", "Sure.")
    assert conversation.title == "Untitled"

    conversation.generated_title = "Bee Help"
    restored = Conversation.from_dict(conversation.to_dict())

    assert restored.generated_title == "Bee Help"
    assert restored.title == "Bee Help"
    assert restored.excerpt == "can you help me with something about bees"


def test_conversations_saved_before_titles_existed_still_load():
    data = chat("old question", "old answer").to_dict()
    del data["generated_title"]
    restored = Conversation.from_dict(data)
    assert restored.generated_title is None
    assert restored.title == "Untitled"
    assert restored.excerpt == "old question"
    assert restored.title_checked_at == 0  # never looked at, so it will be


def test_saving_a_title_does_not_reorder_history(store):
    older = chat("older", "a")
    newer = chat("newer", "b")
    store.save(older)
    store.save(newer)

    older.generated_title = "Older Topic"
    store.save(older, touch=False)

    summaries = store.list()
    assert [s.title for s in summaries] == ["Untitled", "Older Topic"]
    assert summaries[1].generated_title == "Older Topic"
    assert summaries[0].generated_title is None


def test_title_checked_at_is_saved_and_junk_values_are_ignored():
    conversation = chat("hi", "Hello!")
    conversation.title_checked_at = 2
    assert Conversation.from_dict(conversation.to_dict()).title_checked_at == 2

    for junk in [-5, "3", None, True, 2.5]:
        data = conversation.to_dict()
        data["title_checked_at"] = junk
        assert Conversation.from_dict(data).title_checked_at == 0, junk


# ---- Gate 1: did the user write any words? ----------------------------------


@pytest.mark.parametrize("text", ["1", "???", "42 !!", "   "])
def test_messages_without_letters_have_nothing_to_name(text):
    from bedrock.titles import user_wrote_words

    assert not user_wrote_words(chat(text, "What would you like to talk about?"))


@pytest.mark.parametrize("text", ["bees", "蜂", "1 thing", "游戏翻译"])
def test_any_letters_in_any_script_count_as_words(text):
    from bedrock.titles import user_wrote_words

    assert user_wrote_words(chat(text, "ok"))


async def test_the_model_is_not_asked_about_a_conversation_with_no_words():
    backend = FakeBackend()
    assert await generate_title(backend, "test-model", chat("1", "What can I do for you?")) is None
    assert backend.topic_checks == [] and backend.title_requests == []


# ---- Gate 2: YES/NO topic check ---------------------------------------------


def test_topic_check_shows_only_what_the_user_said():
    from bedrock.titles import TOPIC_CHECK_INSTRUCTION, is_topic_check, topic_check_request

    messages = topic_check_request(chat("hi", "Hello! Ask me about DATABASES.", "sea animals", "Sure."))
    assert is_topic_check(messages) and not is_title_request(messages)
    assert messages[0]["content"] == TOPIC_CHECK_INSTRUCTION
    body = messages[1]["content"]
    assert "- hi" in body and "- sea animals" in body
    assert "DATABASES" not in body  # the assistant cannot talk it into a YES


@pytest.mark.parametrize(
    "reply, expected",
    [("YES", True), ("Yes.", True), ("**YES**", True), (" yes\n", True),
     ("NO", False), ("No.", False), ("Maybe", False), ("", False), ("Yesterday", False)],
)
def test_anything_not_clearly_yes_counts_as_no(reply, expected):
    from bedrock.titles import means_yes

    assert means_yes(reply) is expected


async def test_a_no_from_the_topic_check_means_no_title_request():
    backend = FakeBackend(topic_reply="NO")
    assert await generate_title(backend, "test-model", chat("hey how are you", "Good, thanks!")) is None
    assert len(backend.topic_checks) == 1
    assert backend.title_requests == []


async def test_a_yes_still_has_to_produce_a_real_title():
    backend = FakeBackend(topic_reply="YES", title_reply="Getting Started")
    assert await generate_title(backend, "test-model", chat("can you help me", "Sure")) is None
