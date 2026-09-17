"""Conversations must survive the app being closed.

Each test drives the real Textual app, lets it exit, then builds a *second*
app against the same directory - which is what actually happens when an
employee quits the terminal and comes back tomorrow.
"""

import asyncio
from datetime import datetime, timezone

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp, find_summaries
from bedrock.config import Settings
from bedrock.storage import ConversationStore, ConversationSummary
from bedrock.ui.widgets import AssistantMessage, SystemNotice, UserMessage

from tests.conftest import FakeBackend


def make_app(backend: FakeBackend, store: ConversationStore, model: str | None = "test-model", **settings) -> BedrockApp:
    return BedrockApp(settings=Settings(model=model, **settings), backend=backend, store=store)


def reopen(store: ConversationStore, backend: FakeBackend | None = None, **kwargs) -> BedrockApp:
    """A fresh app and a fresh store object, pointed at the same directory."""
    return make_app(backend or FakeBackend(), ConversationStore(store.directory), **kwargs)


async def test_a_conversation_reopens_after_the_app_is_closed(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"what is our VPN policy?", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()
        first_id = app.conversation.id

    second = reopen(store)
    async with second.run_test() as pilot:
        await pilot.pause()

        assert second.conversation.id == first_id
        assert [m.content for m in second.conversation.messages][0] == "what is our VPN policy?"
        # And it is on screen, not just in memory.
        assert str(second.query_one(UserMessage).content) == "what is our VPN policy?"
        assert "Hello from the fake model." in second.query_one(AssistantMessage).source
        assert "Resumed" in second.query_one(SystemNotice).source


async def test_a_resumed_conversation_keeps_its_context_for_the_next_question(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"first question", "enter")
        await wait_for_app_workers(app)

    later_backend = FakeBackend()
    second = reopen(store, later_backend)
    async with second.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"follow-up question", "enter")
        await wait_for_app_workers(second)
        await pilot.pause()

    # The model is sent the whole history, including yesterday's turns.
    sent = later_backend.requests[0][1]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[0]["content"] == "first question"
    assert sent[2]["content"] == "follow-up question"


async def test_the_question_is_saved_before_the_answer_arrives(fake_backend, store):
    """If the app dies mid-answer, what the employee typed is not lost."""
    fake_backend.hold_reply = asyncio.Event()
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"slow question", "enter")
        await pilot.pause()

        saved = ConversationStore(store.directory).latest()
        assert [m.content for m in saved.messages] == ["slow question"]

        fake_backend.hold_reply.set()
        await wait_for_app_workers(app)


async def test_new_keeps_the_old_conversation_on_disk(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"first conversation", "enter")
        await wait_for_app_workers(app)
        first_id = app.conversation.id

        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert app.conversation.id != first_id
        assert not app.query(UserMessage)  # screen is clear

        await pilot.press(*"second conversation", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

    first_questions = [s.excerpt for s in ConversationStore(store.directory).list()]
    assert first_questions == ["second conversation", "first conversation"]


async def test_history_lists_saved_conversations_and_resume_reopens_one(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"older conversation", "enter")
        await wait_for_app_workers(app)
        older_id = app.conversation.id

        await pilot.press(*"/new", "enter")
        await pilot.press(*"newer conversation", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

        await pilot.press(*"/history", "enter")
        await pilot.pause()
        listing = app.query(SystemNotice).last().source
        assert "older conversation" in listing and "newer conversation" in listing
        assert "← open now" in listing

        # `2` is the older one: the list is newest first.
        await pilot.press(*"/resume 2", "enter")
        await pilot.pause()
        assert app.conversation.id == older_id
        assert str(app.query_one(UserMessage).content) == "older conversation"

        # Resuming did not throw the newer conversation away.
        assert len(ConversationStore(store.directory).list()) == 2


async def test_resume_accepts_an_id_and_rejects_nonsense(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"first", "enter")
        await wait_for_app_workers(app)
        first_id = app.conversation.id
        await pilot.press(*"/new", "enter")
        await pilot.press(*"second", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

        await pilot.press(*f"/resume {first_id}", "enter")
        await pilot.pause()
        assert app.conversation.id == first_id

        await pilot.press(*"/resume 99", "enter")
        await pilot.pause()
        assert "No saved conversation matches" in app.query(SystemNotice).last().source
        assert app.conversation.id == first_id  # unchanged


async def test_forget_deletes_the_conversation_from_disk(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"something private", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert len(ConversationStore(store.directory).list()) == 1

        await pilot.press(*"/forget", "enter")
        await pilot.pause()

        assert ConversationStore(store.directory).list() == []
        assert not app.query(UserMessage)
        assert "Deleted" in app.query(SystemNotice).last().source

    # Nothing is written back when the app exits, either.
    assert ConversationStore(store.directory).list() == []


async def test_an_empty_session_leaves_nothing_behind(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"/help", "enter")
        await pilot.press(*"/exit", "enter")
        await pilot.pause()

    assert not store.directory.exists() or list(store.directory.glob("*.json")) == []


async def test_the_model_is_remembered_with_the_conversation(fake_backend, store):
    app = make_app(fake_backend, store, model="test-model")
    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

    # No configured model this time, and a server offering something different:
    # the saved conversation's model wins over the server's first offer.
    second = reopen(store, FakeBackend(models=["other-model"]), model=None)
    async with second.run_test() as pilot:
        await pilot.pause()
        assert second.model_name == "test-model"


async def test_nothing_is_written_when_history_is_switched_off(fake_backend, tmp_path):
    settings = Settings(model="test-model", save_history=False, history_dir=tmp_path / "nope")
    app = BedrockApp(settings=settings, backend=fake_backend)
    async with app.run_test() as pilot:
        await pilot.press(*"a private question", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

        await pilot.press(*"/history", "enter")
        await pilot.pause()
        assert "switched off" in app.query(SystemNotice).last().source

    assert not (tmp_path / "nope").exists()


async def test_an_answer_stopped_half_way_is_still_saved(fake_backend, store):
    fake_backend.fragment_delay = 0.05
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"a question with a long answer", "enter")
        await pilot.pause()
        await pilot.press("escape")
        await wait_for_app_workers(app, cancelled_groups=frozenset({"chat"}))
        await pilot.pause()

    saved = ConversationStore(store.directory).latest()
    assert [m.role.value for m in saved.messages] == ["user", "assistant"]
    assert "(stopped)" in saved.messages[1].content


async def test_the_last_conversation_opened_comes_back_even_if_the_window_is_closed(fake_backend, store):
    """Leaving ``run_test`` without /exit or Ctrl+C is what closing the terminal
    window does: the process ends and no quit-time save runs."""
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"platypus", "enter")
        await wait_for_app_workers(app)
        platypus_id = app.conversation.id
        await pilot.press(*"/new", "enter")
        await pilot.press(*"bees", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

    # Next launch reopens "bees". The employee switches to "platypus", only
    # reads it, and closes the window.
    second = reopen(store)
    async with second.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"/resume 2", "enter")
        await pilot.pause()
        assert second.conversation.id == platypus_id

    third = reopen(store)
    async with third.run_test() as pilot:
        await pilot.pause()
        assert third.conversation.id == platypus_id


async def test_an_empty_new_conversation_does_not_replace_the_last_real_one(fake_backend, store):
    """``/new`` then closing the window should not greet you with a blank screen."""
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"bees", "enter")
        await wait_for_app_workers(app)
        bees_id = app.conversation.id
        await pilot.press(*"/new", "enter")
        await pilot.pause()

    second = reopen(store)
    async with second.run_test() as pilot:
        await pilot.pause()
        assert second.conversation.id == bees_id


async def test_resume_finds_a_conversation_by_words_from_its_title(fake_backend, store):
    """The bug report: ``/resume bees`` said nothing matched a chat titled "bees"."""
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"conversation 1 name it platypus", "enter")
        await wait_for_app_workers(app)
        platypus_id = app.conversation.id
        await pilot.press(*"/new", "enter")
        await pilot.press(*"bees", "enter")
        await wait_for_app_workers(app)
        bees_id = app.conversation.id
        await pilot.press("ctrl+l")  # the easy way to lose sight of a chat
        await pilot.pause()

        await pilot.press(*"/resume platypus", "enter")
        await pilot.pause()
        assert app.conversation.id == platypus_id
        assert str(app.query(UserMessage).first().content) == "conversation 1 name it platypus"

        await pilot.press(*"/resume BEES", "enter")  # case does not matter
        await pilot.pause()
        assert app.conversation.id == bees_id


async def test_resume_asks_instead_of_guessing_when_several_titles_match(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"notes on bees", "enter")
        await wait_for_app_workers(app)
        await pilot.press(*"/new", "enter")
        await pilot.press(*"more notes on wasps", "enter")
        await wait_for_app_workers(app)
        current = app.conversation.id
        await pilot.pause()

        await pilot.press(*"/resume notes", "enter")
        await pilot.pause()

        listing = app.query(SystemNotice).last().source
        assert "more than one" in listing
        assert "`1`" in listing and "`2`" in listing  # numbers usable with /resume
        assert app.conversation.id == current  # nothing was opened


def summary(id: str, title: str, excerpt: str = "") -> ConversationSummary:
    return ConversationSummary(
        id=id,
        title=title,
        model=None,
        updated_at=datetime.now(timezone.utc),
        message_count=2,
        excerpt=excerpt or title,
    )


def test_find_summaries_rules():
    bees = summary("20260914-230044-f66e78", "bees")
    wasps = summary("20260914-230015-799a91", "bees and wasps")
    tomato = summary("20260914-225951-496639", "remember my phrase 2")
    listing = [bees, wasps, tomato]

    assert find_summaries(listing, "2") == [wasps]  # a number is a position, never a title
    assert find_summaries(listing, "9") == []
    assert find_summaries(listing, "20260914-2259") == [tomato]  # start of an id
    assert find_summaries(listing, "bees") == [bees]  # exact title beats "bees and wasps"
    assert find_summaries(listing, "  Bees   AND wasps ") == [wasps]
    assert find_summaries(listing, "phrase") == [tomato]
    assert find_summaries(listing, "e") == [bees, wasps, tomato]  # ambiguous: caller asks
    assert find_summaries(listing, "platypus") == []


async def test_resume_still_finds_a_conversation_after_the_model_renames_it(store):
    """Regression: once "conversation 1 name it platypus" is titled "Sea Animal
    Talk", ``/resume platypus`` must still work - people remember what they typed."""
    backend = FakeBackend(title_reply="Sea Animal Talk")
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"conversation 1 name it platypus", "enter")
        await wait_for_app_workers(app)
        platypus_id = app.conversation.id
        await pilot.press(*"/new", "enter")
        await pilot.pause()
        assert ConversationStore(store.directory).list()[0].title == "Sea Animal Talk"

        await pilot.press(*"/resume platypus", "enter")
        await pilot.pause()
        assert app.conversation.id == platypus_id

        await pilot.press(*"/new", "enter")
        await pilot.press(*"/resume sea animal", "enter")  # the new title works too
        await pilot.pause()
        assert app.conversation.id == platypus_id


def test_find_summaries_matches_the_first_question_as_well_as_the_title():
    renamed = summary("20260914-230015-799a91", "Sea Animal Talk", excerpt="conversation 1 name it platypus")
    other = summary("20260914-230044-f66e78", "Bee Facts", excerpt="bees")

    assert find_summaries([renamed, other], "platypus") == [renamed]
    assert find_summaries([renamed, other], "bee facts") == [other]
    assert find_summaries([renamed, other], "bees") == [other]  # exact first question
