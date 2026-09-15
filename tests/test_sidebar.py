"""The conversation sidebar, and the summary titles it shows."""

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.models import Conversation
from bedrock.storage import ConversationStore
from bedrock.ui.widgets import NEW_CONVERSATION, ConversationList, SystemNotice, UserMessage

from tests.conftest import FakeBackend


def make_app(backend: FakeBackend, store: ConversationStore | None, **settings) -> BedrockApp:
    return BedrockApp(settings=Settings(model="test-model", **settings), backend=backend, store=store)


def rows(app: BedrockApp) -> list[str]:
    """The conversation titles in the sidebar, top to bottom."""
    sidebar = app.query_one(ConversationList)
    return [
        str(option.prompt)
        for option in sidebar.options
        if option.id != NEW_CONVERSATION
    ]


def open_row(app: BedrockApp) -> str | None:
    """The title of the highlighted row - the conversation on screen."""
    sidebar = app.query_one(ConversationList)
    option = sidebar.highlighted_option
    return None if option is None or option.id == NEW_CONVERSATION else str(option.prompt)


async def click_row(pilot, app: BedrockApp, option_id: str) -> None:
    """Click a sidebar row with the mouse, exactly as an employee would.

    Every row is one line tall, plus one line for the divider under
    "+ New conversation" and one line of top padding.
    """
    sidebar = app.query_one(ConversationList)
    index = sidebar.get_option_index(option_id)
    line = index + (1 if index > 0 else 0)
    await pilot.click(ConversationList, offset=(4, 1 + line))
    await pilot.pause()


async def chat(pilot, app: BedrockApp, text: str) -> str:
    await pilot.press(*text, "enter")
    await app.workers.wait_for_complete()
    await pilot.pause()
    return app.conversation.id


async def test_sidebar_lists_conversations_newest_first_with_summary_titles(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "tell me about bees")
        await pilot.press(*"/new", "enter")
        await chat(pilot, app, "tell me about wasps")

        assert rows(app) == ["Re tell me about wasps", "Re tell me about bees"]
        assert open_row(app) == "Re tell me about wasps"


async def test_clicking_a_conversation_opens_it(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        bees = await chat(pilot, app, "tell me about bees")
        await pilot.press(*"/new", "enter")
        await chat(pilot, app, "tell me about wasps")

        await click_row(pilot, app, bees)

        assert app.conversation.id == bees
        assert str(app.query_one(UserMessage).content) == "tell me about bees"
        assert open_row(app) == "Re tell me about bees"
        assert app.focused is app.query_one("#prompt")  # ready to type straight away
        # Opening it made it the most recent, so it moved to the top.
        assert rows(app)[0] == "Re tell me about bees"


async def test_clicking_new_conversation_starts_one_and_keeps_the_old(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        bees = await chat(pilot, app, "tell me about bees")

        await click_row(pilot, app, NEW_CONVERSATION)

        assert app.conversation.id != bees
        assert not app.query(UserMessage)
        assert open_row(app) is None  # nothing saved yet for the blank one
        assert rows(app) == ["Re tell me about bees"]


async def test_a_title_is_only_requested_after_the_reply_finishes(fake_backend, store):
    fake_backend.fragment_delay = 0.05
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await pilot.press(*"slow question", "enter")
        await pilot.pause()
        assert app._streaming
        assert fake_backend.title_requests == []  # the employee's answer comes first
        assert fake_backend.topic_checks == []

        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(fake_backend.title_requests) == 1
        assert ConversationStore(store.directory).latest().generated_title == "Re slow question"


async def test_only_open_conversation_gets_a_title_at_startup(store):
    """Unopened transcripts are never loaded for title backfill."""
    for question in ["first old question", "second old question"]:
        conversation = Conversation(model="test-model")
        conversation.add_user(question)
        conversation.add_assistant("an answer")
        store.save(conversation)
    before = [(s.id, s.updated_at) for s in store.list()]

    backend = FakeBackend()
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert rows(app) == ["Re second old question", "Untitled"]

    after = ConversationStore(store.directory).list()
    assert [(s.id, s.updated_at) for s in after] == before  # same order, same "last used"
    assert len(backend.title_requests) == 1


async def test_each_conversation_is_only_titled_once(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "first question")
        await chat(pilot, app, "follow-up question")
    assert len(fake_backend.title_requests) == 1

    again = FakeBackend()
    second = make_app(again, ConversationStore(store.directory))
    async with second.run_test() as pilot:
        await pilot.pause()
        await second.workers.wait_for_complete()
    assert again.title_requests == []  # already titled on disk


async def test_without_a_server_conversations_stay_untitled_and_are_retried_later(store):
    offline = FakeBackend()
    offline.fail_titles = True
    app = make_app(offline, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "what is our VPN policy?")
        assert rows(app) == ["Untitled"]  # and no crash

    online = FakeBackend()
    second = make_app(online, ConversationStore(store.directory))
    async with second.run_test() as pilot:
        await pilot.pause()
        await second.workers.wait_for_complete()
        await pilot.pause()
        assert rows(second) == ["Re what is our VPN policy"]


async def test_no_name_until_there_is_a_concrete_topic(store):
    """The feature request: "hi" stays Untitled; a real topic later gets a name."""
    backend = FakeBackend(title_reply="NONE")
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "hi")
        assert rows(app) == ["Untitled"]
        assert len(backend.title_requests) == 1

    # Relaunching does not ask about the same two messages again.
    relaunched = FakeBackend(title_reply="NONE")
    second = make_app(relaunched, ConversationStore(store.directory))
    async with second.run_test() as pilot:
        await pilot.pause()
        await second.workers.wait_for_complete()
        assert relaunched.title_requests == []
        assert rows(second) == ["Untitled"]

        # A topic turns up: the model is asked again, with the new message in view.
        relaunched.title_reply = "Python Import Errors"
        await chat(pilot, second, "I need to fix a Python import error")
        assert len(relaunched.title_requests) == 1
        assert "Python import error" in relaunched.title_requests[0][1][1]["content"]
        assert rows(second) == ["Python Import Errors"]


async def test_a_vague_title_is_treated_as_no_topic(store):
    backend = FakeBackend(title_reply="Unclear Conversation Topic")
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "1")
        assert rows(app) == ["Untitled"]
    assert ConversationStore(store.directory).list()[0].generated_title is None


async def test_history_shows_the_first_question_for_untitled_conversations(store):
    app = make_app(FakeBackend(title_reply="NONE"), store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "hi")
        await pilot.press(*"/history", "enter")
        await pilot.pause()
        listing = app.query(SystemNotice).last().source
    assert "Untitled" in listing and "“hi”" in listing


async def test_a_title_arriving_for_a_forgotten_conversation_does_not_bring_it_back(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        doomed = await chat(pilot, app, "something private")
        await pilot.press(*"/forget", "enter")
        await pilot.pause()

        app.apply_title(doomed, "Late Title", 2)  # as if the model had been slow

        assert ConversationStore(store.directory).list() == []
        assert rows(app) == []


async def test_ctrl_b_hides_and_shows_the_sidebar(fake_backend, store):
    app = make_app(fake_backend, store)
    async with app.run_test() as pilot:
        sidebar = app.query_one(ConversationList)
        assert sidebar.display

        await pilot.press("ctrl+b")
        assert not sidebar.display
        assert app.focused is app.query_one("#prompt")

        await pilot.press("ctrl+b")
        assert sidebar.display


async def test_no_sidebar_and_no_titles_when_history_is_switched_off(fake_backend, tmp_path):
    settings = Settings(model="test-model", save_history=False, history_dir=tmp_path / "nope")
    app = BedrockApp(settings=settings, backend=fake_backend)
    async with app.run_test() as pilot:
        assert not app.query(ConversationList)
        await chat(pilot, app, "a private question")
        await pilot.press("ctrl+b")  # harmless
        await pilot.pause()
    assert fake_backend.title_requests == []
    assert fake_backend.topic_checks == []


async def test_long_titles_stay_on_one_line_and_end_with_an_ellipsis(store):
    conversation = Conversation(model="test-model", generated_title="Quarterly Infrastructure Budget Planning")
    conversation.add_user("budget question")
    conversation.add_assistant("an answer")
    store.save(conversation)

    app = make_app(FakeBackend(), store)
    async with app.run_test(size=(100, 20)) as pilot:
        await pilot.pause()
        sidebar = app.query_one(ConversationList)
        # Rendered lines start below the padding: "+ New conversation", divider, title.
        drawn = [sidebar.render_line(y).text.rstrip() for y in range(0, 5)]

    assert drawn[2].endswith("…"), drawn
    assert "Quarterly Infrastructure" in drawn[2]
    assert not any("Planning" in line for line in drawn)  # not wrapped onto a second line


async def test_a_conversation_that_is_just_a_number_stays_untitled_without_asking(store):
    """The "1" chat from the bug report: nothing to name, and no model requests."""
    backend = FakeBackend()
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "1")
        assert rows(app) == ["Untitled"]
    assert backend.topic_checks == [] and backend.title_requests == []


async def test_small_talk_stays_untitled_until_a_topic_turns_up(store):
    backend = FakeBackend(topic_reply="NO")
    app = make_app(backend, store)
    async with app.run_test() as pilot:
        await chat(pilot, app, "hey how are you")
        assert rows(app) == ["Untitled"]
        assert backend.title_requests == []  # a NO stops before asking for a title

        backend.topic_reply = "YES"
        backend.title_reply = "Sea Animals"
        await chat(pilot, app, "sea animals")
        assert rows(app) == ["Sea Animals"]
        assert len(backend.topic_checks) == 2
