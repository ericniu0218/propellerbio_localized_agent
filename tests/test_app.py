"""Drive the real Textual app headlessly with a fake backend."""

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.ui.widgets import AssistantMessage, StatusBar, SystemNotice, UserMessage

from tests.conftest import FakeBackend


def make_app(backend: FakeBackend, model: str | None = None) -> BedrockApp:
    return BedrockApp(settings=Settings(model=model), backend=backend)


async def test_send_message_streams_reply(fake_backend: FakeBackend):
    app = make_app(fake_backend)
    async with app.run_test() as pilot:
        await pilot.pause()  # let on_mount workers pick the default model
        assert app.model_name == "test-model"
        assert app.query_one(StatusBar).connected is True

        await pilot.press(*"hi there", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert str(app.query_one(UserMessage).content) == "hi there"
        reply = app.query_one(AssistantMessage)
        assert "Hello from the fake model." in reply.source
        # The conversation model carries the full history for the next turn.
        assert [m.role.value for m in app.conversation.messages] == ["user", "assistant"]
        assert app.conversation.messages[1].content.strip() == "Hello from the fake model."
        # Backend received exactly what the conversation model produced.
        assert fake_backend.requests[0][0] == "test-model"
        assert fake_backend.requests[0][1] == [{"role": "user", "content": "hi there"}]


async def test_model_command_lists_and_switches(fake_backend: FakeBackend):
    app = make_app(fake_backend, model="test-model")
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await pilot.pause()
        notice = app.query_one(SystemNotice)
        assert "test-model" in notice.source and "other-model" in notice.source

        await pilot.press(*"/model other-model", "enter")
        await pilot.pause()
        assert app.model_name == "other-model"
        assert "other-model" in str(app.query_one("#status-model").content)


async def test_clear_and_help(fake_backend: FakeBackend):
    app = make_app(fake_backend, model="test-model")
    async with app.run_test() as pilot:
        await pilot.press(*"hello", "enter")
        await app.workers.wait_for_complete()
        await pilot.press(*"/clear", "enter")
        await pilot.pause()
        assert not app.query(UserMessage)
        assert app.conversation.messages == []

        await pilot.press(*"/help", "enter")
        await pilot.pause()
        assert "/model" in app.query_one(SystemNotice).source


async def test_disconnected_status(fake_backend: FakeBackend):
    fake_backend.connected = False
    app = make_app(fake_backend, model="test-model")
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(StatusBar).connected is False


async def test_exit_command_closes_backend(fake_backend: FakeBackend):
    app = make_app(fake_backend, model="test-model")
    async with app.run_test() as pilot:
        await pilot.press(*"/exit", "enter")
        await pilot.pause()
    assert app.return_value is None
    assert fake_backend.closed is True


async def test_workspace_command_is_native_and_updates_status(fake_backend, tmp_path):
    app = BedrockApp(
        settings=Settings(
            model="test-model",
            user_id="alice",
            display_name="Alice",
            context_dir=str(tmp_path / "context"),
            workspace_dir=str(tmp_path / "workspaces"),
            history_dir=tmp_path / "history",
        ),
        backend=fake_backend,
    )
    async with app.run_test() as pilot:
        await pilot.press(*"/workspace create Research", "enter")
        await app.workers.wait_for_complete()
        await pilot.pause()

        assert app.workspace_session.store.list_workspaces()[0].name == "Research"
        assert "space: Research" in str(app.query_one("#status-workspace").content)
        assert any("Created and selected" in notice.source for notice in app.query(SystemNotice))

        await pilot.press(*"/exit", "enter")
        await pilot.pause()
