"""Drive the real Textual app headlessly with a fake backend."""

from tests.workers import wait_for_app_workers

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
        await wait_for_app_workers(app)
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


async def test_relevant_stored_context_is_added_to_the_outgoing_message(fake_backend: FakeBackend, tmp_path):
    """Closes the same /digest-never-reaches-the-model gap fixed for
    AgentPackApp - compressing a file used to do nothing for the next
    question asked here, unless the search result was pasted in by hand."""
    app = BedrockApp(
        settings=Settings(model="test-model", context_dir=str(tmp_path / "context")),
        backend=fake_backend,
    )
    (tmp_path / "assay.csv").write_text(
        "\n".join(["sample,ph,passed"] + [f"s{i},{6.0 + i * 0.1:.1f},true" for i in range(10)])
    )
    from bedrock.context import ingest

    ingest(tmp_path / "assay.csv", store=app.context_store)

    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"what was the ph in the assay results?", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

        model_name, api_messages = fake_backend.requests[-1]
        system_text = "\n".join(m["content"] for m in api_messages if m["role"] == "system")
        assert "Relevant stored context" in system_text
        assert "assay.csv" in system_text

        # The user's own bubble and the saved conversation both stay exactly
        # what was typed - the injected context lives only in the one-turn
        # note, never in what's shown or saved.
        assert str(app.query_one(UserMessage).content) == "what was the ph in the assay results?"
        assert app.conversation.messages[0].content == "what was the ph in the assay results?"
        notices = [n.source for n in app.query(SystemNotice)]
        assert any("Added stored context from" in n and "assay.csv" in n for n in notices)


async def test_unrelated_message_has_no_injected_context(fake_backend: FakeBackend, tmp_path):
    app = BedrockApp(
        settings=Settings(model="test-model", context_dir=str(tmp_path / "context")),
        backend=fake_backend,
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"good morning", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()

        model_name, api_messages = fake_backend.requests[-1]
        assert api_messages == [{"role": "user", "content": "good morning"}]


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
        await wait_for_app_workers(app)
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
        await wait_for_app_workers(app)
        await pilot.pause()

        assert app.workspace_session.store.list_workspaces()[0].name == "Research"
        assert "space: Research" in str(app.query_one("#status-workspace").content)
        assert any("Created and selected" in notice.source for notice in app.query(SystemNotice))

        await pilot.press(*"/exit", "enter")
        await pilot.pause()


async def test_the_app_starts_against_a_server_with_no_models_pulled(tmp_path):
    """A fresh Ollama answers `/models` with `"data": null`. The app used to
    crash on launch there; it must start, say it has no model, and tell the
    employee how to get one."""
    import httpx

    from bedrock.inference.openai_compatible import OpenAICompatibleBackend

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "list", "data": None})

    backend = OpenAICompatibleBackend("http://fake/v1", transport=httpx.MockTransport(handler))
    app = BedrockApp(settings=Settings(context_dir=str(tmp_path / "context")), backend=backend)
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await pilot.pause()
        assert app.model_name is None
        assert app.query_one(StatusBar).model_name == "no model"

        await pilot.press(*"/model", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "The server reports no models" in str(app.query(SystemNotice)[-1].source)

        await pilot.press(*"hello", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "No model selected" in str(app.query(SystemNotice)[-1].source)
    assert app._exception is None
