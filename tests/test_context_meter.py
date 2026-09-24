"""The status bar's context counter: local numbers, shown locally."""

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import PROFILES, Settings
from bedrock.inference import Priority, Usage
from bedrock.ui.widgets import StatusBar, _tokens

from tests.conftest import FakeBackend


def meter(app: BedrockApp) -> str:
    return str(app.query_one("#status-context").content)


async def answer_a_chat(app: BedrockApp) -> None:
    await app.scheduler.complete("test-model", [{"role": "user", "content": "hello " * 400}], priority=Priority.CHAT)


async def test_the_counter_is_blank_until_a_reply_reports_its_size():
    app = BedrockApp(settings=Settings(model="test-model"), backend=FakeBackend())
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await pilot.pause()
        assert meter(app) == ""


async def test_the_counter_shows_the_last_prompt_against_the_profile_window():
    backend = FakeBackend()
    backend.usage = Usage(prompt_tokens=3174, completion_tokens=40, cached_prompt_tokens=3000)
    app = BedrockApp(settings=Settings(model="test-model"), backend=backend)
    async with app.run_test() as pilot:
        await answer_a_chat(app)
        app.check_connection()
        await wait_for_app_workers(app)
        await pilot.pause()
        assert meter(app) == "context 3.1k/8k"
        assert not app.query_one("#status-context").has_class("full")


async def test_a_nearly_full_window_is_highlighted():
    backend = FakeBackend()
    backend.usage = Usage(prompt_tokens=7600)
    app = BedrockApp(settings=Settings(model="test-model", profile=PROFILES["laptop"]), backend=backend)
    async with app.run_test() as pilot:
        await answer_a_chat(app)
        app.check_connection()
        await wait_for_app_workers(app)
        await pilot.pause()
        assert meter(app) == "context 7.4k/8k"
        assert app.query_one("#status-context").has_class("full")


def test_token_counts_are_written_short():
    assert _tokens(850) == "850"
    assert _tokens(8192) == "8k"
    assert _tokens(3174) == "3.1k"
    assert _tokens(32768) == "32k"


async def test_the_status_bar_still_shows_model_and_connection():
    app = BedrockApp(settings=Settings(model="test-model"), backend=FakeBackend())
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await pilot.pause()
        bar = app.query_one(StatusBar)
        assert bar.connected is True
        assert "test-model" in str(app.query_one("#status-model").content)
