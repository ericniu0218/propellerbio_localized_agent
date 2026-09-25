"""Starting with a model that fits the machine, and showing sizes in /model."""

from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import PROFILES, Settings
from bedrock.ui.widgets import SystemNotice

from tests.conftest import FakeBackend

GB = 1_000_000_000


def recording_app(backend: FakeBackend, **settings) -> tuple[BedrockApp, list[str]]:
    app = BedrockApp(settings=Settings(**settings), backend=backend)
    toasts: list[str] = []
    original = app.notify

    def notify(message, *args, **kwargs):
        toasts.append(str(message))
        return original(message, *args, **kwargs)

    app.notify = notify
    return app, toasts


async def test_the_largest_model_that_fits_the_laptop_is_picked():
    backend = FakeBackend(
        models=["llama3.1:8b", "llama3.2:1b", "llama3.2:3b"],
        model_sizes={"llama3.1:8b": int(4.9 * GB), "llama3.2:1b": int(1.3 * GB), "llama3.2:3b": int(2.0 * GB)},
    )
    app, toasts = recording_app(backend)
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await pilot.pause()
        assert app.model_name == "llama3.2:3b"
        assert any("`llama3.2:3b`" in toast and "laptop" in toast for toast in toasts)


async def test_without_sizes_the_first_model_is_kept_and_the_reason_given():
    app, toasts = recording_app(FakeBackend())
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await pilot.pause()
        assert app.model_name == "test-model"
        assert any("does not report" in toast for toast in toasts)


async def test_a_configured_model_is_used_as_is_without_a_notice():
    backend = FakeBackend(models=["big"], model_sizes={"big": 40 * GB})
    app, toasts = recording_app(backend, model="big")
    async with app.run_test() as pilot:
        await wait_for_app_workers(app)
        await pilot.pause()
        assert app.model_name == "big"
        assert not any("Picked" in toast for toast in toasts)


async def test_model_list_shows_sizes_and_marks_models_too_large_for_the_profile():
    backend = FakeBackend(models=["small", "huge", "unsized"], model_sizes={"small": 2 * GB, "huge": 40 * GB})
    app, _ = recording_app(backend, model="small")
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()
        text = app.query_one(SystemNotice).source
        assert "`small` (2.0 GB)" in text
        assert "`huge` (40.0 GB, too large for the laptop profile)" in text
        assert "`unsized`\n" in text + "\n"
        assert "You can still choose one" in text

        # An explicit choice is still allowed.
        await pilot.press(*"/model huge", "enter")
        await pilot.pause()
        assert app.model_name == "huge"


async def test_the_server_profile_marks_nothing_too_large():
    backend = FakeBackend(models=["huge"], model_sizes={"huge": 40 * GB})
    app, _ = recording_app(backend, model="huge", profile=PROFILES["server"])
    async with app.run_test() as pilot:
        await pilot.press(*"/model", "enter")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "too large" not in app.query_one(SystemNotice).source
