"""Prompt placement and search threading through the real terminal app."""

import threading
from dataclasses import replace

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.config.profile import PROFILES
from bedrock.context import ingest
from bedrock.context.adapters import Source
from tests.conftest import FakeBackend
from tests.workers import wait_for_app_workers


async def test_auto_inject_search_runs_off_ui_thread_and_is_user_text(tmp_path, monkeypatch):
    backend = FakeBackend()
    app = BedrockApp(Settings(model="test-model", context_dir=str(tmp_path / "context")), backend=backend)
    ingest(Source.from_text("sample ph reading 6.8", name="assay.txt"), store=app.context_store)
    threads = []
    search = app.context_store.search

    def observe_search(*args, **kwargs):
        threads.append(threading.current_thread())
        return search(*args, **kwargs)

    monkeypatch.setattr(app.context_store, "search", observe_search)
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"what was the sample ph reading?", "enter")
        await wait_for_app_workers(app)
    assert threads and all(thread is not threading.main_thread() for thread in threads)
    messages = backend.requests[-1][1]
    assert "sample ph reading" in messages[-1]["content"]
    assert "sample ph reading" not in messages[0]["content"]


async def test_zero_passage_profile_does_not_search(tmp_path, monkeypatch):
    backend = FakeBackend()
    profile = replace(PROFILES["laptop"], auto_inject_passages=0)
    app = BedrockApp(Settings(model="test-model", profile=profile, context_dir=str(tmp_path / "context")), backend=backend)
    monkeypatch.setattr(app.context_store, "search", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("auto-inject searched with limit zero")))
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(*"what was the sample ph reading?", "enter")
        await wait_for_app_workers(app)
    assert backend.requests[-1][1][-1]["content"] == "what was the sample ph reading?"
