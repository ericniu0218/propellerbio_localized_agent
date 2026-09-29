"""Server model calls go through bedrock's Scheduler (bedrock.server.models).

The hub gives each model server one queue, the client turns the agents'
``ModelClient`` calls into scheduled calls, and - the regression this exists
for - token counts reach the per-task budget. The agent pack's old HTTP client
reported 0 tokens when streaming, so no budget was ever enforced in real mode.
"""

from __future__ import annotations

import pytest

from bedrock.inference import Priority, Usage
from bedrock.inference.openai_compatible import OpenAICompatibleBackend
from bedrock.server.agents.base import BudgetExceededError
from bedrock.server.agents.runner import Runner
from bedrock.server.config import ModelConfig, ModelsConfig, Settings, load_models_config
from bedrock.server.models.client import ChatMessage
from bedrock.server.models.hub import ModelHub, get_model_hub
from bedrock.server.models.mock_backend import MockBackend
from bedrock.server.models.registry import get_model_registry
from bedrock.server.models.scheduled import SchedulerModelClient
from tests.conftest import FakeBackend


def _models(**endpoints) -> ModelsConfig:
    def role(name, endpoint, **extra):
        return ModelConfig(role=name, model_name=f"{name}-model", endpoint=endpoint, **extra)

    return ModelsConfig(
        director=role("director", endpoints.get("director", "http://gpu-a:8001/v1"), context_length=131072, max_parallel=2),
        specialist=role("specialist", endpoints.get("specialist", "http://gpu-a:8001/v1"), context_length=65536, max_parallel=6),
        runner=role("runner", endpoints.get("runner", "http://gpu-b:8003/v1"), context_length=32768),
        notes=role("notes", endpoints.get("notes", "http://gpu-b:8003/v1"), context_length=8192, max_parallel=1),
    )


def _settings(tmp_path, mode: str) -> Settings:
    return Settings(mode=mode, data_dir=tmp_path, database_url=f"sqlite:///{tmp_path / 'db.sqlite'}")


def test_roles_on_one_server_share_one_queue_and_its_limits(tmp_path):
    hub = ModelHub(_models(), _settings(tmp_path, "real"))
    director, specialist = hub.scheduler_for("director"), hub.scheduler_for("specialist")
    runner, notes = hub.scheduler_for("runner"), hub.scheduler_for("notes")

    assert director is specialist and runner is notes and director is not runner
    assert isinstance(director.backend, OpenAICompatibleBackend)
    assert director.profile.context_window == 65536   # the smaller window of the two
    assert director.profile.max_parallel == 6          # the larger allowance
    assert runner.profile.context_window == 8192 and runner.profile.max_parallel == 8
    assert director.profile.background == "concurrent"


def test_mock_mode_replaces_the_agents_server_but_never_the_notes_server(tmp_path):
    hub = ModelHub(_models(runner="http://gpu-b:8003/v1", notes="http://gpu-b:8003/v1"), _settings(tmp_path, "mock"))

    agents = {hub.scheduler_for(role) for role in ("director", "specialist", "runner")}
    assert len(agents) == 1 and isinstance(next(iter(agents)).backend, MockBackend)
    notes = hub.scheduler_for("notes")
    assert isinstance(notes.backend, OpenAICompatibleBackend)  # demo-recorder.sh: fixture agents, real notes
    assert notes.profile.context_window == 8192


def test_the_notes_window_follows_its_own_setting(monkeypatch):
    monkeypatch.setenv("MEETING_SUMMARY_CONTEXT", "32768")
    load_models_config.cache_clear()
    try:
        assert load_models_config().for_role("notes").context_length == 32768
    finally:
        load_models_config.cache_clear()


def test_agents_get_scheduled_clients():
    client = get_model_registry().client_for("runner")
    assert isinstance(client, SchedulerModelClient)
    assert get_model_hub().scheduler_for("runner") is get_model_hub().scheduler_for("director")


def _client(backend, *, thinking_switch: bool = True) -> tuple[SchedulerModelClient, FakeBackend]:
    from bedrock.config import PROFILES
    from bedrock.inference import Scheduler

    config = ModelConfig(role="specialist", model_name="test-model", endpoint="http://x/v1",
                         supports_thinking_mode=thinking_switch)
    return SchedulerModelClient(Scheduler(backend, PROFILES["server"]), config, priority=Priority.CHAT), backend


async def test_thinking_is_asked_for_only_where_the_model_has_the_switch():
    client, backend = _client(FakeBackend(reply="ok"))
    await client.complete([ChatMessage(role="user", content="hi")], max_tokens=16, thinking=True)
    await client.complete([ChatMessage(role="user", content="hi")], max_tokens=16, thinking=False)
    assert [o.reasoning for o in backend.request_options] == [True, False]
    assert backend.request_options[0].max_tokens == 16

    client, backend = _client(FakeBackend(reply="ok"), thinking_switch=False)
    await client.complete([ChatMessage(role="user", content="hi")], max_tokens=16, thinking=True)
    assert backend.request_options[0].reasoning is None


async def test_reported_usage_reaches_the_result_and_is_estimated_when_absent():
    backend = FakeBackend(reply="four words right here")
    backend.usage = Usage(prompt_tokens=120, completion_tokens=4)
    client, _ = _client(backend)
    result = await client.complete([ChatMessage(role="user", content="hi")], max_tokens=16)
    assert (result.prompt_tokens, result.completion_tokens) == (120, 4)

    client, _ = _client(FakeBackend(reply="four words right here"))
    result = await client.complete([ChatMessage(role="user", content="hi there")], max_tokens=16)
    assert result.prompt_tokens > 0 and result.completion_tokens > 0


async def test_a_task_over_its_budget_is_stopped(session_id):
    """F3: with real token counts the per-task budget is enforced."""
    backend = FakeBackend(reply="done")
    backend.usage = Usage(prompt_tokens=9_000, completion_tokens=10)  # runner budget: 8,000 per task
    client, _ = _client(backend)
    runner = Runner(session_id=session_id, model_client=client)

    with pytest.raises(BudgetExceededError):
        await runner.call_model(runner.build_prompt("hello"), task_id="task_budget_check")


async def test_mock_mode_streams_through_the_scheduler():
    backend = MockBackend(["runner-model"])
    client, _ = _client(backend)
    chunks = [c async for c in client.stream([ChatMessage(role="user", content="Task: say hello")], max_tokens=64)]
    assert len(chunks) > 2 and chunks[-1].done and chunks[-1].final.completion_tokens > 0
