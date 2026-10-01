"""The literature review in agent-pack mode, end to end in one process.

What runs is the real thing at every layer but the GPU:

    AgentPackApp  `/review ...`                      (the Textual app, headless)
      -> bedrock.litreview pipeline, SchedulerClient (the terminal's own scheduler)
      -> OpenAICompatibleBackend                     (real HTTP client, ASGI transport)
      -> bedrock.server  GET /v1/models, POST /v1/chat/completions
      -> the server's model hub Scheduler at USER_JOB
      -> Upstream: a recording stand-in for vLLM

So these tests prove the contract between the two sides - named model per job,
temperature 0, the seed, the JSON schema, the reported window - over the wire
the two really use, not two halves tested against each other's assumptions.

The same fixture covers `bedrock-litreview ask` with `BEDROCK_BACKEND=agentpack`
and no config file (P0.4), and the server's refusals (P0.3).
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from bedrock.inference import InferenceError, ModelInfo, Usage
from bedrock.inference.openai_compatible import OpenAICompatibleBackend
from bedrock.server.api import routes_inference
from bedrock.server.config import get_settings, load_models_config
from bedrock.server.main import create_app
from bedrock.server.models.hub import ModelHub
from tests.conftest import make_services
from tests.litreview.fixtures import QUESTION, corpus, responder

AGENT_SERVER = "http://127.0.0.1:8765"


class Upstream:
    """The model server behind the agent server: records every call it gets."""

    def __init__(self, models: list[str], *, window: int | None = None, fail_after: int | None = None,
                 down: bool = False) -> None:
        self.models = models
        self.window = window
        self.fail_after = fail_after
        self.down = down
        self.calls: list[tuple[str, list[dict], object]] = []

    async def check_connection(self) -> bool:
        return not self.down

    async def list_models(self) -> list[ModelInfo]:
        if self.down:
            raise InferenceError("connection refused", unavailable=True)
        return [ModelInfo(id=m, context_length=self.window) for m in self.models]

    async def stream_chat(self, model, messages, options=None, on_usage=None):
        self.calls.append((model, messages, options))
        reply = responder(messages[-1]["content"], options.json_schema if options else None)
        if self.fail_after is not None:
            yield reply[: self.fail_after]
            raise InferenceError("the GPU went away")
        yield reply
        if on_usage is not None:
            on_usage(Usage(prompt_tokens=max(1, len(messages[-1]["content"]) // 4), completion_tokens=len(reply) // 4))

    async def close(self) -> None:
        pass


@pytest.fixture
def agent_server(monkeypatch):
    """The real server app, its hub's model servers replaced by one Upstream."""
    models = load_models_config()
    upstream = Upstream([models.for_role(r).model_name for r in ("runner", "specialist", "director")])
    hub = ModelHub(models, get_settings())
    monkeypatch.setattr(hub, "_backend", lambda key, sharing: upstream)
    monkeypatch.setattr(routes_inference, "get_model_hub", lambda: hub)
    routes_inference._upstream.clear()
    app = create_app()

    def backend_for(settings):
        # What bedrock's factory would build, speaking to the in-process app.
        return OpenAICompatibleBackend(settings.api_base, api_key=settings.api_key,
                                       timeout=settings.request_timeout,
                                       transport=httpx.ASGITransport(app=app))

    monkeypatch.setattr("bedrock.inference.create_backend", backend_for)
    yield app, upstream
    routes_inference._upstream.clear()


@pytest.fixture
def review_env(tmp_path, monkeypatch):
    """No bedrock config file at all, agent-pack mode from the environment only."""
    from bedrock.config import settings as bedrock_settings

    monkeypatch.setattr(bedrock_settings, "DEFAULT_CONFIG_PATH", tmp_path / "no-config.toml")
    for name in [k for k in __import__("os").environ if k.startswith("BEDROCK_") and not k.startswith("BEDROCK_SERVER")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("BEDROCK_BACKEND", "agentpack")
    monkeypatch.setenv("AGENT_PACK_API_BASE", AGENT_SERVER)
    monkeypatch.setenv("BEDROCK_LITREVIEW_DIR", str(tmp_path / "litreview"))
    corpus_file = tmp_path / "papers.jsonl"
    from bedrock.litreview.sources.corpus import write_papers

    write_papers(corpus(), corpus_file)
    return corpus_file


def _roles(upstream) -> dict[str, str]:
    models = load_models_config()
    return {models.for_role(r).model_name: r for r in ("runner", "specialist", "director")}


# --- P0.1 / P0.2: /review through the real agent-pack application -------------------------------


async def test_review_runs_in_agent_pack_mode_through_the_agent_servers_inference_api(agent_server, review_env):
    from bedrock.agentpack_app import AgentPackApp
    from bedrock.config import load_settings
    from bedrock.ui.commands import parse_command
    from bedrock.ui.widgets import SystemNotice
    from tests.test_agentpack_app import FakeAgentPackClient

    _, upstream = agent_server
    settings = load_settings()
    assert settings.backend == "agentpack" and settings.profile.name == "laptop"  # the P0.4 default
    app = AgentPackApp(settings=settings, client=FakeAgentPackClient(), services=make_services())
    async with app.run_test() as pilot:
        await pilot.pause()
        await app.run_command(parse_command(f"/review {QUESTION} --corpus {review_env}"))
        # Not wait_for_app_workers: agent-pack mode's live-event listener
        # never finishes. The review's own flag says when it has.
        for _ in range(600):
            await pilot.pause(0.05)
            if not app._review_running:
                break
        await pilot.pause()
        text = "\n".join(str(n.source) for n in app.query(SystemNotice))

    assert "not available in agent-pack mode" not in text
    assert "Literature review finished" in text, text
    assert "Integrity: passed" in text
    assert upstream.calls, "no review call reached the model behind the agent server"
    roles = _roles(upstream)
    used = {roles[model] for model, _, _ in upstream.calls}
    assert used == {"runner", "specialist"}  # relevance on the small tier, the rest on the medium one
    for model, messages, options in upstream.calls:
        assert [m["role"] for m in messages] == ["user"]
        assert options.temperature == 0.0 and options.seed == 0
        assert options.json_schema is not None
    # thinking: false in the agentpack model set reaches every role: a model
    # that does not think ignores it, one that does must get it
    assert all(o.reasoning is False for _, _, o in upstream.calls)
    # The review's connection was the agent server's, opened for the review:
    # nothing reached for a model server on this machine.
    assert app._own_review_scheduler is not None
    assert app._own_review_scheduler.backend.base_url == f"{AGENT_SERVER}/v1"


async def test_quitting_closes_the_review_connection(agent_server, review_env):
    from bedrock.agentpack_app import AgentPackApp
    from bedrock.config import load_settings
    from tests.test_agentpack_app import FakeAgentPackClient

    app = AgentPackApp(settings=load_settings(), client=FakeAgentPackClient(), services=make_services())
    async with app.run_test() as pilot:
        await pilot.pause()
        review = await asyncio.to_thread(app._review_settings, None)
        scheduler = app.review_scheduler(review)
        assert scheduler.backend.base_url == f"{AGENT_SERVER}/v1"
        await app.action_quit()
    assert app._review_cancel.is_set() and app._own_review_scheduler is None


# --- P0.4: the shell command, default install ----------------------------------------------------


def test_litreview_ask_in_agent_pack_mode_with_no_config_uses_the_agent_server(agent_server, review_env, capsys):
    from bedrock.litreview import cli
    from bedrock.litreview.config import load_settings

    settings = load_settings()
    assert settings.base_url == f"{AGENT_SERVER}/v1" and settings.endpoint_source == "agentpack"
    assert settings.model_set == "agentpack" and settings.context_window_source == "unset"

    assert cli.main(["ask", QUESTION, "--corpus", str(review_env)]) == 0, capsys.readouterr().err
    _, upstream = agent_server
    assert upstream.calls and {o.seed for _, _, o in upstream.calls} == {0}
    err = capsys.readouterr().err
    assert "context window of at least" not in err
    run = json.loads((next((review_env.parent / "litreview").rglob("run.json"))).read_text())
    # the run records the weights behind each role, not just the alias
    assert "specialist=" in run["model_versions"]["extract"]


def test_an_agent_server_that_serves_too_small_a_window_is_refused_before_any_call(agent_server, review_env, capsys):
    from bedrock.litreview import cli

    _, upstream = agent_server
    upstream.window = 8192
    routes_inference._upstream.clear()
    assert cli.main(["ask", QUESTION, "--corpus", str(review_env)]) == 1
    err = capsys.readouterr().err
    assert "8,192-token context window" in err and "'runner'" in err
    assert "models.yaml" in err  # the fix is on the agent server, not Ollama
    assert upstream.calls == []


def test_an_absent_agent_server_is_reported_as_such(review_env, monkeypatch, capsys):
    from bedrock.litreview import cli

    def refused(settings):
        def handler(request):
            raise httpx.ConnectError("connection refused", request=request)
        return OpenAICompatibleBackend(settings.api_base, transport=httpx.MockTransport(handler))

    monkeypatch.setattr("bedrock.inference.create_backend", refused)
    assert cli.main(["ask", QUESTION, "--corpus", str(review_env)]) == 1
    err = capsys.readouterr().err
    assert f"agent server's inference API at {AGENT_SERVER}/v1 is not answering" in err
    assert "localhost:11434" not in err


# --- P0.3: the inference API's own contract --------------------------------------------------------


def _client(app, **headers) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=AGENT_SERVER, headers=headers)


async def test_v1_models_lists_roles_with_their_window_weights_and_capabilities(agent_server):
    app, _ = agent_server
    async with _client(app) as c:
        body = (await c.get("/v1/models")).json()
    by_id = {m["id"]: m for m in body["data"]}
    assert set(by_id) == {"runner", "specialist", "director"}
    specialist = by_id["specialist"]
    assert specialist["root"] == load_models_config().specialist.model_name
    assert specialist["max_model_len"] == load_models_config().specialist.context_length
    assert specialist["capabilities"] == {"json_schema": True, "seed": True, "reasoning_switch": True}


async def test_a_smaller_served_window_wins_over_the_configured_one(agent_server):
    app, upstream = agent_server
    upstream.window = 16384
    async with _client(app) as c:
        by_id = {m["id"]: m for m in (await c.get("/v1/models")).json()["data"]}
    assert by_id["specialist"]["max_model_len"] == 16384


async def test_unknown_parameters_are_refused_by_name_never_dropped(agent_server):
    app, upstream = agent_server
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "runner", "top_p": 0.9,
                                                       "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and "top_p" in r.json()["detail"]
    assert upstream.calls == []


@pytest.mark.parametrize("body, words", [
    ({"chat_template_kwargs": {"enable_thinking": True}}, "no thinking mode"),
    ({"response_format": {"type": "json_object"}}, "response_format"),
    ({"reasoning_effort": "high"}, "reasoning_effort"),
    ({"max_tokens": 0}, "max_tokens"),
    ({"temperature": 5}, "temperature"),
])
async def test_settings_the_model_cannot_honour_are_refused(agent_server, body, words):
    app, upstream = agent_server
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "runner", **body,
                                                       "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400 and words in r.json()["detail"]
    assert upstream.calls == []


async def test_a_schema_or_seed_the_engine_ignores_is_refused(agent_server, monkeypatch):
    app, upstream = agent_server
    runner = load_models_config().runner.model_copy(update={"supports_json_schema": False, "supports_seed": False})
    monkeypatch.setattr(routes_inference, "_roles", lambda: {"runner": runner})
    schema = {"type": "json_schema", "json_schema": {"name": "r", "schema": {"type": "object"}}}
    async with _client(app) as c:
        a = await c.post("/v1/chat/completions", json={"model": "runner", "response_format": schema,
                                                       "messages": [{"role": "user", "content": "hi"}]})
        b = await c.post("/v1/chat/completions", json={"model": "runner", "seed": 0,
                                                       "messages": [{"role": "user", "content": "hi"}]})
    assert a.status_code == 400 and "JSON schema" in a.json()["detail"]
    assert b.status_code == 400 and "seeds" in b.json()["detail"]
    assert upstream.calls == []


async def test_a_prompt_too_large_for_the_served_window_is_refused_not_truncated(agent_server):
    app, upstream = agent_server
    upstream.window = 2048
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "runner", "max_tokens": 1024,
                                                       "messages": [{"role": "user", "content": "x " * 6000}]})
    assert r.status_code == 400 and "2,048-token window" in r.json()["detail"]
    assert upstream.calls == []


async def test_structured_settings_reach_the_model_server_unchanged(agent_server):
    app, upstream = agent_server
    schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "specialist", "temperature": 0, "seed": 7, "max_tokens": 64,
            "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}},
            "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    model, _, options = upstream.calls[-1]
    assert model == load_models_config().specialist.model_name
    assert (options.temperature, options.seed, options.max_tokens) == (0, 7, 64)
    assert options.json_schema == schema and options.reasoning is False
    assert r.json()["model"] == "specialist" and r.json()["usage"]["prompt_tokens"] >= 1


async def test_a_model_server_that_fails_mid_reply_is_an_error_not_half_a_reply(agent_server):
    app, upstream = agent_server
    upstream.fail_after = 3
    backend = OpenAICompatibleBackend(f"{AGENT_SERVER}/v1", transport=httpx.ASGITransport(app=app))
    with pytest.raises(InferenceError, match="partway"):
        async for _ in backend.stream_chat("runner", [{"role": "user", "content": "hi"}]):
            pass


async def test_a_role_whose_model_server_is_down_is_unavailable_not_listed(agent_server):
    app, upstream = agent_server
    upstream.down = True
    async with _client(app) as c:
        body = (await c.get("/v1/models")).json()
        r = await c.post("/v1/chat/completions", json={"model": "runner",
                                                       "messages": [{"role": "user", "content": "hi"}]})
    assert body["data"] == [] and "not answering" in body["unavailable"]["runner"]
    assert r.status_code == 503


# --- access -------------------------------------------------------------------------------------------


async def test_without_a_token_only_this_machine_is_served(agent_server):
    app, _ = agent_server
    remote = httpx.ASGITransport(app=app, client=("10.1.2.3", 5000))
    async with httpx.AsyncClient(transport=remote, base_url=AGENT_SERVER) as c:
        r = await c.get("/v1/models")
    assert r.status_code == 403 and "BEDROCK_SERVER_INFERENCE_TOKEN" in r.json()["detail"]


async def test_a_real_mode_server_without_a_token_refuses_even_loopback(agent_server, monkeypatch):
    app, upstream = agent_server
    settings = get_settings().model_copy(update={"mode": "real", "inference_token": None})
    monkeypatch.setattr(routes_inference, "get_settings", lambda: settings)
    async with _client(app) as c:  # ASGITransport's client is 127.0.0.1
        r = await c.post("/v1/chat/completions", json={"model": "runner",
                                                       "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 403 and "needs a token in real mode" in r.json()["detail"]
    assert upstream.calls == []


async def test_with_a_token_every_caller_must_present_it(agent_server, monkeypatch):
    app, _ = agent_server
    settings = get_settings().model_copy(update={"inference_token": "s3cret"})
    monkeypatch.setattr(routes_inference, "get_settings", lambda: settings)
    async with _client(app) as c:
        missing = await c.get("/v1/models")
        wrong = await c.get("/v1/models", headers={"Authorization": "Bearer nope"})
        right = await c.get("/v1/models", headers={"Authorization": "Bearer s3cret"})
    assert missing.status_code == 401 and wrong.status_code == 401 and right.status_code == 200
    assert "s3cret" not in missing.text


async def test_the_api_can_be_switched_off(agent_server, monkeypatch):
    app, _ = agent_server
    settings = get_settings().model_copy(update={"inference_api": False})
    monkeypatch.setattr(routes_inference, "get_settings", lambda: settings)
    async with _client(app) as c:
        assert (await c.get("/v1/models")).status_code == 404


async def test_health_advertises_the_inference_api(agent_server):
    app, _ = agent_server
    async with _client(app) as c:
        assert "inference_api" in (await c.get("/health")).json()["features"]


def test_the_servers_own_research_tab_never_routes_through_itself(review_env):
    """One machine, one config: `backend = agentpack` must not send the
    server's own reviews back into its inference API."""
    from bedrock.litreview.config import load_settings

    assert load_settings().endpoint_source == "agentpack"
    served = load_settings(as_server=True, check_endpoint=False)
    assert served.endpoint_source == "api_base" and served.base_url == "http://localhost:11434/v1"


# --- P0.5: the doctor ------------------------------------------------------------------------------


@pytest.fixture
def health(monkeypatch, agent_server):
    """/health answered by the in-process server."""
    from bedrock.litreview import readiness

    app, _ = agent_server

    async def ask(url):
        async with _client(app) as c:
            r = await c.get("/health")
        return r.status_code, set(r.json().get("features") or [])

    monkeypatch.setattr(readiness, "_health", ask)
    return readiness


async def test_doctor_says_operational_when_everything_a_review_needs_is_there(health, review_env):
    checks = await health.review_readiness()
    lines = "\n".join(c.line() for c in checks)
    assert health.operational(checks), lines
    assert "[ok  ] Agent server: reachable, and offers the inference API" in lines
    assert "specialist=" in lines and "65,536 tokens" in lines
    assert "API key not set" in lines


async def test_doctor_tells_reachable_apart_from_operational_on_an_older_server(health, review_env, monkeypatch):
    async def old_server(url):
        return 200, {"context_blocks", "cancel"}

    monkeypatch.setattr(health, "_health", old_server)
    checks = await health.review_readiness()
    by = {c.name: c for c in checks}
    assert by["Agent server"].ok is True
    assert by["Inference API"].ok is False and "predates" in by["Inference API"].detail
    assert by["Literature review"].ok is False and not health.operational(checks)


async def test_doctor_reports_an_unreachable_agent_server(review_env, monkeypatch):
    from bedrock.litreview import readiness

    async def down(url):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(readiness, "_health", down)
    checks = await readiness.review_readiness()
    assert checks[1].name == "Agent server" and checks[1].ok is False
    assert "AGENT_PACK_API_BASE" in checks[1].detail


async def test_doctor_reports_a_refused_key_without_printing_it(health, review_env, monkeypatch):
    settings = get_settings().model_copy(update={"inference_token": "s3cret"})
    monkeypatch.setattr(routes_inference, "get_settings", lambda: settings)
    monkeypatch.setenv("BEDROCK_API_KEY", "wrong-key")
    checks = await health.review_readiness()
    by = {c.name: c for c in checks}
    assert by["Authentication"].ok is False and "BEDROCK_SERVER_INFERENCE_TOKEN" in by["Authentication"].detail
    assert "wrong-key" not in "\n".join(c.line() for c in checks) and "API key set" in checks[0].detail


async def test_doctor_reports_a_window_too_small_for_a_review(health, review_env, agent_server):
    _, upstream = agent_server
    upstream.window = 8192
    checks = await health.review_readiness()
    assert checks[-1].ok is False and "8,192-token context window" in checks[-1].detail


async def test_doctor_reports_a_model_set_the_server_does_not_offer(health, review_env, monkeypatch):
    monkeypatch.setenv("BEDROCK_LITREVIEW_MODELS", "office")
    checks = await health.review_readiness()
    assert checks[-1].ok is False and "does not have" in checks[-1].detail and "It offers: director" in checks[-1].detail


def test_first_run_banner_in_agent_pack_mode_now_checks_the_review(health, review_env, monkeypatch, agent_server):
    """P0.5: the agent-pack branch used to stop at "reachable" and never said a
    word about reviews."""
    from bedrock import doctor
    from bedrock.config import load_settings

    _, upstream = agent_server
    upstream.window = 8192

    async def reachable(url, timeout=3.0):
        return True

    monkeypatch.setattr(doctor, "_reachable", reachable)
    monkeypatch.setattr(doctor, "review_notes", doctor._review_notes)  # the real check, in-process server
    monkeypatch.setattr(doctor, "DEFAULT_CONFIG_PATH", review_env.parent / "no-config.toml")  # a first run
    issues = doctor.run_first_run_check(load_settings())
    text = "\n".join(issues)
    assert "Literature reviews (/review) cannot run yet" in text and "8,192-token" in text
    assert "OLLAMA_CONTEXT_LENGTH" not in text  # the local-Ollama advice does not apply to a remote model


async def test_every_call_is_audited_as_metadata_only(agent_server):
    from bedrock.server.workspace.routes import store

    app, _ = agent_server
    async with _client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "runner", "messages": [
            {"role": "user", "content": "CONFIDENTIAL-QUESTION-TEXT"}]})
    call_id = r.json()["id"]
    rows = [row for row in store().audit_list("inference-api") if row.get("target") == call_id]
    assert rows and "inference runner ok" in rows[0]["action"]
    assert "CONFIDENTIAL" not in str(rows)
