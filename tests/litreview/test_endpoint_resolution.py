"""Where a review's model calls go, what window they get, and how that is checked.

P0.4: with `BEDROCK_BACKEND=agentpack` and no config file, `bedrock-litreview
ask` used Ollama's default address on this machine and the `laptop` profile's
8,192-token window, and refused to start - while the model that would serve the
review is on the agent server, with a window of its own. And the other half of
the same mistake: a configured window is a claim about the server, and the
Ollama app's default is 4,096, whatever the claim.
"""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from bedrock.config import load_settings as load_bedrock
from bedrock.inference import ModelInfo, Usage
from bedrock.inference.openai_compatible import OpenAICompatibleBackend
from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.llm.scheduled import PromptTruncated, SchedulerClient, truncated
from bedrock.litreview.service import preflight


def settings_for(env: dict[str, str], **kw):
    return load_settings(load_bedrock(env=env), env=env, check_endpoint=False, **kw)


# --- precedence ---------------------------------------------------------------------------


def test_default_mode_uses_api_base_and_the_profile_window():
    s = settings_for({})
    assert (s.base_url, s.endpoint_source) == ("http://localhost:11434/v1", "api_base")
    assert (s.context_window, s.context_window_source, s.model_set) == (8192, "profile", "laptop")


def test_agent_pack_mode_with_nothing_configured_asks_the_agent_server():
    s = settings_for({"BEDROCK_BACKEND": "agentpack"})
    assert (s.base_url, s.endpoint_source) == ("http://localhost:8000/v1", "agentpack")
    # The laptop profile describes Ollama on this machine, not the server's model.
    assert (s.context_window, s.context_window_source) == (0, "unset")
    assert s.model_set == "agentpack" and s.model_for("extract") == "specialist"


def test_agent_pack_address_and_key_come_from_bedrocks_own_settings():
    s = settings_for({"BEDROCK_BACKEND": "agentpack", "AGENT_PACK_API_BASE": "http://ai.lab.internal:9000/",
                      "BEDROCK_API_KEY": "k1"})
    assert s.base_url == "http://ai.lab.internal:9000/v1" and s.api_key == "k1"


def test_explicit_review_settings_win_over_the_mode():
    env = {"BEDROCK_BACKEND": "agentpack", "BEDROCK_LITREVIEW_API_BASE": "http://gpu.lab.internal/v1",
           "BEDROCK_LITREVIEW_API_KEY": "k2", "BEDROCK_LITREVIEW_CONTEXT_WINDOW": "16384",
           "BEDROCK_LITREVIEW_MODELS": "office", "BEDROCK_API_KEY": "k1"}
    s = settings_for(env)
    assert (s.base_url, s.endpoint_source, s.api_key) == ("http://gpu.lab.internal/v1", "litreview", "k2")
    assert (s.context_window, s.context_window_source, s.model_set) == (16384, "litreview", "office")


def test_a_configured_bedrock_window_counts_as_configured_in_every_mode():
    s = settings_for({"BEDROCK_BACKEND": "agentpack", "BEDROCK_CONTEXT_WINDOW": "32768"})
    assert (s.context_window, s.context_window_source) == (32768, "bedrock")


def test_the_config_file_table_is_read(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('backend = "agentpack"\n[litreview]\napi_base = "http://10.0.0.5/v1"\ncontext_window = 20000\n')
    s = load_settings(load_bedrock(config_path=path, env={}), env={}, check_endpoint=True)
    assert (s.base_url, s.context_window, s.context_window_source) == ("http://10.0.0.5/v1", 20000, "litreview")


def test_a_public_review_endpoint_is_still_refused():
    with pytest.raises(ConfigError, match="not a local address"):
        load_settings(load_bedrock(env={}), env={"BEDROCK_LITREVIEW_API_BASE": "https://api.example.com/v1"})


def test_a_nonsense_window_is_refused():
    with pytest.raises(ConfigError, match="at least 1024"):
        settings_for({"BEDROCK_LITREVIEW_CONTEXT_WINDOW": "512"})


# --- preflight against what the server reports ---------------------------------------------------


class Server:
    def __init__(self, *models: ModelInfo) -> None:
        self.models = list(models)

    async def list_models(self):
        return self.models


def _office(**env):
    return settings_for({"BEDROCK_LITREVIEW_MODELS": "office", **env})


async def test_the_reported_window_is_used_when_nothing_is_configured():
    s = settings_for({"BEDROCK_BACKEND": "agentpack"})
    got = await preflight(s, Server(ModelInfo("runner", context_length=32768), ModelInfo("specialist", context_length=65536)))
    assert got.windows == {"runner": 32768, "specialist": 65536}


async def test_nothing_reported_and_nothing_configured_is_refused_not_guessed():
    s = settings_for({"BEDROCK_BACKEND": "agentpack"})
    with pytest.raises(ConfigError, match="does not say how large a context window"):
        await preflight(s, Server(ModelInfo("runner"), ModelInfo("specialist")))


async def test_a_smaller_reported_window_wins_over_a_configured_one():
    s = _office(BEDROCK_CONTEXT_WINDOW="32768")
    got = await preflight(s, Server(ModelInfo("qwen3-8b-instruct", context_length=16384),
                                    ModelInfo("qwen3-30b-a3b-instruct", context_length=65536)))
    assert got.windows == {"qwen3-8b-instruct": 16384, "qwen3-30b-a3b-instruct": 32768}
    assert any("smaller than the 32,768 configured" in n for n in got.notes)


async def test_ollamas_4096_default_is_caught_even_when_12288_is_configured():
    """The machine this was found on: BEDROCK_CONTEXT_WINDOW=12288 in the
    server's environment, and the Ollama app loading qwen3:8b at 4,096."""
    s = settings_for({"BEDROCK_CONTEXT_WINDOW": "12288"})
    with pytest.raises(ConfigError) as exc:
        await preflight(s, Server(ModelInfo("qwen3:8b", context_length=4096)))
    assert "reports a 4,096-token context window" in str(exc.value)
    assert "OLLAMA_CONTEXT_LENGTH=12288" in str(exc.value)


async def test_a_model_without_structured_output_is_refused():
    s = settings_for({"BEDROCK_BACKEND": "agentpack"})
    with pytest.raises(ConfigError, match="JSON schema"):
        await preflight(s, Server(ModelInfo("runner", context_length=32768, capabilities=frozenset({"seed"})),
                                  ModelInfo("specialist", context_length=65536, capabilities=frozenset({"seed"}))))


async def test_a_model_that_ignores_the_seed_is_said_not_hidden():
    s = settings_for({"BEDROCK_BACKEND": "agentpack"})
    both = frozenset({"json_schema"})
    got = await preflight(s, Server(ModelInfo("runner", context_length=32768, capabilities=both),
                                    ModelInfo("specialist", context_length=65536, capabilities=both)))
    assert any("ignores the random seed" in n for n in got.notes)


async def test_a_missing_model_lists_what_the_agent_server_does_offer():
    s = settings_for({"BEDROCK_BACKEND": "agentpack"})
    with pytest.raises(ConfigError) as exc:
        await preflight(s, Server(ModelInfo("director", context_length=131072)))
    assert "It offers: director" in str(exc.value) and "ollama pull" not in str(exc.value)


# --- the backend reports Ollama's real window -----------------------------------------------------


async def test_ollama_reports_the_window_a_loaded_model_was_loaded_with():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen3:8b"}, {"id": "llama3.2:3b"}]})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "digest": "500a1f067a9f", "size": 5}]})
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": "qwen3:8b", "context_length": 4096}]})
        return httpx.Response(404)

    backend = OpenAICompatibleBackend("http://localhost:11434/v1", transport=httpx.MockTransport(handler))
    models = {m.id: m for m in await backend.list_models()}
    assert models["qwen3:8b"].context_length == 4096
    assert models["llama3.2:3b"].context_length is None  # not loaded: unknown, not small


async def test_an_alias_reports_its_weights_and_capabilities():
    def handler(request):
        return httpx.Response(200, json={"data": [{"id": "specialist", "root": "Qwen/Qwen3.8-27B",
                                                   "max_model_len": 65536,
                                                   "capabilities": {"json_schema": True, "seed": False}}]}) \
            if request.url.path == "/v1/models" else httpx.Response(404)

    backend = OpenAICompatibleBackend("http://srv/v1", transport=httpx.MockTransport(handler))
    (m,) = await backend.list_models()
    assert (m.root, m.context_length, m.capabilities) == ("Qwen/Qwen3.8-27B", 65536, frozenset({"json_schema"}))


# --- every call is checked as it runs -----------------------------------------------------------------


def test_a_reported_prompt_count_that_cannot_hold_the_prompt_is_truncation():
    prompt = "word " * 4000  # 20,000 characters: at least ~2,850 tokens with any tokenizer
    assert truncated(prompt, 2050)
    assert not truncated(prompt, 4300)
    assert not truncated(prompt, None)
    assert not truncated("short prompt", 3)


class _Scheduler:
    """Answers like Ollama with a 4,096 window: reads the end of the prompt only."""

    def __init__(self, read: int) -> None:
        self.read = read
        self.backend = None

    async def complete(self, model, messages, *, priority, options, on_usage=None):
        if on_usage is not None:
            on_usage(Usage(prompt_tokens=self.read, completion_tokens=10))
        return '{"ok": true}'


def test_a_truncated_call_stops_the_run_with_the_fix():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        client = SchedulerClient(_Scheduler(read=2050), loop, context_window=12288)
        with pytest.raises(PromptTruncated) as exc:
            client.complete("instructions " * 2000, model="qwen3:8b", max_tokens=100)
        assert "read only 2,050 tokens" in str(exc.value) and "OLLAMA_CONTEXT_LENGTH=12288" in str(exc.value)
        # A server that read the whole prompt is fine.
        ok = SchedulerClient(_Scheduler(read=6000), loop, context_window=12288)
        assert ok.complete("instructions " * 2000, model="qwen3:8b", max_tokens=100) == '{"ok": true}'
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


def test_each_model_is_sized_to_its_own_window():
    from bedrock.litreview.llm import LLMError

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        client = SchedulerClient(_Scheduler(read=6000), loop, context_window=0,
                                 context_windows={"small": 4096, "big": 65536})
        with pytest.raises(LLMError, match="window is 4096"):
            client.complete("x " * 10000, model="small", max_tokens=2048)
        assert client.complete("x " * 10000, model="big", max_tokens=2048)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)


# --- a stream that stops halfway ------------------------------------------------------------------


async def test_an_error_chunk_mid_stream_is_an_error():
    from bedrock.inference import InferenceError

    body = ('data: {"choices":[{"delta":{"content":"{\\"a\\":"}}]}\n\n'
            'data: {"error":{"message":"upstream died"}}\n\n')
    backend = OpenAICompatibleBackend("http://srv/v1", transport=httpx.MockTransport(
        lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})))
    with pytest.raises(InferenceError, match="partway"):
        async for _ in backend.stream_chat("m", [{"role": "user", "content": "x"}]):
            pass
