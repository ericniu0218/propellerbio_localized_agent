"""Exercise the HTTP backend against a fake OpenAI-compatible server."""

import json

import httpx
import pytest

from bedrock.inference import InferenceError, RequestOptions, Usage
from bedrock.inference.openai_compatible import OpenAICompatibleBackend, _parse_sse_line


def sse(fragments: list[str]) -> str:
    lines = []
    for text in fragments:
        chunk = {"choices": [{"delta": {"content": text}}]}
        lines.append(f"data: {json.dumps(chunk)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines)


def make_backend(handler) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend("http://fake/v1", transport=httpx.MockTransport(handler))


async def test_list_models():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(404)  # not Ollama: sizes stay unknown
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "llama3.2"}, {"id": "qwen2.5"}]})

    backend = make_backend(handler)
    assert [m.id for m in await backend.list_models()] == ["llama3.2", "qwen2.5"]


async def test_a_server_with_no_models_pulled_lists_none():
    """What Ollama really returns before `ollama pull`: `data` is null."""
    backend = make_backend(lambda request: httpx.Response(200, json={"object": "list", "data": None}))
    assert await backend.list_models() == []


@pytest.mark.parametrize(
    "body",
    [{"data": "llama3.2"}, {"data": {"id": "llama3.2"}}, {"models": []}, ["llama3.2"]],
)
async def test_an_unexpected_models_response_is_an_inference_error(body):
    backend = make_backend(lambda request: httpx.Response(200, json=body))
    with pytest.raises(InferenceError, match="unexpected /models response"):
        await backend.list_models()


async def test_entries_that_are_not_models_are_skipped():
    """A bare string used to pass `"id" in item` as a substring test and then fail."""
    body = {"data": ["idle", {"id": "qwen2.5"}, {"name": "no id"}, None]}
    backend = make_backend(lambda request: httpx.Response(200, json=body))
    assert [m.id for m in await backend.list_models()] == ["qwen2.5"]


async def test_stream_chat_yields_fragments_and_sends_messages():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, text=sse(["Hel", "lo", " world"]), headers={"content-type": "text/event-stream"})

    backend = make_backend(handler)
    messages = [{"role": "user", "content": "hi"}]
    out = [f async for f in backend.stream_chat("llama3.2", messages)]
    assert "".join(out) == "Hello world"
    assert seen["body"]["model"] == "llama3.2"
    assert seen["body"]["messages"] == messages
    assert seen["body"]["stream"] is True
    # With no options, nothing beyond the basics (and the usage request) is sent.
    assert set(seen["body"]) == {"model", "messages", "stream", "stream_options"}


async def test_http_error_becomes_inference_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "model 'nope' not found"}})

    backend = make_backend(handler)
    with pytest.raises(InferenceError, match="model 'nope' not found"):
        async for _ in backend.stream_chat("nope", []):
            pass


async def test_connection_refused_is_reported_cleanly():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    backend = make_backend(handler)
    assert await backend.check_connection() is False
    with pytest.raises(InferenceError, match="Is the inference server running"):
        await backend.list_models()


def test_sse_line_parsing():
    assert _parse_sse_line("") == ""
    assert _parse_sse_line(": keep-alive") == ""
    assert _parse_sse_line("data: [DONE]") is None
    assert _parse_sse_line('data: {"choices":[{"delta":{"role":"assistant"}}]}') == ""
    assert _parse_sse_line('data: {"choices":[{"delta":{"content":"x"}}]}') == "x"


def usage_chunk(prompt: int, completion: int, cached: int | None = None) -> str:
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"


def streaming(text: str) -> httpx.Response:
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


async def sent_body(options: RequestOptions) -> dict:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return streaming(sse(["ok"]))

    backend = make_backend(handler)
    assert [f async for f in backend.stream_chat("m", [], options=options)] == ["ok"]
    return seen["body"]


async def test_reply_limit_temperature_and_stop_words_are_sent():
    body = await sent_body(RequestOptions(max_tokens=16, temperature=0.0, stop=("\n\n", "END")))
    assert body["max_tokens"] == 16
    assert body["temperature"] == 0.0
    assert body["stop"] == ["\n\n", "END"]


async def test_a_json_schema_is_sent_as_response_format():
    schema = {"type": "object", "properties": {"steps": {"type": "array"}}, "required": ["steps"]}
    body = await sent_body(RequestOptions(json_schema=schema))
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "schema": schema}}


async def test_reasoning_off_is_sent_the_ways_ollama_and_vllm_read_it():
    body = await sent_body(RequestOptions(reasoning=False))
    assert body["reasoning_effort"] == "none"  # Ollama
    assert body["chat_template_kwargs"] == {"enable_thinking": False}  # vLLM, SGLang


async def test_reasoning_left_to_the_server_sends_nothing_about_it():
    for options in (RequestOptions(), RequestOptions(reasoning=True)):
        body = await sent_body(options)
        assert "reasoning_effort" not in body and "chat_template_kwargs" not in body


async def test_a_server_that_rejects_reasoning_effort_is_asked_again_without_it():
    """Older vLLM only allows low/medium/high and answers 400 to "none"."""
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "reasoning_effort" in body:
            return httpx.Response(400, json={"error": {"message": "reasoning_effort: Input should be 'low', 'medium' or 'high'"}})
        return streaming(sse(["fine"]))

    backend = make_backend(handler)
    options = RequestOptions(reasoning=False, max_tokens=4)
    assert [f async for f in backend.stream_chat("m", [], options=options)] == ["fine"]
    assert [f async for f in backend.stream_chat("m", [], options=options)] == ["fine"]
    assert ["reasoning_effort" in body for body in bodies] == [True, False, False]
    assert all(body["chat_template_kwargs"] == {"enable_thinking": False} for body in bodies)


async def test_usage_is_reported_once_with_cached_prompt_tokens():
    reported: list[Usage] = []
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        text = sse(["Hi"]).replace("data: [DONE]", usage_chunk(120, 3, cached=100) + "data: [DONE]")
        return streaming(text)

    backend = make_backend(handler)
    out = [f async for f in backend.stream_chat("m", [], on_usage=reported.append)]
    assert out == ["Hi"]
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert reported == [Usage(prompt_tokens=120, completion_tokens=3, cached_prompt_tokens=100)]


async def test_a_server_that_sends_no_usage_reports_nothing():
    reported: list[Usage] = []
    backend = make_backend(lambda request: streaming(sse(["Hi"])))
    assert [f async for f in backend.stream_chat("m", [], on_usage=reported.append)] == ["Hi"]
    assert reported == []


async def test_ollama_model_sizes_come_from_api_tags():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "llama3.2:3b", "model": "llama3.2:3b", "size": 2019393189},
                {"name": "llama3.1:8b", "model": "llama3.1:8b", "size": 4920753328},
            ]})
        return httpx.Response(200, json={"data": [{"id": "llama3.2:3b"}, {"id": "llama3.1:8b"}, {"id": "mystery"}]})

    backend = OpenAICompatibleBackend("http://localhost:11434/v1", transport=httpx.MockTransport(handler))
    models = {m.id: m for m in await backend.list_models()}
    assert models["llama3.2:3b"].size_bytes == 2019393189
    assert models["llama3.1:8b"].size_bytes == 4920753328
    assert models["mystery"].size_bytes is None  # unknown stays unknown, never a guess
    assert models["mystery"].context_length is None


async def test_vllm_context_length_comes_from_max_model_len():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-4B", "max_model_len": 32768}]})

    [model] = await make_backend(handler).list_models()
    assert model.context_length == 32768
    assert model.size_bytes is None


async def test_a_broken_tags_endpoint_does_not_break_listing_models():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    assert [m.id for m in await make_backend(handler).list_models()] == ["m"]


async def test_a_seed_is_sent_when_set_and_left_out_when_not():
    """The literature review pins a seed so the same question gives the same report."""
    assert (await sent_body(RequestOptions(seed=0)))["seed"] == 0
    assert (await sent_body(RequestOptions(seed=7)))["seed"] == 7
    assert "seed" not in await sent_body(RequestOptions())


async def test_ollama_weight_digests_come_from_api_tags():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "medgemma1.5:4b", "model": "medgemma1.5:4b",
                 "size": 3338801804, "digest": "a1b2c3d4e5f60718293a4b5c6d7e8f90"},
            ]})
        return httpx.Response(200, json={"data": [{"id": "medgemma1.5:4b"}, {"id": "mystery"}]})

    backend = OpenAICompatibleBackend("http://localhost:11434/v1", transport=httpx.MockTransport(handler))
    models = {m.id: m for m in await backend.list_models()}
    assert models["medgemma1.5:4b"].digest == "a1b2c3d4e5f60718293a4b5c6d7e8f90"
    assert models["medgemma1.5:4b"].size_bytes == 3338801804
    assert models["mystery"].digest is None  # unknown stays unknown


async def test_a_server_without_api_tags_reports_no_digest():
    """vLLM has no /api/tags, so it says nothing about the weights it serves."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(404)
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3-8B", "max_model_len": 32768}]})

    [model] = await make_backend(handler).list_models()
    assert model.digest is None
