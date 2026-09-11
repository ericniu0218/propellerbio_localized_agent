"""Exercise the HTTP backend against a fake OpenAI-compatible server."""

import json

import httpx
import pytest

from bedrock.inference import InferenceError
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
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "llama3.2"}, {"id": "qwen2.5"}]})

    backend = make_backend(handler)
    assert [m.id for m in await backend.list_models()] == ["llama3.2", "qwen2.5"]


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
