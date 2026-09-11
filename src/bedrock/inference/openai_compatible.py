"""Backend for any server that speaks the OpenAI chat-completions API.

That covers all of the servers we care about:

* Ollama       - ``http://localhost:11434/v1``
* vLLM         - ``http://<host>:8000/v1``
* SGLang       - ``http://<host>:30000/v1``
* llama.cpp    - ``http://<host>:8080/v1``

We only use two endpoints: ``GET /models`` and ``POST /chat/completions`` with
``stream: true``. The streaming format is Server-Sent Events: each line looks
like ``data: {json}`` and the stream ends with ``data: [DONE]``.

Note: "OpenAI-compatible" is only the *wire format*. Nothing here ever contacts
OpenAI; the base URL is whatever the company configures.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from bedrock.inference.base import InferenceError, ModelInfo


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``transport`` lets tests substitute a fake server; leave it None in real use.
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )

    async def check_connection(self) -> bool:
        try:
            response = await self._client.get("/models", timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Could not list models: {_describe(exc)}") from exc

        try:
            data = response.json()["data"]
        except (ValueError, KeyError, TypeError) as exc:
            raise InferenceError("Server returned an unexpected /models response.") from exc

        return [ModelInfo(id=item["id"]) for item in data if "id" in item]

    async def stream_chat(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        payload = {"model": model, "messages": messages, "stream": True}
        try:
            async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode(errors="replace")
                    raise InferenceError(_http_error_message(response.status_code, body))

                async for line in response.aiter_lines():
                    fragment = _parse_sse_line(line)
                    if fragment is None:
                        break
                    if fragment:
                        yield fragment
        except httpx.HTTPError as exc:
            raise InferenceError(f"Request failed: {_describe(exc)}") from exc

    async def close(self) -> None:
        await self._client.aclose()


def _parse_sse_line(line: str) -> str | None:
    """Turn one SSE line into a text fragment.

    Returns ``""`` for lines that carry no text (keep-alives, role markers),
    and ``None`` when the server signals the end of the stream.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return ""
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return None
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return ""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}
    return delta.get("content") or ""


def _http_error_message(status: int, body: str) -> str:
    # Servers usually put a helpful message in {"error": {"message": ...}} or {"error": "..."}
    try:
        error = json.loads(body).get("error")
        if isinstance(error, dict):
            error = error.get("message")
        if error:
            return f"Server error {status}: {error}"
    except (ValueError, AttributeError):
        pass
    return f"Server error {status}."


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "connection refused. Is the inference server running?"
    if isinstance(exc, httpx.TimeoutException):
        return "the server took too long to respond."
    return str(exc) or exc.__class__.__name__
