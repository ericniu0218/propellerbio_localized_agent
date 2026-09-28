"""Real model client: talks to an OpenAI-compatible endpoint served by vLLM
or SGLang. Used when AGENT_PACK_MODE=real. Requires the `httpx` package and
a running model server — not exercised by the test suite, which runs in
mock mode."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx

from bedrock.server.config import ModelConfig
from bedrock.server.models.client import ChatMessage, CompletionResult, StreamChunk


class HTTPModelClient:
    def __init__(self, config: ModelConfig, timeout_s: float = 120.0) -> None:
        self._config = config
        self._timeout_s = timeout_s

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _body(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        thinking: bool,
        temperature: float,
        stream: bool,
    ) -> dict:
        body = {
            "model": self._config.model_name,
            "messages": [m.model_dump() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        # vLLM/SGLang expose reasoning/"thinking" mode via extra_body / chat
        # template kwargs depending on server version; kept as a pass-through
        # extension field rather than hardcoding one server's flag name.
        if self._config.supports_thinking_mode:
            body["chat_template_kwargs"] = {"enable_thinking": thinking}
        return body

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> CompletionResult:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            resp = await client.post(
                f"{self._config.endpoint}/chat/completions",
                json=self._body(messages, max_tokens, thinking, temperature, stream=False),
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return CompletionResult(
            text=choice,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model_name=self._config.model_name,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> AsyncIterator[StreamChunk]:
        async with httpx.AsyncClient(timeout=self._timeout_s) as client:
            async with client.stream(
                "POST",
                f"{self._config.endpoint}/chat/completions",
                json=self._body(messages, max_tokens, thinking, temperature, stream=True),
                headers=self._headers(),
            ) as resp:
                resp.raise_for_status()
                text_parts: list[str] = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    import json

                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content", "")
                    if delta:
                        text_parts.append(delta)
                        yield StreamChunk(text=delta, done=False)
        yield StreamChunk(
            text="",
            done=True,
            final=CompletionResult(
                text="".join(text_parts),
                prompt_tokens=0,
                completion_tokens=0,
                model_name=self._config.model_name,
            ),
        )
