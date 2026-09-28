"""Fake model client used when AGENT_PACK_MODE=mock. Produces deterministic,
fixture-driven responses with simulated latency and token counts so the
frontend can be built and demoed with zero GPUs — see docs/ARCHITECTURE.md
section 7."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from app.models.client import ChatMessage, CompletionResult, StreamChunk
from app.models.mock_fixtures import fake_reply


class MockModelClient:
    def __init__(
        self,
        model_name: str,
        *,
        fixed_response: str | None = None,
        token_delay_s: float = 0.0,
    ) -> None:
        self._model_name = model_name
        self._fixed_response = fixed_response
        self._token_delay_s = token_delay_s

    def _response_for(self, messages: list[ChatMessage]) -> str:
        if self._fixed_response is not None:
            return self._fixed_response
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        return fake_reply(last_user, self._model_name)

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> CompletionResult:
        text = self._response_for(messages)
        if self._token_delay_s:
            await asyncio.sleep(self._token_delay_s)
        prompt_tokens = sum(len(m.content.split()) for m in messages)
        return CompletionResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(text.split()),
            model_name=self._model_name,
        )

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> AsyncIterator[StreamChunk]:
        text = self._response_for(messages)
        words = text.split(" ")
        for i, word in enumerate(words):
            if self._token_delay_s:
                await asyncio.sleep(self._token_delay_s)
            chunk_text = word if i == 0 else f" {word}"
            yield StreamChunk(text=chunk_text, done=False)
        yield StreamChunk(
            text="",
            done=True,
            final=CompletionResult(
                text=text,
                prompt_tokens=sum(len(m.content.split()) for m in messages),
                completion_tokens=len(words),
                model_name=self._model_name,
            ),
        )
