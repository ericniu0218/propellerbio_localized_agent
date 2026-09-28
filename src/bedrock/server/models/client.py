"""The ModelClient protocol. Agents depend on this and nothing else — never
on vLLM, SGLang, or mock specifics directly — so real/mock and even the
serving engine can be swapped without touching agent code."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


class CompletionResult(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    model_name: str


class StreamChunk(BaseModel):
    text: str
    done: bool = False
    # populated only on the final chunk
    final: CompletionResult | None = None


class ModelClient(Protocol):
    """OpenAI-compatible chat completion, sync and streaming."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> CompletionResult: ...

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> AsyncIterator[StreamChunk]: ...
