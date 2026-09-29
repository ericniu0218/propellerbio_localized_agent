"""The agent pack's ``ModelClient``, served by bedrock's ``Scheduler``.

Agents (Director, Specialist, Runner) only know the ``ModelClient`` protocol in
``client.py``. This is the one implementation of it: every call waits its turn
in the ``Scheduler`` for its model server (``bedrock.inference.scheduler``) and
goes out through bedrock's ``OpenAICompatibleBackend`` - the model-call rule in
CLAUDE.md, and the only way model calls leave this server.

What that buys over the agent pack's old ``HTTPModelClient``:

* **Token counts.** The backend asks the server for usage and the scheduler hands
  this call's own counts back through ``on_usage``. The old client always
  reported 0 when streaming, so per-task budgets were never enforced in real
  mode. When a server reports nothing, the counts are estimated
  (``bedrock.inference.tokens``) rather than left at 0, for the same reason.
* **One queue per model server**, shared by every user and every job on it (see
  ``hub.py``), and one reused HTTP connection pool instead of a new client per
  call.
* **Thinking.** ``thinking=True`` becomes ``RequestOptions(reasoning=True)``,
  ``False`` becomes ``reasoning=False`` - only for models whose config says they
  support the switch, as before; for the rest the server's default stands.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from bedrock.inference import Priority, RequestOptions, Scheduler, Usage, estimate_tokens
from bedrock.server.config import ModelConfig
from bedrock.server.models.client import ChatMessage, CompletionResult, StreamChunk


class SchedulerModelClient:
    def __init__(self, scheduler: Scheduler, config: ModelConfig, *, priority: Priority = Priority.CHAT) -> None:
        self._scheduler = scheduler
        self._config = config
        self._priority = priority

    def _options(self, max_tokens: int, thinking: bool, temperature: float) -> RequestOptions:
        reasoning = thinking if self._config.supports_thinking_mode else None
        return RequestOptions(max_tokens=max_tokens, temperature=temperature, reasoning=reasoning)

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> AsyncIterator[StreamChunk]:
        payload = [m.model_dump() for m in messages]
        reported: list[Usage] = []
        parts: list[str] = []
        async for fragment in self._scheduler.stream(
            self._config.model_name,
            payload,
            priority=self._priority,
            options=self._options(max_tokens, thinking, temperature),
            on_usage=reported.append,
        ):
            parts.append(fragment)
            yield StreamChunk(text=fragment)
        text = "".join(parts)
        usage = reported[-1] if reported else Usage()
        prompt_tokens = usage.prompt_tokens
        if prompt_tokens is None:
            prompt_tokens = estimate_tokens("\n".join(m["content"] for m in payload))
        completion_tokens = usage.completion_tokens
        if completion_tokens is None:
            completion_tokens = estimate_tokens(text)
        yield StreamChunk(
            text="",
            done=True,
            final=CompletionResult(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model_name=self._config.model_name,
            ),
        )

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        max_tokens: int,
        thinking: bool = False,
        temperature: float = 0.2,
    ) -> CompletionResult:
        final: CompletionResult | None = None
        async for chunk in self.stream(messages, max_tokens=max_tokens, thinking=thinking, temperature=temperature):
            if chunk.done:
                final = chunk.final
        assert final is not None
        return final
