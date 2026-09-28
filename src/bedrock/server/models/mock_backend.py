"""An ``InferenceBackend`` for AGENT_PACK_MODE=mock.

Fixture-driven replies (``mock_fixtures.fake_reply``) with simulated latency and
token counts, so the whole server - routing, agents, tools, approvals, the
event stream, the workspace - can be built and demoed with no GPU.

It sits *behind* bedrock's ``Scheduler``, exactly where a real model server
sits, so mock mode exercises the same path real mode does: the same queue, the
same usage reporting, the same ``SchedulerModelClient``. (The agent pack's old
mock client bypassed all of that, which is how its real-mode client could ship
never having been run.)

Defining ``stream_chat`` here is allowed by ``tests/test_model_call_guard.py``;
only *calling* it outside ``bedrock.inference`` is not.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Callable

from bedrock.inference.base import ModelInfo, RequestOptions, Usage
from bedrock.server.models.mock_fixtures import fake_reply


class MockBackend:
    def __init__(self, models: list[str] | None = None, *, token_delay_s: float = 0.0) -> None:
        self._models = list(models or [])
        self._token_delay_s = token_delay_s

    async def check_connection(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=name) for name in self._models]

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        text = fake_reply(last_user, model)
        words = text.split(" ")
        for i, word in enumerate(words):
            if self._token_delay_s:
                await asyncio.sleep(self._token_delay_s)
            yield word if i == 0 else f" {word}"
        if on_usage is not None:
            on_usage(Usage(
                prompt_tokens=sum(len(str(m.get("content") or "").split()) for m in messages),
                completion_tokens=len(words),
            ))

    async def close(self) -> None:
        return None
