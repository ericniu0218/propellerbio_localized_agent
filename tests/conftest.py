"""Shared test helpers.

``FakeBackend`` implements the InferenceBackend protocol in memory, so the UI
can be tested without any server. This is exactly the swap-ability the
architecture is designed for: the app cannot tell the difference.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest

from bedrock.inference import ModelInfo


class FakeBackend:
    def __init__(self, models: list[str] | None = None, reply: str = "Hello from the fake model.", connected: bool = True) -> None:
        self.models = models if models is not None else ["test-model", "other-model"]
        self.reply = reply
        self.connected = connected
        self.requests: list[tuple[str, list[dict[str, str]]]] = []
        self.closed = False
        self.fragment_delay = 0.0

    async def check_connection(self) -> bool:
        return self.connected

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m) for m in self.models]

    async def stream_chat(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        self.requests.append((model, list(messages)))
        for word in self.reply.split(" "):
            if self.fragment_delay:
                await asyncio.sleep(self.fragment_delay)
            yield word + " "

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()
