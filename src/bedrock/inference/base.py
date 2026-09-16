"""The contract between the UI and any inference server.

This is the single most important seam in the codebase. The UI only ever talks
to an ``InferenceBackend``. It does not know or care whether the other end is
Ollama on a laptop, vLLM in a data centre, or a fake used in tests.

If the company later switches inference servers, the only code that changes is
a new class implementing this protocol, plus one line in ``factory.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol


class InferenceError(Exception):
    """Raised for any failure talking to the inference server.

    The UI catches this one type and shows the message to the user, so backends
    should wrap transport errors and bad responses in it with a readable message.
    """

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.unavailable = unavailable


@dataclass(frozen=True)
class ModelInfo:
    id: str
    """The identifier the server expects in requests, e.g. ``llama3.2:3b``."""


class InferenceBackend(Protocol):
    """Everything the client needs from a model server."""

    async def check_connection(self) -> bool:
        """Return True if the server is reachable. Must not raise."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Models the server can serve right now."""
        ...

    def stream_chat(self, model: str, messages: list[dict[str, str]]) -> AsyncIterator[str]:
        """Send a conversation and yield the reply as text fragments arrive.

        ``messages`` is the OpenAI-style list of ``{"role": ..., "content": ...}``.
        """
        ...

    async def close(self) -> None:
        """Release network resources."""
        ...
