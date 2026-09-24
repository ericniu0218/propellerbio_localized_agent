"""The one door every model call goes through.

Why a scheduler. A laptop's model server can work on one request at a time:
each extra request running in parallel needs another full KV cache (see
``bedrock.config.profile``). If a title request is running when the employee
sends a message, the employee waits for the title. The title request has also
replaced the server's saved work on the chat, so the next chat turn is read
again from the very start.

So nothing outside ``bedrock.inference`` calls ``stream_chat`` itself. It asks
the scheduler instead, saying how urgent the call is (``Priority``), and the
scheduler decides when it runs.

This version passes every call straight through. It keeps track of the
model that last answered a chat and the token counts the server reported.
Queueing and cancellation come next, behind the same methods.
"""

from __future__ import annotations

import inspect
from typing import AsyncIterator

from bedrock.config.profile import MachineProfile
from bedrock.inference import tokens
from bedrock.inference.base import InferenceBackend, InferenceError, Priority, RequestOptions, Usage


class Preempted(InferenceError):
    """A lower-priority call was cancelled because the employee sent a message. Retry later; not a failure."""

    def __init__(self, message: str = "Stopped to make room for the chat; will try again later.") -> None:
        super().__init__(message)


class Scheduler:
    """Decides when each model call reaches the server. See the module docstring."""

    def __init__(self, backend: InferenceBackend, profile: MachineProfile) -> None:
        self.backend = backend
        self.profile = profile
        self._last_usage: Usage | None = None
        self._loaded_model: str | None = None
        self._takes_options: bool | None = None

    @property
    def last_usage(self) -> Usage | None:
        """Token counts from the last chat reply, when the server reported them."""
        return self._last_usage

    @property
    def loaded_model(self) -> str | None:
        """The model that last answered a chat, which is the one the server has in memory."""
        return self._loaded_model

    async def stream(
        self,
        model: str | None,
        messages: list[dict[str, str]],
        *,
        priority: Priority,
        options: RequestOptions | None = None,
    ) -> AsyncIterator[str]:
        """Run one call when its turn comes and yield the reply as it arrives.

        ``model=None`` is allowed only for ``BACKGROUND`` calls and means "the
        model that last answered a chat". If no chat has been answered yet,
        this raises ``Preempted`` rather than make the server load a model
        just for a title.
        """
        model = self._resolve_model(model, priority)
        characters = sum(len(str(message.get("content") or "")) for message in messages)
        usage: list[Usage] = []
        answered = False
        stream = self._open(model, messages, options, usage.append)
        try:
            async for fragment in stream:
                if priority is Priority.CHAT and not answered:
                    self._loaded_model = model
                answered = True
                yield fragment
        finally:
            await _close(stream)
        if priority is Priority.CHAT:
            self._loaded_model = model
            if usage:
                self._last_usage = usage[-1]
                tokens.observe(characters, usage[-1])

    async def complete(
        self,
        model: str | None,
        messages: list[dict[str, str]],
        *,
        priority: Priority,
        options: RequestOptions | None = None,
    ) -> str:
        """Like ``stream``, but wait for the whole reply and return it as one string."""
        parts: list[str] = []
        async for fragment in self.stream(model, messages, priority=priority, options=options):
            parts.append(fragment)
        return "".join(parts)

    def _resolve_model(self, model: str | None, priority: Priority) -> str:
        if model:
            return model
        if priority is not Priority.BACKGROUND:
            raise ValueError("Only BACKGROUND calls may leave the model to the scheduler.")
        if self._loaded_model is None:
            raise Preempted("No chat has been answered yet, so no model is loaded to ask.")
        return self._loaded_model

    def _open(self, model, messages, options, on_usage) -> AsyncIterator[str]:
        """Start the backend's stream, with options if the backend understands them.

        A backend written for the original two-argument ``stream_chat`` (some
        test doubles, or a company's own adapter) still works: it just cannot
        honour limits or report token counts.
        """
        if self._takes_options is None:
            self._takes_options = _accepts_options(self.backend)
        if self._takes_options:
            return self.backend.stream_chat(model, messages, options=options, on_usage=on_usage)
        return self.backend.stream_chat(model, messages)


def _accepts_options(backend: InferenceBackend) -> bool:
    try:
        parameters = inspect.signature(backend.stream_chat).parameters.values()
    except (TypeError, ValueError):
        return True
    names = {parameter.name for parameter in parameters}
    return "options" in names or any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)


async def _close(stream: AsyncIterator[str]) -> None:
    """Close the backend's stream, so an early stop also closes the HTTP connection."""
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()
