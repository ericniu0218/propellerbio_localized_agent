"""The contract between the UI and any inference server.

This is the single most important seam in the codebase. The UI only ever talks
to an ``InferenceBackend``. It does not know or care whether the other end is
Ollama on a laptop, vLLM in a data centre, or a fake used in tests.

If the company later switches inference servers, the only code that changes is
a new class implementing this protocol, plus one line in ``factory.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import AsyncIterator, Callable, Protocol


class InferenceError(Exception):
    """Raised for any failure talking to the inference server.

    The UI catches this one type and shows the message to the user, so backends
    should wrap transport errors and bad responses in it with a readable message.
    """

    def __init__(self, message: str, *, unavailable: bool = False) -> None:
        super().__init__(message)
        self.unavailable = unavailable


class Priority(IntEnum):
    """How urgently a model call is wanted. A lower number goes first.

    On a laptop the server can only work on one request at a time, so every
    call waits its turn in the ``Scheduler`` (``bedrock.inference.scheduler``).
    The priority decides who goes next, and whether a running call may be
    cancelled to make room for a more urgent one.
    """

    CHAT = 0
    """The employee's own turn: they are watching the screen, waiting for it."""

    USER_JOB = 1
    """Something the employee asked for that is not a chat reply: a ``/digest``
    summary, meeting notes, an agent plan. Never cancelled for a chat, but a
    chat that arrives meanwhile goes next."""

    BACKGROUND = 2
    """Titles and anything else nobody is waiting for. Cancelled the moment the
    employee sends a message, and tried again later."""


@dataclass(frozen=True)
class RequestOptions:
    """Limits and switches for one model call. Every field is optional.

    ``None`` (or an empty tuple) always means "leave it to the server", so a
    call that passes ``RequestOptions()`` behaves exactly like one that passes
    nothing at all.
    """

    max_tokens: int | None = None
    """The longest reply the server may write, in tokens (roughly four
    characters each). Without it a small model asked for a YES/NO answer can
    write for minutes."""

    temperature: float | None = None
    """How random the reply is: 0 always picks the likeliest word."""

    stop: tuple[str, ...] = ()
    """The server stops writing as soon as it produces one of these strings."""

    json_schema: dict | None = None
    """A JSON Schema the reply must match. Servers that support it constrain the
    model so it can only produce matching JSON; the reply is still checked on
    arrival, because not every server does."""

    reasoning: bool | None = None
    """``False`` asks a reasoning model to skip its hidden "thinking" and answer
    straight away. ``None`` leaves the server's default."""

    seed: int | None = None
    """The random seed the server should start from. With ``temperature=0`` a
    fixed seed is what makes the same prompt give back the same words, which the
    literature review depends on. ``None`` leaves it to the server."""


@dataclass(frozen=True)
class Usage:
    """Token counts the server reports for one call, when it reports them.

    Any count the server did not send stays ``None`` - never a guess. These
    numbers stay on this machine: they feed the context meter in the status bar
    and correct the token estimate in ``bedrock.inference.tokens``.
    """

    prompt_tokens: int | None = None
    """How many tokens the request filled on the server."""

    completion_tokens: int | None = None
    """How many tokens the reply took."""

    cached_prompt_tokens: int | None = None
    """How much of the prompt the server reused from its previous request
    instead of reading it again. High is good: it means the prompt start
    stayed the same."""


@dataclass(frozen=True)
class ModelInfo:
    """One model the server offers, with what we know about it.

    Only ``id`` is guaranteed. The rest is filled in when the server says so
    and stays ``None`` otherwise, so code must treat "unknown" as unknown,
    not as small or as zero.
    """

    id: str
    """The identifier the server expects in requests, e.g. ``llama3.2:3b``."""

    size_bytes: int | None = None
    """Size of the model's weights on disk, which is roughly the memory it needs
    before any context. Used to avoid auto-picking a model a laptop can't hold."""

    context_length: int | None = None
    """The longest prompt plus reply, in tokens, the server will accept."""

    digest: str | None = None
    """A hash of the model's weights, when the server reports one (Ollama does,
    on ``/api/tags``). Two servers that report the same digest are running the
    same weights, so a record of it says which model actually wrote something.
    vLLM does not report one, so there it stays ``None``."""


class InferenceBackend(Protocol):
    """Everything the client needs from a model server."""

    async def check_connection(self) -> bool:
        """Return True if the server is reachable. Must not raise."""
        ...

    async def list_models(self) -> list[ModelInfo]:
        """Models the server can serve right now."""
        ...

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        """Send a conversation and yield the reply as text fragments arrive.

        ``messages`` is the OpenAI-style list of ``{"role": ..., "content": ...}``.
        ``options`` carries limits such as the reply length. ``on_usage`` is
        called at most once, after the reply, if the server reported token counts.

        Only code in ``bedrock.inference`` calls this. Everything else goes
        through ``Scheduler``, which decides when a call may run.
        """
        ...

    async def close(self) -> None:
        """Release network resources."""
        ...
