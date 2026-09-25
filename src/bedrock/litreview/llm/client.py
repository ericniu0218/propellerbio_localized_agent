"""The seam between this pipeline and whatever is serving the models.

Nothing outside this package may import a concrete client. Stages depend on the
`LLMClient` protocol only, and get a client from
`bedrock.litreview.llm.make_client`, so the fake used in tests and the real
thing are interchangeable.

The one live client is `SchedulerClient` (`llm/scheduled.py`), which reaches
the model server through bedrock's `Scheduler`. This module holds the protocol
it implements, the fake that stands in for it, and the two pieces of reply
handling every client shares: `strip_thinking` and `parse_json_reply`. There is
no HTTP here - the source's own vLLM and Ollama clients were dropped in the
move, because a second route to the server would be a route around the
scheduler, and the employee's chat would end up queued behind a review.

Two things here are not negotiable:

`temperature=0.0` is the default and should stay that way. The client's loudest
complaint about existing chatbots is getting a different answer to the same
question, and sampling is the direct cause.

`schema` is not a suggestion. Guided decoding constrains generation to a JSON
schema, meaning the model is *physically unable* to emit a malformed reply.
This removes an entire category of bug rather than mitigating it, so every
structured call must pass one.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Protocol, runtime_checkable

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 2048

log = logging.getLogger(__name__)

# Thinking traces some models write before a free-text answer: MedGemma 1.5
# (`<unused94>thought ... <unused95>`, with no switch to turn it off in
# Ollama) and Qwen3 (`<think>...</think>`). The trace is not the answer.
_TRACES = [("<unused94>", "<unused95>"), ("<think>", "</think>")]


def strip_thinking(text: str) -> str:
    for start, end in _TRACES:
        while start in text:
            head, rest = text.split(start, 1)
            # An unterminated trace was cut off before any answer began.
            text = head + (rest.split(end, 1)[1] if end in rest else "")
    return text.strip()


class LLMError(RuntimeError):
    """The server failed, or the reply was cut off before it was complete."""


@runtime_checkable
class LLMClient(Protocol):
    """What every stage is allowed to assume about a model server."""

    def complete(
        self,
        prompt: str,
        *,
        model: str,
        schema: dict[str, Any] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Return the model's reply as a string.

        When `schema` is given, the reply is guaranteed to parse as JSON
        matching it.
        """
        ...

    @property
    def model_version(self) -> str:
        """Pinned identifier, recorded on every Run so a report can always say
        which model produced it."""
        ...


Responder = Callable[[str, dict[str, Any] | None], str]


class FakeLLMClient:
    """Replays canned replies. Lets every stage be tested with no GPU at all.

    Either a list of replies, consumed in order, or a `responder` function
    that sees the prompt and schema and returns the reply — which lets an
    end-to-end test answer each kind of call differently.

    Keep this working. The whole pipeline should be runnable on a laptop, or
    the team is blocked whenever the office GPU is busy.
    """

    def __init__(
        self,
        replies: list[str] | None = None,
        version: str = "fake-1",
        responder: Responder | None = None,
    ) -> None:
        self._replies = list(replies or [])
        self._version = version
        self._responder = responder
        self.prompts: list[str] = []
        self.schemas: list[dict[str, Any] | None] = []

    def complete(self, prompt: str, *, model: str, schema=None, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS) -> str:
        self.prompts.append(prompt)
        self.schemas.append(schema)
        if self._responder is not None:
            return self._responder(prompt, schema)
        return self._replies.pop(0) if self._replies else "{}"

    @property
    def model_version(self) -> str:
        return self._version

    def version_of(self, model: str) -> str:
        return self._version


def parse_json_reply(raw: str) -> dict[str, Any]:
    """Parse a structured reply. Tolerates a stray code fence or think block,
    which some servers emit even under guided decoding."""
    text = strip_thinking(raw)
    if text.startswith("```"):
        text = text.strip("`")
        text = text.split("\n", 1)[1] if "\n" in text else text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"reply is not JSON: {raw[:120]!r}")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMError(f"reply is not valid JSON: {exc}") from exc
