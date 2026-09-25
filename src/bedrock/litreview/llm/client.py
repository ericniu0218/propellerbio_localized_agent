"""The seam between this pipeline and whatever is serving the models.

Nothing outside this package may import a concrete client. Stages depend on the
`LLMClient` protocol only, and get a client from `litreview.llm.make_client`,
so swapping vLLM for Ollama — or for the fake, in tests — is a configuration
change rather than a code change.

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

import httpx

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


class OpenAICompatibleClient:
    """Talks to a local vLLM server over its OpenAI-compatible endpoint.

    `structured_output` picks how the schema is sent: `response_format` (the
    OpenAI-style json_schema field current vLLM accepts) or `guided_json`
    (older vLLM releases). Confirm against the deployed version.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_options: dict[str, dict[str, Any]] | None = None,
        structured_output: str = "response_format",
        seed: int = 0,
        timeout: float = 600.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_options = model_options or {}
        self.structured_output = structured_output
        self.seed = seed
        self._http = httpx.Client(timeout=timeout)
        self._last_model = ""

    def complete(self, prompt: str, *, model: str, schema=None, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS) -> str:
        self._last_model = model
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": self.seed,
        }
        if not self.model_options.get(model, {}).get("thinking", False):
            # Qwen3 hybrid models think out loud unless told not to. The
            # thinking is not part of the answer and costs most of the tokens.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if schema is not None:
            if self.structured_output == "guided_json":
                body["guided_json"] = schema
            else:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": schema.get("title", "output"), "schema": schema, "strict": True},
                }
        try:
            resp = self._http.post(f"{self.base_url}/chat/completions", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"model server error: {exc}") from exc
        choice = resp.json()["choices"][0]
        content = strip_thinking(choice["message"]["content"] or "")
        if choice.get("finish_reason") == "length":
            if schema is not None or not content:
                raise LLMError(f"reply cut off at max_tokens={max_tokens}")
            # Free text cut short is still an answer, as a chatbot's would be.
            log.warning("free-text reply cut off at max_tokens=%d; keeping the partial answer", max_tokens)
        return content

    @property
    def model_version(self) -> str:
        return self._last_model or "openai-compatible"


class OllamaClient:
    """Talks to a local Ollama server over its native API.

    The native API rather than Ollama's OpenAI shim, because only the native
    one lets us set the context window per request — Ollama's default is a
    few thousand tokens, and a results section plus its methods context does
    not fit in that. It also takes a JSON schema directly in `format`.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model_options: dict[str, dict[str, Any]] | None = None,
        seed: int = 0,
        timeout: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_options = model_options or {}
        self.seed = seed
        self._http = httpx.Client(timeout=timeout)
        self._digests: dict[str, str] = {}
        self._last_model = ""
        self.stats = {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "seconds": 0.0}

    def _digest(self, model: str) -> str:
        """Pin the exact weights: a re-pulled tag with new weights is a new model."""
        if model not in self._digests:
            try:
                tags = self._http.get(f"{self.base_url}/api/tags").json().get("models", [])
                for tag in tags:
                    if tag.get("name") == model or tag.get("model") == model:
                        self._digests[model] = str(tag.get("digest", ""))[:12]
            except (httpx.HTTPError, ValueError):
                pass
            self._digests.setdefault(model, "")
        return self._digests[model]

    def complete(self, prompt: str, *, model: str, schema=None, temperature=DEFAULT_TEMPERATURE, max_tokens=DEFAULT_MAX_TOKENS) -> str:
        self._last_model = model
        opts = self.model_options.get(model, {})
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "seed": self.seed,
                "num_predict": max_tokens,
                "num_ctx": int(opts.get("num_ctx", 8192)),
            },
        }
        if "thinking" in opts:
            # Only for models that have a thinking mode (Qwen3). Sent to a
            # model without one, some Ollama versions refuse the request.
            body["think"] = bool(opts["thinking"])
        if schema is not None:
            body["format"] = schema
        try:
            resp = self._http.post(f"{self.base_url}/api/chat", json=body)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama error: {exc}") from exc
        data = resp.json()
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += int(data.get("prompt_eval_count", 0) or 0)
        self.stats["output_tokens"] += int(data.get("eval_count", 0) or 0)
        self.stats["seconds"] += float(data.get("total_duration", 0) or 0) / 1e9
        content = strip_thinking((data.get("message") or {}).get("content", "") or "")
        if data.get("done_reason") == "length":
            if schema is not None or not content:
                raise LLMError(f"reply cut off at max_tokens={max_tokens}")
            # Free text cut short is still an answer, as a chatbot's would be.
            log.warning("free-text reply cut off at max_tokens=%d; keeping the partial answer", max_tokens)
        return content

    @property
    def model_version(self) -> str:
        if not self._last_model:
            return "ollama"
        digest = self._digest(self._last_model)
        return f"{self._last_model}@{digest}" if digest else self._last_model

    def version_of(self, model: str) -> str:
        digest = self._digest(model)
        return f"{model}@{digest}" if digest else model


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
