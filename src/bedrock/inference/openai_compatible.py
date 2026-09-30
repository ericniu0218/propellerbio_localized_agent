"""Backend for any server that speaks the OpenAI chat-completions API.

That covers all of the servers we care about:

* Ollama       - ``http://localhost:11434/v1``
* vLLM         - ``http://<host>:8000/v1``
* SGLang       - ``http://<host>:30000/v1``
* llama.cpp    - ``http://<host>:8080/v1``

We use ``GET /models`` and ``POST /chat/completions`` with ``stream: true``.
The streaming format is Server-Sent Events: each line looks like
``data: {json}`` and the stream ends with ``data: [DONE]``. On Ollama we also
read ``GET /api/tags`` (outside ``/v1``), because that is the only place it
says how big each model is and the hash of the weights it is serving.

How ``RequestOptions`` reach the server, field by field:

* ``max_tokens``, ``temperature``, ``stop``, ``seed``: the standard fields;
  every server above reads them. ``seed`` is sent only when it is set, so a
  call that leaves it ``None`` looks exactly as it did before.
* ``json_schema``: sent as ``response_format = {"type": "json_schema", ...}``.
  Ollama (0.5+), vLLM and SGLang then constrain the reply to the schema.
* ``reasoning=False`` (skip the hidden "thinking"): no single field works
  everywhere, so two are sent:

  - ``reasoning_effort: "none"`` - **Ollama** reads this on ``/v1`` and turns
    thinking off. Current vLLM also accepts it.
  - ``chat_template_kwargs: {"enable_thinking": false}`` - **vLLM** and
    **SGLang** hand this to the model's chat template; Qwen3-style templates
    use it to switch thinking off. Ollama ignores the field.

  Older vLLM releases reject ``reasoning_effort: "none"`` with a 400. When a
  server does that, the request is sent once more without it, and it is not
  sent to that server again.

* ``reasoning=True`` (ask for the thinking): ``chat_template_kwargs:
  {"enable_thinking": true}`` for vLLM and SGLang, whose hybrid models default
  either way depending on the template. Ollama has no switch for it and keeps
  its default. The agent-pack server's Director asks for this.

* Token counts: ``stream_options = {"include_usage": true}`` asks the server
  to end the stream with a chunk carrying ``usage``, including
  ``prompt_tokens_details.cached_tokens`` where the server reports reuse.

Note: "OpenAI-compatible" is only the *wire format*. Nothing here ever contacts
OpenAI; the base URL is whatever the company configures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import httpx

from bedrock.inference.base import InferenceError, ModelInfo, RequestOptions, Usage


@dataclass(frozen=True)
class _Tag:
    """What Ollama's ``/api/tags`` says about one model. Unknown stays ``None``."""

    size: int | None = None
    digest: str | None = None


class OpenAICompatibleBackend:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``transport`` lets tests substitute a fake server; leave it None in real use.
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=httpx.Timeout(timeout, connect=5.0),
            transport=transport,
        )
        # Set once the server has refused ``reasoning_effort``; see the module docstring.
        self._skip_reasoning_effort = False
        self._observers: list[Callable[[dict[str, Any]], None]] = []

    @property
    def base_url(self) -> str:
        return self._base_url

    def add_request_observer(self, observer: Callable[[dict[str, Any]], None]) -> None:
        """Be told, for every attempt, what was actually sent and what came back.

        For audit and the determinism experiment: the request that reaches the
        server can differ from the one asked for (a server that refuses
        ``reasoning_effort`` gets a second attempt without it), and only the
        wire payload says which. Observers receive hashes and the non-message
        fields only - never the messages, never a header, so never a key. An
        observer that raises is ignored: auditing must not break a call.
        """
        self._observers.append(observer)

    def _observe(self, event: dict[str, Any]) -> None:
        for observer in self._observers:
            try:
                observer(event)
            except Exception:  # noqa: BLE001 - see add_request_observer
                pass

    async def check_connection(self) -> bool:
        try:
            response = await self._client.get("/models", timeout=5.0)
            return response.status_code < 500
        except httpx.HTTPError:
            return False

    async def list_models(self) -> list[ModelInfo]:
        try:
            response = await self._client.get("/models")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise InferenceError(f"Could not list models: {_describe(exc)}", unavailable=True) from exc

        try:
            data = response.json()["data"]
        except (ValueError, KeyError, TypeError) as exc:
            raise InferenceError("Server returned an unexpected /models response.") from exc

        # Ollama answers `"data": null`, not `[]`, until a model has been
        # pulled. That is an empty list, not a broken server: treating it as
        # one to iterate crashed the app on startup on a fresh install.
        if data is None:
            return []
        if not isinstance(data, list):
            raise InferenceError("Server returned an unexpected /models response.")
        entries = [item for item in data if isinstance(item, dict) and "id" in item]
        if not entries:
            return []
        tags = await self._ollama_tags()
        loaded = await self._ollama_loaded_windows() if tags else {}
        return [
            ModelInfo(
                id=str(item["id"]),
                size_bytes=_positive_int(item.get("size")) or tags.get(str(item["id"]), _Tag()).size,
                # vLLM (and bedrock's agent server) report the window a model
                # was started with as ``max_model_len``. Ollama reports none,
                # except for a model that is loaded right now: ``/api/ps``
                # gives the window it was loaded with, which is what a prompt
                # really gets - not the model's own maximum.
                context_length=_positive_int(item.get("max_model_len")) or loaded.get(str(item["id"])),
                digest=tags.get(str(item["id"]), _Tag()).digest,
                root=item["root"] if isinstance(item.get("root"), str) and item["root"] != item["id"] else None,
                capabilities=_capabilities(item.get("capabilities")),
            )
            for item in entries
        ]

    async def _ollama_loaded_windows(self) -> dict[str, int]:
        """Model -> the context window Ollama loaded it with (``/api/ps``).

        Only models in memory right now are listed, so a missing entry means
        "not known", never "small". Any failure means the same."""
        root = self._base_url[: -len("/v1")] if self._base_url.endswith("/v1") else self._base_url
        try:
            response = await self._client.get(f"{root}/api/ps", timeout=5.0)
            if response.status_code != 200:
                return {}
            models = response.json().get("models") or []
        except (httpx.HTTPError, ValueError, AttributeError):
            return {}
        windows: dict[str, int] = {}
        for entry in models if isinstance(models, list) else []:
            if not isinstance(entry, dict):
                continue
            window = _positive_int(entry.get("context_length"))
            for key in ("model", "name"):
                name = entry.get(key)
                if window and isinstance(name, str):
                    windows[name] = window
        return windows

    async def _ollama_tags(self) -> dict[str, _Tag]:
        """Model size and weight digest from Ollama's ``/api/tags``.

        Empty for any other server: only Ollama offers this. Always the
        configured server, never anywhere else. Any failure (a 404 from vLLM, a
        timeout, an odd body) just means "we do not know", so every caller has
        to treat a missing entry as unknown rather than as zero or as absent.
        """
        root = self._base_url[: -len("/v1")] if self._base_url.endswith("/v1") else self._base_url
        try:
            response = await self._client.get(f"{root}/api/tags", timeout=5.0)
            if response.status_code != 200:
                return {}
            models = response.json().get("models") or []
        except (httpx.HTTPError, ValueError, AttributeError):
            return {}
        tags: dict[str, _Tag] = {}
        for entry in models if isinstance(models, list) else []:
            if not isinstance(entry, dict):
                continue
            digest = entry.get("digest")
            tag = _Tag(
                size=_positive_int(entry.get("size")),
                digest=digest if isinstance(digest, str) and digest else None,
            )
            if tag == _Tag():
                continue
            for key in ("model", "name"):
                name = entry.get(key)
                if isinstance(name, str):
                    tags[name] = tag
        return tags

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        payload = self._payload(model, messages, options)
        observed = bool(self._observers)
        try:
            for attempt in (1, 2):
                if observed:
                    self._observe({"event": "request", "attempt": attempt, **wire_summary(payload)})
                async with self._client.stream("POST", "/chat/completions", json=payload) as response:
                    if response.status_code >= 400:
                        body = (await response.aread()).decode(errors="replace")
                        retry = attempt == 1 and _rejects_reasoning_effort(response.status_code, body, payload)
                        if observed:
                            self._observe({"event": "rejected", "attempt": attempt,
                                           "status": response.status_code, "retrying": retry})
                        if retry:
                            self._skip_reasoning_effort = True
                            payload = self._payload(model, messages, options)
                            continue
                        raise InferenceError(_http_error_message(response.status_code, body))

                    usage: Usage | None = None
                    raw = hashlib.sha256() if observed else None
                    async for line in response.aiter_lines():
                        fragment, reported = _parse_sse_chunk(line)
                        if reported is not None:
                            usage = reported
                        if fragment is None:
                            break
                        if fragment:
                            if raw is not None:
                                raw.update(fragment.encode("utf-8"))
                            yield fragment
                    if usage is not None and on_usage is not None:
                        on_usage(usage)
                    if raw is not None:
                        # The generated text as the server sent it, before any
                        # thinking trace is stripped: SSE ids and timestamps
                        # are transport, not output, and are not part of it.
                        self._observe({
                            "event": "response", "attempt": attempt, "raw_sha256": raw.hexdigest(),
                            "completion_tokens": usage.completion_tokens if usage else None,
                            "prompt_tokens": usage.prompt_tokens if usage else None,
                        })
                    return
        except httpx.HTTPError as exc:
            raise InferenceError(f"Request failed: {_describe(exc)}", unavailable=True) from exc

    def _payload(self, model: str, messages: list[dict[str, str]], options: RequestOptions | None) -> dict:
        """The request body, with each option in the field the servers read (module docstring)."""
        payload: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if options is None:
            return payload
        if options.max_tokens is not None:
            payload["max_tokens"] = options.max_tokens
        if options.temperature is not None:
            payload["temperature"] = options.temperature
        if options.stop:
            payload["stop"] = list(options.stop)
        if options.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": options.json_schema},
            }
        if options.reasoning is False:
            if not self._skip_reasoning_effort:
                payload["reasoning_effort"] = "none"
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        elif options.reasoning is True:
            # vLLM and SGLang switch a hybrid model's thinking on this way; Ollama
            # ignores the field and keeps its default.
            payload["chat_template_kwargs"] = {"enable_thinking": True}
        if options.seed is not None:
            payload["seed"] = options.seed
        return payload

    async def close(self) -> None:
        await self._client.aclose()


def wire_summary(payload: dict) -> dict[str, Any]:
    """What an audit may keep of a request body: a hash of all of it, and the
    fields that steer generation - never the messages themselves."""
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    schema = (payload.get("response_format") or {}).get("json_schema", {}).get("schema")
    return {
        "payload_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "resolved": {
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "seed": payload.get("seed"),
            "stop": payload.get("stop"),
            "reasoning_effort": payload.get("reasoning_effort"),
            "chat_template_kwargs": payload.get("chat_template_kwargs"),
            "schema_sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest() if schema is not None else None,
        },
    }


def _parse_sse_line(line: str) -> str | None:
    """Turn one SSE line into a text fragment.

    Returns ``""`` for lines that carry no text (keep-alives, role markers),
    and ``None`` when the server signals the end of the stream.
    """
    return _parse_sse_chunk(line)[0]


def _parse_sse_chunk(line: str) -> tuple[str | None, Usage | None]:
    """One SSE line as (text fragment, token counts if this chunk carried them).

    The fragment follows ``_parse_sse_line``: ``""`` for no text, ``None`` at
    the end. Servers send the counts in a last chunk whose ``choices`` is empty.
    """
    line = line.strip()
    if not line or not line.startswith("data:"):
        return "", None
    data = line[len("data:"):].strip()
    if data == "[DONE]":
        return None, None
    try:
        chunk = json.loads(data)
    except json.JSONDecodeError:
        return "", None
    if not isinstance(chunk, dict):
        return "", None
    error = chunk.get("error")
    if error:
        # A server that fails after the reply has started can only say so in
        # the stream. Keeping the text so far would pass off half a reply
        # (half a JSON object) as the whole of one.
        message = error.get("message") if isinstance(error, dict) else str(error)
        raise InferenceError(f"The server failed partway through its reply: {str(message)[:300]}")
    usage = _usage(chunk.get("usage"))
    choices = chunk.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return "", usage
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return "", usage
    return delta.get("content") or "", usage


def _usage(value: object) -> Usage | None:
    if not isinstance(value, dict):
        return None
    details = value.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    usage = Usage(
        prompt_tokens=_count(value.get("prompt_tokens")),
        completion_tokens=_count(value.get("completion_tokens")),
        cached_prompt_tokens=_count(cached),
    )
    return usage if usage != Usage() else None


def _count(value: object) -> int | None:
    """A token count as the server sent it; 0 is a real count, so only >= 0 is kept."""
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _capabilities(value: object) -> frozenset[str] | None:
    """``{"json_schema": true, "seed": false}`` -> ``{"json_schema"}``; None if not stated."""
    if not isinstance(value, dict):
        return None
    return frozenset(str(k) for k, v in value.items() if v is True)


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _rejects_reasoning_effort(status: int, body: str, payload: dict) -> bool:
    return status in (400, 422) and "reasoning_effort" in payload and "reasoning_effort" in body


def _http_error_message(status: int, body: str) -> str:
    # Servers usually put a helpful message in {"error": {"message": ...}} or {"error": "..."}
    try:
        error = json.loads(body).get("error")
        if isinstance(error, dict):
            error = error.get("message")
        if error:
            return f"Server error {status}: {error}"
    except (ValueError, AttributeError):
        pass
    return f"Server error {status}."


def _describe(exc: httpx.HTTPError) -> str:
    if isinstance(exc, httpx.ConnectError):
        return "connection refused. Is the inference server running?"
    if isinstance(exc, httpx.TimeoutException):
        return "the server took too long to respond."
    return str(exc) or exc.__class__.__name__
