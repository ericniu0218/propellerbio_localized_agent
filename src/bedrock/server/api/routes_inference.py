"""The inference API: named-model calls for clients that need a model, not an agent.

The terminal app's literature review (``bedrock.litreview``) runs on the
employee's machine and needs the company's models for hundreds of narrow,
structured calls: grade this abstract, extract from this chunk, write this
section. None of that fits ``POST /messages``, which is a conversation: a tier
classifier picks who answers, the answer is reviewed and reformatted, and the
reply is prose. A review needs the opposite - an explicitly named model per
job, ``temperature=0``, a seed, JSON-schema output returned exactly as the
model wrote it, and model discovery with each model's context window.

So this router speaks the part of the OpenAI chat-completions protocol that
bedrock's own client sends (``bedrock.inference.openai_compatible``)::

    GET  /v1/models             the roles served, each with its window and capabilities
    POST /v1/chat/completions   one call to one named role, streamed or not

Models are named by **role** (``runner``, ``specialist``, ``director``), so a
client's configuration survives the server swapping weights; each entry reports
the weights behind it as ``root`` (vLLM's own field for this), and the client
records that on every run.

What it will and will not do:

* Every call goes through the role's ``Scheduler`` from the model hub at
  ``USER_JOB`` - the same queue the agents use for that model server - and out
  through ``OpenAICompatibleBackend``. Nothing here calls a model directly.
* It passes on exactly the settings it can honour and refuses the rest with a
  400 that names them: an unknown parameter, a schema for a model whose engine
  cannot constrain output, a seed for one that ignores seeds, a request for
  "thinking" from a model that has no switch. Nothing is silently dropped.
  Temperature 0 and a seed remove sampling randomness; they do not make a
  batched GPU server bit-for-bit repeatable, and nothing here claims so.
* It never serves fixtures. In mock mode the agents' model is ``MockBackend``,
  which cannot honour a schema; this API always calls the role's configured
  model server, and says the role is unavailable when that server is down.
* Access: with ``BEDROCK_SERVER_INFERENCE_TOKEN`` set on the server, every
  caller must present it as a bearer token (a client sets it as its
  ``api_key``). Without one, a real-mode server refuses every caller (behind a
  reverse proxy on the same host, everyone looks like loopback), and a mock or
  development server serves only callers on this machine. The token
  is compared in constant time and never logged. Each call is logged as
  metadata - role, sizes, status, duration - never its messages.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from bedrock.inference import InferenceError, Priority, RequestOptions, estimate_tokens
from bedrock.server.config import ModelConfig, get_settings, load_models_config
from bedrock.server.models.hub import get_model_hub

log = logging.getLogger("bedrock.server.inference")

router = APIRouter(prefix="/v1", tags=["inference"])

ALLOWED = frozenset({
    "model", "messages", "stream", "stream_options", "max_tokens", "temperature", "seed",
    "response_format", "reasoning_effort", "chat_template_kwargs", "stop",
})
"""The parameters bedrock's client sends. Anything else is refused by name."""

UPSTREAM_CHECK_SECONDS = 30.0
_upstream: dict[str, tuple[float, dict[str, Any]]] = {}


# --------------------------------------------------------------------------- access


def _loopback(request: Request) -> bool:
    host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"


def inference_access(request: Request) -> None:
    settings = get_settings()
    if not settings.inference_api:
        raise HTTPException(404, "The inference API is switched off on this server (inference_api = false).")
    token = settings.inference_token
    if token:
        given = request.headers.get("authorization", "")
        scheme, _, value = given.partition(" ")
        if scheme.lower() != "bearer" or not hmac.compare_digest(value.strip().encode(), token.encode()):
            raise HTTPException(401, "A valid bearer token is required: set the client's api_key "
                                     "(BEDROCK_API_KEY) to the server's BEDROCK_SERVER_INFERENCE_TOKEN.",
                                headers={"WWW-Authenticate": "Bearer"})
        return
    if settings.mode == "real":
        # A real server usually sits behind a reverse proxy on the same host,
        # and then every caller on the network arrives as loopback unless the
        # proxy's forwarded headers are trusted. So in real mode the token is
        # required, whoever is asking.
        raise HTTPException(403, "This server's inference API needs a token in real mode. Set "
                                 "BEDROCK_SERVER_INFERENCE_TOKEN on the server and give clients the same value "
                                 "as their api_key (BEDROCK_API_KEY).")
    if not _loopback(request):
        raise HTTPException(403, "This server serves its inference API only to callers on the same machine, "
                                 "because no token is set. Set BEDROCK_SERVER_INFERENCE_TOKEN on the server and "
                                 "give clients the same value as their api_key (BEDROCK_API_KEY).")


# --------------------------------------------------------------------------- roles


def _roles() -> dict[str, ModelConfig]:
    models = load_models_config()
    served = {}
    for role in get_settings().inference_roles:
        try:
            served[role] = models.for_role(role)
        except AttributeError:
            continue
    return served


def _resolve(name: str) -> tuple[str, ModelConfig]:
    """A role name, or the model name a role is configured with."""
    roles = _roles()
    if name in roles:
        return name, roles[name]
    for role, config in roles.items():
        if config.model_name == name:
            return role, config
    raise HTTPException(404, f"This server does not serve a model called {name!r}. "
                             f"It serves: {', '.join(sorted(roles)) or 'none'} (GET /v1/models).")


async def _upstream_state(role: str, config: ModelConfig) -> dict[str, Any]:
    """Is the role's model server up and serving its model, and with what window?

    Asked of the model server itself, briefly cached. A server that reports a
    smaller window than the configured ``context_length`` wins: the configured
    number is a claim, the served one is what a prompt gets."""
    now = time.monotonic()
    cached = _upstream.get(role)
    if cached and now - cached[0] < UPSTREAM_CHECK_SECONDS:
        return cached[1]
    state: dict[str, Any] = {"ready": False, "window": config.context_length, "reason": ""}
    try:
        offered = {m.id: m for m in await get_model_hub().inference_scheduler_for(role).backend.list_models()}
    except InferenceError as exc:
        state["reason"] = f"its model server is not answering ({exc})"
    else:
        info = offered.get(config.model_name)
        if info is None:
            state["reason"] = f"its model server does not serve {config.model_name!r}"
        else:
            state["ready"] = True
            if info.context_length:
                state["window"] = min(config.context_length, info.context_length)
    _upstream[role] = (now, state)
    return state


@router.get("/models")
async def list_models(_=Depends(inference_access)) -> dict:
    data, unavailable = [], {}
    for role, config in _roles().items():
        state = await _upstream_state(role, config)
        if not state["ready"]:
            unavailable[role] = state["reason"]
            continue
        data.append({
            "id": role,
            "object": "model",
            "owned_by": "bedrock-server",
            "root": config.model_name,
            "max_model_len": state["window"],
            "capabilities": {
                "json_schema": config.supports_json_schema,
                "seed": config.supports_seed,
                "reasoning_switch": config.supports_thinking_mode,
            },
        })
    return {"object": "list", "data": data, "unavailable": unavailable}


# --------------------------------------------------------------------------- calls


def _bad(message: str) -> HTTPException:
    return HTTPException(400, message)


def _options(body: dict, role: str, config: ModelConfig) -> RequestOptions:
    unknown = sorted(set(body) - ALLOWED)
    if unknown:
        raise _bad(f"unsupported parameter(s): {', '.join(unknown)}. This API passes on only "
                   f"{', '.join(sorted(ALLOWED))}, and refuses the rest rather than ignore them.")
    max_tokens = body.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
        raise _bad("max_tokens must be a positive whole number")
    temperature = body.get("temperature")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                                    or not 0 <= temperature <= 2):
        raise _bad("temperature must be a number from 0 to 2")
    seed = body.get("seed")
    if seed is not None:
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise _bad("seed must be a whole number")
        if not config.supports_seed:
            raise _bad(f"{role!r} is served by an engine that ignores seeds (supports_seed: false), so a seed "
                       "would promise repeatability it cannot give. Send the request without one.")
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop is not None and (not isinstance(stop, list) or not all(isinstance(s, str) for s in stop)):
        raise _bad("stop must be a string or a list of strings")

    schema = None
    fmt = body.get("response_format")
    if fmt is not None:
        kind = fmt.get("type") if isinstance(fmt, dict) else None
        if kind == "json_schema":
            schema = (fmt.get("json_schema") or {}).get("schema")
            if not isinstance(schema, dict):
                raise _bad("response_format json_schema needs a schema object")
            if not config.supports_json_schema:
                raise _bad(f"{role!r} is served by an engine that cannot constrain output to a JSON schema "
                           "(supports_json_schema: false). Choose another model.")
        elif kind != "text":
            raise _bad("response_format must be {\"type\": \"json_schema\", ...} or {\"type\": \"text\"}")

    reasoning = None
    kwargs = body.get("chat_template_kwargs")
    if kwargs is not None:
        if not isinstance(kwargs, dict) or set(kwargs) - {"enable_thinking"}:
            raise _bad("chat_template_kwargs may carry only enable_thinking")
        reasoning = bool(kwargs.get("enable_thinking"))
    effort = body.get("reasoning_effort")
    if effort is not None:
        if effort != "none":
            raise _bad("reasoning_effort may only be \"none\" (turn thinking off)")
        reasoning = False
    if reasoning is True and not config.supports_thinking_mode:
        raise _bad(f"{role!r} has no thinking mode to switch on (supports_thinking_mode: false)")
    # "Thinking off" is always passed on. For a model that does not think it
    # changes nothing; for one whose config says it has no switch but does
    # think (qwen3:8b served as the runner, measured live on 29 September),
    # dropping it let the thinking spend the reply budget and cut the JSON off.
    return RequestOptions(max_tokens=max_tokens, temperature=temperature, stop=tuple(stop or ()),
                          json_schema=schema, reasoning=reasoning, seed=seed)


def _messages(body: dict) -> list[dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise _bad("messages must be a non-empty list")
    out = []
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") \
                or not isinstance(m.get("content"), str):
            raise _bad("each message needs a role (system, user or assistant) and string content")
        out.append({"role": m["role"], "content": m["content"]})
    return out


def _usage(usage) -> dict | None:
    if usage is None:
        return None
    body = {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
    if usage.prompt_tokens is not None and usage.completion_tokens is not None:
        body["total_tokens"] = usage.prompt_tokens + usage.completion_tokens
    return {k: v for k, v in body.items() if v is not None}


def _status_for(exc: InferenceError) -> int:
    return 503 if getattr(exc, "unavailable", False) else 502


@router.post("/chat/completions")
async def chat_completions(request: Request, _=Depends(inference_access)):
    try:
        body = await request.json()
    except ValueError:
        raise _bad("the body must be JSON") from None
    if not isinstance(body, dict) or not isinstance(body.get("model"), str):
        raise _bad("model is required: one of the ids from GET /v1/models")
    role, config = _resolve(body["model"])
    options = _options(body, role, config)
    messages = _messages(body)
    state = await _upstream_state(role, config)
    if not state["ready"]:
        raise HTTPException(503, f"{role!r} is unavailable: {state['reason']}.")
    needed = estimate_tokens("\n".join(m["content"] for m in messages)) + (options.max_tokens or 0)
    if needed > state["window"]:
        raise _bad(f"the prompt and a {options.max_tokens or 0}-token reply need about {needed:,} tokens; "
                   f"{role!r} is served with a {state['window']:,}-token window. The request was not sent, "
                   "because the model server would drop the start of the prompt instead of refusing it.")

    call_id = "chatcmpl-" + uuid.uuid4().hex[:24]
    started = time.monotonic()
    reported: list = []
    scheduler = get_model_hub().inference_scheduler_for(role)
    fragments = scheduler.stream(config.model_name, messages, priority=Priority.USER_JOB,
                                 options=options, on_usage=reported.append)

    def audit(status: str, characters: int) -> None:
        # Durable, in the workspace store's audit table beside every other
        # server action; metadata only - never a message, never the token.
        try:
            from bedrock.server.workspace.routes import store

            store().audit("inference-api", f"inference {role} {status} "
                          f"prompt_chars={sum(len(m['content']) for m in messages)} reply_chars={characters} "
                          f"schema={options.json_schema is not None}", call_id)
        except Exception:  # noqa: BLE001 - an audit write must not fail the call
            log.exception("could not record inference call %s", call_id)
        log.info("inference call=%s role=%s model=%s prompt_chars=%d reply_chars=%d schema=%s seed=%s "
                 "status=%s seconds=%.1f", call_id, role, config.model_name,
                 sum(len(m["content"]) for m in messages), characters, options.json_schema is not None,
                 options.seed is not None, status, time.monotonic() - started)

    # The first fragment is awaited before any response starts, so a server
    # that is down or refuses the request gets a real HTTP status rather than
    # a stream that ends early.
    try:
        first = await anext(fragments)
    except StopAsyncIteration:
        first = ""
    except InferenceError as exc:
        await fragments.aclose()
        audit("failed", 0)
        raise HTTPException(_status_for(exc), f"the model server failed: {exc}") from exc

    if not body.get("stream"):
        parts = [first]
        try:
            async for fragment in fragments:
                parts.append(fragment)
        except InferenceError as exc:
            audit("failed", sum(map(len, parts)))
            raise HTTPException(_status_for(exc), f"the model server failed: {exc}") from exc
        finally:
            await fragments.aclose()
        text = "".join(parts)
        audit("ok", len(text))
        return JSONResponse({
            "id": call_id, "object": "chat.completion", "model": role,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            **({"usage": u} if (u := _usage(reported[-1] if reported else None)) else {}),
        })

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    def chunk(payload: dict) -> str:
        return "data: " + json.dumps({"id": call_id, "object": "chat.completion.chunk", "model": role,
                                      **payload}, ensure_ascii=False) + "\n\n"

    async def stream():
        characters = 0
        try:
            if first:
                characters += len(first)
                yield chunk({"choices": [{"index": 0, "delta": {"content": first}}]})
            async for fragment in fragments:
                characters += len(fragment)
                yield chunk({"choices": [{"index": 0, "delta": {"content": fragment}}]})
            if include_usage and (u := _usage(reported[-1] if reported else None)):
                yield chunk({"choices": [], "usage": u})
            yield "data: [DONE]\n\n"
            audit("ok", characters)
        except InferenceError as exc:
            # Mid-stream: the status line has gone, so say it in the stream.
            # bedrock's client raises on an `error` chunk rather than keeping
            # a reply that stopped halfway.
            audit("failed", characters)
            yield "data: " + json.dumps({"error": {"message": f"the model server failed: {exc}",
                                                   "type": "upstream_error"}}) + "\n\n"
        finally:
            await fragments.aclose()

    return StreamingResponse(stream(), media_type="text/event-stream")
