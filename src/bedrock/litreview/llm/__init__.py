"""The model layer. Everything outside this package talks to models through
`make_client` and `ask_json`, never through a concrete client class."""

from __future__ import annotations

import copy
from typing import Any

from bedrock.litreview.llm.cache import CachedClient, CallCache
from bedrock.litreview.llm.client import (
    FakeLLMClient,
    LLMClient,
    LLMError,
    OllamaClient,
    OpenAICompatibleClient,
    parse_json_reply,
)
from bedrock.litreview.llm.prompts import Prompt, load_prompt

__all__ = [
    "FakeLLMClient",
    "LLMClient",
    "LLMError",
    "Prompt",
    "ask_json",
    "load_prompt",
    "make_client",
    "strict_schema",
    "version_of",
]


def make_client(settings, *, cache: bool = True, read_cache: bool = True, fake: LLMClient | None = None) -> LLMClient:
    """Build the client the settings describe, wrapped in the call cache."""
    options = {spec.name: spec.options for spec in settings.models.values()}
    if fake is not None or settings.backend == "fake":
        inner: Any = fake or FakeLLMClient()
    elif settings.backend == "ollama":
        inner = OllamaClient(settings.base_url, model_options=options, seed=settings.seed)
    elif settings.backend == "openai":
        inner = OpenAICompatibleClient(
            settings.base_url,
            model_options=options,
            structured_output=settings.structured_output,
            seed=settings.seed,
        )
    else:
        raise ValueError(f"unknown backend {settings.backend!r}")
    if not cache:
        return inner
    return CachedClient(inner, CallCache(settings.cache_dir), read=read_cache)


def version_of(client: Any, model: str) -> str:
    fn = getattr(client, "version_of", None)
    return fn(model) if fn else getattr(client, "model_version", model)


def strict_schema(schema: dict[str, Any], enums: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    """Make a pydantic JSON schema safe and strict for guided decoding.

    - `$ref`s are inlined, because not every server's grammar compiler
      follows them.
    - Every property becomes required. A property the model may skip is a
      property it will sometimes skip, and "sometimes" is not deterministic
      enough for the client; nullable fields stay nullable.
    - `enums` narrows string fields to a runtime list, by property name —
      e.g. `{"facet": ["mechanism", "clinical"]}` or `{"citations": ["E1", "E2"]}`.
      Guided decoding then cannot produce an id that does not exist.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})
    enums = enums or {}

    def inline(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = defs[node["$ref"].split("/")[-1]]
                return inline(copy.deepcopy(target))
            out = {k: inline(v) for k, v in node.items()}
            if out.get("type") == "object" and "properties" in out:
                out["required"] = list(out["properties"])
                out["additionalProperties"] = False
                for prop, values in enums.items():
                    if prop in out["properties"]:
                        out["properties"][prop] = _narrow(out["properties"][prop], values)
            return out
        if isinstance(node, list):
            return [inline(v) for v in node]
        return node

    return inline(schema)


def _narrow(prop: dict[str, Any], values: list[Any]) -> dict[str, Any]:
    enum_node = {"type": "string", "enum": list(values)}
    if prop.get("type") == "array":
        out = dict(prop)
        out["items"] = enum_node
        out["minItems"] = 1 if values else 0
        return out
    keep = {k: v for k, v in prop.items() if k in ("title", "description")}
    return {**keep, **enum_node}


def ask_json(client: LLMClient, prompt: str, *, model: str, schema: dict[str, Any], max_tokens: int = 2048) -> dict[str, Any]:
    """One structured call: prompt in, parsed JSON out. Always temperature 0."""
    raw = client.complete(prompt, model=model, schema=schema, temperature=0.0, max_tokens=max_tokens)
    return parse_json_reply(raw)
