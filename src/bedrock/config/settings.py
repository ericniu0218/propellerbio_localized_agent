"""Application settings.

Where settings come from, in order of priority (highest wins):

1. Environment variables (``BEDROCK_API_BASE``, ``BEDROCK_MODEL``, ``BEDROCK_API_KEY``)
2. A config file at ``~/.config/bedrock/config.toml``
3. Built-in defaults, which point at a local Ollama server.

Why this design: employees must never have an inference endpoint hard-coded into
the client. In development the endpoint is a local Ollama instance; in production
IT sets ``BEDROCK_API_BASE`` (or ships a config file) pointing at the company's
private inference server. The rest of the app never needs to change.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "bedrock" / "config.toml"


@dataclass(frozen=True)
class Settings:
    """Everything the client needs to know to talk to an inference server."""

    api_base: str = DEFAULT_API_BASE
    """Base URL of an OpenAI-compatible API, e.g. ``http://localhost:11434/v1``."""

    model: str | None = None
    """Model to use at startup. ``None`` means "ask the server and pick the first"."""

    api_key: str | None = None
    """Optional bearer token. Ollama ignores it; vLLM/SGLang can require one."""

    request_timeout: float = 120.0
    """Seconds to wait for a response before giving up."""

    system_prompt: str | None = None
    """Optional instruction sent at the start of every conversation."""

    extra: dict = field(default_factory=dict)
    """Any unrecognised keys from the config file, kept for forward compatibility."""


def _read_config_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_settings(config_path: Path | None = None, env: dict[str, str] | None = None) -> Settings:
    """Build a :class:`Settings` from defaults, config file, then environment.

    Both arguments exist so tests can supply their own values instead of
    touching the real home directory or process environment.
    """
    env = os.environ if env is None else env
    file_values = _read_config_file(config_path or DEFAULT_CONFIG_PATH)

    known = {"api_base", "model", "api_key", "request_timeout", "system_prompt"}
    extra = {k: v for k, v in file_values.items() if k not in known}

    api_base = env.get("BEDROCK_API_BASE") or file_values.get("api_base") or DEFAULT_API_BASE
    model = env.get("BEDROCK_MODEL") or file_values.get("model") or None
    api_key = env.get("BEDROCK_API_KEY") or file_values.get("api_key") or None
    timeout = float(env.get("BEDROCK_TIMEOUT") or file_values.get("request_timeout") or 120.0)
    system_prompt = file_values.get("system_prompt") or None

    return Settings(
        api_base=api_base.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=timeout,
        system_prompt=system_prompt,
        extra=extra,
    )
