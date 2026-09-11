"""Application settings.

Where settings come from, in order of priority (highest wins):

1. Environment variables (``BEDROCK_API_BASE``, ``BEDROCK_MODEL``, ``BEDROCK_API_KEY``, ...)
2. A config file at ``~/.config/bedrock/config.toml``
3. Built-in defaults, which point at a local Ollama server.

Why this design: employees must never have an inference endpoint hard-coded into
the client. In development the endpoint is a local Ollama instance; in production
IT sets ``BEDROCK_API_BASE`` (or ships a config file) pointing at the company's
private inference server. The rest of the app never needs to change.

``backend`` picks which of two very different things this client talks to:

- ``"openai"`` (default, unchanged from the original design): a single
  OpenAI-compatible model server (Ollama/vLLM/SGLang) — one message in, one
  streamed reply out. See ``inference/``.
- ``"agentpack"``: the biotech agent-pack backend — real sessions, multiple
  agents working a request, and human-approval gates. See ``agentpack/``
  and ``agentpack_app.py``. This is what IT points at the company's actual
  AI system; ``agent_pack_api_base`` is that system's URL, and ``user_id``
  identifies the employee until real SSO login is wired in (see
  docs/DEPLOYMENT.md in the agent-pack repo for the plan there).
"""

from __future__ import annotations

import getpass
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_API_BASE = "http://localhost:11434/v1"
DEFAULT_AGENT_PACK_API_BASE = "http://localhost:8000"
DEFAULT_CONFIG_PATH = Path.home() / ".config" / "bedrock" / "config.toml"


@dataclass(frozen=True)
class Settings:
    """Everything the client needs to know to talk to an inference server."""

    backend: str = "openai"
    """"openai" (a single model server) or "agentpack" (the full agent system)."""

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

    agent_pack_api_base: str = DEFAULT_AGENT_PACK_API_BASE
    """Base URL of the agent-pack backend, used when backend="agentpack"."""

    user_id: str = field(default_factory=getpass.getuser)
    """Identifies the employee to the agent-pack backend. A stand-in until
    real SSO login is wired into this client — see the note above."""

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

    known = {
        "backend", "api_base", "model", "api_key", "request_timeout",
        "system_prompt", "agent_pack_api_base", "user_id",
    }
    extra = {k: v for k, v in file_values.items() if k not in known}

    backend = env.get("BEDROCK_BACKEND") or file_values.get("backend") or "openai"
    api_base = env.get("BEDROCK_API_BASE") or file_values.get("api_base") or DEFAULT_API_BASE
    model = env.get("BEDROCK_MODEL") or file_values.get("model") or None
    api_key = env.get("BEDROCK_API_KEY") or file_values.get("api_key") or None
    timeout = float(env.get("BEDROCK_TIMEOUT") or file_values.get("request_timeout") or 120.0)
    system_prompt = file_values.get("system_prompt") or None
    agent_pack_api_base = (
        env.get("AGENT_PACK_API_BASE")
        or file_values.get("agent_pack_api_base")
        or DEFAULT_AGENT_PACK_API_BASE
    )
    user_id = env.get("BEDROCK_USER") or file_values.get("user_id") or getpass.getuser()

    return Settings(
        backend=backend,
        api_base=api_base.rstrip("/"),
        model=model,
        api_key=api_key,
        request_timeout=timeout,
        system_prompt=system_prompt,
        agent_pack_api_base=agent_pack_api_base.rstrip("/"),
        user_id=user_id,
        extra=extra,
    )
