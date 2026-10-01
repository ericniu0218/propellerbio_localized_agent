"""Settings for the agent-pack server (``bedrock.server``).

Two kinds of configuration live here:

* ``models.yaml`` and ``budgets.yaml`` - the only place model names, endpoints
  and token budgets are defined (docs/server/ARCHITECTURE.md section 3). They
  ship as package data in ``bedrock/server/defaults/``. To change them on one
  server, put a whole replacement file in the config directory
  (``BEDROCK_SERVER_CONFIG_DIR``, default ``~/.config/bedrock/server``) - the
  same arrangement as the literature review's ``models.yaml``.
* ``Settings`` - the server's switches. Each is read, in order, from its
  environment variable (the agent pack's names, unchanged, so existing
  deployments keep working), then the ``[server]`` table of bedrock's
  ``config.toml``, then a default.

Everything is read when first asked for, never at import: importing a module
must not decide where the database lives, or tests and tools that import the
server could not point it somewhere else first.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Mapping

import yaml
from pydantic import BaseModel

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-(.*?))?\}")


def _expand_env(value):
    if isinstance(value, str):
        def repl(m: re.Match) -> str:
            env_name, default = m.group(1), m.group(2)
            return os.environ.get(env_name, default if default is not None else "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class ModelFallback(BaseModel):
    note: str | None = None
    model_name: str | None = None
    context_length: int | None = None


class ModelConfig(BaseModel):
    role: str
    model_name: str
    endpoint: str
    served_via: str = "vllm"
    precision: str = "fp8"
    context_length: int = 32768
    supports_thinking_mode: bool = False
    thinking_mode_default: bool = False
    api_key_env: str | None = None
    # How many of this role's calls may run at once on its model server. Roles
    # that share one server share one queue, which allows the largest of their
    # values; vLLM batches concurrent requests, so this is well above 1.
    max_parallel: int = 8
    # What the serving engine can do for this model, as the inference API
    # (api/routes_inference.py) reports it and enforces it. vLLM and Ollama do
    # both; set false for an engine that silently ignores them, and the API
    # refuses such a request instead of passing it on.
    supports_json_schema: bool = True
    supports_seed: bool = True
    fallback: ModelFallback | None = None
    alternatives: list[str] = []

    @property
    def api_key(self) -> str | None:
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


class ModelsConfig(BaseModel):
    director: ModelConfig
    specialist: ModelConfig
    runner: ModelConfig
    # Meeting notes (bedrock.server.meetings.pipeline). Optional so an older
    # replacement models.yaml without it still loads; then it comes from the
    # recorder's own variables, which is where it came from before.
    notes: ModelConfig | None = None
    defaults: dict = {}

    def for_role(self, role: str) -> ModelConfig:
        if role == "notes" and self.notes is None:
            return ModelConfig(
                role="notes",
                model_name=os.environ.get("MEETING_SUMMARY_MODEL", "llama3.1:8b"),
                endpoint=os.environ.get("MEETING_SUMMARY_ENDPOINT", "http://localhost:11434/v1"),
                context_length=int(os.environ.get("MEETING_SUMMARY_CONTEXT", "8192")),
            )
        return getattr(self, role)


class TokenBudget(BaseModel):
    max_tokens_per_task: int
    max_tokens_per_call: int


class BudgetsConfig(BaseModel):
    token_budgets: dict[str, TokenBudget]
    summarization: dict
    context_compaction: dict
    rag: dict
    meeting_transcripts: dict
    caching: dict
    retries: dict

    def for_role(self, role: str) -> TokenBudget:
        return self.token_budgets[role]


def config_dir(env: Mapping[str, str] | None = None) -> Path:
    """Where a server's own replacement ``models.yaml``/``budgets.yaml`` live."""
    env = os.environ if env is None else env
    table = _server_table(env)
    value = env.get("BEDROCK_SERVER_CONFIG_DIR") or table.get("config_dir")
    return Path(value).expanduser() if value else Path.home() / ".config" / "bedrock" / "server"


def config_file(name: str) -> Path:
    """The server's own copy of ``name`` if it has one, else the packaged default."""
    override = config_dir() / name
    if override.is_file():
        return override
    return Path(str(resources.files("bedrock.server") / "defaults" / name))


@lru_cache
def load_models_config() -> ModelsConfig:
    raw = yaml.safe_load(config_file("models.yaml").read_text())
    return ModelsConfig.model_validate(_expand_env(raw))


@lru_cache
def load_budgets_config() -> BudgetsConfig:
    raw = yaml.safe_load(config_file("budgets.yaml").read_text())
    return BudgetsConfig.model_validate(raw)


def _server_table(env: Mapping[str, str]) -> dict:
    """The ``[server]`` table of bedrock's config file, or nothing.

    ``BEDROCK_SERVER_CONFIG_FILE`` names a different file; set it to an empty
    string to read none (the test suite does, so a developer's own config file
    never reaches it).
    """
    from bedrock.config.settings import DEFAULT_CONFIG_PATH, _read_config_file

    path = env.get("BEDROCK_SERVER_CONFIG_FILE", str(DEFAULT_CONFIG_PATH))
    if not path:
        return {}
    return dict(_read_config_file(Path(path).expanduser()).get("server") or {})


def default_data_dir(env: Mapping[str, str]) -> Path:
    xdg = env.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "bedrock" / "server"


def _flag(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    mode: str = "mock"  # "mock" | "real"
    data_dir: Path
    """Where the server keeps its database, workspace and recordings when their
    own variables are not set. Created 0700."""
    database_url: str
    redis_url: str | None = None
    airgap_check_enabled: bool = True
    # Mock mode responds instantly by default (good for tests). Set this
    # above 0 for a demo/screenshot run so the agent panel is actually
    # visible for a moment instead of flashing past.
    mock_token_delay_s: float = 0.0

    # SSO (docs/server/WORKFLOW.md section 1). Login is disabled (falls back to
    # the plain development sign-in) unless all four are set - that's what lets
    # mock-mode demos and tests keep working without standing up Keycloak.
    keycloak_base_url: str | None = None
    keycloak_realm: str | None = None
    keycloak_client_id: str | None = None
    keycloak_client_secret: str | None = None
    # Where Keycloak should send the browser back to after login. Must
    # exactly match a Redirect URI configured on the Keycloak client.
    app_base_url: str = "http://localhost:8000"
    host: str = "127.0.0.1"
    port: int = 8000
    # The company's own documents for rag_search, and a local literature mirror
    # for literature_search. Unset in real mode means none: the packaged
    # fictional samples are only ever read by a mock server.
    knowledge_dir: Path | None = None
    literature_dir: Path | None = None
    # [server.connectors.<name>] tables, the same shape as the terminal app's
    # [connectors.<name>] (bedrock.connectors): systems of record eln_lims reads.
    connectors: dict = {}
    # The inference API (api/routes_inference.py): the roles it serves, and
    # the bearer token a client must present. With no token, only callers on
    # this machine (loopback) are served - a client elsewhere on the network
    # would otherwise reach the models with no Director review, no redaction
    # and no audit of who asked.
    inference_api: bool = True
    inference_roles: tuple[str, ...] = ("runner", "specialist", "director")
    inference_token: str | None = None

    @property
    def sso_enabled(self) -> bool:
        return bool(
            self.keycloak_base_url and self.keycloak_realm
            and self.keycloak_client_id and self.keycloak_client_secret
        )

    @classmethod
    def load(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Environment, then the ``[server]`` table, then defaults."""
        env = os.environ if env is None else env
        table = _server_table(env)

        def pick(env_names: tuple[str, ...], key: str, default=None):
            for name in env_names:
                if env.get(name):
                    return env[name]
            return table.get(key, default)

        data_value = pick(("BEDROCK_SERVER_DATA_DIR",), "data_dir")
        data_dir = Path(data_value).expanduser() if data_value else default_data_dir(env)
        return cls(
            mode=pick(("AGENT_PACK_MODE", "BEDROCK_SERVER_MODE"), "mode", "mock"),
            data_dir=data_dir,
            database_url=pick(("DATABASE_URL",), "database_url") or f"sqlite:///{data_dir / 'agent_pack.db'}",
            redis_url=pick(("REDIS_URL",), "redis_url"),
            airgap_check_enabled=_flag(pick(("AIRGAP_CHECK",), "airgap_check", "true")),
            mock_token_delay_s=float(pick(("MOCK_TOKEN_DELAY_S",), "mock_token_delay_s", 0) or 0),
            keycloak_base_url=pick(("KEYCLOAK_BASE_URL",), "keycloak_base_url"),
            keycloak_realm=pick(("KEYCLOAK_REALM",), "keycloak_realm"),
            keycloak_client_id=pick(("KEYCLOAK_CLIENT_ID",), "keycloak_client_id"),
            keycloak_client_secret=pick(("KEYCLOAK_CLIENT_SECRET",), "keycloak_client_secret"),
            app_base_url=pick(("APP_BASE_URL",), "app_base_url", "http://localhost:8000"),
            host=pick(("BEDROCK_SERVER_HOST",), "host", "127.0.0.1"),
            port=int(pick(("BEDROCK_SERVER_PORT",), "port", 8000)),
            knowledge_dir=_path(pick(("BEDROCK_SERVER_KNOWLEDGE_DIR",), "knowledge_dir")),
            literature_dir=_path(pick(("BEDROCK_SERVER_LITERATURE_DIR",), "literature_dir")),
            connectors=dict(table.get("connectors") or {}),
            inference_api=_flag(pick(("BEDROCK_SERVER_INFERENCE_API",), "inference_api", "true")),
            inference_roles=_roles(pick(("BEDROCK_SERVER_INFERENCE_ROLES",), "inference_roles")),
            # A secret: environment only, never the config file.
            inference_token=env.get("BEDROCK_SERVER_INFERENCE_TOKEN") or None,
        )


def _roles(value) -> tuple[str, ...]:
    """Roles the inference API serves: a list, or a comma-separated string."""
    if value in (None, ""):
        return ("runner", "specialist", "director")
    items = value.split(",") if isinstance(value, str) else list(value)
    roles = tuple(dict.fromkeys(str(r).strip() for r in items if str(r).strip()))
    unknown = [r for r in roles if r not in ("runner", "specialist", "director", "notes")]
    if unknown:
        raise ValueError(f"inference_roles: unknown role(s) {', '.join(unknown)}")
    return roles


def _path(value) -> Path | None:
    return Path(value).expanduser() if value else None


def private_dir(path: Path) -> Path:
    """Create ``path`` (and parents) readable only by this account."""
    path.mkdir(parents=True, exist_ok=True)
    try:
        path.chmod(0o700)
    except OSError:  # a filesystem without POSIX modes keeps its own rules
        pass
    return path


@lru_cache
def get_settings() -> Settings:
    return Settings.load()
