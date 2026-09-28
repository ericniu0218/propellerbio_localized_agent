"""Loads config/models.yaml and config/budgets.yaml. These two files are the
only place model names, endpoints, and token budgets should ever be defined
— see docs/ARCHITECTURE.md section 3."""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

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
    defaults: dict = {}

    def for_role(self, role: str) -> ModelConfig:
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


@lru_cache
def load_models_config() -> ModelsConfig:
    raw = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())
    return ModelsConfig.model_validate(_expand_env(raw))


@lru_cache
def load_budgets_config() -> BudgetsConfig:
    raw = yaml.safe_load((CONFIG_DIR / "budgets.yaml").read_text())
    return BudgetsConfig.model_validate(raw)


class Settings(BaseModel):
    mode: str = os.environ.get("AGENT_PACK_MODE", "mock")  # "mock" | "real"
    database_url: str = os.environ.get(
        "DATABASE_URL", "sqlite:///./agent_pack.db"
    )
    redis_url: str | None = os.environ.get("REDIS_URL")
    airgap_check_enabled: bool = os.environ.get("AIRGAP_CHECK", "true").lower() == "true"
    # Mock mode responds instantly by default (good for tests). Set this
    # above 0 for a demo/screenshot run so the agent panel is actually
    # visible for a moment instead of flashing past — see
    # MockModelClient's token_delay_s.
    mock_token_delay_s: float = float(os.environ.get("MOCK_TOKEN_DELAY_S", "0"))

    # SSO (Workstream 2 / docs/WORKFLOW.md section 1). Login is disabled
    # (falls back to the old "type any name" behavior) unless all of these
    # are set — that's what lets mock-mode demos and tests keep working
    # without standing up Keycloak.
    keycloak_base_url: str | None = os.environ.get("KEYCLOAK_BASE_URL")
    keycloak_realm: str | None = os.environ.get("KEYCLOAK_REALM")
    keycloak_client_id: str | None = os.environ.get("KEYCLOAK_CLIENT_ID")
    keycloak_client_secret: str | None = os.environ.get("KEYCLOAK_CLIENT_SECRET")
    # Where Keycloak should send the browser back to after login. Must
    # exactly match a Redirect URI configured on the Keycloak client.
    app_base_url: str = os.environ.get("APP_BASE_URL", "http://localhost:8000")

    @property
    def sso_enabled(self) -> bool:
        return bool(
            self.keycloak_base_url and self.keycloak_realm
            and self.keycloak_client_id and self.keycloak_client_secret
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
