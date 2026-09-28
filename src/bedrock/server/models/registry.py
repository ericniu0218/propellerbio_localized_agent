"""Wires config/models.yaml to either HTTPModelClient (real) or
MockModelClient (mock), keyed by agent role. Agents ask the registry for
their client; they never construct one directly."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings, load_models_config
from app.models.client import ModelClient
from app.models.http_client import HTTPModelClient
from app.models.mock_client import MockModelClient


class ModelRegistry:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._models = load_models_config()
        self._clients: dict[str, ModelClient] = {}

    def client_for(self, role: str) -> ModelClient:
        if role not in self._clients:
            config = self._models.for_role(role)
            if self._settings.mode == "real":
                self._clients[role] = HTTPModelClient(config)
            else:
                self._clients[role] = MockModelClient(
                    config.model_name, token_delay_s=self._settings.mock_token_delay_s
                )
        return self._clients[role]

    def config_for(self, role: str):
        return self._models.for_role(role)


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
