"""Hands each agent role its model client and config.

Agents ask the registry; they never construct a client themselves. Every client
is a ``SchedulerModelClient`` from the ``ModelHub``, so a call waits its turn in
bedrock's ``Scheduler`` for its model server - in mock mode as much as in real
mode (see ``hub.py`` and ``mock_backend.py``).
"""

from __future__ import annotations

from functools import lru_cache

from bedrock.server.config import load_models_config
from bedrock.server.models.client import ModelClient
from bedrock.server.models.hub import get_model_hub


class ModelRegistry:
    def __init__(self) -> None:
        self._models = load_models_config()

    def client_for(self, role: str) -> ModelClient:
        return get_model_hub().client_for(role)

    def config_for(self, role: str):
        return self._models.for_role(role)


@lru_cache
def get_model_registry() -> ModelRegistry:
    return ModelRegistry()
