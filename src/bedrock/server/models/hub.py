"""One ``Scheduler`` per model server, shared by everything that uses it.

The Director, Specialist and Runner may each have their own model server
(``models.yaml``), or share one - on a development machine all three and the
meeting notes are usually one Ollama. Whatever the layout, calls to *one* server
wait in *one* queue: that is what stops a burst of meeting notes starving
everyone's chat on the same GPU, and what lets the server's parallel-call limit
mean anything. So schedulers are keyed by (endpoint, API key), not by role.

Each scheduler runs bedrock's ``server`` machine profile (concurrent calls,
background work allowed alongside), with two values taken from the roles that
share it: the smallest ``context_length`` (the window callers such as the
meeting summariser size their prompts against) and the largest
``max_parallel``.

Priorities: agent calls answer a person who is waiting, so they are ``CHAT``;
meeting notes are ``USER_JOB``. Nothing on the server is ``BACKGROUND``.

Mock mode (``AGENT_PACK_MODE=mock``) puts ``MockBackend`` where the *agents'*
model server would be; everything else is identical. Meeting notes are not
agents: they always go to their configured server, as they did before the
merge - ``scripts/server/demo-recorder.sh`` runs fixture agents next to a real
local model for the notes. ``aclose`` closes every backend's connections
(lifespan shutdown).
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from bedrock.config import PROFILES
from bedrock.inference import Priority, Scheduler
from bedrock.server.config import ModelConfig, ModelsConfig, Settings, get_settings, load_models_config
from bedrock.server.models.mock_backend import MockBackend
from bedrock.server.models.scheduled import SchedulerModelClient

AGENT_ROLES = ("director", "specialist", "runner")
ROLES = AGENT_ROLES + ("notes",)
MOCK = ("mock://agents", None)


def _key(config: ModelConfig) -> tuple[str, str | None]:
    return config.endpoint.rstrip("/"), config.api_key


class ModelHub:
    def __init__(self, models: ModelsConfig, settings: Settings) -> None:
        self._models = models
        self._settings = settings
        self._schedulers: dict[tuple[str, str | None], Scheduler] = {}

    def _key_for(self, role: str) -> tuple[str, str | None]:
        if role in AGENT_ROLES and self._settings.mode != "real":
            return MOCK
        return _key(self._models.for_role(role))

    def scheduler_for(self, role: str) -> Scheduler:
        key = self._key_for(role)
        if key not in self._schedulers:
            sharing = [self._models.for_role(r) for r in ROLES if self._key_for(r) == key]
            profile = replace(
                PROFILES["server"],
                name=f"server:{key[0]}",
                context_window=min(c.context_length for c in sharing),
                max_parallel=max(c.max_parallel for c in sharing),
            )
            self._schedulers[key] = Scheduler(self._backend(key, sharing), profile)
        return self._schedulers[key]

    def client_for(self, role: str, *, priority: Priority = Priority.CHAT) -> SchedulerModelClient:
        return SchedulerModelClient(self.scheduler_for(role), self._models.for_role(role), priority=priority)

    def _backend(self, key: tuple[str, str | None], sharing: list[ModelConfig]):
        if key == MOCK:
            return MockBackend([c.model_name for c in sharing], token_delay_s=self._settings.mock_token_delay_s)
        from bedrock.inference.openai_compatible import OpenAICompatibleBackend

        timeout = float(self._models.defaults.get("request_timeout_s", 120))
        return OpenAICompatibleBackend(base_url=key[0], api_key=key[1], timeout=timeout)

    async def aclose(self) -> None:
        schedulers, self._schedulers = list(self._schedulers.values()), {}
        for scheduler in schedulers:
            await scheduler.backend.close()


@lru_cache
def get_model_hub() -> ModelHub:
    return ModelHub(load_models_config(), get_settings())
