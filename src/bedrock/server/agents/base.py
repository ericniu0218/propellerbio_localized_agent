"""Shared machinery for Director/Specialist/Runner: model calls with
token-usage tracking + budget enforcement, event emission for the frontend
panel, and the static-prefix-first prompt convention that makes serving-
engine prefix caching effective (docs/ARCHITECTURE.md section 6).
"""

from __future__ import annotations

import time
from uuid import uuid4

from bedrock.server.config import BudgetsConfig, load_budgets_config, load_models_config
from bedrock.server.db import Database, get_database
from bedrock.server.events import EventBus, get_event_bus
from bedrock.server.models.client import ChatMessage, ModelClient
from bedrock.server.models.registry import get_model_registry
from bedrock.server.schemas.agents import AgentCard
from bedrock.server.schemas.enums import AgentRole, AgentStatus, EventType
from bedrock.server.schemas.events import (
    AgentStartedPayload,
    AgentStatusPayload,
    AgentTokenPayload,
    Event,
    UsageUpdatedPayload,
)


class BudgetExceededError(Exception):
    def __init__(self, role: AgentRole, task_id: str, used: int, budget: int) -> None:
        self.role = role
        self.task_id = task_id
        self.used = used
        self.budget = budget
        super().__init__(
            f"{role.value} exceeded its per-task token budget on {task_id}: {used} > {budget}"
        )


class BaseAgent:
    role: AgentRole
    static_prompt_prefix: str = ""  # unchanging part, first, for prefix caching

    def __init__(
        self,
        *,
        session_id: str,
        model_client: ModelClient | None = None,
        event_bus: EventBus | None = None,
        db: Database | None = None,
        budgets: BudgetsConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.agent_id = f"{self.role.value}-{uuid4().hex[:6]}"
        registry = get_model_registry()
        self._client = model_client or registry.client_for(self.role.value)
        self._model_name = registry.config_for(self.role.value).model_name
        self._bus = event_bus or get_event_bus()
        self._db = db or get_database()
        self._budgets = budgets or load_budgets_config()
        self._started_at = time.monotonic()

    # -- prompt construction --------------------------------------------------
    def build_prompt(self, dynamic_context: str) -> list[ChatMessage]:
        """Static prefix first (cacheable), variable content after — see
        docs/ARCHITECTURE.md section 6."""
        return [
            ChatMessage(role="system", content=self.static_prompt_prefix),
            ChatMessage(role="user", content=dynamic_context),
        ]

    # -- events -----------------------------------------------------------------
    async def emit(self, event_type: EventType, payload: dict) -> None:
        await self._bus.publish(
            Event(type=event_type, session_id=self.session_id, payload=payload)
        )

    async def emit_started(self, task_id: str | None, task_title: str | None) -> None:
        await self.emit(
            EventType.AGENT_STARTED,
            AgentStartedPayload(
                agent=AgentCard(
                    agent_id=self.agent_id,
                    name=self.role.value.capitalize(),
                    role=self.role,
                    model_name=self._model_name,
                    status=AgentStatus.THINKING,
                    current_task_id=task_id,
                    current_task_title=task_title,
                )
            ).model_dump(),
        )

    async def emit_status(
        self,
        status: AgentStatus,
        task_id: str | None = None,
        task_title: str | None = None,
        tokens_used: int = 0,
    ) -> None:
        elapsed_ms = int((time.monotonic() - self._started_at) * 1000)
        await self.emit(
            EventType.AGENT_STATUS,
            AgentStatusPayload(
                agent_id=self.agent_id,
                role=self.role,
                status=status,
                task_id=task_id,
                task_title=task_title,
                tokens_used=tokens_used,
                elapsed_ms=elapsed_ms,
            ).model_dump(),
        )

    # -- model calls with streaming + budget enforcement -------------------------
    async def call_model(
        self,
        messages: list[ChatMessage],
        *,
        task_id: str | None,
        thinking: bool = False,
        max_tokens: int | None = None,
    ) -> str:
        budget = self._budgets.for_role(self.role.value)
        max_tokens = max_tokens or budget.max_tokens_per_call

        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        async for chunk in self._client.stream(
            messages, max_tokens=max_tokens, thinking=thinking
        ):
            if chunk.text:
                text_parts.append(chunk.text)
                await self.emit(
                    EventType.AGENT_TOKEN,
                    AgentTokenPayload(
                        agent_id=self.agent_id,
                        role=self.role,
                        task_id=task_id,
                        text=chunk.text,
                        done=False,
                    ).model_dump(),
                )
            if chunk.done and chunk.final:
                prompt_tokens = chunk.final.prompt_tokens
                completion_tokens = chunk.final.completion_tokens

        await self.emit(
            EventType.AGENT_TOKEN,
            AgentTokenPayload(
                agent_id=self.agent_id, role=self.role, task_id=task_id, text="", done=True
            ).model_dump(),
        )

        self._db.record_usage(
            session_id=self.session_id,
            task_id=task_id,
            agent_id=self.agent_id,
            role=self.role,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_for_task = self._db.total_tokens_for_task(task_id) if task_id else (
            prompt_tokens + completion_tokens
        )
        await self.emit(
            EventType.USAGE_UPDATED,
            UsageUpdatedPayload(
                session_id=self.session_id,
                agent_id=self.agent_id,
                role=self.role,
                task_id=task_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens_for_task=total_for_task,
                budget_for_task=budget.max_tokens_per_task,
            ).model_dump(),
        )

        if task_id and total_for_task > budget.max_tokens_per_task:
            raise BudgetExceededError(self.role, task_id, total_for_task, budget.max_tokens_per_task)

        return "".join(text_parts)
