"""A task that exceeds its agent's per-task token budget must stop and
report to the Director rather than continuing to spend tokens —
docs/ARCHITECTURE.md section 6 and config/budgets.yaml.
"""

import pytest

from bedrock.server.agents.base import BudgetExceededError
from bedrock.server.agents.runner import Runner
from bedrock.server.config import load_budgets_config
from bedrock.server.db import get_database
from bedrock.server.models.client import ChatMessage, CompletionResult, StreamChunk
from bedrock.server.task_store.store import get_task_store


class _OversizedResponseClient:
    """Always reports a completion far larger than any configured budget."""

    def __init__(self, completion_tokens: int) -> None:
        self._completion_tokens = completion_tokens

    async def complete(self, messages, *, max_tokens, thinking=False, temperature=0.2):
        return CompletionResult(
            text="x", prompt_tokens=10, completion_tokens=self._completion_tokens, model_name="fake"
        )

    async def stream(self, messages, *, max_tokens, thinking=False, temperature=0.2):
        yield StreamChunk(text="x", done=False)
        yield StreamChunk(
            text="",
            done=True,
            final=CompletionResult(
                text="x", prompt_tokens=10, completion_tokens=self._completion_tokens, model_name="fake"
            ),
        )


async def test_call_exceeding_budget_raises_and_still_records_usage(session_id):
    budget = load_budgets_config().for_role("runner")
    oversized = budget.max_tokens_per_task + 1000

    task_id = get_task_store().new_task_id()
    runner = Runner(session_id=session_id, model_client=_OversizedResponseClient(oversized))

    with pytest.raises(BudgetExceededError) as exc_info:
        await runner.call_model(
            [ChatMessage(role="user", content="hello")], task_id=task_id
        )

    assert exc_info.value.task_id == task_id
    assert exc_info.value.budget == budget.max_tokens_per_task

    # Usage should still be recorded even though the budget was exceeded —
    # the point is to stop *further* spending, not to hide what was spent.
    total = get_database().total_tokens_for_task(task_id)
    assert total > budget.max_tokens_per_task


async def test_call_within_budget_does_not_raise(session_id):
    task_id = get_task_store().new_task_id()
    runner = Runner(session_id=session_id, model_client=_OversizedResponseClient(10))

    result_text = await runner.call_model(
        [ChatMessage(role="user", content="hello")], task_id=task_id
    )
    assert result_text  # no exception raised
