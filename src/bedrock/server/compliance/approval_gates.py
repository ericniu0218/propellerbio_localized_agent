"""Human approval gates: any tool that writes to a company system, or any
task flagged high-stakes, blocks until a human approves/rejects via the
REST endpoint. See docs/server/ARCHITECTURE.md section 8."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from bedrock.server.schemas.enums import ApprovalDecision


class ApprovalGate:
    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}

    def open(self, task_id: str) -> asyncio.Future[ApprovalDecision]:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_event_loop().create_future()
        self._pending[task_id] = future
        return future

    async def wait_for_decision(self, task_id: str) -> ApprovalDecision:
        future = self._pending.get(task_id) or self.open(task_id)
        try:
            return await future
        finally:
            # Decided, or the run waiting was cancelled: either way nothing is
            # pending any more, and the entry must not outlive its run.
            if self._pending.get(task_id) is future:
                del self._pending[task_id]

    def discard(self, task_id: str) -> None:
        """Nothing will wait for this decision any more (the run ended or was
        cancelled, possibly before it began waiting)."""
        future = self._pending.pop(task_id, None)
        if future is not None and not future.done():
            future.cancel()

    def resolve(self, task_id: str, decision: ApprovalDecision) -> bool:
        future = self._pending.get(task_id)
        if future is None or future.done():
            return False
        future.set_result(decision)
        return True

    def is_pending(self, task_id: str) -> bool:
        future = self._pending.get(task_id)
        return future is not None and not future.done()


@lru_cache
def get_approval_gate() -> ApprovalGate:
    return ApprovalGate()
