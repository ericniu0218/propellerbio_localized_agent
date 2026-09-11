"""Human approval gates: any tool that writes to a company system, or any
task flagged high-stakes, blocks until a human approves/rejects via the
REST endpoint. See docs/ARCHITECTURE.md section 8."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from app.schemas.enums import ApprovalDecision


class ApprovalGate:
    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}

    def open(self, task_id: str) -> asyncio.Future[ApprovalDecision]:
        future: asyncio.Future[ApprovalDecision] = asyncio.get_event_loop().create_future()
        self._pending[task_id] = future
        return future

    async def wait_for_decision(self, task_id: str) -> ApprovalDecision:
        future = self._pending.get(task_id) or self.open(task_id)
        return await future

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
