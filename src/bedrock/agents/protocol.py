"""The frozen interface implemented by every agent capability.

Capabilities are the only agent code allowed to touch the outside world.  The
kernel depends on this small structural interface instead of importing any
particular adapter, so new capabilities can be added without changing the
planner or supervisor.
"""

from __future__ import annotations

from typing import Protocol

from bedrock.agents.contract import Outcome, Risk


class Capability(Protocol):
    """A named operation that can describe itself before it is allowed to run."""

    name: str
    risk: Risk

    def describe(self, arguments: dict[str, str]) -> str:
        """Return the exact, human-readable action proposed by ``arguments``."""

        ...

    async def run(self, arguments: dict[str, str]) -> Outcome:
        """Perform the operation after the supervisor has validated a grant."""

        ...
