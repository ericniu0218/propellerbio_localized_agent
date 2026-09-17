"""A no-I/O capability used to exercise the agent kernel in tests."""

from __future__ import annotations

import asyncio

from bedrock.agents.contract import Outcome, Risk


class EchoCapability:
    """Describe and return text without reading, writing, or sending anything."""

    name = "echo"
    risk = Risk.READ_ONLY

    def describe(self, arguments: dict[str, str]) -> str:
        message = arguments.get("message", "")
        return f"Echo {message!r} locally."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        delay_text = arguments.get("delay", "0")
        try:
            delay = float(delay_text)
        except ValueError as exc:
            raise ValueError("echo delay must be a number") from exc
        if delay < 0:
            raise ValueError("echo delay must not be negative")
        if delay:
            await asyncio.sleep(delay)
        return Outcome(plan_id="", completed=(self.describe(arguments),))


CAPABILITY = EchoCapability()
