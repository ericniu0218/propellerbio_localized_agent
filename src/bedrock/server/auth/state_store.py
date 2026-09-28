"""Tracks the `state` value issued at /auth/login until /auth/callback
consumes it — a single-use CSRF guard for the OIDC redirect round-trip.
In-process and short-lived (5 minutes) is enough for a login flow; no
database table needed for something this transient."""

from __future__ import annotations

import time
from functools import lru_cache

_TTL_SECONDS = 300


class StateStore:
    def __init__(self) -> None:
        self._states: dict[str, tuple[float, str]] = {}

    def issue(self, state: str, next_path: str = "/") -> None:
        """Remember ``state`` and where the browser goes after signing in."""
        self._prune()
        self._states[state] = (time.monotonic(), next_path)

    def consume(self, state: str) -> str | None:
        """The page to return to, exactly once per issued state, before it
        expires; None for an unknown, used or expired state."""
        self._prune()
        entry = self._states.pop(state, None)
        return entry[1] if entry is not None else None

    def _prune(self) -> None:
        cutoff = time.monotonic() - _TTL_SECONDS
        expired = [s for s, (issued_at, _) in self._states.items() if issued_at < cutoff]
        for s in expired:
            del self._states[s]


@lru_cache
def get_state_store() -> StateStore:
    return StateStore()
