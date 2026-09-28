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
        self._states: dict[str, float] = {}

    def issue(self, state: str) -> None:
        self._prune()
        self._states[state] = time.monotonic()

    def consume(self, state: str) -> bool:
        """Returns True exactly once per issued state, before it expires."""
        self._prune()
        return self._states.pop(state, None) is not None

    def _prune(self) -> None:
        cutoff = time.monotonic() - _TTL_SECONDS
        expired = [s for s, issued_at in self._states.items() if issued_at < cutoff]
        for s in expired:
            del self._states[s]


@lru_cache
def get_state_store() -> StateStore:
    return StateStore()
