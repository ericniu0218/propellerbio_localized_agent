"""Lazy access to the local history journal and the run catalogue.

Both apps and the biotech CLI need the same two objects:

* ``HistoryStore`` - the durable local action journal
* ``RunCatalog``   - a read-only view over the per-run manifests already on disk

Three properties matter enough to justify a class rather than two globals.

**Lazy.** Building either one touches the disk (creating the history
directory, opening SQLite). An employee who only ever chats should never pay
for that, exactly as ``BedrockApp.context_store`` is built on first use.

**Single-flight.** ``/runs`` and ``/audit`` can be typed a moment apart and run
in separate Textual workers. Without the lock, both would build their own
store. With it, the second waits for the first and reuses it.

**Injectable.** Tests pass fakes with the same shape, so the UI is driven
headlessly against a temporary directory and never opens an employee's vault.

Construction happens in ``asyncio.to_thread``: nothing that touches a
filesystem belongs on the Textual event loop.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

DEFAULT_HISTORY_DIR = Path.home() / ".local" / "share" / "bedrock" / "history"

UNAVAILABLE_MESSAGE = (
    "Local history is not available in this installation: the `bedrock.provenance` "
    "module is missing. Runs, confidence and audit browsing need it."
)


class ServicesUnavailable(RuntimeError):
    """The provenance modules are not installed. User-presentable."""


class ComplianceServices:
    """Holds - and builds at most once - the history journal and run catalogue."""

    def __init__(
        self,
        settings: Any,
        *,
        history: Any | None = None,
        runs: Any | None = None,
    ) -> None:
        self.settings = settings
        self._history = history
        self._runs = runs
        self._lock = asyncio.Lock()
        self.actor_id: str = getattr(settings, "user_id", "") or "unknown"

    # ---- The two services -------------------------------------------------

    async def history(self) -> Any:
        if self._history is None:
            await self._build()
        return self._history

    async def runs(self) -> Any:
        if self._runs is None:
            await self._build()
        return self._runs

    async def optional_history(self) -> Any | None:
        """The journal, or ``None`` in a build that has no provenance module.

        `/digest` predates local history and must keep working without it, so
        it asks this way. `/runs` and `/audit` ask for the real thing and
        report the missing module, because without it they have nothing to
        show at all.
        """
        try:
            return await self.history()
        except ServicesUnavailable:
            return None

    async def _build(self) -> None:
        async with self._lock:
            # A concurrent command may have finished building while we waited.
            if self._history is not None and self._runs is not None:
                return
            history, runs = await asyncio.to_thread(self._build_blocking)
            if self._history is None:
                self._history = history
            if self._runs is None:
                self._runs = runs

    def _build_blocking(self) -> tuple[Any, Any]:
        try:
            from bedrock.provenance import HistoryStore, RunCatalog
        except ImportError as exc:  # pragma: no cover - depends on the installed build
            raise ServicesUnavailable(UNAVAILABLE_MESSAGE) from exc
        history = self._history or HistoryStore(self.history_dir)
        runs = self._runs or RunCatalog(self.settings.biotech_work_dir, history)
        return history, runs

    @property
    def history_dir(self) -> Path:
        """Where the journal lives.

        ``Settings.history_dir`` is the supported answer and follows the usual
        environment-over-config-over-default precedence. The environment
        fallback below only matters in a build whose settings module predates
        that field; it keeps this module working either way rather than
        reaching into configuration loading itself.
        """
        configured = getattr(self.settings, "history_dir", None) or os.environ.get(
            "BEDROCK_HISTORY_DIR"
        )
        return Path(configured).expanduser() if configured else DEFAULT_HISTORY_DIR

    # ---- Error handling ---------------------------------------------------

    @property
    def errors(self) -> tuple[type[BaseException], ...]:
        """Exceptions a command handler should show as an error notice.

        ``HistoryError`` and ``RunError`` are the provenance layer's
        user-presentable exceptions; the standard-library types cover a fake
        in tests and the ordinary filesystem failures underneath.
        """
        return (ValueError, OSError, LookupError, ServicesUnavailable, *_domain_errors())


_DOMAIN_ERRORS: tuple[type[BaseException], ...] | None = None


def _domain_errors() -> tuple[type[BaseException], ...]:
    global _DOMAIN_ERRORS
    if _DOMAIN_ERRORS is None:
        try:
            from bedrock.provenance import HistoryError, RunError

            _DOMAIN_ERRORS = (HistoryError, RunError)
        except ImportError:
            _DOMAIN_ERRORS = ()
    return _DOMAIN_ERRORS
