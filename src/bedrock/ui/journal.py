"""Writing one action to local history: intent first, outcome after.

Every tracked action in this client - ingesting a digest, forgetting one,
answering an approval - follows the same three-step shape, and getting the
shape wrong is how audit trails start lying:

1. **Commit the intent before doing anything.** If the intent cannot be
   written, the action does not start. A mutation nobody recorded wanting is
   the one thing local history can never reconstruct.
2. **Do the work exactly once.**
3. **Commit what actually happened.** Success, failure, cancellation, or -
   the honest answer more often than people expect - *unknown*.

The asymmetry in step 3 is the part worth reading twice. If the work
succeeded and only the outcome write failed, the result is kept and the gap
is reported. Repeating the work to make the journal tidy would be choosing a
neat record over a correct one: local files, a remote HTTP call and SQLite
cannot be made into one transaction, so something has to give, and it must
never be the real-world effect.

A ``history`` of ``None`` makes every method a no-op, which is what tests and
any caller without a journal get.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Awaitable, Callable

Notify = Callable[[str, str], Awaitable[None]]


class JournalledAction:
    """One operation ID, one intent slot, one outcome slot."""

    def __init__(
        self,
        history: Any | None,
        actor_id: str,
        action: str,
        *,
        subject_id: str | None = None,
        operation_id: str | None = None,
    ) -> None:
        self.history = history
        self.actor_id = actor_id or "unknown"
        self.action = action
        self.subject_id = subject_id
        # One UUID per actual invocation. Re-ingesting identical content is a
        # second act by the employee and gets its own ID, even though it will
        # reuse the same stored artifact.
        self.operation_id = operation_id or str(uuid.uuid4())
        self.intent_committed = False

    async def begin(self, notify: Notify, **details: Any) -> bool:
        """Commit the intent. ``False`` means: do not start the action."""
        if self.history is None:
            return True
        try:
            await self._append("intent", details)
        except Exception as exc:  # noqa: BLE001 - any failure here stops the action
            await notify(
                f"Nothing was started: this action could not be recorded in local "
                f"history ({_describe(exc)}).",
                "error",
            )
            return False
        self.intent_committed = True
        return True

    async def record(self, notify: Notify, state: str, *, already_done: bool = False, **details: Any) -> bool:
        """Commit the outcome. ``False`` means the journal now has a gap.

        ``already_done`` changes only the wording: it tells the employee that
        the real-world effect stands and will not be repeated, which is the
        difference between a bookkeeping problem and a lost result.
        """
        if self.history is None or not self.intent_committed:
            return self.history is None
        try:
            await self._append(state, details)
        except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
            tail = (
                " The action itself succeeded and has NOT been repeated."
                if already_done
                else ""
            )
            await notify(
                f"Local history has a gap: the outcome of operation "
                f"`{self.operation_id}` could not be written ({_describe(exc)}).{tail}",
                "error",
            )
            return False
        return True

    async def _append(self, state: str, details: dict[str, Any]) -> None:
        payload = {key: value for key, value in details.items() if value is not None}
        payload.setdefault("actor_source", "config")
        # SQLite writes belong off the Textual event loop; every history call
        # owns and closes its own connection, so a thread is safe here.
        await asyncio.to_thread(
            self.history.append,
            self.operation_id,
            actor_id=self.actor_id,
            action=self.action,
            state=state,
            subject_id=self.subject_id,
            details=payload,
        )


def _describe(exc: BaseException) -> str:
    """A bounded description. Raw exception text never enters history itself."""
    text = str(exc).strip() or exc.__class__.__name__
    return text[:200]
