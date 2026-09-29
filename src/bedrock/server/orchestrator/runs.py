"""Orchestration runs started by ``POST /messages``: watched, and stoppable.

``handle_message`` runs in the background so the request can return at once;
the client follows it over the event stream. Two things went wrong with a bare
``asyncio.create_task`` there:

* **A failure was silent.** An exception - a task over its token budget, a model
  server that stopped answering - ended the task with nothing published, so a
  client waiting for ``task.completed`` or ``task.failed`` waited for ever.
  ``asyncio`` also keeps only a weak reference to a task nobody holds, so one
  could in principle vanish mid-run.
* **Cancel did not cancel.** ``/tasks/cancel`` marked the task cancelled and
  published ``task.failed``, while the work carried on and could still publish
  ``task.completed`` afterwards.

``MessageRuns`` holds every run it starts. An uncaught exception is logged and
becomes exactly one ``task.failed`` in plain words; a cancellation stops the
coroutine (including one waiting at a human-approval gate) and becomes exactly
one ``task.failed`` saying so.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache

from bedrock.inference import InferenceError
from bedrock.server.agents.base import BudgetExceededError
from bedrock.server.events import get_event_bus
from bedrock.server.schemas.enums import EventType, TaskStatus
from bedrock.server.schemas.events import TaskFailedPayload
from bedrock.server.task_store.store import get_task_store

logger = logging.getLogger(__name__)

CANCELLED = "Cancelled by user."
OVER_BUDGET = "This request used up its token budget before it finished. Try a narrower question."
MODEL_DOWN = "The language model server did not answer. Try again in a minute."
UNEXPECTED = "Something went wrong on the server. The error was logged."


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, BudgetExceededError):
        return OVER_BUDGET
    if isinstance(exc, InferenceError) and exc.unavailable:
        return MODEL_DOWN
    return UNEXPECTED


@dataclass
class _Run:
    session_id: str
    task: asyncio.Task


class MessageRuns:
    def __init__(self) -> None:
        self._runs: dict[str, _Run] = {}

    def start(self, *, session_id: str, root_task_id: str, coro) -> asyncio.Task:
        task = asyncio.create_task(self._supervise(session_id, root_task_id, coro), name=f"message:{root_task_id}")
        self._runs[root_task_id] = _Run(session_id, task)
        return task

    def running(self, task_id: str) -> bool:
        return self._find(task_id) is not None

    async def cancel(self, task_id: str) -> bool:
        """Stop the run that ``task_id`` belongs to: its root id, or any step of its plan."""
        run = self._find(task_id)
        if run is None:
            return False
        run.task.cancel()
        # Let the supervisor publish its task.failed before the caller answers.
        await asyncio.wait([run.task], timeout=5)
        return True

    def _find(self, task_id: str) -> _Run | None:
        if task_id in self._runs:
            return self._runs[task_id]
        session_id = get_task_store().get_task_session_id(task_id)
        return next((run for run in self._runs.values() if session_id and run.session_id == session_id), None)

    async def _supervise(self, session_id: str, root_task_id: str, coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            await self._fail(session_id, root_task_id, CANCELLED, TaskStatus.CANCELLED)
        except Exception as exc:
            logger.exception("Request %s in session %s failed", root_task_id, session_id)
            await self._fail(session_id, root_task_id, friendly_error(exc), TaskStatus.FAILED)
        finally:
            self._runs.pop(root_task_id, None)

    async def _fail(self, session_id: str, root_task_id: str, message: str, status: TaskStatus) -> None:
        store = get_task_store()
        node = store.get_task(root_task_id)
        if node is not None:
            node.status = status
            store.update_task(session_id, node)
        await get_event_bus().publish_typed(
            session_id,
            EventType.TASK_FAILED,
            TaskFailedPayload(
                task_id=root_task_id,
                title=node.title if node is not None else "Your request",
                error=message,
                blocked_on_human=False,
            ),
        )


@lru_cache
def get_message_runs() -> MessageRuns:
    return MessageRuns()
