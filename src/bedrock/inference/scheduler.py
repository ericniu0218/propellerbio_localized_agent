"""The one door every model call goes through.

Why a scheduler. A laptop's model server can work on one request at a time:
each extra request running in parallel needs another full KV cache (see
``bedrock.config.profile``). If a title request is running when the employee
sends a message, the employee waits for the title. The title request has also
replaced the server's saved work on the chat, so the next chat turn is read
again from the very start.

So nothing outside ``bedrock.inference`` calls ``stream_chat`` itself. It asks
the scheduler instead, saying how urgent the call is (``Priority``), and the
scheduler decides when it runs. The rules:

* **Order.** ``CHAT`` before ``USER_JOB`` before ``BACKGROUND``; within one
  priority, first come, first served.
* **How many at once.** At most ``profile.max_parallel`` calls run together:
  one on a laptop. The rest wait their turn.
* **Chat interrupts background.** When a chat call arrives while a background
  call is running, the background call is cancelled: its HTTP stream is
  closed, so the server stops too, and whoever was reading it gets
  ``Preempted``, which means "try again later", not "it failed". A running
  ``USER_JOB`` is never cancelled - the employee asked for it - but the chat
  goes next.
* **Background waits for quiet.** With ``profile.background == "idle"``, a
  background call starts only when nothing else is running or waiting, and
  ``profile.idle_seconds`` have passed since the last call somebody was
  waiting for - a chat reply or a ``USER_JOB`` - finished. A long job made of
  many short calls, such as a literature review, therefore counts as busy in
  the gaps between its calls rather than looking idle in each one. With
  ``"concurrent"`` (a shared server that batches requests), background calls
  simply run at the lowest priority.
* **Background uses the loaded model.** A background call made with
  ``model=None`` uses the model that last answered a chat - the one already
  in the server's memory - so a title never makes the server swap gigabytes
  of weights. If no chat has been answered yet, it gets ``Preempted``.

How cancelling works: each call's backend stream is read by a small helper
task (the "pump") that passes fragments to the caller through a queue.
Cancelling that task stops the backend mid-read, which closes the HTTP
response, and the scheduler then hands the caller ``Preempted`` through the
same queue. The caller's own task is never cancelled, so a title worker can
tell "stopped for the chat" apart from "the employee closed the app".

It also keeps the token counts the server reported for the last chat
(``last_usage``, shown by the status bar's context meter) and uses them to
correct the token estimate (``bedrock.inference.tokens``). All of this stays
in memory, on this machine.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import itertools
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

from bedrock.config.profile import MachineProfile
from bedrock.inference import tokens
from bedrock.inference.base import InferenceBackend, InferenceError, Priority, RequestOptions, Usage


class Preempted(InferenceError):
    """A lower-priority call was cancelled because the employee sent a message. Retry later; not a failure."""

    def __init__(self, message: str = "Stopped to make room for the chat; will try again later.") -> None:
        super().__init__(message)


_END = object()
"""Put on a call's queue when the backend's reply is complete."""

QUEUE_FRAGMENTS = 64
"""How many fragments the pump may read ahead of the caller."""


@dataclass(eq=False)
class _Ticket:
    """One call's place in line, and, once it runs, the handles to stop it."""

    priority: Priority
    order: int
    granted: asyncio.Event = field(default_factory=asyncio.Event)
    queue: asyncio.Queue | None = None
    pump: asyncio.Task | None = None
    preempted: bool = False

    @property
    def rank(self) -> tuple[int, int]:
        return (int(self.priority), self.order)


class Scheduler:
    """Decides when each model call reaches the server. See the module docstring."""

    def __init__(self, backend: InferenceBackend, profile: MachineProfile) -> None:
        self.backend = backend
        self.profile = profile
        self._last_usage: Usage | None = None
        self._loaded_model: str | None = None
        self._takes_options: bool | None = None
        self._waiting: list[_Ticket] = []
        self._running: set[_Ticket] = set()
        self._numbers = itertools.count()
        self._last_foreground_finished = float("-inf")
        self._wake: asyncio.TimerHandle | None = None

    @property
    def last_usage(self) -> Usage | None:
        """Token counts from the last chat reply, when the server reported them."""
        return self._last_usage

    @property
    def loaded_model(self) -> str | None:
        """The model that last answered a chat, which is the one the server has in memory."""
        return self._loaded_model

    @property
    def busy(self) -> bool:
        """True while any call is running or waiting."""
        return bool(self._running or self._waiting)

    async def stream(
        self,
        model: str | None,
        messages: list[dict[str, str]],
        *,
        priority: Priority,
        options: RequestOptions | None = None,
    ) -> AsyncIterator[str]:
        """Run one call when its turn comes and yield the reply as it arrives.

        ``model=None`` is allowed only for ``BACKGROUND`` calls and means "the
        model that last answered a chat", decided when the call starts rather
        than when it was asked for. If no chat has been answered yet, this
        raises ``Preempted`` rather than make the server load a model just
        for a title.

        Raises ``Preempted`` if a chat arrives while this ``BACKGROUND`` call
        is running, and ``InferenceError`` for any server failure.
        """
        if model is None and priority is not Priority.BACKGROUND:
            raise ValueError("Only BACKGROUND calls may leave the model to the scheduler.")
        ticket = _Ticket(priority=priority, order=next(self._numbers))
        await self._wait_for_turn(ticket)
        answered = False
        try:
            if model is None:
                if self._loaded_model is None:
                    raise Preempted("No chat has been answered yet, so no model is loaded to ask.")
                model = self._loaded_model
            characters = sum(len(str(message.get("content") or "")) for message in messages)
            usage: list[Usage] = []
            ticket.queue = asyncio.Queue(maxsize=QUEUE_FRAGMENTS)
            ticket.pump = asyncio.create_task(self._pump(ticket.queue, model, messages, options, usage.append))
            while True:
                item = await ticket.queue.get()
                if item is _END:
                    break
                if isinstance(item, BaseException):
                    raise item
                if priority is Priority.CHAT and not answered:
                    self._loaded_model = model
                answered = True
                yield item
            if priority is Priority.CHAT:
                self._loaded_model = model
                if usage:
                    self._last_usage = usage[-1]
                    tokens.observe(characters, usage[-1])
        finally:
            await self._finish(ticket)

    async def complete(
        self,
        model: str | None,
        messages: list[dict[str, str]],
        *,
        priority: Priority,
        options: RequestOptions | None = None,
    ) -> str:
        """Like ``stream``, but wait for the whole reply and return it as one string."""
        parts: list[str] = []
        stream = self.stream(model, messages, priority=priority, options=options)
        try:
            async for fragment in stream:
                parts.append(fragment)
        finally:
            await stream.aclose()
        return "".join(parts)

    # ---- Taking turns -------------------------------------------------------

    async def _wait_for_turn(self, ticket: _Ticket) -> None:
        self._waiting.append(ticket)
        if ticket.priority is Priority.CHAT:
            self._make_room_for_chat()
        self._dispatch()
        try:
            await ticket.granted.wait()
        except BaseException:
            # Cancelled while waiting (the employee pressed Esc, say): leave
            # the line, or give back the turn if it was granted meanwhile.
            if ticket in self._waiting:
                self._waiting.remove(ticket)
            self._running.discard(ticket)
            self._dispatch()
            raise

    def _make_room_for_chat(self) -> None:
        """Cancel running background calls that stand in the chat's way."""
        free = len(self._running) < self.profile.max_parallel
        if free and self.profile.background != "idle":
            return  # a shared server has a slot for the chat anyway
        for running in list(self._running):
            if running.priority is Priority.BACKGROUND and not running.preempted:
                self._preempt(running)

    def _preempt(self, ticket: _Ticket) -> None:
        ticket.preempted = True
        if ticket.pump is not None:
            ticket.pump.cancel()
        if ticket.queue is not None:
            # Whatever the cancelled call had produced is thrown away: the
            # caller is only told that it was stopped.
            while not ticket.queue.empty():
                ticket.queue.get_nowait()
            ticket.queue.put_nowait(Preempted())

    def _dispatch(self) -> None:
        """Start as many waiting calls as the rules allow, most urgent first."""
        if self._wake is not None:
            self._wake.cancel()
            self._wake = None
        self._waiting.sort(key=lambda ticket: ticket.rank)
        while self._waiting and len(self._running) < self.profile.max_parallel:
            ticket = self._waiting[0]
            if ticket.priority is Priority.BACKGROUND and self.profile.background == "idle":
                if self._running:
                    return
                quiet_left = self.profile.idle_seconds - (time.monotonic() - self._last_foreground_finished)
                if quiet_left > 0:
                    self._wake = asyncio.get_running_loop().call_later(quiet_left, self._dispatch)
                    return
            self._waiting.pop(0)
            self._running.add(ticket)
            ticket.granted.set()

    async def _finish(self, ticket: _Ticket) -> None:
        """Stop the call's backend stream if it is still going, then free its turn."""
        if ticket.pump is not None and not ticket.pump.done():
            ticket.pump.cancel()
        if ticket.pump is not None:
            # ``wait`` rather than ``await pump``: the pump's own cancellation
            # is expected and must not surface, while a cancellation of the
            # caller's task still must.
            await asyncio.wait([ticket.pump])
        if ticket in self._running:
            self._running.discard(ticket)
            # Any call somebody is waiting for restarts the idle clock, not just
            # a chat reply. A literature review is hundreds of ``USER_JOB``
            # calls with short gaps between them, and counting only ``CHAT``
            # meant the machine looked idle in every one of those gaps: a title
            # would slot in between two review calls, and on a laptop holding
            # one model that unloads the review's model and loads the chat's,
            # gigabytes each way. "Idle" has to mean nobody is waiting.
            if ticket.priority is not Priority.BACKGROUND:
                self._last_foreground_finished = time.monotonic()
        self._dispatch()

    # ---- Talking to the backend --------------------------------------------

    async def _pump(self, queue: asyncio.Queue, model, messages, options, on_usage) -> None:
        """Read the backend's stream into ``queue``. Cancelling this closes the stream."""
        stream = self._open(model, messages, options, on_usage)
        try:
            async for fragment in stream:
                # ``await put`` waits while the queue is full, so a server
                # (or test double) that sends faster than the caller reads
                # cannot pile up the whole reply in memory.
                await queue.put(fragment)
            await queue.put(_END)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # handed to the caller, who raises it
            await queue.put(exc)
        finally:
            await _close(stream)

    def _open(self, model, messages, options, on_usage) -> AsyncIterator[str]:
        """Start the backend's stream, with options if the backend understands them.

        A backend written for the original two-argument ``stream_chat`` (some
        test doubles, or a company's own adapter) still works: it just cannot
        honour limits or report token counts.
        """
        if self._takes_options is None:
            self._takes_options = _accepts_options(self.backend)
        if self._takes_options:
            return self.backend.stream_chat(model, messages, options=options, on_usage=on_usage)
        return self.backend.stream_chat(model, messages)


def _accepts_options(backend: InferenceBackend) -> bool:
    try:
        parameters = inspect.signature(backend.stream_chat).parameters.values()
    except (TypeError, ValueError):
        return True
    names = {parameter.name for parameter in parameters}
    return "options" in names or any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters)


async def _close(stream: AsyncIterator[str]) -> None:
    """Close the backend's stream, so an early stop also closes the HTTP connection."""
    close = getattr(stream, "aclose", None)
    if close is not None:
        with contextlib.suppress(Exception):
            await close()
