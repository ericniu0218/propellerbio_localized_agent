"""First-in, first-out queueing of prompts typed while the model is busy.

Why this exists
---------------
One request runs at a time. Before this module a second prompt sent during a
reply was refused ("Still responding - press Esc first"), so the employee had
to sit and watch. Now the prompt joins a queue and runs when its turn comes.

What is deliberately *not* here
-------------------------------
No Textual widgets, no transcripts, no disk access. The queue holds the text
the employee typed, which conversation (or agent-pack session) it belongs to,
and which model was selected - nothing else. The conversation itself is built
when the prompt actually starts, so a prompt that waits ten minutes still sees
the answers that arrived while it waited, and a queued item never pins a copy
of a transcript in memory.

Scope and lifetime
------------------
In memory, and only for this window. Queues are per conversation: switching
chats pauses the one you left, so nothing runs unwatched in a chat you are no
longer looking at, and returning to it needs an explicit resume.

The UI interface
----------------
``queue_snapshot``, ``remove_queued_prompt``, ``pause_queue``, ``resume_queue``
and ``clear_queue`` live on the app. They hand out ``QueueSnapshot``, whose
items are frozen: a panel can render the queue but cannot reach into it. Every
change posts a ``QueueChanged`` message.

``QueueSnapshot.can_skip`` gates the panel's "Stop & run next" control
(``PromptQueuePanel.SkipRequested``). The app computes it, not this module -
whether stopping in-flight work can be done safely and confirmed depends on
the backend, which this module never touches.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass
from typing import Iterable

from textual.message import Message

MAX_QUEUE_ITEMS = 50
MAX_PROMPT_CHARS = 64_000
MAX_QUEUE_CHARS = 256_000


class QueueFull(ValueError):
    """The draft was refused, not truncated or discarded."""


@dataclass(frozen=True)
class QueuedPrompt:
    """One queued prompt as the UI sees it: no conversation internals."""

    id: str
    text: str
    model: str | None = None


@dataclass(frozen=True)
class PromptItem:
    """A queued prompt as the app sees it, including where it belongs."""

    id: str
    text: str
    conversation_id: str
    model: str | None = None

    def public(self) -> QueuedPrompt:
        return QueuedPrompt(id=self.id, text=self.text, model=self.model)


@dataclass(frozen=True)
class QueueSnapshot:
    """What one conversation's queue looks like right now.

    ``running`` is only set when the running prompt belongs to *this*
    conversation, so a panel never shows another chat's work as its own.
    """

    pending: tuple[QueuedPrompt, ...] = ()
    running: QueuedPrompt | None = None
    paused: bool = False
    conversation_id: str | None = None
    """Which conversation this is a snapshot of, so the panel can reset its
    selection when the employee switches chats."""
    can_skip: bool = False
    """Whether "Stop & run next" should be offered right now.

    Computed by the app, not this module: it depends on whether stopping the
    backend's in-flight work can be done safely and confirmed, which this
    module has no way to know. False unless the app says otherwise."""

    @property
    def is_idle(self) -> bool:
        return self.running is None and not self.pending

    def __len__(self) -> int:
        return len(self.pending)


class QueueChanged(Message):
    """Posted after any change to queue state, for the queue panel to render."""

    def __init__(self, conversation_id: str, snapshot: QueueSnapshot) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self.snapshot = snapshot


class PromptQueue:
    """Per-conversation FIFO queues with one prompt running at a time.

    Every method that changes something returns whether it changed anything,
    so callers can post ``QueueChanged`` exactly when there is news.
    """

    def __init__(self) -> None:
        self._pending: dict[str, deque[PromptItem]] = {}
        self._paused: set[str] = set()
        self._running: PromptItem | None = None

    # ---- Questions --------------------------------------------------------

    @property
    def running(self) -> PromptItem | None:
        return self._running

    @property
    def busy(self) -> bool:
        """True from the moment a prompt is dispatched until it finishes.

        Set *before* any awaiting preparation (recall, saving) so a prompt
        submitted during that preparation queues instead of racing it.
        """
        return self._running is not None

    def is_paused(self, conversation_id: str) -> bool:
        return conversation_id in self._paused

    def pending_for(self, conversation_id: str) -> tuple[PromptItem, ...]:
        return tuple(self._pending.get(conversation_id, ()))

    @property
    def pending_count(self) -> int:
        return sum(map(len, self._pending.values()))

    def snapshot(self, conversation_id: str, *, can_skip: bool = False) -> QueueSnapshot:
        running = self._running
        return QueueSnapshot(
            pending=tuple(item.public() for item in self._pending.get(conversation_id, ())),
            running=running.public() if running is not None and running.conversation_id == conversation_id else None,
            paused=conversation_id in self._paused,
            conversation_id=conversation_id,
            can_skip=can_skip,
        )

    def conversations(self) -> set[str]:
        """Every conversation this queue knows about, including the running one."""
        known = {cid for cid, items in self._pending.items() if items} | set(self._paused)
        if self._running is not None:
            known.add(self._running.conversation_id)
        return known

    # ---- Changes ----------------------------------------------------------

    def add(self, text: str, conversation_id: str, model: str | None) -> PromptItem:
        retained = [item for waiting in self._pending.values() for item in waiting]
        if self._running is not None:
            retained.append(self._running)
        if len(retained) >= MAX_QUEUE_ITEMS:
            raise QueueFull(f"This window can hold {MAX_QUEUE_ITEMS} prompts, including the running one. Clear some waiting prompts or wait.")
        if len(text) > MAX_PROMPT_CHARS or sum(len(item.text) for item in retained) + len(text) > MAX_QUEUE_CHARS:
            raise QueueFull("This draft exceeds the queue's text limit. Shorten it or clear some waiting prompts.")
        if not self._pending.get(conversation_id) and not (self._running and self._running.conversation_id == conversation_id):
            self._paused.discard(conversation_id)
        item = PromptItem(id=uuid.uuid4().hex, text=text, conversation_id=conversation_id, model=model)
        self._pending.setdefault(conversation_id, deque()).append(item)
        return item

    def start_next(self, conversation_id: str) -> PromptItem | None:
        """Take the next prompt for a conversation, or None if it cannot start.

        Nothing starts while another prompt is running, while this
        conversation is paused, or when it has nothing waiting.
        """
        if self.busy or conversation_id in self._paused:
            return None
        waiting = self._pending.get(conversation_id)
        if not waiting:
            return None
        self._running = waiting.popleft()
        if not waiting:
            self._pending.pop(conversation_id, None)
        return self._running

    def requeue_front(self, item: PromptItem) -> bool:
        """Return the current unsent item to its queue and pause for review.

        Only the current item may be returned: late cleanup cannot resurrect
        work removed by /forget. The running item already reserves capacity.
        """
        if not self.finish(item.id):
            return False
        self._pending.setdefault(item.conversation_id, deque()).appendleft(item)
        self._paused.add(item.conversation_id)
        return True

    def start(self, item: PromptItem) -> bool:
        """Mark a prompt as running without it having waited in the queue."""
        if self.busy:
            return False
        self._running = item
        return True

    def finish(self, item_id: str) -> bool:
        """End the running prompt. False if ``item_id`` is not the running one.

        This is the guard against duplicate, delayed or unrelated completion
        events: only the prompt that is actually running can end a run, ids are
        never reused, and a second attempt to finish the same prompt does
        nothing rather than releasing whatever started next.
        """
        if self._running is None or self._running.id != item_id:
            return False
        self._running = None
        return True

    def pause(self, conversation_id: str) -> bool:
        if conversation_id in self._paused:
            return False
        self._paused.add(conversation_id)
        return True

    def resume(self, conversation_id: str) -> bool:
        if conversation_id not in self._paused:
            return False
        self._paused.discard(conversation_id)
        return True

    def clear(self, conversation_id: str) -> tuple[PromptItem, ...]:
        """Drop everything waiting for a conversation. The running prompt continues."""
        dropped = tuple(self._pending.pop(conversation_id, ()))
        self._paused.discard(conversation_id)
        return dropped

    def remove(self, item_id: str) -> PromptItem | None:
        """Remove one waiting prompt. A running prompt cannot be removed."""
        for conversation_id, waiting in self._pending.items():
            for item in waiting:
                if item.id == item_id:
                    waiting.remove(item)
                    if not waiting:
                        self._pending.pop(conversation_id, None)
                        self._paused.discard(conversation_id)
                    return item
        return None

    def drop_conversations(self, conversation_ids: Iterable[str]) -> tuple[PromptItem, ...]:
        """Forget queues for deleted conversations, including linked copies."""
        dropped: list[PromptItem] = []
        for conversation_id in set(conversation_ids):
            dropped.extend(self._pending.pop(conversation_id, ()))
            self._paused.discard(conversation_id)
            if self._running is not None and self._running.conversation_id == conversation_id:
                self._running = None
        return tuple(dropped)
