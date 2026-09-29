"""How a running review says it is still working.

There are two kinds of message, and they answer different questions.

`progress` (the `say` in `runner.py`) is a line per *milestone*: "gate: 84 of
108 concern the exact entity". It is the record of what happened, it is worth
keeping, and there are about a dozen of them in a run.

A `Tick` is the answer to "is this thing still alive?". The three stages that
make one model call per item - grading each paper, extracting from each chunk,
writing each section - used to say nothing at all between their milestones.
Grading 84 papers is one call each and was measured at about four minutes of
complete silence; a whole laptop run is over two hours. Nothing distinguished
that from a hung process.

Ticks are deliberately *not* progress lines. A line per graded paper would be
84 lines, and in the terminal app each one is a toast that covers the screen
and then fades, so the screen would be either full of noise or blank again.
A tick instead overwrites one place - the status bar in the app, a throttled
stderr line in the shell - so the cost of a tick is bounded no matter how many
there are.

What a tick must not do is change the run. It carries no information the
pipeline reads back, a caller may ignore every one, and `safe_tick` makes a
callback that raises harmless: a run that died because a progress indicator
failed would be the worst possible trade. The report is byte-identical whether
or not anyone is listening, which `tests/litreview/test_progress.py` checks.
"""

from __future__ import annotations

import logging
from typing import Callable, Protocol

log = logging.getLogger(__name__)


class Tick(Protocol):
    """Called as each item of a stage finishes.

    `done` counts items completed, out of `total`. `detail` is a short
    sub-position such as "chunk 2/6" when a stage's items have parts of their
    own; it is free-form and for display only.
    """

    def __call__(self, stage: str, done: int, total: int, detail: str = "") -> None:
        ...


def safe_tick(tick: Tick | None) -> Tick:
    """`tick`, wrapped so nothing it does can end the run, or a no-op.

    Callers hand these across a thread boundary and into a UI. `progress` is
    already wrapped this way for the same reason (`ui/review_command.py`'s
    `progress_to_toasts`); this keeps the stages from having to care whether
    anyone passed one.
    """
    if tick is None:
        return _ignore

    def guarded(stage: str, done: int, total: int, detail: str = "") -> None:
        try:
            tick(stage, done, total, detail)
        except Exception:  # noqa: BLE001 - see the module docstring
            log.debug("progress tick failed", exc_info=True)

    return guarded


def _ignore(stage: str, done: int, total: int, detail: str = "") -> None:
    return None


def format_tick(stage: str, done: int, total: int, detail: str = "", elapsed: float | None = None) -> str:
    """One line for a status bar: `review: rank 12/40 · 2m 10s`.

    Counts and elapsed time, never an estimate of time remaining. Call
    durations vary by more than an order of magnitude between a title-and-
    abstract grade and a 4,096-token extraction, so an ETA computed from the
    calls so far would be confidently wrong, and a wrong ETA is worse than
    none: it is the number someone plans their afternoon around.
    """
    parts = [f"{stage} {done}/{total}" if total else stage]
    if detail:
        parts.append(detail)
    if elapsed is not None:
        parts.append(format_elapsed(elapsed))
    return "review: " + " · ".join(parts)


def format_elapsed(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def throttled(
    emit: Callable[[str], None],
    *,
    now: Callable[[], float],
    min_seconds: float = 60.0,
    min_fraction: float = 0.1,
) -> Tick:
    """A tick that writes a line, but rarely: for the shell, where there is no
    status bar to overwrite and every line is permanent.

    `min_fraction` is what bounds the noise: a line every tenth of a stage, so
    about eleven lines however many items there are, instead of one per paper.

    `min_seconds` only breaks a long silence, and is deliberately much larger
    than a fast call. It has to be: the two conditions are an OR, so a small
    `min_seconds` would make a *slower* stage print *more* lines - at 30
    seconds a call, a 15-second floor would emit on every single item, which
    is the behaviour this function exists to avoid. At a minute it fires only
    when the fraction alone would leave someone staring at nothing.
    """
    state: dict[str, tuple[float, int]] = {}

    def tick(stage: str, done: int, total: int, detail: str = "") -> None:
        last_at, last_done = state.get(stage, (0.0, -1))
        first, final = last_done < 0, bool(total) and done >= total
        step = max(1, int(total * min_fraction)) if total else 1
        due = (now() - last_at) >= min_seconds or (done - last_done) >= step
        if not (first or final or due):
            return
        state[stage] = (now(), done)
        emit(format_tick(stage, done, total, detail))

    return tick
