"""Recording a review in local history: intent before it starts, outcome after.

This is litreview's own copy of the intent/outcome pattern that
``bedrock.ui.journal`` uses for `/digest` and approvals - not an import of it.
``ui/journal.py`` belongs to the Textual layer (its steps are async and end
with a ``notify`` callback that posts a chat notice); this package's own rule
is that nothing outside ``service.py`` and ``llm/scheduled.py`` reaches into
another bedrock subsystem it does not need, and ``bedrock.provenance`` is no
exception. So ``history`` here is duck-typed, exactly the shape
``ui/journal.py`` uses: one method, ``append(operation_id, *, actor_id,
action, state, subject_id=None, details=None)``. ``history=None`` makes every
function in this module a no-op, which is what every existing caller and
every test without a journal gets.

The shape is the same three steps as everywhere else in bedrock that journals
an action, and getting it wrong is how an audit trail starts lying:

1. **Commit the intent before doing anything.** If it cannot be written, the
   action does not start - see :class:`ReviewNotRecorded`.
2. **Do the work once.**
3. **Commit what actually happened.** Succeeded, failed with a bounded error
   code, cancelled, or - the honest answer when the pipeline thread has
   already finished but a journal write itself then fails - a report of the
   gap rather than a retry.

The asymmetry in step 3 is deliberate and is explained at length in
``ui/journal.py``: if the work already happened, redoing it to make the
journal tidy would trade a correct result for a neat record, and local files,
a remote HTTP call and SQLite cannot be made into one transaction, so the
real-world effect - the finished report - must be the thing that is kept.

Only two kinds of event are written here, and both are metadata only:

* ``litreview.review`` - one operation per run. Never the question text; the
  outcome's ``run_id`` is the join back to litreview's own database, which
  already holds the question.
* ``litreview.search`` - one operation per external host a run is
  *authorised to contact*, written once whether the run goes on to make one
  request to that host or a hundred. The literature search and the automatic
  retraction check are different hosts under different decisions (see this
  package's ``CLAUDE.md`` on ``SOURCE_HOSTS`` vs ``INTEGRITY_HOSTS``), so each
  gets its own event, named by ``source_kind`` (``literature_search``,
  ``retraction_check``, and ``figure_download`` for the deck's paper figures:
  the images, and the article text re-read to find which figure a quote's
  paragraph cites, both fetched from Europe PMC only when the review was asked
  to both search and make the deck). A run that contacts nothing -
  a corpus file, retraction checks set to manual - writes no ``litreview.search``
  event at all. Read this as "the run was authorised and began", not as a
  request-by-request log.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable

Progress = Callable[[str], None]


class ReviewNotRecorded(RuntimeError):
    """Local history could not record that this review was starting.

    Raised before any pipeline work runs - see ``service.run_review``'s
    docstring, which lists this beside its other exceptions. A front end
    shows it exactly like ``ConfigError``: the review did not start, nothing
    was contacted, nothing needs to be undone.
    """


async def commit_intent(
    history: Any | None,
    operation_id: str,
    *,
    actor_id: str,
    action: str,
    subject_id: str | None = None,
    details: dict | None = None,
) -> None:
    """Commit one intent. Raises :class:`ReviewNotRecorded` if it cannot be
    written - the caller must not go on to do the work that intent describes."""
    if history is None:
        return
    try:
        await _append(history, operation_id, actor_id=actor_id, action=action,
                      state="intent", subject_id=subject_id, details=details)
    except Exception as exc:  # noqa: BLE001 - any failure here stops the run
        raise ReviewNotRecorded(
            f"local history could not record the start of this review "
            f"({_describe(exc)}); nothing was run."
        ) from exc


async def commit_outcome(
    history: Any | None,
    operation_id: str,
    *,
    actor_id: str,
    action: str,
    state: str,
    subject_id: str | None = None,
    details: dict | None = None,
    progress: Progress | None,
) -> None:
    """Commit one outcome. Never raises: the pipeline has already produced
    (or failed to produce) its real result by the time this is called, and
    that result is never re-run to tidy the journal. A write that fails here
    is reported through ``progress`` instead - the same callback
    ``run_review`` already threads end to end - so the gap is visible rather
    than silently dropped."""
    if history is None:
        return
    try:
        await _append(history, operation_id, actor_id=actor_id, action=action,
                      state=state, subject_id=subject_id, details=details)
    except Exception as exc:  # noqa: BLE001 - reported, never silently dropped
        _safe_progress(
            progress,
            f"Local history has a gap: the outcome of operation `{operation_id}` "
            f"could not be recorded ({_describe(exc)}). The result itself stands.",
        )


def new_operation_id() -> str:
    """One UUID per actual attempt, the same rule ``ui/journal.py`` uses:
    resuming or re-running a review is a new act and gets its own id."""
    return str(uuid.uuid4())


async def _append(
    history: Any,
    operation_id: str,
    *,
    actor_id: str,
    action: str,
    state: str,
    subject_id: str | None,
    details: dict | None,
) -> None:
    payload = {key: value for key, value in (details or {}).items() if value is not None}
    payload.setdefault("actor_source", "config")
    # SQLite writes belong off the event loop; every HistoryStore call owns
    # and closes its own connection, so a thread is safe here.
    await asyncio.to_thread(
        history.append,
        operation_id,
        actor_id=actor_id or "unknown",
        action=action,
        state=state,
        subject_id=subject_id,
        details=payload,
    )


def _safe_progress(progress: Progress | None, message: str) -> None:
    if progress is None:
        return
    try:
        progress(message)
    except Exception:  # noqa: BLE001 - a progress callback must never raise
        pass


def _describe(exc: BaseException) -> str:
    """A bounded description for a user-facing message. Raw exception text
    never enters the journal itself - only ``details["error_code"]`` does,
    and that is chosen by code, never built from ``str(exc)``."""
    text = str(exc).strip() or exc.__class__.__name__
    return text[:200]
