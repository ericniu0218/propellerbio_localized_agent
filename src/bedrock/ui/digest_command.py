"""The `/digest` command, shared by both apps.

Both ``BedrockApp`` and ``AgentPackApp`` offer this command and neither should
word it differently, so the logic lives here once. It is written against a
``notify`` callback rather than against Textual, which keeps it testable on its
own and keeps this module free of widget imports - the same reason
``bedrock.ui.commands`` is plain functions.

Sub-commands::

    /digest <path>          compress a file into the store
    /digest <source>://<id> pull a record from a company source and compress it
    /digest                 list what is stored
    /digest search <words>  find exact content across everything stored
    /digest show <id>       print one digest
    /digest forget <id>     delete one artifact

The ingest itself runs on a thread. Compressing a large file takes a moment and
the interface must keep painting while it does - the app stays responsive and
the input box keeps working.

Ingesting and forgetting are also recorded in local history when a journal is
supplied. Two rules there are easy to get wrong and are spelled out in the
code below: the ingest outcome is committed as soon as the artifact is stored
and *before* the optional model summary, because the data is already saved at
that point; and a cancelled ingest is recorded as **unknown**, because
``asyncio.to_thread`` cannot stop a thread that is already compressing a file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

from bedrock.connectors import ConnectorError, SourceHub, SourceRef, parse_source_ref
from bedrock.context import ContextStore, ingest
from bedrock.context.adapters import Source
from bedrock.context.digest import summarize, would_benefit_from_model
from bedrock.context.report import (
    format_digest,
    format_hits,
    format_ingested,
    format_listing,
)
from bedrock.ui.journal import JournalledAction

Notify = Callable[[str, str], Awaitable[None]]

USAGE = """\
**`/digest` — compress data so a model can read it cheaply**

| Command | What it does |
|---|---|
| `/digest <path>` | Compress a file into the store |
| `/digest <source>://<id>` | Pull a record from a company source (see `/sources`) and compress it |
| `/digest` | List what is stored |
| `/digest search <words>` | Find exact content across everything stored |
| `/digest show <id>` | Print one digest |
| `/digest forget <id>` | Delete one artifact |

Handles text, Markdown, CSV/TSV, JSON, source code, log files, and FASTA \
(sequence count, length, and DNA/RNA/protein alphabet instead of treating it \
as prose). Each one is stored twice over: a small digest a model can read on \
every turn, and the full content compressed and indexed for exact lookup.

Relevant stored context is now pulled into your questions automatically — \
see the note after each answer that used any.
"""


async def run_digest_command(
    args: str,
    *,
    store: ContextStore,
    notify: Notify,
    backend=None,
    model: str | None = None,
    sources: SourceHub | None = None,
    history: Any | None = None,
    actor_id: str | None = None,
) -> None:
    """Handle one ``/digest ...`` invocation.

    ``history`` and ``actor_id`` are optional so this function stays usable on
    its own; both production apps always supply them.
    """
    args = args.strip()
    verb, _, rest = args.partition(" ")
    verb, rest = verb.lower(), rest.strip()

    if not args:
        await notify(format_listing(store.list(), store.totals()), "info")
        return

    if verb in {"help", "?"}:
        await notify(USAGE, "info")
        return

    if verb == "search":
        if not rest:
            await notify("Give me something to search for: `/digest search <words>`.", "error")
            return
        await notify(format_hits(rest, store.search(rest, limit=5)), "info")
        return

    if verb == "show":
        artifact = store.resolve(rest) if rest else None
        if artifact is None:
            await notify(_not_found(rest), "error")
            return
        await notify(format_digest(artifact), "info")
        return

    if verb in {"forget", "delete", "rm"}:
        artifact = store.resolve(rest) if rest else None
        if artifact is None:
            await notify(_not_found(rest), "error")
            return
        await _forget_artifact(
            artifact, store=store, notify=notify, history=history, actor_id=actor_id
        )
        return

    source = parse_source_ref(args.strip("'\""))
    if source is not None:
        await _ingest_source(
            source,
            store=store,
            notify=notify,
            sources=sources,
            backend=backend,
            model=model,
            history=history,
            actor_id=actor_id,
        )
        return

    await _ingest_path(
        args,
        store=store,
        notify=notify,
        backend=backend,
        model=model,
        history=history,
        actor_id=actor_id,
    )


async def _ingest_path(
    raw_path: str,
    *,
    store: ContextStore,
    notify: Notify,
    backend,
    model: str | None,
    history: Any | None = None,
    actor_id: str | None = None,
) -> None:
    path = Path(raw_path.strip().strip("\'\"")).expanduser()
    if not path.exists():
        await notify(
            f"No file at `{path}`. Type `/digest help` for what this command takes.", "error"
        )
        return
    if path.is_dir():
        await notify(f"`{path}` is a folder. Point me at a single file for now.", "error")
        return

    action = JournalledAction(history, actor_id or "", "digest.ingest")
    if not await action.begin(notify):
        return

    # Off the event loop: a large file takes a moment to read, chunk and
    # compress, and the interface has to keep painting while it does.
    work = asyncio.ensure_future(asyncio.to_thread(ingest, path, store=store))
    try:
        # Shielded, because cancelling the await does not cancel the thread.
        artifact = await asyncio.shield(work)
    except (OSError, ValueError) as exc:
        await action.record(notify, "failed", error_code="ingest_read_failed")
        await notify(f"Could not read `{path.name}`: {exc}", "error")
        return
    except asyncio.CancelledError:
        # The thread is still compressing and will finish on its own. Claiming
        # it stopped would be a lie; "unknown" is the true answer.
        work.add_done_callback(_discard_orphaned_result)
        await action.record(notify, "unknown", error_code="ingest_cancelled_outcome_unknown")
        await notify(
            "Stopped waiting for that digest. The compression may still finish in "
            "the background, so its result is recorded as unknown.",
            "error",
        )
        raise

    # The artifact is committed to the store here. Record that before the
    # optional model pass, which may be slow, may fail, and changes nothing
    # about whether the data was ingested.
    await action.record(
        notify,
        "succeeded",
        already_done=True,
        artifact_id=artifact.id,
        source_kind=artifact.source_kind,
        source_bytes=artifact.stats.source_bytes,
    )
    await notify(format_ingested(artifact), "info")
    await _maybe_summarize(artifact, store=store, notify=notify, backend=backend, model=model)


async def _ingest_source(
    source: SourceRef,
    *,
    store: ContextStore,
    notify: Notify,
    sources: SourceHub | None,
    backend,
    model: str | None,
    history: Any | None = None,
    actor_id: str | None = None,
) -> None:
    """Pull a record from a company system and compress it - no manual export.

    Journalled exactly like a local-file ingest: this is the same act, and
    "where did this digest come from" is a question the audit trail has to be
    able to answer for a pulled record as much as a copied one. The intent
    commits before the fetch, so a source that cannot be reached is a recorded
    attempt rather than a silent nothing.
    """
    if sources is None:
        await notify("Company data sources are not available here.", "error")
        return

    action = JournalledAction(history, actor_id or "", "digest.ingest")
    if not await action.begin(notify):
        return

    try:
        record = await sources.fetch(source)
    except ConnectorError as exc:
        await action.record(notify, "failed", error_code="source_fetch_failed")
        await notify(str(exc), "error")
        return

    # The record's filename carries its suffix, which is how the adapter
    # registry tells a CSV export from a FASTA or a JSON payload.
    data = Source(
        name=record.filename,
        text=record.text(),
        suffix=record.suffix,
        byte_size=len(record.content),
    )
    try:
        artifact = await asyncio.to_thread(ingest, data, store=store)
    except ValueError as exc:
        await action.record(notify, "failed", error_code="ingest_unsupported_source")
        await notify(f"Could not compress `{source}`: {exc}", "error")
        return

    # Committed to the store here, so recorded here - before the optional
    # model pass, exactly as in the local-file path.
    await action.record(
        notify,
        "succeeded",
        already_done=True,
        artifact_id=artifact.id,
        source_kind=artifact.source_kind,
        source_bytes=len(record.content),
        source_sha256=record.sha256,
    )

    version = f", version {record.version}" if record.version else ""
    await notify(
        f"Fetched `{source}` as `{record.fetched_by}`{version} "
        f"(sha256 `{record.sha256[:12]}`).\n\n{format_ingested(artifact)}",
        "info",
    )
    await _maybe_summarize(artifact, store=store, notify=notify, backend=backend, model=model)


async def _maybe_summarize(
    artifact, *, store: ContextStore, notify: Notify, backend, model: str | None
) -> None:
    # The model pass is optional, additive, and skipped for data whose
    # computed summary is already better than prose - see RULES_SUFFICE. The
    # artifact is stored and fully usable before this runs at all.
    if backend is not None and model and would_benefit_from_model(artifact.source_kind):
        upgraded = await summarize(artifact, backend, model, store=store)
        if upgraded.generated_by != "rules":
            await notify(f"**{artifact.name}** — {upgraded.gist}", "info")


async def _forget_artifact(
    artifact,
    *,
    store: ContextStore,
    notify: Notify,
    history: Any | None,
    actor_id: str | None,
) -> None:
    action = JournalledAction(
        history, actor_id or "", "digest.forget", subject_id=artifact.id
    )
    if not await action.begin(notify, artifact_id=artifact.id):
        return
    try:
        await asyncio.to_thread(store.delete, artifact.id)
    except OSError as exc:
        await action.record(notify, "failed", error_code="digest_delete_failed")
        await notify(f"Could not forget `{artifact.name}`: {exc}", "error")
        return
    await action.record(notify, "succeeded", already_done=True, artifact_id=artifact.id)
    await notify(f"Forgot `{artifact.name}` (`{artifact.id}`).", "info")


def _discard_orphaned_result(work: asyncio.Future) -> None:
    """Collect the result of an ingest nobody is waiting for any more.

    The thread outlives the cancelled command, so something has to retrieve
    whatever it ends with or asyncio warns about an exception that was never
    handled. Its outcome is already recorded as unknown; there is nothing
    truthful left to say about it here.
    """
    if not work.cancelled():
        work.exception()


def _not_found(reference: str) -> str:
    if not reference:
        return "Which one? Try `/digest` to see what is stored."
    return f"No stored item matches `{reference}`. Try `/digest` to see the list."
