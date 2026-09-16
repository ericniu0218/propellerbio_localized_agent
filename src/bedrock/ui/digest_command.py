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
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Awaitable, Callable

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
) -> None:
    """Handle one ``/digest ...`` invocation."""
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
        store.delete(artifact.id)
        await notify(f"Forgot `{artifact.name}` (`{artifact.id}`).", "info")
        return

    source = parse_source_ref(args.strip("'\""))
    if source is not None:
        await _ingest_source(
            source, store=store, notify=notify, sources=sources, backend=backend, model=model
        )
        return

    await _ingest_path(args, store=store, notify=notify, backend=backend, model=model)


async def _ingest_path(
    raw_path: str,
    *,
    store: ContextStore,
    notify: Notify,
    backend,
    model: str | None,
) -> None:
    path = Path(raw_path.strip().strip("'\"")).expanduser()
    if not path.exists():
        await notify(
            f"No file at `{path}`. Type `/digest help` for what this command takes.", "error"
        )
        return
    if path.is_dir():
        await notify(f"`{path}` is a folder. Point me at a single file for now.", "error")
        return

    try:
        # Off the event loop: a large file takes a moment to read, chunk and
        # compress, and the interface has to keep painting while it does.
        artifact = await asyncio.to_thread(ingest, path, store=store)
    except (OSError, ValueError) as exc:
        await notify(f"Could not read `{path.name}`: {exc}", "error")
        return

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
) -> None:
    """Pull a record from a company system and compress it - no manual export."""
    if sources is None:
        await notify("Company data sources are not available here.", "error")
        return
    try:
        record = await sources.fetch(source)
    except ConnectorError as exc:
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
        await notify(f"Could not compress `{source}`: {exc}", "error")
        return

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


def _not_found(reference: str) -> str:
    if not reference:
        return "Which one? Try `/digest` to see what is stored."
    return f"No stored item matches `{reference}`. Try `/digest` to see the list."
