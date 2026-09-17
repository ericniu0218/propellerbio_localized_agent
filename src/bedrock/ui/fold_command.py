"""Shared pieces of the `/fold`, `/pymol` and `/tools` entry points.

Both apps and ``bedrock-biotech`` must agree about how a fold's arguments are
read, where a sequence or structure comes from (a path on this machine, or a
company source reference such as ``registry://SEQ-00123``), and what is said
while a fold runs and when it finishes. The cheapest way to guarantee that is
for all of them to call the same functions, so those functions live here.

The workflow itself is not here. Folding is owned by ``ProteinWorkflow``,
which also writes the one journal entry a fold produces - wrapping it in a
second event from the UI would double-count every prediction in the history.

What differs between the apps is only *when* these run. Agent-pack mode puts a
fold in its prompt queue, because the agent system answers one request at a
time per session. The plain model mode runs a fold beside the chat instead: the
model server and the GPU host are different machines, and a prediction that
takes hours must not stop anyone asking the model a question meanwhile.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any, Awaitable, Callable

from bedrock.biotech import ToolIntegrationError
from bedrock.connectors import ConnectorError, SourceHub, SourceRef, parse_source_ref
from bedrock.ui.argparsing import CommandParser, CommandUsageError, split_arguments
from bedrock.ui.confidence_view import short_confidence_summary
from bedrock.ui.rendering import as_code, safe_line

Notify = Callable[[str, str], Awaitable[None]]

FOLD_TARGET_MISSING = (
    "Tell me which one-protein FASTA file to fold, or use "
    "`/fold <path or source://id> [--project X]`."
)


def parse_fold_arguments(args: str) -> tuple[str, str | None]:
    """``<fasta or source://id> [--project X]``, with shell-style quoting.

    The target is returned as written. A company source reference has no file
    suffix and is not a path, so classifying it is the caller's job - this
    function only separates the target from the options, without a shell.
    """
    parser = CommandParser("/fold")
    parser.add_argument("fasta")
    parser.add_argument("--project")
    options = parser.parse_args(split_arguments(args))
    return options.fasta, options.project


def parse_fold_request(args: str) -> tuple[Path | SourceRef, str | None]:
    """What ``/fold`` should fold, and the optional project label.

    Everything that must not start a GPU job - nothing typed, a typo in the
    options - raises :class:`CommandUsageError` here, before any job exists.
    """
    if not args.strip():
        raise CommandUsageError(FOLD_TARGET_MISSING)
    target, project_id = parse_fold_arguments(args)
    return path_or_source(target), project_id


def parse_structure_argument(args: str) -> Path | SourceRef | None:
    """``/pymol [structure or source://id]`` - quoted paths included.

    None means "the latest prediction on this machine".
    """
    parser = CommandParser("/pymol")
    parser.add_argument("structure", nargs="?")
    options = parser.parse_args(split_arguments(args))
    return path_or_source(options.structure) if options.structure else None


def path_or_source(text: str) -> Path | SourceRef:
    """``registry://SEQ-00123`` is fetched from a company system; anything else
    is a path on this machine."""
    source = parse_source_ref(text)
    return source if source is not None else Path(text).expanduser()


async def predict_and_render(
    workflow: Any,
    fasta_path: Path,
    project_id: str | None,
    *,
    source_provenance: dict | None = None,
):
    """Start a fold, with the project label the employee asked for.

    Thin on purpose. The workflow validates the label, records the run and
    writes the single journal entry a fold produces; this exists so that the
    app and the CLI cannot start a fold in two subtly different ways.

    ``source_provenance`` is passed straight through when the sequence came
    from a company system of record rather than a file on disk.
    """
    return await workflow.predict_and_render(
        fasta_path, project_id=project_id, source_provenance=source_provenance
    )


async def confidence_summary(services: Any, run_id: str) -> str:
    """One line about a finished fold's confidence, or why there isn't one.

    Confidence never decides whether a fold counts as successful. An expensive
    prediction whose confidence JSON is missing or unreadable still has usable
    coordinates, and this line says so rather than hiding the result.
    """
    try:
        catalog = await services.runs()
        report = await asyncio.to_thread(catalog.confidence, run_id)
    except Exception as exc:  # noqa: BLE001 - never let this fail a fold
        return f"Confidence is unavailable for this run ({safe_line(exc, limit=200)})."
    return short_confidence_summary(report)


async def run_fold(
    target: Path | SourceRef,
    project_id: str | None,
    *,
    workflow: Any,
    sources: SourceHub,
    services: Any,
    notify: Notify,
) -> bool:
    """Fetch the sequence if it lives in a company system, fold it, report.

    Returns True only for a finished prediction. A refused fetch, a tool that
    is not set up and a failed prediction are each reported to the employee
    and return False; none of them raises, so the caller's own bookkeeping
    (one job at a time, the prompt queue) always gets to run.

    A source record is fetched and staged with its provenance first, and that
    provenance goes into the run manifest, so the prediction names the
    registry record and version it came from rather than only the staging
    file. ``project_id`` is orthogonal to where the sequence came from: a run
    can be labelled for a project whether it was pulled or handed over.
    """
    provenance = None
    label = f" for project {as_code(project_id)}" if project_id else ""
    try:
        if isinstance(target, SourceRef):
            await notify(f"Fetching {as_code(target)}…", "info")
            record, fasta_path = await sources.fetch_to_file(target)
            provenance = record.provenance()
        else:
            fasta_path = target
        await notify(
            f"Started AlphaFold 2 for {as_code(target)}{label}. This may take a long "
            "time; the run is preserved if PyMOL rendering fails.",
            "info",
        )
        result = await predict_and_render(
            workflow, fasta_path, project_id, source_provenance=provenance
        )
    except (ToolIntegrationError, ConnectorError) as exc:
        await notify(safe_line(exc, limit=400), "error")
        return False
    except asyncio.CancelledError:
        # The prediction itself may still be running in its own process
        # group; the workflow owns that outcome, not this notice. Saying so is
        # best effort: quitting the app is what usually cancels a fold, and by
        # then there is no screen left to say it on. Whatever that attempt
        # raises must not replace the cancellation, or quitting mid-fold ends
        # in a traceback instead of a closed window.
        with contextlib.suppress(Exception):
            await notify("The scientific job was cancelled.", "error")
        raise

    lines = [
        f"**Protein prediction completed** — run {as_code(result.run_id)}",
        "",
        f"- Structure: {as_code(result.structure_path)}",
        f"- Provenance: {as_code(result.manifest_path)}",
    ]
    if provenance:
        version = f", version {provenance['version']}" if provenance.get("version") else ""
        lines.insert(
            2,
            f"- Source: `{provenance['ref']}`{version} (sha256 `{provenance['sha256'][:12]}`)",
        )
    if result.preview_path:
        lines.append(f"- PyMOL preview: {as_code(result.preview_path)}")
    if result.session_path:
        lines.append(f"- PyMOL session: {as_code(result.session_path)}")
    if getattr(result, "confidence_path", None):
        lines.append(f"- Confidence summary: {as_code(result.confidence_path)}")
    lines.extend(f"- Warning: {safe_line(warning, limit=300)}" for warning in result.warnings)
    # A short line only: the full bands, intervals and PAE overview are a page
    # of their own, and a failed confidence read must never take a successful,
    # expensive structure down with it.
    lines.extend(
        [
            "",
            f"Confidence: {await confidence_summary(services, result.run_id)}",
            f"Full detail: `/confidence {safe_line(result.run_id, limit=64)}`",
            "",
            "Use `/pymol` to open the latest structure interactively.",
        ]
    )
    await notify("\n".join(lines), "info")
    return True


async def run_pymol(
    target: Path | SourceRef | None,
    *,
    workflow: Any,
    sources: SourceHub,
    notify: Notify,
) -> bool:
    """Open a structure in PyMOL: the latest prediction, a file, or a record.

    A record pulled from a company system is staged with its provenance
    first, exactly as a fold's sequence is. It opens with neutral styling,
    like any structure the employee names: its B-factor column is whatever
    the source put there - crystallographic displacement, for an experimental
    structure - and colouring it as model confidence would be a false claim.
    Only the latest prediction, verified from its own run, is coloured by
    confidence. Returns True if PyMOL was started; never raises for a refused
    fetch or a missing tool.
    """
    try:
        if isinstance(target, SourceRef):
            await notify(f"Fetching {as_code(target)}…", "info")
            record, structure_path = await sources.fetch_to_file(target)
            opened = await workflow.open_in_pymol(structure_path)
            version = f"version {safe_line(record.version, limit=80)}, " if record.version else ""
            await notify(
                f"Opened {as_code(target)} in PyMOL ({version}sha256 "
                f"`{record.sha256[:12]}`, saved at {as_code(opened)}).",
                "info",
            )
        else:
            opened = await workflow.open_in_pymol(target)
            await notify(f"Opened {as_code(opened)} in PyMOL.", "info")
    except (ToolIntegrationError, ConnectorError) as exc:
        await notify(safe_line(exc, limit=400), "error")
        return False
    return True


def tool_status_text(workflow: Any) -> str:
    """``/tools``: whether AlphaFold 2 and PyMOL are set up, and if not, why."""
    lines = ["**Scientific tools**", ""]
    for status in workflow.status():
        lines.append(f"- {'Ready' if status.ready else 'Not ready'}: **{status.name}**")
        lines.extend(f"  - {issue}" for issue in status.issues)
    return "\n".join(lines)
