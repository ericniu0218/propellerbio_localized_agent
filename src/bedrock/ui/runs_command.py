"""The `/runs` command: browse the prediction runs already on this machine.

Per-run manifests have existed since the first AlphaFold 2 slice, but nothing
could read them back, so an employee asked "what did we fold for project X
last quarter?" had to open JSON files by hand. This command is the answer to
that question and nothing more: it lists, filters, opens and integrity-checks
runs that are already there.

Three things it deliberately does not do:

* It never asks a language model. These are deterministic questions about
  local files, and an answer that was *plausible* rather than *correct* would
  be worse than no answer at all.
* It never needs a network. Browsing works with the model server switched off.
* It never rewrites a manifest. ``/runs tag`` records a new project assignment
  as a journal event; the original manifest stays byte-for-byte as it was.

Reading and hashing happen in ``asyncio.to_thread``; only the notice is
produced on the event loop.
"""

from __future__ import annotations

import asyncio
import shlex
import uuid
from typing import Any

from bedrock.ui.argparsing import CommandParser, CommandUsageError, split_arguments
from bedrock.ui.rendering import as_cell, as_code, as_code_cell, safe_line

RUNS_HELP = """\
**`/runs` — prediction runs on this machine**

| Command | What it does |
|---|---|
| `/runs` | Most recent runs (25 per page, 100 maximum) |
| `/runs --project X --since 2026-07-01 --before 2026-10-01` | Project X, this quarter |
| `/runs --status completed --limit 50 --offset 50` | Filter and page |
| `/runs show RUN_ID` | Settings, source and output lineage, integrity check |
| `/runs verify RUN_ID` | Re-hash every recorded artifact |
| `/runs tag RUN_ID --project X` | Assign a run to a project |

Dates are UTC: `--since` includes that day, `--before` excludes it. Quoted
paths and labels work as they do in a shell. Runs made before project labels
existed stay visible as unassigned until you tag them. A project label
organises results; it grants nobody access to anything.

Status is always the **last recorded** status. A run left as `running_alphafold2`
may still be running, or may have died with the machine; local history cannot
tell those apart and does not guess.
"""

_STATE_NOTES = {
    "match": "hash agrees with the manifest",
    "mismatch": "file changed since the run",
    "missing": "file is no longer there",
    "unrecorded": "no hash was recorded at the time",
    "invalid": "entry could not be read",
}


async def run_runs_command(args: str, *, services: Any, notify) -> None:
    """Handle one ``/runs ...`` invocation."""
    text = args.strip()
    verb, _, rest = text.partition(" ")
    try:
        if verb in {"help", "?"}:
            await notify(RUNS_HELP, "info")
        elif verb == "show":
            await _show(rest.strip(), services=services, notify=notify)
        elif verb == "verify":
            await _verify(rest.strip(), services=services, notify=notify)
        elif verb == "tag":
            await _tag(rest.strip(), services=services, notify=notify)
        else:
            await _list(text, services=services, notify=notify)
    except CommandUsageError as exc:
        # A syntax mistake is a message, never an exit: this is a full-screen
        # application and the employee is in the middle of something.
        await notify(f"{safe_line(exc)}\n\nType `/runs help` for the options.", "error")
    except services.errors as exc:
        await notify(safe_line(exc, limit=400), "error")


# ---- The four sub-commands -------------------------------------------------


async def _list(text: str, *, services: Any, notify) -> None:
    parser = CommandParser("/runs")
    parser.add_argument("--project")
    parser.add_argument("--since")
    parser.add_argument("--before")
    parser.add_argument("--status")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--offset", type=int, default=0)
    options = parser.parse_args(split_arguments(text))

    catalog = await services.runs()
    page = await asyncio.to_thread(
        catalog.list,
        project_id=options.project,
        since=options.since,
        before=options.before,
        status=options.status,
        limit=options.limit,
        offset=options.offset,
    )
    await notify(format_run_page(page, options), "info")


async def _show(run_id: str, *, services: Any, notify) -> None:
    run_id = _one_run_id(run_id, "/runs show")
    catalog = await services.runs()
    detail, verification = await asyncio.to_thread(_read_detail, catalog, run_id)
    await notify(format_run_detail(detail, verification), "info")


async def _verify(run_id: str, *, services: Any, notify) -> None:
    run_id = _one_run_id(run_id, "/runs verify")
    catalog = await services.runs()
    rows = await asyncio.to_thread(catalog.verify, run_id)
    await notify(format_verification(run_id, rows), "info")


async def _tag(text: str, *, services: Any, notify) -> None:
    parser = CommandParser("/runs tag")
    parser.add_argument("run_id")
    parser.add_argument("--project", required=True)
    options = parser.parse_args(split_arguments(text))

    catalog = await services.runs()
    # One UUID per actual invocation: tagging the same run twice is two
    # deliberate acts and both belong in history.
    await asyncio.to_thread(
        catalog.tag,
        options.run_id,
        options.project,
        actor_id=services.actor_id,
        operation_id=str(uuid.uuid4()),
    )
    await notify(
        f"Run {as_code(options.run_id)} is now assigned to project "
        f"{as_code(options.project)}. The original manifest is unchanged, and this "
        "label grants no access to anything.",
        "info",
    )


def _read_detail(catalog: Any, run_id: str) -> tuple[dict, list[dict]]:
    """Both reads in one thread hop, so the page is drawn once."""
    detail = catalog.show(run_id)
    try:
        verification = catalog.verify(run_id)
    except Exception:  # noqa: BLE001 - the detail page is still worth showing
        verification = []
    return detail, verification


def _one_run_id(text: str, prog: str) -> str:
    words = split_arguments(text)
    if len(words) != 1:
        raise CommandUsageError(f"{prog}: expected exactly one run ID")
    return words[0]


# ---- Rendering --------------------------------------------------------------


def format_run_page(page: dict, options: Any, prefix: str = "/runs") -> str:
    """``prefix`` is how the reader would type the next page: the slash command
    in the app, ``bedrock-biotech runs`` on the command line."""
    items = list(page.get("items", []))
    total = int(page.get("total", len(items)))
    if not items:
        body = ["_No runs match._ Runs appear here as soon as a fold starts."]
    else:
        body = [
            "| Run | Created (UTC) | Last recorded status | Project | Residues |",
            "|---|---|---|---|---|",
        ]
        for item in items:
            body.append(
                "| {run} | {created} | {status} | {project} | {residues} |".format(
                    run=as_code_cell(item.get("run_id", "")),
                    created=as_cell(item.get("created_at", "")),
                    status=as_cell(item.get("status", "unknown")),
                    project=as_code_cell(item["project_id"])
                    if item.get("project_id")
                    else "_unassigned_",
                    residues=item.get("residue_count") if item.get("residue_count") else "—",
                )
            )

    offset = int(getattr(options, "offset", 0) or 0)
    heading = (
        f"**Runs** — showing {offset + 1}–{offset + len(items)} of {total}"
        if items
        else "**Runs** — none match"
    )
    lines = [heading, "", *body]
    next_offset = page.get("next_offset")
    if next_offset is not None:
        lines.extend(["", f"Next page: `{prefix} {_repeat_filters(options, next_offset)}`"])
    for warning in page.get("warnings", ()):
        lines.append(f"- Warning: {as_cell(warning, limit=300)}")
    detail = f"{prefix} show RUN_ID"
    confidence = "/confidence RUN_ID" if prefix == "/runs" else "bedrock-biotech confidence RUN_ID"
    lines.extend(["", f"Open one with `{detail}`, or its confidence with `{confidence}`."])
    return "\n".join(lines)


def _repeat_filters(options: Any, offset: int) -> str:
    parts = []
    for name in ("project", "since", "before", "status"):
        value = getattr(options, name, None)
        if value:
            # shlex.quote, because the same splitting reads this line back:
            # whatever the employee typed has to survive a round trip.
            parts.append(f"--{name} {shlex.quote(safe_line(value, limit=64))}")
    limit = int(getattr(options, "limit", 25) or 25)
    if limit != 25:
        parts.append(f"--limit {limit}")
    parts.append(f"--offset {offset}")
    return " ".join(parts)


def format_run_detail(
    detail: dict,
    verification: list[dict],
    confidence_command: str = "/confidence",
) -> str:
    manifest = detail.get("manifest", {}) or {}
    source = manifest.get("source", {}) or {}
    alphafold = manifest.get("alphafold2", {}) or {}
    pymol = manifest.get("pymol", {}) or {}

    lines = [
        f"**Run** {as_code(detail.get('run_id', ''))}",
        "",
        f"- Project: {as_code(detail['project_id']) if detail.get('project_id') else '_unassigned_'}",
        f"- Created (UTC): {as_cell(manifest.get('created_at', 'unknown'))}",
        f"- Last recorded status: {as_cell(manifest.get('status', 'unknown'))}",
        f"- Finished (UTC): {as_cell(manifest.get('finished_at', '—'))}",
        f"- Manifest: {as_code(detail.get('manifest_path', ''))}",
        "",
        "**Settings used**",
        "",
        f"- Model preset: {as_cell(alphafold.get('model_preset', '—'))}, "
        f"database preset: {as_cell(alphafold.get('db_preset', '—'))}",
        f"- Template cutoff: {as_cell(alphafold.get('max_template_date', '—'))}",
        f"- Container image: {as_code(alphafold.get('docker_image') or '—')}",
        f"- Database snapshot: {as_cell(alphafold.get('database_version', '—'))}",
        f"- Relaxed models: {as_cell(alphafold.get('models_to_relax', '—'))}, "
        f"GPUs: {as_cell(alphafold.get('gpu_devices', '—'))}",
        f"- Selected model: {as_code(alphafold.get('selected_model') or 'not recorded')}",
        f"- PyMOL: {as_code(pymol.get('executable', '—'))} "
        f"(confidence colouring: {as_cell(pymol.get('confidence_metric', 'none'))})",
        "",
        "**Source**",
        "",
        f"- Original file: {as_code(source.get('original_filename', '—'))}",
        f"- Description: {as_cell(source.get('description', '—'), limit=200)}",
        f"- Residues: {source.get('residue_count', '—')}",
        f"- SHA-256 of the submitted bytes: {as_code(source.get('sha256', '—'))}",
        "",
        "**Outputs**",
        "",
    ]
    outputs = _output_entries(manifest.get("outputs", {}) or {})
    if outputs:
        lines.append("| Output | Path | Bytes |")
        lines.append("|---|---|---|")
        for name, entry in outputs:
            lines.append(
                f"| {as_cell(name)} | {as_code_cell(entry.get('path', '—'))} | "
                f"{entry.get('bytes', '—')} |"
            )
    else:
        lines.append("_No outputs were recorded for this run._")

    lines.extend(["", format_verification(detail.get("run_id", ""), verification, heading=False)])

    warnings = list(detail.get("warnings", ()) or []) + list(manifest.get("warnings", ()) or [])
    if warnings:
        lines.extend(["", "**Warnings**", ""])
        lines.extend(f"- {as_cell(warning, limit=300)}" for warning in warnings)
    lines.extend(
        ["", f"Confidence: `{confidence_command} {safe_line(detail.get('run_id', ''), limit=64)}`"]
    )
    return "\n".join(lines)


def _output_entries(outputs: dict) -> list[tuple[str, dict]]:
    entries: list[tuple[str, dict]] = []
    for name, value in outputs.items():
        if isinstance(value, dict):
            entries.append((name, value))
        elif isinstance(value, list):
            entries.extend((f"{name}[{index}]", item) for index, item in enumerate(value) if isinstance(item, dict))
    return entries


def format_verification(run_id: str, rows: list[dict], heading: bool = True) -> str:
    lines = []
    if heading:
        lines.extend([f"**Integrity check** — run {as_code(run_id)}", ""])
    else:
        lines.extend(["**Integrity check**", ""])
    if not rows:
        lines.append("_Nothing recorded to check._")
        return "\n".join(lines)
    lines.append("| Artifact | Result | Recorded SHA-256 | On disk now |")
    lines.append("|---|---|---|---|")
    for row in rows:
        state = row.get("state", "invalid")
        note = _STATE_NOTES.get(state, state)
        lines.append(
            "| {path} | {state} | {expected} | {actual} |".format(
                path=as_code_cell(row.get("path", "—")),
                state=f"{as_cell(state)} — {note}",
                expected=as_code_cell(_short_hash(row.get("expected_sha256"))),
                actual=as_code_cell(_short_hash(row.get("actual_sha256"))),
            )
        )
    lines.extend(
        [
            "",
            "*Matching hashes show the files are unchanged since this run recorded "
            "them. That is an integrity comparison with a local manifest, not proof "
            "of who produced them.*",
        ]
    )
    return "\n".join(lines)


def _short_hash(value: str | None) -> str:
    if not value:
        return "—"
    return f"{value[:12]}…"
