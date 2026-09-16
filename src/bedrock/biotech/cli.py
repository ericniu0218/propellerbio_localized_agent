"""Operator-friendly command line for the local biotech integration slice.

The same operations the terminal app offers, for people and scripts that want
them without a full-screen interface: check the tools, run a fold, open a
structure, and - the reason this file grew - read back what is already on
this machine.

``runs``, ``confidence`` and ``audit`` print exactly what the app's `/runs`,
`/confidence` and `/audit` show, because they call the same renderers. Two
interfaces onto one set of facts is fine; two descriptions of those facts
that drift apart is how a compliance answer ends up depending on which
program the employee happened to open.

Output goes through Rich, so the Markdown those renderers produce arrives as
formatted terminal text rather than raw pipes and asterisks.
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from bedrock.biotech import ProteinWorkflow, ToolIntegrationError
from bedrock.connectors import ConnectorError, SourceHub, parse_source_ref
from bedrock.config import load_settings
from bedrock.provenance import HistoryError, HistoryStore, RunCatalog, RunError
from bedrock.ui.audit_command import format_events
from bedrock.ui.confidence_view import format_confidence_report
from bedrock.ui.runs_command import format_run_detail, format_run_page, format_verification


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bedrock-biotech")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="check AlphaFold 2 and PyMOL configuration")

    fold = subparsers.add_parser("fold", help="predict one monomer FASTA and render it")
    fold.add_argument(
        "fasta",
        help="a FASTA file, or a company source reference such as registry://SEQ-00123",
    )
    fold.add_argument("--project", help="project label recorded with the run")

    pymol = subparsers.add_parser("pymol", help="open a structure, or the latest result")
    pymol.add_argument("structure", nargs="?", type=Path)

    # `runs` with no sub-action lists; the filters live on the parent parser so
    # that `bedrock-biotech runs --project X --since 2026-07-01` reads the way
    # the documentation writes it.
    runs = subparsers.add_parser("runs", help="browse local prediction runs")
    runs.add_argument("--project")
    runs.add_argument("--since", help="UTC date, inclusive (YYYY-MM-DD)")
    runs.add_argument("--before", help="UTC date, exclusive (YYYY-MM-DD)")
    runs.add_argument("--status")
    runs.add_argument("--limit", type=int, default=25)
    runs.add_argument("--offset", type=int, default=0)
    run_actions = runs.add_subparsers(dest="runs_action")
    run_actions.add_parser("show", help="settings, lineage and integrity").add_argument("run_id")
    run_actions.add_parser("verify", help="re-hash recorded artifacts").add_argument("run_id")
    tag = run_actions.add_parser("tag", help="assign a run to a project")
    tag.add_argument("run_id")
    tag.add_argument("--project", required=True)

    confidence = subparsers.add_parser("confidence", help="pLDDT bands, low regions and PAE")
    confidence.add_argument("run_id")

    audit = subparsers.add_parser("audit", help="read the local action history")
    audit.add_argument("--after", type=int, default=0)
    audit.add_argument("--limit", type=int, default=25)
    audit_actions = audit.add_subparsers(dest="audit_action")
    audit_actions.add_parser("export", help="write the journal as JSONL").add_argument(
        "destination", type=Path
    )
    return parser


def _print(markdown: str) -> None:
    Console().print(Markdown(markdown))


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings()

    if args.command == "status":
        workflow = ProteinWorkflow.from_settings(settings)
        failed = False
        for status in workflow.status():
            print(f"{'ready' if status.ready else 'not ready'}: {status.name}")
            for issue in status.issues:
                print(f"  - {issue}")
            failed = failed or not status.ready
        return 1 if failed else 0

    if args.command == "fold":
        workflow = ProteinWorkflow.from_settings(settings)
        provenance = None
        source = parse_source_ref(args.fasta)
        if source is not None:
            record, fasta_path = await SourceHub.from_settings(settings).fetch_to_file(source)
            provenance = record.provenance()
            print(f"fetched: {source} as {record.fetched_by} ({record.version or 'no version reported'})")
            print(f"sha256: {record.sha256}")
        else:
            fasta_path = Path(args.fasta)
        result = await workflow.predict_and_render(
            fasta_path, project_id=args.project, source_provenance=provenance
        )
        print(f"run: {result.run_id}")
        print(f"structure: {result.structure_path}")
        if result.preview_path:
            print(f"preview: {result.preview_path}")
        if result.session_path:
            print(f"PyMOL session: {result.session_path}")
        if result.confidence_path:
            print(f"confidence summary: {result.confidence_path}")
        print(f"manifest: {result.manifest_path}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        print(f"confidence detail: bedrock-biotech confidence {result.run_id}")
        return 0

    if args.command == "pymol":
        workflow = ProteinWorkflow.from_settings(settings)
        opened = await workflow.open_in_pymol(args.structure)
        print(f"opened in PyMOL: {opened}")
        return 0

    # Everything below reads local files only: no model server, no network,
    # no scientific tool needs to be installed.
    history = HistoryStore(Path(settings.action_history_dir))

    if args.command == "audit":
        if args.audit_action == "export":
            count = history.export(args.destination.expanduser())
            print(f"exported {count} event(s) to {args.destination}")
            return 0
        _print(
            format_events(
                history.events(after=args.after, limit=args.limit),
                after=args.after,
                prefix="bedrock-biotech audit",
            )
        )
        return 0

    catalog = RunCatalog(settings.biotech_work_dir, history)

    if args.command == "confidence":
        _print(format_confidence_report(catalog.confidence(args.run_id)))
        return 0

    action = getattr(args, "runs_action", None)
    if action == "show":
        detail = catalog.show(args.run_id)
        _print(format_run_detail(detail, catalog.verify(args.run_id), "bedrock-biotech confidence"))
    elif action == "verify":
        _print(format_verification(args.run_id, catalog.verify(args.run_id)))
    elif action == "tag":
        catalog.tag(
            args.run_id,
            args.project,
            actor_id=settings.user_id,
            operation_id=str(uuid.uuid4()),
        )
        print(f"run {args.run_id} is now assigned to project {args.project}")
    else:
        page = catalog.list(
            project_id=args.project,
            since=args.since,
            before=args.before,
            status=args.status,
            limit=args.limit,
            offset=args.offset,
        )
        _print(format_run_page(page, args, prefix="bedrock-biotech runs"))
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run(build_parser().parse_args())))
    except (ToolIntegrationError, ConnectorError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    except (HistoryError, RunError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
