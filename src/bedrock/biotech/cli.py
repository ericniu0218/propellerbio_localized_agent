"""Operator-friendly command line for the local biotech integration slice."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bedrock.biotech import ProteinWorkflow, ToolIntegrationError
from bedrock.connectors import ConnectorError, SourceHub, parse_source_ref
from bedrock.config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bedrock-biotech")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="check AlphaFold 2 and PyMOL configuration")
    fold = subparsers.add_parser("fold", help="predict one monomer FASTA and render it")
    fold.add_argument(
        "fasta",
        help="a FASTA file, or a company source reference such as registry://SEQ-00123",
    )
    pymol = subparsers.add_parser("pymol", help="open a structure, or the latest result")
    pymol.add_argument("structure", nargs="?", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings()
    workflow = ProteinWorkflow.from_settings(settings)
    if args.command == "status":
        failed = False
        for status in workflow.status():
            print(f"{'ready' if status.ready else 'not ready'}: {status.name}")
            for issue in status.issues:
                print(f"  - {issue}")
            failed = failed or not status.ready
        return 1 if failed else 0
    if args.command == "fold":
        provenance = None
        source = parse_source_ref(args.fasta)
        if source is not None:
            record, fasta_path = await SourceHub.from_settings(settings).fetch_to_file(source)
            provenance = record.provenance()
            print(f"fetched: {source} as {record.fetched_by} ({record.version or 'no version reported'})")
            print(f"sha256: {record.sha256}")
        else:
            fasta_path = Path(args.fasta)
        result = await workflow.predict_and_render(fasta_path, source_provenance=provenance)
        print(f"run: {result.run_id}")
        print(f"structure: {result.structure_path}")
        if result.preview_path:
            print(f"preview: {result.preview_path}")
        if result.session_path:
            print(f"PyMOL session: {result.session_path}")
        print(f"manifest: {result.manifest_path}")
        for warning in result.warnings:
            print(f"warning: {warning}")
        return 0
    opened = await workflow.open_in_pymol(args.structure)
    print(f"opened in PyMOL: {opened}")
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(_run(build_parser().parse_args())))
    except (ToolIntegrationError, ConnectorError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
