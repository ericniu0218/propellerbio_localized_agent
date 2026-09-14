"""Operator-friendly command line for the local biotech integration slice."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bedrock.biotech import ProteinWorkflow, ToolIntegrationError
from bedrock.config import load_settings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bedrock-biotech")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="check AlphaFold 2 and PyMOL configuration")
    fold = subparsers.add_parser("fold", help="predict one monomer FASTA and render it")
    fold.add_argument("fasta", type=Path)
    pymol = subparsers.add_parser("pymol", help="open a structure, or the latest result")
    pymol.add_argument("structure", nargs="?", type=Path)
    return parser


async def _run(args: argparse.Namespace) -> int:
    workflow = ProteinWorkflow.from_settings(load_settings())
    if args.command == "status":
        failed = False
        for status in workflow.status():
            print(f"{'ready' if status.ready else 'not ready'}: {status.name}")
            for issue in status.issues:
                print(f"  - {issue}")
            failed = failed or not status.ready
        return 1 if failed else 0
    if args.command == "fold":
        result = await workflow.predict_and_render(args.fasta)
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
    except ToolIntegrationError as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
