"""Shared pieces of the `/fold` entry points.

The agent-pack app and ``bedrock-biotech fold`` must agree about three small
things, and the cheapest way to guarantee that is for both to call the same
functions: how a fold's arguments are read, how the optional project label
reaches the workflow, and what is said about confidence when a fold finishes.

The workflow itself is not here. Folding is owned by ``ProteinWorkflow``,
which also writes the one journal entry a fold produces - wrapping it in a
second event from the UI would double-count every prediction in the history.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from bedrock.ui.argparsing import CommandParser, split_arguments
from bedrock.ui.confidence_view import short_confidence_summary
from bedrock.ui.rendering import safe_line


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


def parse_structure_argument(args: str) -> Path | None:
    """``/pymol [structure]`` - quoted paths included."""
    parser = CommandParser("/pymol")
    parser.add_argument("structure", nargs="?")
    options = parser.parse_args(split_arguments(args))
    return Path(options.structure).expanduser() if options.structure else None


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
