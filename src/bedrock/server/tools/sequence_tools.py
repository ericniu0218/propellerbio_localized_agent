"""Sequence & structure tools: Biopython for parsing/basic ops, local BLAST
for search, and a local protein structure tool. All three are optional
dependencies / external binaries — each is stubbed with a clear "not
installed" result if unavailable, per the spec's "stub anything that needs
a company system we don't have yet" instruction.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import ClassVar, Literal

from pydantic import BaseModel

from bedrock.server.tools.base import Tool, ToolResult

try:
    from Bio.Seq import Seq  # type: ignore

    _BIOPYTHON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when Biopython is absent
    _BIOPYTHON_AVAILABLE = False


class SequenceToolArgs(BaseModel):
    operation: Literal["transcribe", "translate", "reverse_complement", "blastn", "structure_predict"]
    sequence: str
    database: str | None = None  # for blastn


class SequenceStructureTool(Tool):
    name: ClassVar[str] = "sequence_structure"
    description: ClassVar[str] = (
        "Biopython sequence operations (transcribe/translate/reverse-complement), "
        "local BLAST search, and local protein structure prediction. Stubs a "
        "clear 'not installed/configured' result for any missing dependency."
    )
    args_model: ClassVar[type[BaseModel]] = SequenceToolArgs

    async def run(self, args: SequenceToolArgs) -> ToolResult:
        if args.operation in ("transcribe", "translate", "reverse_complement"):
            return self._biopython_op(args)
        if args.operation == "blastn":
            return self._blast(args)
        if args.operation == "structure_predict":
            return self._structure_predict(args)
        raise ValueError(f"Unsupported operation: {args.operation}")

    def _biopython_op(self, args: SequenceToolArgs) -> ToolResult:
        if not _BIOPYTHON_AVAILABLE:
            return ToolResult(
                summary="Biopython is not installed; this is a stub result.",
                content="",
                citations=[],
            )
        seq = Seq(args.sequence)
        if args.operation == "transcribe":
            result = str(seq.transcribe())
        elif args.operation == "translate":
            result = str(seq.translate())
        else:
            result = str(seq.reverse_complement())
        return ToolResult(summary=f"{args.operation} result: {result}", content=result, citations=[])

    def _blast(self, args: SequenceToolArgs) -> ToolResult:
        binary = shutil.which("blastn")
        if not binary:
            return ToolResult(
                summary=(
                    "Local BLAST (blastn) is not installed/configured on this host. "
                    "Stub interface: call with {operation: 'blastn', sequence, database} "
                    "once a local BLAST DB is available."
                ),
                content="",
                citations=[],
            )
        # Real invocation, only reached once `blastn` is actually on PATH.
        proc = subprocess.run(
            [binary, "-query", "-", "-db", args.database or "nt", "-outfmt", "6"],
            input=args.sequence,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return ToolResult(
            summary="Local BLAST search completed.",
            content=proc.stdout,
            citations=[f"local-blast-db:{args.database or 'nt'}"],
        )

    def _structure_predict(self, args: SequenceToolArgs) -> ToolResult:
        # No local structure-prediction tool (e.g. a local ColabFold/ESMFold
        # install) is assumed present. Stub with a clear interface to wire a
        # real one later.
        return ToolResult(
            summary=(
                "Local protein structure prediction is not configured on this "
                "host. Stub interface: call with {operation: 'structure_predict', "
                "sequence} once a local structure tool (e.g. ESMFold/ColabFold "
                "run locally) is installed."
            ),
            content="",
            citations=[],
        )
