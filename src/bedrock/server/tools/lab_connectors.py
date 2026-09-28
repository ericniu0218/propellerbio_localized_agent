"""Lab data connector stubs: open-source ELN/LIMS systems and plate-reader /
instrument CSV exports. No specific company system was named (see
docs/ARCHITECTURE.md open question 1), so this ships a typed interface
loosely shaped like common open-source ELN/LIMS REST APIs, returning a
clear "not connected" result until a real backend is wired in.
"""

from __future__ import annotations

import csv
import io
from typing import ClassVar, Literal

from pydantic import BaseModel

from bedrock.server.tools.base import Tool, ToolResult


class ELNLIMSArgs(BaseModel):
    operation: Literal["get_experiment", "get_sample", "parse_instrument_csv"]
    record_id: str | None = None      # for get_experiment / get_sample
    csv_content: str | None = None    # for parse_instrument_csv


class ELNLIMSTool(Tool):
    name: ClassVar[str] = "eln_lims"
    description: ClassVar[str] = (
        "Stub interface for the company's electronic lab notebook / LIMS "
        "system, plus a working parser for plate-reader/instrument CSV "
        "exports. get_experiment/get_sample are not connected to a real "
        "backend yet — wire a real client behind this same interface."
    )
    args_model: ClassVar[type[BaseModel]] = ELNLIMSArgs

    async def run(self, args: ELNLIMSArgs) -> ToolResult:
        if args.operation == "parse_instrument_csv":
            return self._parse_csv(args.csv_content or "")
        return ToolResult(
            summary=(
                f"No ELN/LIMS system is connected. This is a stub for "
                f"'{args.operation}' (record_id={args.record_id!r}). "
                "Provide the company's actual ELN/LIMS endpoint to wire a "
                "real client behind this interface."
            ),
            content="",
            citations=[],
        )

    def _parse_csv(self, csv_content: str) -> ToolResult:
        if not csv_content.strip():
            return ToolResult(summary="No CSV content provided.", content="", citations=[])
        reader = csv.DictReader(io.StringIO(csv_content))
        rows = list(reader)
        return ToolResult(
            summary=f"Parsed {len(rows)} row(s) from instrument CSV export.",
            content=str(rows[:50]),  # cap what gets passed upstream
            citations=[],
        )
