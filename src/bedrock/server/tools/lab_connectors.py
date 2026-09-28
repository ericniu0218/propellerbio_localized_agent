"""Lab data connectors: open-source ELN/LIMS systems and plate-reader /
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
    # A record in a configured system of record, as `name://id` (bedrock's
    # connectors, [server.connectors.<name>]); a record_id of that shape works too.
    source: str | None = None
    user_id: str | None = None        # sent to the source system as who asked


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
        reference = args.source or (args.record_id if args.record_id and "://" in args.record_id else None)
        hub = _source_hub(args.user_id)
        if hub is not None and reference:
            return await _fetch(hub, reference)
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


# --- systems of record, through bedrock's connectors ---------------------------------
#
# get_experiment/get_sample were stubs. bedrock-private-ai already had generic
# connectors for company systems (bedrock.connectors: `rest` for HTTP/JSON APIs,
# `share` for export folders), with provenance and path-escape guards; the
# server now uses them. Access is decided by the source system, as in the
# terminal app. Still read-only.

MAX_CONTENT_CHARS = 20_000


def _source_hub(user_id: str | None):
    from bedrock.connectors import SourceHub, build_connectors
    from bedrock.server.config import get_settings

    settings = get_settings()
    connectors, _errors = build_connectors(settings.connectors)
    if not connectors:
        return None
    return SourceHub(connectors, user=user_id or "agent-pack", staging_dir=settings.data_dir / "sources")


async def _fetch(hub, reference: str) -> ToolResult:
    from bedrock.connectors import ConnectorError

    try:
        record = await hub.fetch(reference)
    except ConnectorError as exc:
        return ToolResult(summary=f"Could not fetch {reference}: {exc}", content="", citations=[])
    text = record.content.decode("utf-8", errors="replace")
    version = record.version or record.sha256[:12]
    return ToolResult(
        summary=f"Fetched {record.filename} from {record.ref.connector} ({len(record.content):,} bytes).",
        content=text[:MAX_CONTENT_CHARS],
        citations=[f"{record.ref}@{version}"],
    )
