"""`results/history.jsonl`: one row per benchmark run, readable at any age.

`run.evaluate` appends a row and never rewrites one, so the file holds rows
written by every version of the benchmark. A field added later is missing
from every older row, and a missing field reads as "not recorded" - never as
a default that looks like a measurement. A row written before gates could be
"not measured" counted an unmeasured gate as passed, and a row written before
preflight ran says nothing about the context window its report was written at;
both are shown as exactly that.

    python -m bedrock.litreview.bench history [--question ID]
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

NOT_RECORDED = "not recorded"


class HistoryRow(BaseModel):
    """One line of `history.jsonl`. Every field after `report_id` was added
    later than the first rows, so every one of them may be absent."""

    model_config = ConfigDict(extra="allow")

    question_id: str
    created: str
    report_id: str
    prompt_versions: dict[str, str] = {}
    model_versions: dict[str, str] = {}
    overall: dict[str, float | None] = {}
    gates_passed: int | None = None
    gates_total: int | None = None
    context_windows: dict[str, int] | None = None
    """Model -> the window preflight established for this run's model calls."""
    report_context_windows: dict[str, int] | None = None
    """The windows the scored report was written at."""


def read_history(path: Path) -> list[HistoryRow]:
    """Every row, oldest first. A missing file is an empty history."""
    path = Path(path)
    if not path.exists():
        return []
    return [HistoryRow.model_validate(json.loads(line))
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def window_text(windows: dict[str, int] | None) -> str:
    """"qwen3:8b 12,288 tokens", or "not recorded"."""
    if not windows:
        return NOT_RECORDED
    return ", ".join(f"{model} {tokens:,} tokens" for model, tokens in sorted(windows.items()))


def recorded_report_windows(path: Path, report_id: str) -> tuple[dict[str, int], str] | None:
    """The windows a stored report was written at, and when that was recorded.

    A report id is a hash of the report's bytes, so the first row that
    recorded windows for this id is the run that wrote it (or one that
    re-scored it and carried the same record forward)."""
    for row in read_history(path):
        if row.report_id == report_id and row.report_context_windows:
            return row.report_context_windows, row.created
    return None


def render_history(rows: list[HistoryRow]) -> str:
    """A table of runs, oldest first, for reading in a terminal."""
    lines = ["| Created | Question | Report | Pipeline overall | Gates | Report context window |",
             "|---|---|---|---|---|---|"]
    for row in rows:
        gates = f"{row.gates_passed} of {row.gates_total}" if row.gates_total is not None else NOT_RECORDED
        overall = row.overall.get("pipeline")
        lines.append(
            f"| {row.created} | {row.question_id} | {row.report_id} | "
            f"{'–' if overall is None else overall} | {gates} | {window_text(row.report_context_windows)} |"
        )
    return "\n".join(lines) + "\n"
