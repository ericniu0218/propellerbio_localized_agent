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
    """Measured gates. Before `gates_not_measured` existed, every gate."""
    gates_not_measured: dict[str, str] | None = None
    """Gate -> why it was not measured. None: the row was written before a
    gate could be "not measured", when a gate with no input counted as passed."""
    context_windows: dict[str, int] | None = None
    """Model -> the window preflight established for this run's model calls."""
    report_context_windows: dict[str, int] | None = None
    """The windows the scored report was written at."""
    judged: dict[str, dict[str, tuple[int, int]]] | None = None
    """System -> judged score -> (judged, total). See `SystemScore.judged`."""
    judged_fraction: dict[str, float | None] | None = None
    """System -> the overall's judged fraction."""
    raters: dict[str, int] | None = None
    """Rater -> answers used; `code` for verdicts settled in code."""


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


def gates_text(row: HistoryRow) -> str:
    """"8 of 8 measured, 1 not measured", or, for a row written before gates
    could be unmeasured, "9 of 9 †" - see `OLD_GATES`."""
    if row.gates_total is None:
        return NOT_RECORDED
    if row.gates_not_measured is None:
        return f"{row.gates_passed} of {row.gates_total} †"
    text = f"{row.gates_passed} of {row.gates_total} measured"
    return text + (f", {len(row.gates_not_measured)} not measured" if row.gates_not_measured else "")


OLD_GATES = ("† written before a gate could be \"not measured\": a gate whose input was never built "
             "counted as passed.")


def render_history(rows: list[HistoryRow]) -> str:
    """A table of runs, oldest first, for reading in a terminal."""
    lines = ["| Created | Question | Report | Pipeline overall | Judged | Raters | Gates passed "
             "| Report context window |",
             "|---|---|---|---|---|---|---|---|"]
    for row in rows:
        overall = row.overall.get("pipeline")
        fraction = (row.judged_fraction or {}).get("pipeline")
        judged = NOT_RECORDED if row.judged_fraction is None else "–" if fraction is None else f"{fraction:.0%}"
        raters = NOT_RECORDED if row.raters is None else ", ".join(f"{k} {v}" for k, v in row.raters.items()) or "none"
        lines.append(
            f"| {row.created} | {row.question_id} | {row.report_id} | {'–' if overall is None else overall} | "
            f"{judged} | {raters} | {gates_text(row)} | {window_text(row.report_context_windows)} |"
        )
    if any(r.gates_total is not None and r.gates_not_measured is None for r in rows):
        lines += ["", OLD_GATES]
    return "\n".join(lines) + "\n"
