"""Showing AlphaFold 2 confidence honestly in a terminal.

Everything here is presentation. The numbers arrive already parsed, paired
with their own model and validated by ``bedrock.biotech.confidence``; this
module bins them into the four published pLDDT bands, draws a small text
heatmap of the predicted aligned error, and - just as importantly - carries
the caveats that stop a confident-looking picture being read as experimental
evidence.

Two display decisions are worth explaining, because getting either wrong
would quietly mislead a scientist.

**Intervals are input sequence positions.** A coordinate file's residue
identifiers can start anywhere and can skip. "Residues 41-58 are below 70"
means the 41st to 58th residue *of the sequence the employee submitted*.

**The heatmap keeps its direction and its worst value.** PAE is asymmetric:
cell ``[i][j]`` is the error in residue *j* when the structure is aligned on
residue *i*. Rows are the aligned residue, columns the scored one, and each
displayed block shows the *maximum* error inside it, never the average, so a
small patch of high error cannot be averaged out of sight. The symbols are
readable without colour on purpose.
"""

from __future__ import annotations

import asyncio
from typing import Any

from bedrock.ui.argparsing import CommandUsageError, split_arguments
from bedrock.ui.rendering import as_cell, as_code, safe_line

# The four bands AlphaFold and the EMBL-EBI guidance use, in the same order
# as the PyMOL palette in ``biotech/_pymol_style.py``.
PLDDT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("Very low", 0.0, 50.0),
    ("Low", 50.0, 70.0),
    ("Confident", 70.0, 90.0),
    ("Very high", 90.0, 100.0),
)

PLDDT_CAVEAT = (
    "pLDDT estimates local model confidence; it is not experimental validation "
    "or a stability measurement. Low scores can also reflect disorder."
)

PAE_CAVEAT = (
    "This is an overview: high local confidence does not establish correct "
    "domain placement."
)

# Increasing predicted aligned error. A legend spells these out below every
# heatmap so the meaning never depends on colour or on remembering the order.
PAE_SYMBOLS: tuple[tuple[float, str], ...] = (
    (5.0, "."),
    (10.0, ":"),
    (15.0, "+"),
    (20.0, "*"),
    (float("inf"), "#"),
)

MAX_HEATMAP_BLOCKS = 32

CONFIDENCE_HELP = """\
**`/confidence` — AlphaFold 2 confidence for one run**

| Command | What it does |
|---|---|
| `/confidence RUN_ID` | pLDDT bands, low-confidence regions and a PAE overview |

Find a run ID with `/runs`. Confidence is read from the selected model's own
JSON next to the coordinates it scored; a run whose installation never wrote
those files is reported as unavailable, and its structure stays usable.
"""


async def run_confidence_command(args: str, *, services: Any, notify) -> None:
    """Handle one ``/confidence ...`` invocation."""
    try:
        words = split_arguments(args.strip())
    except CommandUsageError as exc:
        await notify(str(exc), "error")
        return
    if not words or words[0] in {"help", "?"}:
        await notify(CONFIDENCE_HELP, "info")
        return
    if len(words) > 1:
        await notify("`/confidence` takes one run ID. Try `/runs` to find it.", "error")
        return

    run_id = words[0]
    try:
        catalog = await services.runs()
        # Reading and hashing confidence JSON is expensive; it belongs off
        # the event loop, like every other disk read in this application.
        report = await asyncio.to_thread(catalog.confidence, run_id)
    except services.errors as exc:
        await notify(f"Could not read confidence for {as_code(run_id)}: {safe_line(exc)}", "error")
        return
    await notify(format_confidence_report(report), "info")


# ---- Rendering ------------------------------------------------------------


def format_confidence_report(report: Any) -> str:
    """The full ``/confidence`` page for one run."""
    lines = [
        f"**Confidence** — run {as_code(report.run_id)}",
        "",
        f"- Selected model: {as_code(report.model_id) if report.model_id else '_unavailable_'}",
        f"- Input sequence length: {int(report.residue_count)} residues",
    ]

    plddt = getattr(report, "plddt", None)
    if plddt:
        lines.extend(["", "**pLDDT bands**", "", *_band_table(plddt)])
        lines.extend(["", *_region_lines(report)])
    else:
        lines.extend(["", "_Per-residue pLDDT is unavailable for this run._"])

    lines.extend(["", f"*{PLDDT_CAVEAT}*"])

    pae = getattr(report, "pae", None)
    if pae is not None:
        lines.extend(["", format_pae_overview(pae)])
    else:
        lines.extend(["", "_Predicted aligned error is unavailable for this run._"])

    notices = tuple(getattr(report, "notices", ()) or ())
    if notices:
        lines.extend(["", "**Notices**", ""])
        lines.extend(f"- {as_cell(notice, limit=300)}" for notice in notices)
    return "\n".join(lines)


def _band_table(plddt: tuple[float, ...]) -> list[str]:
    total = len(plddt)
    rows = [
        "| Band | pLDDT | Residues | Share |",
        "|---|---|---|---|",
    ]
    for index, (label, lower, upper) in enumerate(PLDDT_BANDS):
        last = index == len(PLDDT_BANDS) - 1
        count = sum(
            1 for value in plddt if value >= lower and (value <= upper if last else value < upper)
        )
        bounds = f"{lower:g}–{upper:g}" + ("" if last else " (exclusive)")
        share = (count / total * 100) if total else 0.0
        rows.append(f"| {label} | {bounds} | {count} | {share:.1f}% |")
    return rows


def _region_lines(report: Any) -> list[str]:
    lines = ["**Low-confidence regions** (input sequence positions, 1-based inclusive)", ""]
    for label, regions in (
        ("below 70", tuple(getattr(report, "low_regions", ()) or ())),
        ("below 50", tuple(getattr(report, "very_low_regions", ()) or ())),
    ):
        if regions:
            shown = ", ".join(f"{start}–{end}" for start, end in regions[:40])
            more = " …" if len(regions) > 40 else ""
            lines.append(f"- pLDDT {label}: {shown}{more}")
        else:
            lines.append(f"- pLDDT {label}: none")
    return lines


def format_pae_overview(pae: Any) -> str:
    """The bounded text heatmap, its axes, its legend and its source file."""
    values = [list(row) for row in pae.values][:MAX_HEATMAP_BLOCKS]
    values = [row[:MAX_HEATMAP_BLOCKS] for row in values]
    ranges = list(pae.ranges)[:MAX_HEATMAP_BLOCKS]
    maximum = float(pae.maximum_angstrom)

    label_width = max((len(_range_label(item)) for item in ranges), default=5)
    grid = [
        "rows = aligned residue (down), columns = scored residue (across)",
        "each cell is the MAXIMUM error in its block",
        "",
    ]
    for row_index, row in enumerate(values):
        label = _range_label(ranges[row_index]) if row_index < len(ranges) else "?"
        cells = " ".join(_symbol(value) for value in row)
        grid.append(f"{label:>{label_width}} | {cells}")

    column_labels = "  ".join(
        f"{index + 1}:{_range_label(item)}" for index, item in enumerate(ranges)
    )
    legend = " · ".join(
        f"`{symbol}` {_symbol_range(index)}" for index, (_, symbol) in enumerate(PAE_SYMBOLS)
    )
    return "\n".join(
        [
            "**Predicted aligned error** — maximum per block, in ångström (Å)",
            "",
            "```text",
            *grid,
            "```",
            "",
            f"- Scale: {legend}",
            f"- Highest error in this overview: {max((max(row) for row in values), default=0.0):.1f} Å "
            f"(matrix maximum {maximum:.1f} Å)",
            f"- Blocks, in order, on both axes: {column_labels}",
            f"- Full matrix (unaggregated): {as_code(pae.source_path)}",
            "",
            f"*{PAE_CAVEAT}*",
        ]
    )


def short_confidence_summary(report: Any) -> str:
    """One line for the end of a fold, where the page itself would be noise."""
    plddt = getattr(report, "plddt", None)
    if not plddt:
        return "Per-residue confidence is unavailable for this run."
    very_high = sum(1 for value in plddt if value >= 90.0)
    below_70 = len(tuple(getattr(report, "low_regions", ()) or ()))
    model = f"model {report.model_id}" if report.model_id else "model pairing unavailable"
    share = very_high / len(plddt) * 100
    return (
        f"{share:.0f}% of residues at pLDDT 90 or above, "
        f"{below_70} region(s) below 70 ({model})."
    )


def _range_label(item: tuple[int, int]) -> str:
    start, end = item
    return f"{start}-{end}"


def _symbol(value: float) -> str:
    for threshold, symbol in PAE_SYMBOLS:
        if float(value) < threshold:
            return symbol
    return PAE_SYMBOLS[-1][1]


def _symbol_range(index: int) -> str:
    lower = 0.0 if index == 0 else PAE_SYMBOLS[index - 1][0]
    upper = PAE_SYMBOLS[index][0]
    if upper == float("inf"):
        return f"{lower:g} Å and above"
    return f"{lower:g}–{upper:g} Å"
