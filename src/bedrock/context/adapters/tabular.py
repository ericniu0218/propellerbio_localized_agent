"""CSV and TSV - where the two-layer idea pays off most.

A 2 MB spreadsheet is around half a million tokens, which no local model will
read. But almost every question about a table is answered by its *shape*: what
the columns are, what type each one holds, what range the numbers fall in, how
many values are missing. That summary is a few hundred tokens and it is exact,
because it is computed rather than guessed.

So this adapter puts the schema and column statistics in the overview, and
keeps every row as segments. The digest costs a model almost nothing to read,
and the rows are still there, compressed, when someone asks about a specific
one - which is the whole point of the two-layer store.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter

from bedrock.context.adapters.base import Source
from bedrock.context.types import Extraction, Segment

ROWS_PER_SEGMENT = 25
_MAX_SAMPLE_VALUES = 3
_SNIFF_LINES = 20
_NUMERIC = re.compile(r"^-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?)?")
_BOOL = frozenset({"true", "false", "yes", "no", "y", "n", "0", "1"})


class TabularAdapter:
    name = "tabular"

    def can_handle(self, source: Source) -> bool:
        if source.suffix in {".csv", ".tsv"}:
            return True
        return _looks_delimited(source.text)

    def extract(self, source: Source) -> Extraction:
        delimiter = _delimiter(source.text, source.suffix)
        reader = csv.reader(io.StringIO(source.text), delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            return Extraction(source_kind="tabular", overview="Empty table.", segments=[])

        header = [name.strip() or f"column_{i + 1}" for i, name in enumerate(header)]
        rows = [row for row in reader if any(cell.strip() for cell in row)]

        columns = [_profile(header[i], [_cell(row, i) for row in rows]) for i in range(len(header))]
        overview = _render_overview(len(rows), columns, delimiter)

        segments = [
            Segment(
                text=overview,
                kind="schema",
                label="schema",
                # Weighted well above rows: when we pick a small sample to show
                # a model, the schema should always make the cut.
                weight=5.0,
            )
        ]
        header_line = delimiter.join(header)
        for start in range(0, len(rows), ROWS_PER_SEGMENT):
            block = rows[start : start + ROWS_PER_SEGMENT]
            body = "\n".join(delimiter.join(cell for cell in row) for row in block)
            segments.append(
                Segment(
                    text=f"{header_line}\n{body}",
                    kind="rows",
                    label=f"rows {start + 1}-{start + len(block)}",
                )
            )

        return Extraction(
            source_kind="tabular",
            overview=overview,
            segments=segments,
            metadata={
                "rows": len(rows),
                "columns": len(header),
                "column_names": header,
                "delimiter": delimiter,
            },
        )


def _cell(row: list[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def _profile(name: str, values: list[str]) -> dict:
    """Work out what one column holds, and summarise it exactly."""
    present = [v for v in values if v != ""]
    missing = len(values) - len(present)
    profile: dict = {"name": name, "missing": missing, "count": len(values)}

    if not present:
        profile["type"] = "empty"
        return profile

    numeric = [_to_number(v) for v in present if _NUMERIC.match(v)]
    if len(numeric) == len(present):
        profile["type"] = "number"
        profile["min"] = min(numeric)
        profile["max"] = max(numeric)
        profile["mean"] = sum(numeric) / len(numeric)
        return profile

    if all(_DATE.match(v) for v in present):
        profile["type"] = "date"
        profile["min"] = min(present)
        profile["max"] = max(present)
        return profile

    distinct = Counter(present)
    if len(distinct) <= 2 and all(v.lower() in _BOOL for v in distinct):
        profile["type"] = "boolean"
    elif len(distinct) <= max(12, len(present) // 20):
        profile["type"] = "category"
    else:
        profile["type"] = "text"
    profile["distinct"] = len(distinct)
    profile["common"] = distinct.most_common(_MAX_SAMPLE_VALUES)
    return profile


def _to_number(value: str) -> float:
    return float(value.replace(",", ""))


def _render_overview(row_count: int, columns: list[dict], delimiter: str) -> str:
    kind = {",": "CSV", "\t": "TSV", ";": "semicolon-delimited", "|": "pipe-delimited"}.get(
        delimiter, "delimited"
    )
    lines = [f"{kind} table: {row_count:,} rows x {len(columns)} columns.", "", "Columns:"]
    for column in columns:
        lines.append(f"- {column['name']} ({column['type']}){_describe(column)}")
    return "\n".join(lines)


def _describe(column: dict) -> str:
    parts: list[str] = []
    if column["type"] == "number":
        parts.append(f"min {_short(column['min'])}, max {_short(column['max'])}, mean {_short(column['mean'])}")
    elif column["type"] == "date":
        parts.append(f"{column['min']} to {column['max']}")
    elif column.get("common"):
        shown = ", ".join(f"{value} x{count}" for value, count in column["common"])
        parts.append(f"{column.get('distinct', 0)} distinct; most common {shown}")
    if column["missing"]:
        parts.append(f"{column['missing']} missing")
    return ": " + "; ".join(parts) if parts else ""


def _short(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4g}"


def _delimiter(text: str, suffix: str) -> str:
    if suffix == ".tsv":
        return "\t"
    sample = "\n".join(text.splitlines()[:_SNIFF_LINES])
    counts = {d: sample.count(d) for d in (",", "\t", ";", "|")}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else ","


def _looks_delimited(text: str) -> bool:
    """True when the first lines all split into the same number of fields.

    A consistent field count across several lines is a strong signal and a
    cheap one. Requiring at least two columns keeps this from claiming plain
    prose that happens to contain commas.
    """
    lines = [line for line in text.splitlines()[:_SNIFF_LINES] if line.strip()]
    if len(lines) < 3:
        return False
    delimiter = _delimiter(text, "")
    counts = {line.count(delimiter) for line in lines}
    return len(counts) == 1 and counts.pop() >= 1
