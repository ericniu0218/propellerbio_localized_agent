"""Log files - the biggest deterministic win in the whole pipeline.

Logs are enormous and almost entirely repetition. A hundred thousand lines are
usually a few hundred distinct *shapes* with different numbers and timestamps
substituted in. Replacing those varying parts with placeholders turns each
line into a template, and counting the templates gives a summary that is both
tiny and genuinely more useful than the raw file: "this error happened 4,812
times" is the thing you actually wanted to know.

No model is involved and none is needed. The raw lines stay in the chunks for
when someone needs the specific occurrence.
"""

from __future__ import annotations

import re
from collections import Counter

from bedrock.context.adapters.base import Source
from bedrock.context.types import Extraction, Segment

LINES_PER_SEGMENT = 50
_TOP_TEMPLATES = 12
_SNIFF_LINES = 40

_LEVEL = re.compile(r"\b(TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|CRITICAL|FATAL|SEVERE)\b")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"|\[\d{2}/\w{3}/\d{4}[^\]]*\]"
)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{16,}\b")
_PATH = re.compile(r"(?:/[\w.\-]+){2,}")
_QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'')
# No trailing \b on purpose: a number welded to a unit ("146ms", "12kB") is
# still a varying number, and requiring a boundary after it left those
# uncollapsed - which quietly defeats the whole point of templating.
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?")


class LogAdapter:
    name = "logs"

    def can_handle(self, source: Source) -> bool:
        if source.suffix in {".log", ".jsonl.log"}:
            return True
        lines = [line for line in source.lines(_SNIFF_LINES) if line.strip()]
        if len(lines) < 5:
            return False
        # Treat it as a log when most lines carry a timestamp or a level word.
        # A high bar on purpose: prose that mentions one date should not be
        # run through template extraction.
        matched = sum(1 for line in lines if _TIMESTAMP.search(line) or _LEVEL.search(line))
        return matched / len(lines) >= 0.6

    def extract(self, source: Source) -> Extraction:
        lines = [line for line in source.text.splitlines() if line.strip()]
        templates = Counter(templatise(line) for line in lines)
        levels = Counter(match.group(1).upper() for line in lines for match in [_LEVEL.search(line)] if match)
        timestamps = [match.group(0) for line in lines[:1] + lines[-1:] for match in [_TIMESTAMP.search(line)] if match]

        body = [
            f"Log file: {len(lines):,} lines, {len(templates):,} distinct patterns.",
        ]
        if timestamps:
            body.append(f"Spans {timestamps[0]} to {timestamps[-1]}.")
        if levels:
            body.append("Levels: " + ", ".join(f"{name} {count:,}" for name, count in levels.most_common()))
        body.append("")
        body.append(f"Most frequent patterns (top {min(_TOP_TEMPLATES, len(templates))}):")
        for template, count in templates.most_common(_TOP_TEMPLATES):
            share = count / len(lines) * 100 if lines else 0
            body.append(f"- {count:,}x ({share:.0f}%) {_truncate(template)}")
        overview = "\n".join(body)

        segments = [Segment(text=overview, kind="patterns", label="patterns", weight=5.0)]
        for start in range(0, len(lines), LINES_PER_SEGMENT):
            block = lines[start : start + LINES_PER_SEGMENT]
            segments.append(
                Segment(
                    text="\n".join(block),
                    kind="lines",
                    label=f"lines {start + 1}-{start + len(block)}",
                )
            )

        return Extraction(
            source_kind="logs",
            overview=overview,
            segments=segments,
            metadata={
                "lines": len(lines),
                "templates": len(templates),
                "levels": dict(levels),
            },
        )


def templatise(line: str) -> str:
    """Replace everything that varies between otherwise-identical log lines.

    Order matters: the most specific patterns run first, so a timestamp is
    recognised as a whole rather than being eaten piecewise by the number rule.
    """
    for pattern, placeholder in (
        (_TIMESTAMP, "<TS>"),
        (_UUID, "<UUID>"),
        (_IP, "<IP>"),
        (_HEX, "<HEX>"),
        (_PATH, "<PATH>"),
        (_QUOTED, "<STR>"),
        (_NUMBER, "<N>"),
    ):
        line = pattern.sub(placeholder, line)
    return " ".join(line.split())


def _truncate(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
