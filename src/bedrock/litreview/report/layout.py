"""Which sections a report has, per question type — read from report.yaml.

The file is `defaults/report.yaml` inside the package, unless a site has put a
whole replacement in ~/.config/bedrock/litreview/report.yaml. `config_file` is
called on every load rather than once at import, so a test (or an override
dropped in after the process started) is read rather than a path frozen at
import time."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from bedrock.litreview.config import config_file

GENERATED = {"best_source", "background", "gaps", "plan", "ranking", "evidence_table", "method", "excluded",
             "glance", "synthesis", "numbers", "glossary"}

APPENDIX = ("best_source", "plan", "evidence_table", "ranking", "method", "excluded")
"""Sections a scientist reaches for to check the work, not to read the answer:
rendered after the references, under an Appendix heading (stages/s8_render.py)."""


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    kind: str
    source: str | None = None
    sentences: str = "2 to 5"
    instructions: str = ""
    findings: bool = True
    """Whether every sentence in this section must cite a quote that reports
    a result (`report/guard.py`'s `_reports_result` check). True for every
    `kind: written` section unless report.yaml says otherwise: every written
    section draws from extracted findings, so a sentence with nothing that
    reports one is filler, not an answer. `kind: generated` sections never
    go through the guard, so this field does nothing for them."""
    min_sentences: int = 2
    max_sentences: int = 5
    """Parsed from `sentences`. The maximum is enforced in code after the
    citation guard (`s7_write.write_section_text`); the minimum is only
    asked for - a section never gets filler to reach it."""


_RANGE = re.compile(r"^\s*(\d{1,2})\s*(?:to|-|–)\s*(\d{1,2})\s*$|^\s*(\d{1,2})\s*$")


def sentence_bounds(value: str, section_id: str = "?") -> tuple[int, int]:
    """"3 to 5" -> (3, 5); "4" -> (4, 4). Anything else is a config error."""
    m = _RANGE.match(value)
    if not m:
        raise ValueError(f"report.yaml: section {section_id!r}: sentences must look like \"3 to 5\", not {value!r}")
    low, high = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(3)), int(m.group(3)))
    if not 0 < low <= high <= 10:
        raise ValueError(f"report.yaml: section {section_id!r}: sentences {value!r} must be between 1 and 10, "
                         "low to high")
    return low, high


def load_layout(question_type: str, path: Path | None = None) -> list[SectionSpec]:
    raw = yaml.safe_load(Path(path or config_file("report.yaml")).read_text(encoding="utf-8")) or {}
    entries = raw.get(question_type) or raw.get("default") or []
    out = []
    for entry in entries:
        sentences = str(entry.get("sentences", "2 to 5"))
        low, high = sentence_bounds(sentences, str(entry["id"]))
        spec = SectionSpec(
            id=str(entry["id"]),
            title=str(entry.get("title", entry["id"])),
            kind=str(entry.get("kind", "written")),
            source=entry.get("source"),
            sentences=sentences,
            instructions=str(entry.get("instructions", "")).strip(),
            findings=bool(entry.get("findings", True)),
            min_sentences=low,
            max_sentences=high,
        )
        if spec.kind == "generated" and spec.id not in GENERATED:
            raise ValueError(f"report.yaml: no generator for section {spec.id!r}")
        if spec.kind == "written" and not spec.source:
            raise ValueError(f"report.yaml: written section {spec.id!r} needs a source")
        out.append(spec)
    return out
