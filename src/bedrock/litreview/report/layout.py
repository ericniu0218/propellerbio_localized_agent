"""Which sections a report has, per question type — read from config/report.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from bedrock.litreview.config import ROOT

LAYOUT = ROOT / "config" / "report.yaml"

GENERATED = {"best_source", "gaps", "plan", "ranking", "evidence_table", "method", "excluded"}


@dataclass(frozen=True)
class SectionSpec:
    id: str
    title: str
    kind: str
    source: str | None = None
    sentences: str = "2 to 5"
    instructions: str = ""


def load_layout(question_type: str, path: Path | None = None) -> list[SectionSpec]:
    raw = yaml.safe_load(Path(path or LAYOUT).read_text(encoding="utf-8")) or {}
    entries = raw.get(question_type) or raw.get("default") or []
    out = []
    for entry in entries:
        spec = SectionSpec(
            id=str(entry["id"]),
            title=str(entry.get("title", entry["id"])),
            kind=str(entry.get("kind", "written")),
            source=entry.get("source"),
            sentences=str(entry.get("sentences", "2 to 5")),
            instructions=str(entry.get("instructions", "")).strip(),
        )
        if spec.kind == "generated" and spec.id not in GENERATED:
            raise ValueError(f"report.yaml: no generator for section {spec.id!r}")
        if spec.kind == "written" and not spec.source:
            raise ValueError(f"report.yaml: written section {spec.id!r} needs a source")
        out.append(spec)
    return out
