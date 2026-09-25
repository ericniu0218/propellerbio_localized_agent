"""External references: what published sources say a good answer contains.

Two kinds, both frozen to disk when a benchmark corpus is built so a
benchmark run never depends on the network:

  reviews             published review articles on the question. Reviews are
                      never evidence in a report (a review citing a trial is
                      not a trial) — which is exactly what makes them a fair
                      independent reference to grade the report against.
  reference targets   for "what should I target in disease X", the targets of
                      drugs that reached late-stage trials or approval for X,
                      from Open Targets. The clinical record's answer to the
                      question, independent of any paper we collected.

Completeness is scored only against references the collected papers could
support ("reachable"), so the report is not blamed for what the searcher did
not find. Raw recall is reported alongside, as the searcher's number.
"""

from __future__ import annotations

import json
from pathlib import Path

from bedrock.litreview.knowledge import Knowledge, TermMatcher
from bedrock.litreview.models import Paper


def load_reference_targets(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return json.loads(Path(path).read_text(encoding="utf-8"))


def target_names(symbol: str, kb: Knowledge) -> list[str]:
    canonical = kb.normalize(symbol, "target")
    known = kb.get(canonical)
    names = [symbol, canonical]
    if known:
        names += list(known.terms)
    return list(dict.fromkeys(n for n in names if n))


def reachable_targets(targets: list[dict], papers: list[Paper], kb: Knowledge) -> list[dict]:
    """Reference targets that at least one collected (gate-passing) paper names."""
    text = "\n".join(p.all_text() for p in papers)
    out = []
    for t in targets:
        if TermMatcher(target_names(t["symbol"], kb)).exact_count(text):
            out.append(t)
    return out


def target_recall(targets: list[dict], answer: str, kb: Knowledge) -> tuple[float | None, list[str], list[str]]:
    """Fraction of reference targets the answer names; plus found and missed."""
    if not targets:
        return None, [], []
    found, missed = [], []
    for t in targets:
        if TermMatcher(target_names(t["symbol"], kb)).exact_count(answer):
            found.append(t["symbol"])
        else:
            missed.append(t["symbol"])
    return len(found) / len(targets), found, missed
