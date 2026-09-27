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


def coverage_metrics(
    targets: list[dict],
    reachable: list[dict],
    report,
    kb: Knowledge,
    ledger: dict | None,
) -> dict[str, float | int | str | None]:
    """Target coverage at each stage, each with its own denominator.

    Scoring only - reference targets never reach the pipeline. The families,
    kept apart because they answer different questions:

      raw mention recall         reference targets the report's text names / all of them
      corpus-conditional recall  the same, over those the collected papers name (the
                                 legacy `target_recall`, computed by the caller)
      adjudicated recall         over targets a scientist judged the corpus supports:
                                 not available until such a gold list exists
      verified-unit coverage     reference targets an accepted, quote-verified unit is about
      narrative / Answer         reference targets a unit cited by a kept sentence is about
      human numbers in Answer    human-evidence units with a number, cited by a kept Answer sentence
      negative and categories    negative units found and cited; facets realized by verified units

    A name in text is a mention, not support: only the unit-based rows say a
    verified finding about the target exists.
    """
    by_short = {short: uid for uid, short in report.short_ids.items()}
    units = {u.id: u for u in report.evidence}

    def cited(section_ids: set[str] | None) -> set[str]:
        out: set[str] = set()

        def walk(section, top):
            if section.kind == "written" and (section_ids is None or top in section_ids):
                for s in section.sentences:
                    if s.kept:
                        out.update(by_short[c] for c in s.citations if c in by_short)
            for sub in section.subsections:
                walk(sub, top)

        for section in report.sections:
            walk(section, section.id)
        return out

    def about(unit_ids) -> set[str]:
        found = set()
        for t in targets:
            matcher = TermMatcher(target_names(t["symbol"], kb))
            if any(matcher.exact_count(units[u].target) or matcher.exact_count(units[u].target_raw)
                   for u in unit_ids if u in units):
                found.add(t["symbol"])
        return found

    verified = about(units)
    narrative = about(cited(None))
    answer = about(cited({"answer"}))
    reach = {t["symbol"] for t in reachable}
    counts = (ledger or {}).get("counts") or {}
    facets = (ledger or {}).get("facets") or {}
    realized = [f for f in facets.values() if f.get("realized_status") is not None]

    def share(n: int, d: int) -> float | None:
        return round(n / d, 4) if d else None

    return {
        "cov_ref_targets": len(targets),
        "cov_ref_targets_named_in_collected": len(reach),
        "cov_adjudicated_recall": None,
        "cov_adjudicated_denominator": "not_available",
        "cov_verified_unit_targets": len(verified),
        "cov_verified_unit_recall_reachable": share(len(verified & reach), len(reach)),
        "cov_narrative_targets": len(narrative),
        "cov_answer_targets": len(answer),
        "cov_answer_recall_reachable": share(len(answer & reach), len(reach)),
        "cov_human_numeric_units": counts.get("human_numeric_units"),
        "cov_human_numeric_units_in_answer": counts.get("human_numeric_units_in_answer"),
        "cov_negative_units": counts.get("negative_units"),
        "cov_negative_units_cited": counts.get("negative_units_cited"),
        "cov_facets_realized": (f"{sum(1 for f in realized if f['realized_status'] == 'covered')}/{len(realized)}"
                                if realized else None),
        "cov_ledger": "coverage.v1" if ledger else "not_recorded",
    }
