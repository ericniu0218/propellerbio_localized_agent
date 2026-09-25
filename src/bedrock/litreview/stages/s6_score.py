"""Stage 6 — group the evidence. PURE CODE, no model, ever.

Produces the `EvidenceSummary` the report is written from:

- short ids ("E1", "E2", ...) in a deterministic order, which are what the
  writing model cites and what the reader clicks
- units grouped by facet
- one `TargetSummary` per target: how many independent papers, which kinds of
  evidence, which outcomes, which drugs
- conflicts: the same target and direction reported as helping in one paper
  and not helping (or harming) in another. Two papers contradicting each
  other is a finding, not an error, and it is surfaced loudly

Targets are ordered by a rule written down here and printed in the report,
not by a judgement:

  1. papers with human clinical evidence of benefit
  2. papers with human genetic evidence
  3. papers with animal evidence of benefit
  4. independent papers overall
  5. name, to break ties

This orders *evidence*, not paper quality: it counts what kinds of study
exist, and never says whether any one of them is good. That call stays with
the scientist, who sees the journal, citations and preprint status beside
every reference.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from bedrock.litreview.knowledge import TermMatcher, fold, load_knowledge, looks_like_target
from bedrock.litreview.models import (
    Conflict,
    EvidencePlan,
    EvidenceSummary,
    EvidenceUnit,
    Facet,
    QuestionSpec,
    RankedPaper,
    TargetSummary,
)

TARGET_ORDER_RULE = (
    "Targets are ordered by: papers with human clinical evidence of benefit, then papers with "
    "human genetic evidence, then papers with animal evidence of benefit, then the number of "
    "independent papers. This counts kinds of evidence; it does not judge any paper's quality."
)

_SECTION_ORDER = {"abstract": 1, "results": 0, "discussion": 2, "methods": 3, "other": 4}
_TIER = {"human_clinical": 0, "human_genetic": 1, "in_vivo": 2, "in_vitro": 3}
_GENERIC_TARGETS = {"", "none", "n/a", "unknown", "not stated", "unclear"}
# Cell lines are where an experiment was done, not something to target. Their
# findings stay in the evidence; they just do not become a row of targets.
_CELL_LINES = re.compile(
    r"\b(HaCaT|HEK[\s-]?293\w*|HeLa|THP-1|Jurkat|RAW\s?264\.7|A549|HEp-2|Vero|MDCK|BEAS-2B|"
    r"cell lines?|BMDMs?)\b", re.I)


def _is_subject(unit: EvidenceUnit, spec: QuestionSpec) -> bool:
    """A 'target' that is the disease or virus itself, a cell line, or a word that
    names nothing one could act on ("dropout", "participants"), says nothing
    about what to target. Its findings stay in the evidence; it gets no row."""
    if unit.target.strip().lower() in _GENERIC_TARGETS or _CELL_LINES.search(unit.target):
        return True
    if not looks_like_target(unit.target) and not load_knowledge().lookup(unit.target):
        return True
    for entity in spec.entities:
        if entity.kind in ("disease", "pathogen"):
            matcher = TermMatcher(entity.exact_terms)
            if matcher.exact_count(unit.target) and len(unit.target) <= len(max(entity.exact_terms, key=len)) + 2:
                return True
    return False


def summarize(
    units: list[EvidenceUnit],
    spec: QuestionSpec,
    facets: list[Facet],
    ranked: list[RankedPaper],
    plan: EvidencePlan | None = None,
) -> EvidenceSummary:
    facet_index = {f.id: i for i, f in enumerate(facets)}
    rank_of = {r.paper_id: r.rank for r in ranked}
    ordered = sorted(
        units,
        key=lambda u: (
            facet_index.get(u.facet or "", len(facet_index)),
            rank_of.get(u.paper_id, 10**6),
            _SECTION_ORDER.get(u.section, 9),
            u.quote,
            u.id,
        ),
    )
    short_ids = {u.id: f"E{i}" for i, u in enumerate(ordered, 1)}

    facet_units: dict[str, list[str]] = defaultdict(list)
    for u in ordered:
        facet_units[u.facet or "other"].append(u.id)

    # Group case-insensitively: "Nirsevimab" and "nirsevimab" are one target,
    # and two rows for one drug read as two independent lines of evidence.
    groups: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for u in ordered:
        if not _is_subject(u, spec):
            groups[fold(u.target)].append(u)
    by_target: dict[str, list[EvidenceUnit]] = {}
    for units_for_key in groups.values():
        spellings = Counter(u.target for u in units_for_key)
        by_target[spellings.most_common(1)[0][0]] = units_for_key

    targets: list[TargetSummary] = []
    conflicts: list[Conflict] = []
    for target, tunits in by_target.items():
        by_type: dict[str, set[str]] = defaultdict(set)
        by_outcome: dict[str, set[str]] = defaultdict(set)
        for u in tunits:
            by_type[u.evidence_type].add(u.paper_id)
            by_outcome[u.outcome].add(u.paper_id)
        summary = TargetSummary(
            target=target,
            names_seen=sorted({u.target_raw for u in tunits}),
            directions=sorted({u.direction for u in tunits}),
            paper_ids=sorted({u.paper_id for u in tunits}, key=lambda p: (rank_of.get(p, 10**6), p)),
            unit_ids=[u.id for u in tunits],
            by_evidence_type={k: len(v) for k, v in sorted(by_type.items())},
            by_outcome={k: len(v) for k, v in sorted(by_outcome.items())},
            prevention_units=sum(1 for u in tunits if u.setting == "prevention"),
            interventions=sorted({u.intervention for u in tunits if u.intervention}),
        )
        for direction in ("block", "activate"):
            helped = {u.paper_id for u in tunits if u.direction == direction and u.outcome == "improved"}
            against = {u.paper_id for u in tunits if u.direction == direction and u.outcome in ("no_effect", "worsened")}
            if helped and against and helped != against:
                summary.conflict = True
                conflicts.append(
                    Conflict(
                        target=target,
                        direction=direction,
                        supporting_papers=sorted(helped),
                        opposing_papers=sorted(against),
                        unit_ids=[u.id for u in tunits if u.direction == direction and u.outcome != "association"],
                    )
                )
        targets.append(summary)

    def papers_with(tunits: list[EvidenceUnit], types: tuple[str, ...], outcomes: tuple[str, ...] | None) -> int:
        return len({
            u.paper_id for u in tunits
            if u.evidence_type in types and (outcomes is None or u.outcome in outcomes)
        })

    def target_key(s: TargetSummary) -> tuple:
        tunits = by_target[s.target]
        return (
            -papers_with(tunits, ("human_clinical",), ("improved",)),
            -papers_with(tunits, ("human_genetic",), None),
            -papers_with(tunits, ("in_vivo",), ("improved",)),
            -len(s.paper_ids),
            s.target.lower(),
        )

    targets.sort(key=target_key)
    return EvidenceSummary(
        question_id=spec.id,
        units=ordered,
        short_ids=short_ids,
        targets=targets,
        conflicts=conflicts,
        facet_units=dict(facet_units),
    )


def best_units(units: list[EvidenceUnit], rank_of: dict[str, int], limit: int) -> list[EvidenceUnit]:
    """Pick the most informative units for a section: strongest evidence tier,
    then most relevant paper, spreading across papers before doubling up."""
    ordered = sorted(
        units,
        key=lambda u: (_TIER.get(u.evidence_type, 9), rank_of.get(u.paper_id, 10**6), u.quote),
    )
    picked: list[EvidenceUnit] = []
    seen_papers: set[str] = set()
    for u in ordered:
        if u.paper_id not in seen_papers:
            picked.append(u)
            seen_papers.add(u.paper_id)
        if len(picked) >= limit:
            return picked
    for u in ordered:
        if u not in picked:
            picked.append(u)
        if len(picked) >= limit:
            break
    return picked
