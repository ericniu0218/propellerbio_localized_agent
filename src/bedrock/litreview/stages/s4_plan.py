"""Stage 4 — decide what to read: which information, from how many papers, which ones.

Input: the facets the question needs (`litreview/facets.py`) and the ranking.
Output: an `EvidencePlan` — for every facet, the papers that will supply it,
whether that is enough, and the extraction jobs (paper + sections) that follow.

The selection is code, in two passes over the ranking:

  1. minimums first — every facet gets up to `min_sources` papers before any
     facet gets more, so one well-covered facet cannot eat the whole budget
  2. then top up each facet to `max_sources`, within `max_papers` overall
  3. then spend whatever budget is left on the best papers still unread

A facet the collected papers cannot supply is marked `gap`, and the report says
so plainly. It is never filled from anywhere else: a gap is an answer ("the
papers found contain no prevention trials"), where filling it from the
model's memory would be an unverifiable claim wearing a citation.

The most relevant source (Stage 3's top usable paper) is always read.
"""

from __future__ import annotations

from bedrock.litreview.facets import paper_facets
from bedrock.litreview.models import (
    EvidencePlan,
    ExtractionJob,
    Facet,
    FacetPlan,
    Paper,
    QuestionSpec,
    RankedPaper,
)


def eligible_papers(ranked: list[RankedPaper], *, min_grade: int, graded: bool) -> list[RankedPaper]:
    out = []
    for r in ranked:
        if not r.usable:
            continue
        if graded:
            if r.grade is None or r.grade < min_grade:
                continue
        out.append(r)
    return out


def build_plan(
    spec: QuestionSpec,
    facets: list[Facet],
    ranked: list[RankedPaper],
    papers: dict[str, Paper],
    *,
    max_papers: int = 12,
    min_grade: int = 2,
    sections: tuple[str, ...] = ("abstract", "results"),
    discussion_when_no_results: bool = True,
    reasons: dict[str, str] | None = None,
    policy: str = "lexical",
    designs: dict[str, dict] | None = None,
) -> EvidencePlan:
    """`reasons`, when given, is filled with why each read paper was chosen
    (`best_source`, `facet_min:<id>`, `facet_max:<id>`, `budget_fill`), for
    the coverage ledger. It changes nothing about the choice.

    `policy="evidence_aware"` changes two things. For the clinical need, when
    the pool holds any paper whose own publication type or design wording says
    it is a human trial or cohort, only those count towards it. For the
    clinical and prevention needs, such papers are tried first - by grade,
    then rank, with full text no longer a tie-breaker - so a grade-2
    abstract-only trial takes the clinical slot ahead of a grade-3 mouse paper
    with full text. Every other need, the minimums and their order, the
    budget and the most-relevant-source rule are as in the lexical policy."""
    reasons = reasons if reasons is not None else {}
    graded = any(r.grade is not None for r in ranked)
    pool = eligible_papers(ranked, min_grade=min_grade, graded=graded)
    # Relevance decides; among equally relevant papers, full text supplies
    # results and methods where an abstract supplies one sentence per finding.
    pool.sort(key=lambda r: (-(r.grade or 0), not r.has_full_text, r.rank))

    # Which needs each paper can supply: assigned in Stage 3 by code, from the
    # abstract's study-design words and whether the paper changed something.
    tags: dict[str, list[str]] = {
        r.paper_id: list(r.facets) or paper_facets(papers[r.paper_id], facets) for r in pool
    }
    # A need no paper was tagged with falls back to words alone, so a strict
    # rule does not turn into a false "gap".
    for facet in facets:
        if not any(facet.id in t for t in tags.values()):
            for r in pool:
                if facet.id in paper_facets(papers[r.paper_id], [facet]):
                    tags[r.paper_id].append(facet.id)

    order_for = {f.id: pool for f in facets}
    if policy == "evidence_aware" and designs:
        from bedrock.litreview.study_design import grounded_labels

        human = {"human_rct", "human_interventional_other", "human_observational"}
        grounded = sorted(
            (r for r in pool if grounded_labels(designs.get(r.paper_id)) & human),
            key=lambda r: (-(r.grade or 0), r.rank),
        )
        for facet in facets:
            if facet.id not in ("clinical", "prevention"):
                continue
            if facet.id == "clinical" and grounded:
                # The clinical need is met by human studies. When the pool has
                # any the paper itself identifies as one, only those count
                # towards it; the word list's "cohort" in a mouse paper does
                # not. With none, the word tags stand, and the ledger shows
                # what was actually found.
                ids = {r.paper_id for r in grounded}
                for r in pool:
                    if r.paper_id in ids and facet.id not in tags[r.paper_id]:
                        tags[r.paper_id].append(facet.id)
                    elif r.paper_id not in ids and facet.id in tags[r.paper_id]:
                        tags[r.paper_id].remove(facet.id)
            first = [r for r in grounded if facet.id in tags[r.paper_id]]
            order_for[facet.id] = first + [r for r in pool if r not in first]

    selected: list[str] = []
    best = pool[0].paper_id if pool else None
    if best:
        selected.append(best)
        reasons[best] = "best_source"

    def count(facet_id: str) -> int:
        return sum(1 for pid in selected if facet_id in tags.get(pid, []))

    for limit_attr in ("min_sources", "max_sources"):
        for facet in facets:
            limit = getattr(facet, limit_attr)
            for r in order_for[facet.id]:
                if len(selected) >= max_papers or count(facet.id) >= limit:
                    break
                if r.paper_id not in selected and facet.id in tags[r.paper_id]:
                    selected.append(r.paper_id)
                    reasons[r.paper_id] = f"{'facet_min' if limit_attr == 'min_sources' else 'facet_max'}:{facet.id}"

    # The facet caps, not the paper budget, decide how much gets read: one
    # paper can satisfy four needs at once, so the per-facet maxima fill long
    # before `max_papers` does. Spend what is left on the best papers still
    # unread — they passed the gate and were graded relevant, and the first
    # benchmark with a strong judge showed the reports were short of facts
    # the collected papers do support.
    for r in pool:
        if len(selected) >= max_papers:
            break
        if r.paper_id not in selected:
            selected.append(r.paper_id)
            reasons[r.paper_id] = "budget_fill"

    facet_plans = []
    for facet in facets:
        ids = [pid for pid in selected if facet.id in tags.get(pid, [])]
        if not ids:
            status = "gap"
        elif len(ids) < facet.min_sources:
            status = "thin"
        else:
            status = "covered"
        facet_plans.append(
            FacetPlan(facet=facet, paper_ids=ids, status=status, shortfall=max(0, facet.min_sources - len(ids)))
        )

    jobs = []
    for pid in selected:
        paper = papers[pid]
        wanted = [s for s in sections if paper.sections.get(s)]
        if discussion_when_no_results and "results" not in paper.sections and paper.sections.get("discussion"):
            wanted.append("discussion")
        if not wanted and paper.abstract:
            wanted = ["abstract"]
        facet_ids = [f.id for f in facets if f.id in tags.get(pid, [])] or [f.id for f in facets]
        jobs.append(
            ExtractionJob(
                paper_id=pid, sections=wanted, facets=facet_ids, abstract_only=not paper.has_full_text
            )
        )
    return EvidencePlan(question_id=spec.id, facets=facet_plans, jobs=jobs, best_source_id=best)
