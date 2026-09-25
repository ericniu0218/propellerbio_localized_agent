"""Distractor papers: the exactness test.

Take a real, highly relevant abstract. Replace every mention of the entity
asked about with a near-miss — psoriasis becomes atopic dermatitis, RSV
becomes human metapneumovirus — then add one sentence that mentions the real
entity in passing. The result is a paper about the wrong thing that still
names the right thing: exactly the "close enough" paper the client does not
want in an answer. It passes the lexical gate on purpose, so the relevance
grading is what has to catch it.

Leak rate = distractors that would be selected as evidence / distractors made.
"""

from __future__ import annotations

from bedrock.litreview.facets import subject_label
from bedrock.litreview.knowledge import TermMatcher, fold
from bedrock.litreview.models import Facet, Paper, QuestionSpec, RankedPaper
from bedrock.litreview.stages.s2_gate import gate_papers
from bedrock.litreview.stages.s3_rank import rank_papers


def pick_near_miss(spec: QuestionSpec) -> str | None:
    subject = spec.subject
    if subject is None:
        return None
    exact = [fold(t) for t in subject.exact_terms]
    for term in subject.near_miss_terms:
        f = fold(term)
        if len(term) >= 6 and not any(e in f for e in exact):
            return term
    return None


def _swap(text: str, matcher: TermMatcher, replacement: str) -> str:
    out, last = [], 0
    from bedrock.litreview.knowledge import clean

    text = clean(text)
    for m in matcher.matches(text):
        if m.kind != "exact":
            continue
        out.append(text[last : m.start])
        out.append(replacement)
        last = m.end
    out.append(text[last:])
    return "".join(out)


def make_distractors(papers: dict[str, Paper], ranked: list[RankedPaper], spec: QuestionSpec, k: int = 5) -> list[Paper]:
    near = pick_near_miss(spec)
    subject = spec.subject
    if near is None or subject is None:
        return []
    matcher = TermMatcher(subject.exact_terms)
    out = []
    for r in [r for r in ranked if r.usable and (r.grade or 0) >= 2][:k]:
        p = papers[r.paper_id]
        abstract = _swap(p.abstract or "", matcher, near)
        abstract += f" These findings may not extend to {subject.text}."
        out.append(Paper(
            id=f"DISTRACTOR-{p.id}",
            title=_swap(p.title, matcher, near),
            abstract=abstract,
            authors=p.authors,
            signals=p.signals.model_copy(),
            synthetic=True,
            text_source="abstract_only",
            sections={"abstract": abstract},
        ))
    return out


def leak_test(
    distractors: list[Paper], spec: QuestionSpec, facets: list[Facet], client, *, model: str, min_grade: int = 2
) -> dict:
    if not distractors:
        return {"made": 0, "passed_gate": 0, "leaked": 0, "leak_rate": None, "details": []}
    gates = gate_papers(distractors, spec)
    ranked = rank_papers(distractors, gates, spec, facets, client, model=model, max_graded=len(distractors))
    details, leaked = [], 0
    for r in ranked:
        leak = r.gate == "pass" and r.usable and (r.grade or 0) >= min_grade
        leaked += int(leak)
        details.append({
            "paper_id": r.paper_id, "gate": r.gate, "grade": r.grade,
            "about_exact_entity": r.about_exact_entity, "leaked": leak,
        })
    return {
        "made": len(distractors),
        "passed_gate": sum(1 for r in ranked if r.gate == "pass"),
        "leaked": leaked,
        "leak_rate": leaked / len(distractors),
        "near_miss_used": pick_near_miss(spec),
        "subject": subject_label(spec),
        "details": details,
    }
