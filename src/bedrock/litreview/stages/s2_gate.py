"""Stage 2 — the exact-entity gate. Pure code.

The client asked for answers about the exact drug, species or virus in the
question, with no alternatives and nothing that is merely close. This stage
is where that happens, before any model reads anything:

  pass       the paper names the entity (by an exact term) in its title or
             abstract, or repeatedly in its full text
  near_miss  it does not, but it names a sibling — hMPV for an RSV question,
             psoriatic arthritis for a psoriasis question — or its title is
             about the sibling and the sibling outnumbers the subject. Excluded,
             and the reason says which sibling, so a scientist can see the call
  absent     it does not name the entity at all, or only in passing
  review     a review article. Reviews are not evidence: a review citing a
             trial is not a trial, and counting it would inflate the weight
             of evidence
  retracted  retracted. Shown in the excluded list, never used

Every excluded paper stays visible in the report's appendix. The excluded
list is the recall safety net: it is how a scientist notices that something
important was filtered away.
"""

from __future__ import annotations

from bedrock.litreview.knowledge import TermMatcher
from bedrock.litreview.models import GateResult, Paper, QuestionSpec

# A paper that names the entity only in its full text must do so this often to
# count: one or two mentions is an introduction citing somebody else.
MIN_BODY_MENTIONS = 3


def gate_paper(paper: Paper, spec: QuestionSpec, matchers: dict[str, TermMatcher]) -> GateResult:
    if paper.signals.is_retracted:
        return GateResult(paper_id=paper.id, status="retracted", reason="retracted publication")
    if paper.is_review:
        return GateResult(paper_id=paper.id, status="review", reason="review article — not primary evidence")

    head = paper.head_text()
    body = "\n\n".join(t for name, t in paper.sections.items() if name != "abstract")
    hits: dict[str, int] = {}
    near_hits: dict[str, list[str]] = {}
    failures: list[tuple[str, str]] = []

    for canonical, matcher in matchers.items():
        exact_head, near_head = matcher.summary(head)
        exact_body, near_body = matcher.summary(body) if body else (0, [])
        hits[canonical] = exact_head + exact_body
        near = sorted(set(near_head) | set(near_body))
        if near:
            near_hits[canonical] = near
        title_exact, title_near = matcher.summary(paper.title)
        if title_near and not title_exact and exact_head:
            # "Bovine RSV uses the human IGF1R" names RSV once in its abstract
            # ("like human RSV") and bovine RSV throughout. A title about the
            # sibling, with the sibling outnumbering the subject, is a paper
            # about the sibling.
            near_count = sum(1 for m in matcher.matches(head) if m.kind == "near_miss")
            if near_count > exact_head:
                failures.append((
                    "near_miss",
                    f"mainly about {title_near[0]} ({near_count} mentions vs {exact_head} of {canonical})",
                ))
                continue
        if exact_head >= 1 or exact_body >= MIN_BODY_MENTIONS:
            continue
        if near:
            failures.append(("near_miss", f"about {', '.join(near[:3])} — not {canonical}"))
        elif exact_body:
            failures.append(("absent", f"mentions {canonical} only in passing ({exact_body}x in full text)"))
        else:
            failures.append(("absent", f"does not mention {canonical}"))

    if failures:
        status = "near_miss" if any(kind == "near_miss" for kind, _ in failures) else "absent"
        return GateResult(
            paper_id=paper.id,
            status=status,  # type: ignore[arg-type]
            reason="; ".join(reason for _, reason in failures),
            exact_hits=hits,
            near_miss_hits=near_hits,
        )
    reason = ", ".join(f"{name} x{count}" for name, count in hits.items())
    return GateResult(paper_id=paper.id, status="pass", reason=f"names {reason}", exact_hits=hits, near_miss_hits=near_hits)


def gate_papers(papers: list[Paper], spec: QuestionSpec) -> dict[str, GateResult]:
    matchers = {
        e.canonical: TermMatcher(e.exact_terms, e.near_miss_terms, e.near_miss_patterns)
        for e in spec.required_entities
    }
    return {p.id: gate_paper(p, spec, matchers) for p in papers}
