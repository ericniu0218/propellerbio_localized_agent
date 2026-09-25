"""Search queries, built in code from the question's exact terms.

No model touches a query. Each required entity contributes one OR-group of
the exact names the knowledge table accepts for it; groups are ANDed. A
second query per question type (and per goal) adds a group of focus words
from `config/pipeline.yaml` (`search.focus`), so that "what should I target
to prevent psoriasis?" also asks for knockout, blockade and trial papers
rather than only the top of 60,000 psoriasis hits.

Searching broader than the gate is fine — the gate discards near-misses.
Searching narrower is not: recall lost here is lost for good. So a query
never drops an exact term, and focus queries are added to the plain one,
never substituted for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from bedrock.litreview.sources.base import SearchRequest

# Characters that would end a quoted phrase or open a field tag.
_UNSAFE = re.compile(r'["\[\]{}]')


@dataclass(frozen=True)
class Query:
    label: str
    """"entities", "focus:<type-or-goal>", or "question_text"."""
    groups: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    free_text: str | None = None

    def render(self, phrase) -> str:
        """`phrase(term)` renders one quoted term in a source's syntax."""
        if self.free_text is not None:
            return self.free_text
        parts = []
        for group in self.groups:
            rendered = " OR ".join(phrase(t) for t in group)
            parts.append(f"({rendered})" if len(group) > 1 else rendered)
        return " AND ".join(parts)


def clean_term(term: str) -> str:
    return re.sub(r"\s+", " ", _UNSAFE.sub(" ", term)).strip()


def _group(terms: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        term = clean_term(term)
        if term and term.lower() not in seen:
            seen.add(term.lower())
            out.append(term)
    return tuple(out)


def entity_groups(request: SearchRequest) -> list[tuple[str, ...]]:
    """One OR-group per required entity, in the order the question names them."""
    if request.spec is not None:
        names = [e.canonical for e in request.spec.entities if e.required]
    else:
        names = list(request.exact_terms)
    groups = []
    for name in names:
        group = _group(request.exact_terms.get(name) or [name])
        if group:
            groups.append(group)
    return groups


def build_queries(request: SearchRequest, focus: dict[str, list[str]] | None = None) -> list[Query]:
    groups = entity_groups(request)
    if not groups:
        # Nothing recognised: the question as typed is all there is to search.
        return [Query(label="question_text", free_text=clean_term(request.question))]

    queries = [Query(label="entities", groups=tuple(groups))]
    focus = focus or {}
    keys = []
    if request.spec is not None:
        keys = [request.spec.question_type, request.spec.goal]
    for key in keys:
        words = _group(focus.get(key) or [])
        if words:
            queries.append(Query(label=f"focus:{key}", groups=tuple(groups) + (words,)))
    return queries
