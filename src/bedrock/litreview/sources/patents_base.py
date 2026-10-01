"""The seam where a patent source plugs in. Deliberately not `PaperSearcher`.

    searcher.search(PatentSearchRequest) -> list[PatentRecord]

Patents are a separate, lower-confidence lane — see `sources/literature.py`'s
own docstring for the line this draws. A `PatentSearcher` never enters Stage 1,
never produces an `EvidenceUnit`, and nothing it returns is ever cited as
evidence or read by a model for a biological claim. What it feeds is one report
section, the competitor landscape: who else has filed a patent naming the same
targets, and what each filing says it covers.

A model copies what a filing says it covers - from its title, abstract and
claim 1 only (`stages/s6c_patent_extract.py`) - and a model may write sentences
about those copies (`stages/s7b_patent_write.py`). It may not judge a filing: every
sentence is attributed to the filer ("claims", "describes") and checked by
`report/patent_guard.py`, and each quote is re-proved against the stored filing by
`report/invariants.py` `check_patents`. `report/guard.py`, the paper guard, still
never touches this data: its checks are about results, and a patent reports none.

Same discipline as `SearchRequest`: `exact_terms` carries the pipeline's own
exact names, never a model's; searching with anything broader than these is
fine, searching with less loses recall nothing downstream recovers.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from bedrock.litreview.models import PatentRecord, QuestionSpec


class PatentSearchRequest(BaseModel):
    question: str
    spec: QuestionSpec | None = None
    exact_terms: dict[str, list[str]] = Field(default_factory=dict)
    max_results: int = 100
    targets: dict[str, list[str]] = Field(default_factory=dict)
    """Candidate targets to search for *with* the question's other entities:
    canonical name -> the exact terms the knowledge table accepts for it. Chosen
    by code from the evidence the run has already read (never by a model, and
    never a name the knowledge table does not list). Empty: one search for the
    question's own entities, which is the only thing to search before any
    target is known (a run without a model, or a question naming none)."""
    per_target: int = 40
    """Most filings kept per target, so one crowded target cannot use the whole
    `max_results` and leave the others unsearched."""


@runtime_checkable
class PatentSearcher(Protocol):
    name: str

    def search(self, request: PatentSearchRequest) -> list[PatentRecord]:
        ...
