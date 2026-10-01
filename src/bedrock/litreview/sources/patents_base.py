"""The seam where a patent source plugs in. Deliberately not `PaperSearcher`.

    searcher.search(PatentSearchRequest) -> list[PatentRecord]

Patents are a separate, lower-confidence lane — see `sources/literature.py`'s
own docstring for the line this draws. A `PatentSearcher` never enters Stage 1,
never produces an `EvidenceUnit`, and nothing it returns is ever read by a
model for a biological claim or cited as evidence. What it feeds is one
generated (code-written, not model-written) report section: who else has
filed or been granted a patent naming the same entities. `report/guard.py`
never touches this data because nothing here is asserted as fact requiring a
citation check — a list of filings is simply what the office record says.

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


@runtime_checkable
class PatentSearcher(Protocol):
    name: str

    def search(self, request: PatentSearchRequest) -> list[PatentRecord]:
        ...
