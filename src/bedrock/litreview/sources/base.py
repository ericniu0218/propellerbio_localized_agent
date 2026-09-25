"""The seam where the paper searcher plugs in.

This file is the whole contract between the searcher and everything downstream:

    searcher.search(SearchRequest) -> list[Paper]

Anything that satisfies `PaperSearcher` works: the literature searcher
(`sources/literature.py` — Europe PMC, preprints and PubMed, with open-access
full text), the frozen JSONL corpora the benchmark uses and `litreview search`
writes (`sources/corpus.py`), or a list of papers handed straight to
`runner.answer_question(..., papers=...)`.

What the searcher owes the pipeline, per Paper:

- `title`, and `abstract` whenever one exists. Relevance is judged on these.
- Full text where legally available, either already split into `sections`
  ({"abstract", "methods", "results", "discussion", "other"} -> text) or raw
  in `full_text` with `full_text_format` = "jats" | "tei" | "text", which
  ingestion splits. Prefer Europe PMC JATS, then GROBID TEI, then flat text.
- Identifiers (`doi`, `pmid`, `pmcid`) and `url`, for the reference list.
- `signals`: journal, year, citation count, preprint, retraction, open
  access and publication types — objective facts, looked up in code. A
  publication type of "Review" is how reviews are kept out of the evidence.
- `found_by_queries`: which query found it, to explain coverage.

What it should know it does NOT have to do: rank, filter for relevance, or
decide which papers to read in full. The pipeline does that, question by
question, and shows its reasons. Return generously; the exact-entity gate and
the ranking are built to discard.

The contracts a searcher carries (`literature.py` meets the first; the rest
of the ladder is still to build):

- Search every source, deduplicate on DOI, then PMID, then normalised title,
  and record which query found each hit.
- Search with at least the exact names. `SearchRequest.exact_terms` carries
  the pipeline's own exact names for each entity; searching with anything
  broader is fine — the gate removes near-misses — but searching with less
  loses recall that nothing downstream can recover. (Alias lookup against
  HGNC / NCBI Gene / UniProt and MeSH / EFO would widen this; not built.)
- Acquisition ladder: Europe PMC XML (built), then open-access PDF via
  GROBID, then preprint PDF, then abstract only (flagged). Plus the drop
  folder. Never Sci-Hub or any pirated source.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from bedrock.litreview.models import Paper, QuestionSpec


class SearchRequest(BaseModel):
    question: str
    spec: QuestionSpec | None = None
    exact_terms: dict[str, list[str]] = Field(default_factory=dict)
    """Entity canonical name -> the exact names the pipeline will accept."""
    max_results: int = 200


@runtime_checkable
class PaperSearcher(Protocol):
    name: str

    def search(self, request: SearchRequest) -> list[Paper]:
        ...
