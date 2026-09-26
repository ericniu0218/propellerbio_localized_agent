"""Stage 1 — collect papers from the searcher, and make them safe to read.

The searcher itself lives outside this stage (see `sources/base.py` for the
contract). This stage asks it for papers, or accepts papers handed in
directly, and runs them through ingestion: full text split into sections,
reviews and retractions flagged, duplicates removed.

The paper set is frozen per question once collected. Re-running a report
reuses it; going back to the sources is a separate, explicit action, or the
same question would silently change answer as the literature moves.
"""

from __future__ import annotations

from bedrock.litreview.ingest import ingest
from bedrock.litreview.models import Paper, QuestionSpec
from bedrock.litreview.sources.base import PaperSearcher, SearchRequest


class NoPapers(ValueError):
    """Nothing to read: no searcher, no papers, and no frozen set."""


class NothingFound(NoPapers):
    """A search or a corpus file was given, and it held no papers. The
    message says which; a caller shows it as it is."""


def search_request(spec: QuestionSpec, max_results: int = 200) -> SearchRequest:
    """What the searcher is asked: the question, and the exact names the
    pipeline will accept for each entity — nothing a model added."""
    return SearchRequest(
        question=spec.text,
        spec=spec,
        exact_terms={e.canonical: e.exact_terms for e in spec.entities},
        max_results=max_results,
    )


def collect(
    spec: QuestionSpec,
    *,
    searcher: PaperSearcher | None = None,
    papers: list[Paper] | None = None,
    max_results: int = 200,
) -> list[Paper]:
    if papers is None:
        if searcher is None:
            raise NoPapers("collect needs a searcher or a list of papers")
        papers = searcher.search(search_request(spec, max_results))
        for p in papers:
            p.source = p.source or getattr(searcher, "name", None)
    return ingest(papers)
