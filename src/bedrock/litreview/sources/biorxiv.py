"""bioRxiv / medRxiv — preprints. api.biorxiv.org browses by DOI or date and
has no keyword search, so preprints are searched through Europe PMC's preprint
index (`SRC:PPR`), which ingests bioRxiv, medRxiv and Research Square.

A separate lane rather than a filter on the main Europe PMC results: ranked
together with journal papers, preprints rarely reach the first page, and a
fresh preprint is often the only evidence for a new target. Every hit is
flagged `is_preprint`, so the report can say so.
"""

from __future__ import annotations

from bedrock.litreview.models import Paper
from bedrock.litreview.sources.europepmc import EuropePMC


def preprint_query(query: str) -> str:
    return f"({query}) AND SRC:PPR"


class Preprints:
    def __init__(self, epmc: EuropePMC) -> None:
        self.epmc = epmc

    def papers(self, query: str, limit: int, *, found_by: str) -> list[Paper]:
        papers = self.epmc.papers(preprint_query(query), limit, found_by=found_by, source="biorxiv")
        for paper in papers:
            paper.signals.is_preprint = True
        return papers
