"""The paper searcher: Europe PMC, bioRxiv/medRxiv preprints and PubMed,
merged, deduplicated, with open-access full text attached.

    searcher = LiteratureSearcher.from_settings(settings)
    papers = searcher.search(SearchRequest(...))     # satisfies PaperSearcher
    searcher.last_report                              # what happened, per source

Every step is code. Queries are built from the question's exact terms
(`sources/query.py`); results from every source are merged in a fixed order
and interleaved by rank, so the same responses always give the same list;
full text is fetched only where it is open access (the first rung of the
acquisition ladder, Europe PMC JATS). Nothing ranks for relevance and nothing
scores quality — Stages 2 and 3 decide relevance, and quality signals are
copied from the source records for a human to read.

Failure is loud where silence would mislead. One source failing is logged and
reported, and the others carry on. Every source failing raises `SearchFailed`
rather than returning nothing: an empty list would be frozen as the paper set
for the question and rendered as "no literature found".

Deliberately not here, from the standalone retrieval agent this was ported
from: model-based "credibility" scores (no model scores quality in this
system, and that one was a cloud model), web/news/SEC/Google Scholar results
(not papers; snippets are not abstracts), and patents (a separate,
lower-confidence lane — never evidence). Never Sci-Hub or any pirated source:
paywalled papers come in through the drop folder.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import httpx

from bedrock.litreview.ingest import doi_key, merge_records
from bedrock.litreview.models import Paper
from bedrock.litreview.sources import europepmc, pubmed
from bedrock.litreview.sources.base import SearchRequest
from bedrock.litreview.sources.biorxiv import Preprints
from bedrock.litreview.sources.europepmc import EuropePMC
from bedrock.litreview.sources.http import SourceError, SourceHTTP
from bedrock.litreview.sources.pubmed import PubMed
from bedrock.litreview.sources.query import Query, build_queries

log = logging.getLogger(__name__)

# Merge priority: on a duplicate, the first source's record is kept, because
# Europe PMC's carries citation counts and the open-access flag and PubMed's
# does not. The others fill in only what it lacks.
SOURCES = ("europepmc", "biorxiv", "pubmed")
_PHRASE = {"europepmc": europepmc.phrase, "biorxiv": europepmc.phrase, "pubmed": pubmed.phrase}
_TITLE_FOLD = re.compile(r"[^a-z0-9]+")


class SearchFailed(SourceError):
    """No source answered at all."""


@dataclass
class SearchReport:
    queries: dict[str, list[str]] = field(default_factory=dict)
    """Source -> the exact query strings sent to it."""
    hits: dict[str, int] = field(default_factory=dict)
    """Source -> records returned, before merging."""
    failures: dict[str, str] = field(default_factory=dict)
    """"source [query label]" -> the error, for every query that failed."""
    searched_question_text: bool = False
    """No entity was recognised, so the question itself was sent as the query."""
    unique: int = 0
    returned: int = 0
    full_text_tried: int = 0
    full_text: int = 0
    title_only: int = 0
    seconds: float = 0.0

    def lines(self) -> list[str]:
        out = [f"searched {', '.join(self.queries) or 'nothing'} in {self.seconds:.1f}s"]
        if self.searched_question_text:
            out.append("no known entity in the question: searched the question text as typed")
        for source, n in self.hits.items():
            out.append(f"  {source}: {n} records")
        for where, error in self.failures.items():
            out.append(f"  FAILED {where}: {error}")
        abstract_only = self.returned - self.full_text - self.title_only
        out.append(f"{self.unique} unique papers, {self.returned} kept: {self.full_text} with open-access "
                   f"full text ({self.full_text_tried} tried), {abstract_only} abstract only, "
                   f"{self.title_only} title only")
        return out


def _keys(paper: Paper) -> list[str]:
    keys = []
    if paper.doi:
        keys.append(f"doi:{doi_key(paper.doi)}")
    if paper.pmid:
        keys.append(f"pmid:{paper.pmid.strip()}")
    title = _TITLE_FOLD.sub(" ", (paper.title or "").lower()).strip()
    if title:
        keys.append(f"title:{title}")
    return keys


def merge(results: list[tuple[int, int, list[Paper]]]) -> list[Paper]:
    """Deduplicate on DOI, then PMID, then normalised title, and interleave
    the lists in proportion to their length: a list of 25 preprints is spread
    across the same span as a list of 100 journal papers, so every query's
    best hits come first without a small lane crowding out a large one.

    `results` holds (source priority, query index, papers best-first). The
    record kept for a duplicate is the one from the highest-priority source;
    later copies are folded into it by `ingest.merge_records`, the same rule
    ingestion uses. They donate identifiers, an abstract and metadata it
    lacks, and the queries that found them. Their publication types are
    unioned in, because this runs before `ingest.prepare` reads those types:
    a PubMed copy that alone says "Retracted Publication" or "Review" must
    still make the kept Europe PMC record retracted or a review.
    """
    ordered = sorted(results, key=lambda r: (r[0], r[1]))
    kept: list[Paper] = []
    best: list[tuple[float, int, int]] = []
    index: dict[str, int] = {}
    for priority, qi, papers in ordered:
        for rank, paper in enumerate(papers):
            position = (rank + 0.5) / len(papers)
            keys = _keys(paper)
            at = next((index[k] for k in keys if k in index), None)
            if at is None:
                at = len(kept)
                kept.append(paper)
                best.append((position, priority, qi))
            else:
                merge_records(kept[at], paper)
                best[at] = min(best[at], (position, priority, qi))
            for k in _keys(kept[at]):
                index.setdefault(k, at)
    order = sorted(range(len(kept)), key=lambda i: best[i])
    return [kept[i] for i in order]


class LiteratureSearcher:
    name = "literature"

    def __init__(
        self,
        *,
        sources: tuple[str, ...] | list[str] = SOURCES,
        per_query: int = 100,
        preprints_per_query: int = 25,
        full_text_max: int = 80,
        full_text_workers: int = 4,
        focus: dict[str, list[str]] | None = None,
        ncbi_api_key: str = "",
        contact_email: str = "",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        unknown = [s for s in sources if s not in SOURCES]
        if unknown:
            raise ValueError(f"unknown source(s) {unknown}; available: {list(SOURCES)}")
        if not sources:
            raise ValueError("the searcher needs at least one source")
        self.sources = tuple(s for s in SOURCES if s in sources)
        self.per_query = per_query
        self.preprints_per_query = preprints_per_query
        self.full_text_max = full_text_max
        self.full_text_workers = max(1, full_text_workers)
        self.focus = focus or {}
        self.ncbi_api_key = ncbi_api_key
        self.contact_email = contact_email
        self.timeout = timeout
        self.transport = transport
        self.sleep = sleep
        self.last_report: SearchReport | None = None

    @classmethod
    def from_settings(cls, settings, **overrides) -> LiteratureSearcher:
        knob = settings.knob
        kwargs = {
            "sources": tuple(knob("search.sources", list(SOURCES))),
            "per_query": int(knob("search.per_query", 100)),
            "preprints_per_query": int(knob("search.preprints_per_query", 25)),
            "full_text_max": int(knob("search.full_text_max", 80)),
            "full_text_workers": int(knob("search.full_text_workers", 4)),
            "focus": dict(knob("search.focus", {}) or {}),
            "ncbi_api_key": settings.ncbi_api_key,
            "contact_email": settings.contact_email,
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    def _http(self, min_interval: float, opened: list[SourceHTTP], retries: int = 5) -> SourceHTTP:
        http = SourceHTTP(min_interval=min_interval, timeout=self.timeout, retries=retries,
                          transport=self.transport, sleep=self.sleep)
        opened.append(http)
        return http

    def search(self, request: SearchRequest) -> list[Paper]:
        started = time.monotonic()
        queries = build_queries(request, self.focus)
        report = SearchReport(searched_question_text=queries[0].label == "question_text")

        opened: list[SourceHTTP] = []
        clients = {
            "europepmc": EuropePMC(http=self._http(0.1, opened)),
            "biorxiv": Preprints(EuropePMC(http=self._http(0.1, opened))),
            "pubmed": PubMed(api_key=self.ncbi_api_key, email=self.contact_email,
                             http=self._http(0.11 if self.ncbi_api_key else 0.34, opened)),
        }
        try:
            results = self._run_sources(clients, queries, report)
            if not results:
                self.last_report = report
                raise SearchFailed("every source failed: " + "; ".join(
                    f"{where}: {error}" for where, error in report.failures.items()))
            papers = merge(results)
            report.unique = len(papers)
            papers = papers[: request.max_results]
            report.returned = len(papers)
            # Europe PMC answers a paper it holds no open text for with a
            # 500, every time, so one retry is plenty.
            texts = EuropePMC(http=self._http(0.1, opened, retries=2))
            self._attach_full_text(texts, papers, report)
            report.title_only = sum(1 for p in papers if not p.abstract and not p.full_text)
        finally:
            for http in opened:
                http.close()
        report.seconds = time.monotonic() - started
        self.last_report = report
        for line in report.lines():
            log.info("search: %s", line.strip())
        return papers

    def _run_sources(self, clients, queries: list[Query], report: SearchReport):
        def run(priority: int, source: str) -> list[tuple[int, int, list[Paper]]]:
            client = clients[source]
            limit = self.preprints_per_query if source == "biorxiv" else self.per_query
            out = []
            for qi, query in enumerate(queries):
                rendered = query.render(_PHRASE[source])
                report.queries.setdefault(source, []).append(rendered)
                found_by = f"{source} [{query.label}] {rendered}"
                try:
                    papers = client.papers(rendered, limit, found_by=found_by)
                except (SourceError, httpx.HTTPError, ValueError) as exc:
                    report.failures[f"{source} [{query.label}]"] = str(exc)[:300]
                    log.warning("search: %s [%s] failed: %s", source, query.label, exc)
                    continue
                report.hits[source] = report.hits.get(source, 0) + len(papers)
                out.append((priority, qi, papers))
            return out

        with ThreadPoolExecutor(max_workers=len(self.sources)) as pool:
            futures = [pool.submit(run, SOURCES.index(s), s) for s in self.sources]
            results = [r for f in futures for r in f.result()]
        # Reports read in source order, whatever order the threads finished in.
        report.queries = {s: report.queries[s] for s in self.sources if s in report.queries}
        report.hits = {s: report.hits.get(s, 0) for s in self.sources if s in report.queries}
        report.failures = dict(sorted(report.failures.items(), key=lambda kv: (
            SOURCES.index(kv[0].split(" ", 1)[0]), kv[0])))
        return results

    def _attach_full_text(self, epmc: EuropePMC, papers: list[Paper], report: SearchReport) -> None:
        # Europe PMC's records say whether a paper is open access; PubMed's do
        # not, so a PubMed paper with a PMCID is worth one try.
        wanted = [p for p in papers if p.pmcid and not p.full_text
                  and (p.signals.open_access or p.source == "pubmed")][: self.full_text_max]
        report.full_text_tried = len(wanted)
        if not wanted:
            return
        with ThreadPoolExecutor(max_workers=self.full_text_workers) as pool:
            texts = list(pool.map(lambda p: epmc.full_text(p.pmcid), wanted))
        for paper, xml in zip(wanted, texts):
            if xml:
                paper.full_text, paper.full_text_format = xml, "jats"
                report.full_text += 1
