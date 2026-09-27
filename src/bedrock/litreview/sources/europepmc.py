"""Europe PMC — the workhorse. Free, no API key. It indexes PubMed, PMC and the
preprint servers (bioRxiv, medRxiv, Research Square: `SRC:PPR`), and serves
open-access full text as JATS XML already split into sections — the first rung
of the acquisition ladder, tried before any PDF.
https://europepmc.org/RestfulWebService

The benchmark's corpus builder uses this same client, so a frozen sample
corpus and a live search map records identically.
"""

from __future__ import annotations

import html
import logging
import re
from datetime import date

from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.sources.http import SourceError, SourceHTTP

log = logging.getLogger(__name__)

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
_TAG = re.compile(r"<[^>]+>")


def clean(text: str | None) -> str:
    if not text:
        return ""
    # Unescape first: Europe PMC sends "&lt;sup&gt;+&lt;/sup&gt;", which only
    # becomes a tag to strip once it is unescaped.
    text = _TAG.sub("", html.unescape(html.unescape(text)))
    return re.sub(r"\s+", " ", text).strip()


def phrase(term: str) -> str:
    """One exact term, searched in title and abstract."""
    return f'TITLE_ABS:"{term}"'


def paper_from_hit(hit: dict, query: str, *, source: str = "europepmc") -> Paper:
    pmid, pmcid, doi = hit.get("pmid"), hit.get("pmcid"), hit.get("doi")
    pid = f"PMID{pmid}" if pmid else (pmcid or f"EPMC{hit.get('id')}")
    authors = [a.get("fullName", "") for a in (hit.get("authorList") or {}).get("author", []) if a.get("fullName")]
    journal = ((hit.get("journalInfo") or {}).get("journal") or {}).get("title") or (hit.get("bookOrReportDetails") or {}).get("publisher")
    year = hit.get("pubYear")
    published = None
    if hit.get("firstPublicationDate"):
        try:
            published = date.fromisoformat(hit["firstPublicationDate"])
        except ValueError:
            published = None
    types = list((hit.get("pubTypeList") or {}).get("pubType", []))
    return Paper(
        id=pid,
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        title=clean(hit.get("title")),
        abstract=clean(hit.get("abstractText")) or None,
        authors=authors[:20],
        published=published,
        url=f"https://europepmc.org/article/{hit.get('source', 'MED')}/{hit.get('id')}",
        signals=PaperSignals(
            journal=journal,
            year=int(year) if year and str(year).isdigit() else None,
            citation_count=hit.get("citedByCount"),
            is_preprint=hit.get("source") == "PPR",
            is_retracted=retracted_by_own_record(hit),
            open_access=hit.get("isOpenAccess") == "Y",
            publication_types=types,
        ),
        found_by_queries=[query],
        source=source,
    )


def retracted_by_own_record(hit: dict) -> bool:
    """True when the record's own `commentCorrectionList` says "Retraction in":
    this paper was retracted. "Retraction of" is on the notice, which is not a
    retracted study. The core search already returns the list, so this costs
    no extra request."""
    from bedrock.litreview.publication_integrity import parse_europepmc_hit

    _, events, _ = parse_europepmc_hit(hit)
    return any(e.status == "retracted" and e.asserted_on == "subject_record" for e in events)


class EuropePMC:
    def __init__(self, timeout: float = 60.0, retries: int = 5, *, http: SourceHTTP | None = None) -> None:
        self.http = http or SourceHTTP(min_interval=0.1, timeout=timeout, retries=retries)

    def search(self, query: str, limit: int) -> list[dict]:
        """Raw hits, best first, paging with Europe PMC's cursor."""
        hits, cursor = [], "*"
        while len(hits) < limit:
            resp = self.http.get(f"{EPMC}/search", params={
                "query": query, "format": "json", "resultType": "core",
                "pageSize": min(100, limit - len(hits)), "cursorMark": cursor,
            })
            resp.raise_for_status()
            data = resp.json()
            page = (data.get("resultList") or {}).get("result", [])
            if not page:
                break
            hits.extend(page)
            nxt = data.get("nextCursorMark")
            if not nxt or nxt == cursor:
                break
            cursor = nxt
        return hits[:limit]

    def papers(self, query: str, limit: int, *, found_by: str, source: str = "europepmc") -> list[Paper]:
        papers = []
        for hit in self.search(query, limit):
            try:
                paper = paper_from_hit(hit, found_by, source=source)
            except (ValueError, TypeError) as exc:
                log.warning("europepmc: skipping malformed record %s: %s", hit.get("id"), exc)
                continue
            if paper.title:
                papers.append(paper)
        return papers

    def full_text(self, pmcid: str) -> str | None:
        """Open-access JATS XML, or None when Europe PMC has no full text."""
        try:
            resp = self.http.get(f"{EPMC}/{pmcid}/fullTextXML")
        except SourceError as exc:
            log.debug("europepmc: no full text for %s: %s", pmcid, exc)
            return None
        if resp.status_code != 200 or "<article" not in resp.text[:2000]:
            return None
        return resp.text
