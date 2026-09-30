"""PubMed E-utilities — MeSH-mapped, curated indexing. 3 requests/s without a
key, 10 with a free one (`NCBI_API_KEY`). Register a key; the difference is
large. NCBI also asks every tool to send its name and a contact email.

Two calls per query: esearch for PMIDs, best first, then one efetch for the
records. Publication types are kept — they are how reviews and retractions
are recognised downstream — and the PMCID, which is what full text is
fetched by.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import date

from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.sources.http import SourceError, SourceHTTP

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL = "litreview"
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1)}
_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")


#: elink's names for the two citation lists.
LINK_NAMES = {"citations": "pubmed_pubmed_citedin", "references": "pubmed_pubmed_refs"}


def phrase(term: str) -> str:
    """One exact term, searched in title and abstract."""
    return f'"{term}"[tiab]'


def _text(el: ET.Element | None) -> str:
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip() if el is not None else ""


def _month(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value) if 1 <= int(value) <= 12 else None
    return _MONTHS.get(value[:3])


def _date(el: ET.Element | None) -> date | None:
    """A full date only; a year alone goes into signals.year instead."""
    if el is None:
        return None
    year, month, day = _text(el.find("Year")), _month(_text(el.find("Month"))), _text(el.find("Day"))
    if not (year.isdigit() and month and day.isdigit()):
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def _year(article: ET.Element) -> int | None:
    pub = article.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pub is not None:
        text = _text(pub.find("Year")) or _text(pub.find("MedlineDate"))
        match = _YEAR.search(text)
        if match:
            return int(match.group(1))
    electronic = _date(article.find("MedlineCitation/Article/ArticleDate"))
    return electronic.year if electronic else None


def _abstract(article: ET.Element) -> str:
    parts = []
    for el in article.findall("MedlineCitation/Article/Abstract/AbstractText"):
        text = _text(el)
        if not text:
            continue
        label = el.get("Label")
        parts.append(f"{label}: {text}" if label and label.upper() != "UNLABELLED" else text)
    return " ".join(parts)


def _authors(article: ET.Element) -> list[str]:
    names = []
    for author in article.findall("MedlineCitation/Article/AuthorList/Author"):
        collective = _text(author.find("CollectiveName"))
        last = _text(author.find("LastName"))
        if last:
            initials = _text(author.find("Initials"))
            names.append(f"{last} {initials}".strip())
        elif collective:
            names.append(collective)
    return names


def paper_from_article(article: ET.Element, query: str) -> Paper | None:
    pmid = _text(article.find("MedlineCitation/PMID"))
    title = _text(article.find("MedlineCitation/Article/ArticleTitle"))
    if not pmid or not title:
        return None

    # Only the article's own identifiers: PubmedData also carries a
    # ReferenceList whose ArticleIds belong to the papers it cites.
    ids = {el.get("IdType"): (el.text or "").strip()
           for el in article.findall("PubmedData/ArticleIdList/ArticleId")}
    doi = ids.get("doi") or None
    if not doi:
        for el in article.findall("MedlineCitation/Article/ELocationID"):
            if el.get("EIdType") == "doi" and el.text:
                doi = el.text.strip()
                break
    pmcid = ids.get("pmc") or None

    types = [_text(el) for el in article.findall("MedlineCitation/Article/PublicationTypeList/PublicationType")]
    types = [t for t in types if t]
    journal = _text(article.find("MedlineCitation/Article/Journal/Title")) or None
    published = _date(article.find("MedlineCitation/Article/ArticleDate")) or _date(
        article.find("MedlineCitation/Article/Journal/JournalIssue/PubDate"))

    return Paper(
        id=f"PMID{pmid}",
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        title=title,
        abstract=_abstract(article) or None,
        authors=_authors(article)[:20],
        published=published,
        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        signals=PaperSignals(
            journal=journal,
            year=_year(article),
            is_preprint=any(t.lower() == "preprint" for t in types),
            is_retracted=retracted_by_own_record(article),
            publication_types=types,
        ),
        found_by_queries=[query],
        source="pubmed",
    )


def retracted_by_own_record(article: ET.Element) -> bool:
    """True when the article's own record links a retraction *of it*
    (`RetractionIn`). A retracted paper can carry that link without a
    "Retracted Publication" type; `RetractionOf` marks the notice, which is
    not itself a retracted study. Direction is `publication_integrity`'s rule."""
    from bedrock.litreview.publication_integrity import parse_pubmed_article

    _, events, _ = parse_pubmed_article(article)
    return any(e.status == "retracted" and e.asserted_on == "subject_record" for e in events)


def parse_articles(xml_text: str, query: str) -> list[Paper]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise SourceError(f"PubMed returned XML that does not parse: {exc}") from exc
    papers = []
    for article in root.findall("PubmedArticle"):
        try:
            paper = paper_from_article(article, query)
        except (ValueError, TypeError) as exc:
            log.warning("pubmed: skipping malformed record: %s", exc)
            continue
        if paper is not None:
            papers.append(paper)
    return papers


class PubMed:
    def __init__(
        self, *, api_key: str = "", email: str = "", http: SourceHTTP | None = None, timeout: float = 60.0,
    ) -> None:
        # NCBI's published limits, with a little headroom.
        self.http = http or SourceHTTP(min_interval=0.11 if api_key else 0.34, timeout=timeout)
        self.api_key = api_key
        self.email = email

    def _params(self, **params) -> dict:
        params["tool"] = TOOL
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def search_ids(self, query: str, limit: int) -> list[str]:
        resp = self.http.get(f"{EUTILS}/esearch.fcgi", params=self._params(
            db="pubmed", term=query, retmode="json", retmax=min(limit, 10_000), sort="relevance",
        ))
        resp.raise_for_status()
        data = resp.json()
        error = (data.get("esearchresult") or {}).get("ERROR") or data.get("error")
        if error:
            raise SourceError(f"PubMed esearch: {error}")
        return list((data.get("esearchresult") or {}).get("idlist", []))[:limit]

    def fetch(self, pmids: list[str], query: str) -> list[Paper]:
        papers: list[Paper] = []
        for start in range(0, len(pmids), 200):
            batch = pmids[start:start + 200]
            # POST: a GET with a few hundred ids outgrows NCBI's URL limit.
            resp = self.http.post(f"{EUTILS}/efetch.fcgi", data=self._params(
                db="pubmed", id=",".join(batch), retmode="xml", rettype="abstract",
            ))
            resp.raise_for_status()
            papers.extend(parse_articles(resp.text, query))
        order = {pmid: i for i, pmid in enumerate(pmids)}
        return sorted(papers, key=lambda p: order.get(p.pmid or "", len(order)))

    def linked(self, pmid: str, relation: str, limit: int) -> list[str]:
        """PubMed ids of the papers that cite this one (`relation`
        "citations") or that it cites ("references"), from elink, in
        PubMed's order. Used only by citation snowballing
        (`sources/snowball.py`)."""
        if relation not in LINK_NAMES:
            raise ValueError(f"unknown relation {relation!r}")
        resp = self.http.get(f"{EUTILS}/elink.fcgi", params=self._params(
            dbfrom="pubmed", db="pubmed", id=pmid, linkname=LINK_NAMES[relation], retmode="json",
        ))
        resp.raise_for_status()
        data = resp.json()
        if data.get("ERROR") or data.get("error"):
            raise SourceError(f"PubMed elink: {data.get('ERROR') or data.get('error')}")
        ids: list[str] = []
        for linkset in data.get("linksets") or []:
            for db in linkset.get("linksetdbs") or []:
                if db.get("linkname") == LINK_NAMES[relation]:
                    ids.extend(str(i) for i in db.get("links") or [])
        return list(dict.fromkeys(ids))[:limit]

    def papers(self, query: str, limit: int, *, found_by: str) -> list[Paper]:
        pmids = self.search_ids(query, limit)
        return self.fetch(pmids, found_by) if pmids else []
