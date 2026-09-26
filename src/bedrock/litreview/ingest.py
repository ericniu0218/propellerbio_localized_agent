"""Make whatever the searcher returned safe for the pipeline to read.

Splits raw full text into sections, flags reviews and retractions from their
publication types, and deduplicates on DOI, then PMID, then normalised title.
Pure code; nothing here guesses.
"""

from __future__ import annotations

import logging
import re

from bedrock.litreview.models import Paper, stable_id
from bedrock.litreview.parsing.sections import split

log = logging.getLogger(__name__)

_REVIEW_TYPES = re.compile(r"\b(review|meta-analysis|systematic review)\b", re.I)
_RETRACTED = re.compile(r"retract", re.I)
_TITLE_FOLD = re.compile(r"[^a-z0-9]+")


_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.I)


def _norm_title(title: str) -> str:
    return _TITLE_FOLD.sub(" ", (title or "").lower()).strip()


def doi_key(doi: str) -> str:
    """A DOI as a dedupe key: "https://doi.org/10.1/X" and "10.1/x" are the
    same paper, and a retraction flag on one copy must reach the other."""
    return _DOI_PREFIX.sub("", doi.strip()).strip().lower()


def prepare(paper: Paper) -> Paper:
    paper = paper.model_copy(deep=True)
    if not paper.id:
        paper.id = stable_id(paper.doi or paper.pmid or paper.pmcid or _norm_title(paper.title))

    if not paper.sections and paper.full_text:
        fmt = paper.full_text_format or "text"
        paper.sections = split(
            paper.full_text if fmt == "text" else None,
            jats=paper.full_text if fmt == "jats" else None,
            tei=paper.full_text if fmt == "tei" else None,
        )
        if paper.text_source == "none":
            paper.text_source = {"jats": "europepmc_xml", "tei": "grobid_pdf"}.get(fmt, "flat_text")
    # Raw text is parsed now; keeping it would double the size of every copy.
    paper.full_text = None

    if paper.abstract and "abstract" not in paper.sections:
        paper.sections["abstract"] = paper.abstract
    if not paper.abstract and paper.sections.get("abstract"):
        paper.abstract = paper.sections["abstract"]
    if paper.text_source == "none" and paper.abstract:
        paper.text_source = "abstract_only"

    types = " ".join(paper.signals.publication_types)
    if _REVIEW_TYPES.search(types):
        paper.is_review = True
    if _RETRACTED.search(types) or (paper.title or "").lower().startswith("retracted"):
        paper.signals.is_retracted = True
    return paper


def _record(paper: Paper) -> dict[str, object]:
    return {
        "source": paper.source,
        "id": paper.id,
        "doi": paper.doi,
        "pmid": paper.pmid,
        "pmcid": paper.pmcid,
        "publication_types": list(paper.signals.publication_types),
        "is_review": paper.is_review,
        "is_retracted": paper.signals.is_retracted,
    }


def _add_all(into: list, items: list) -> None:
    for item in items:
        if item not in into:
            into.append(item)


def merge_records(kept: Paper, incoming: Paper) -> None:
    """Fold a duplicate record into the one being kept, in place.

    The searcher and ingestion both use this, so there is one rule for what a
    duplicate carries over. A review or retraction flag on any copy survives:
    publication types are unioned and the two flags are OR-ed, so a retracted
    review found by PubMed is still one after its Europe PMC twin wins the
    merge, whichever order the copies arrive in. Identifiers and metadata the
    kept record lacks are filled in from the other copy. Where both copies
    have a year, journal or preprint status and they disagree, the kept value
    stays and the disagreement is written to `signal_conflicts` for a human to
    see; nothing here decides which copy was right.

    `merged_from` lists every copy folded in, the kept record first, with the
    ids and flags each carried. It stays empty on a paper that was never
    merged, and a copy identical to one already listed (the same record found
    by two queries) is not listed twice.
    """
    records = list(kept.merged_from) or [_record(kept)]
    _add_all(records, list(incoming.merged_from) or [_record(incoming)])
    if len(records) > 1:
        kept.merged_from = records

    ks, inc = kept.signals, incoming.signals
    where = incoming.source or incoming.id
    conflicts = []
    if ks.year is not None and inc.year is not None and ks.year != inc.year:
        conflicts.append(f"year: {ks.year} vs {inc.year} ({where})")
    if ks.journal and inc.journal and ks.journal.strip().lower() != inc.journal.strip().lower():
        conflicts.append(f"journal: {ks.journal} vs {inc.journal} ({where})")
    if ks.is_preprint != inc.is_preprint:
        conflicts.append(f"preprint: {ks.is_preprint} vs {inc.is_preprint} ({where})")
    _add_all(kept.signal_conflicts, conflicts + incoming.signal_conflicts)

    # Flags: added by a merge, never lost.
    _add_all(ks.publication_types, inc.publication_types)
    kept.is_review = kept.is_review or incoming.is_review
    ks.is_retracted = ks.is_retracted or inc.is_retracted
    ks.open_access = ks.open_access or inc.open_access

    for name in ("doi", "pmid", "pmcid", "abstract", "url"):
        if not getattr(kept, name) and getattr(incoming, name):
            setattr(kept, name, getattr(incoming, name))
    if not kept.authors and incoming.authors:
        kept.authors = list(incoming.authors)
    if kept.published is None and incoming.published is not None:
        kept.published = incoming.published
    for name in ("journal", "year", "citation_count"):
        if getattr(ks, name) is None and getattr(inc, name) is not None:
            setattr(ks, name, getattr(inc, name))

    if not kept.has_full_text and incoming.has_full_text:
        kept.sections = incoming.sections
        kept.text_source = incoming.text_source
    elif not kept.has_full_text and not kept.full_text and incoming.full_text:
        # Before `prepare`, full text is still raw; it is split later.
        kept.full_text = incoming.full_text
        kept.full_text_format = incoming.full_text_format

    _add_all(kept.found_by_queries, incoming.found_by_queries)


def dedupe(papers: list[Paper]) -> list[Paper]:
    """First occurrence wins, matched on DOI, then PMID, then title. Every
    later duplicate is folded into it by `merge_records`, so it donates what
    the first lacks (identifiers, full text, the queries that found it) and a
    review or retraction flag it carries is kept rather than dropped with it.
    This runs after `prepare`, so each copy's flags are already set from its
    own publication types; they are OR-ed here.

    Repeated until nothing merges: a copy with only a PMID arrives before the
    copy that links that PMID to the kept record's DOI, and one pass leaves it
    standing beside its twin - with the retraction flag only it carried."""
    kept = _dedupe_once(papers)
    while True:
        again = _dedupe_once(kept)
        if len(again) == len(kept):
            return again
        kept = again


def _dedupe_once(papers: list[Paper]) -> list[Paper]:
    kept: list[Paper] = []
    index: dict[str, Paper] = {}
    for paper in papers:
        keys = [
            f"doi:{doi_key(paper.doi)}" if paper.doi else None,
            f"pmid:{paper.pmid}" if paper.pmid else None,
            f"title:{_norm_title(paper.title)}" if paper.title else None,
        ]
        keys = [k for k in keys if k]
        existing = next((index[k] for k in keys if k in index), None)
        if existing is None:
            kept.append(paper)
            for k in keys:
                index[k] = paper
            continue
        merge_records(existing, paper)
        for k in keys:
            index.setdefault(k, existing)
    return kept


def ingest(papers: list[Paper]) -> list[Paper]:
    prepared = [prepare(p) for p in papers]
    unique = dedupe(prepared)
    if len(unique) < len(prepared):
        log.info("ingest: %d duplicates removed", len(prepared) - len(unique))
    return unique
