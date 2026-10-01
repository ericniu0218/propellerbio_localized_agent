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

import json

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


def terms_signature(spec: QuestionSpec, extra: dict | None = None) -> str:
    """What a search for this reading of the question asks for: each required
    entity and the exact terms that count as it. Two runs of the same question
    text can read it differently - a run without a model sees only the
    entities the knowledge table knows, a run with one also sees names the
    table does not know - and papers searched for one reading are not a
    search for the other.

    `extra` is what the searcher itself adds (`LiteratureSearcher.signature`:
    the citation-snowballing policy when it is on). It is left out entirely
    when None, so a search without it keeps the signature it always had."""
    entities = sorted(
        [e.canonical.lower(), sorted({t.lower() for t in e.exact_terms})] for e in spec.required_entities
    )
    return json.dumps({"kind": "search", "entities": entities, **(extra or {})}, sort_keys=True)


def collection_signature(spec: QuestionSpec, searcher: PaperSearcher | None) -> str | None:
    """The signature a paper set collected by `searcher` for `spec` is
    recorded under. A corpus file is identified by its content hash, since it
    returns the same papers whatever the question asks; a live search by the
    terms it searched for. None for anything else (papers handed in)."""
    if searcher is None:
        return None
    sha = getattr(searcher, "sha256", None)
    if getattr(searcher, "path", None) is not None:
        return json.dumps({"kind": "corpus", "sha256": sha or ""}, sort_keys=True)
    extra = searcher.signature() if callable(getattr(searcher, "signature", None)) else None
    return terms_signature(spec, extra)


def stale_collection(recorded: str | None, spec: QuestionSpec) -> str | None:
    """Why a frozen set was not searched for this reading of the question, in
    words for the report, or None when it was (or cannot be told)."""
    if not recorded:
        return None
    try:
        body = json.loads(recorded)
    except ValueError:
        return None
    if body.get("kind") != "search":
        return None
    then = {name: set(terms) for name, terms in body.get("entities", [])}
    now = {e.canonical.lower(): e for e in spec.required_entities}
    missing = [now[name].canonical for name in sorted(now) if name not in then]
    if missing:
        return (
            "The papers were collected by an earlier search that did not look for "
            + ", ".join(f"**{m}**" for m in missing)
            + ", because that run read the question differently. Papers that mention it only "
            "under this reading were never searched for. Search again with `--search --refresh` "
            "(`/review --search --refresh` in the app) for complete coverage."
        )
    narrower = [now[name].canonical for name in sorted(now)
                if {t.lower() for t in now[name].exact_terms} - then.get(name, set())]
    if narrower:
        return (
            "The papers were collected by an earlier search that used fewer names for "
            + ", ".join(f"**{n}**" for n in narrower)
            + " than this run accepts. Search again with `--search --refresh` for complete coverage."
        )
    return None


def incomplete_search(report) -> str | None:
    """A searcher's failed queries, in words for the report, or None."""
    failures = getattr(report, "failures", None) or {}
    if not failures:
        return None
    total = sum(len(v) for v in (getattr(report, "queries", None) or {}).values()) or len(failures)
    sources = sorted({where.split(" [", 1)[0] for where in failures})
    return (
        f"The literature search was incomplete: {len(failures)} of its {total} queries failed "
        f"({', '.join(sources)} did not answer). Papers that only those queries would have found are missing "
        "from this report. Search again with `--search --refresh` when the services are back."
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
