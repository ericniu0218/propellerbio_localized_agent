"""Merging duplicate papers never loses a review or retraction flag.

Both merge paths — the searcher's (`sources/literature.merge`, before
`ingest.prepare`) and ingestion's (`ingest.dedupe`, after it) — go through
`ingest.merge_records`. The gate excludes papers by these flags, so a flag lost
in a merge turns a retracted review into primary evidence.
"""

from __future__ import annotations

from itertools import permutations
from pathlib import Path

import pytest

from bedrock.litreview.ingest import ingest, merge_records
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.sources.corpus import read_papers
from bedrock.litreview.sources.literature import merge

CORPUS = Path(__file__).resolve().parents[2] / "evals/litreview/corpora-searcher/psoriasis-prevention-targets/papers.jsonl"


def _copy(pid: str, source: str, types: list[str] | None = None, **kw) -> Paper:
    signals = kw.pop("signals", None) or PaperSignals(publication_types=list(types or []))
    kw.setdefault("title", "IL-17 in psoriasis")
    return Paper(id=pid, doi="10.1/x", abstract="A.",
                 source=source, signals=signals, **kw)


def _via_searcher(papers: list[Paper]) -> Paper:
    """Each copy as its own source list, in the order given, then ingested as
    the searcher's output would be."""
    merged = merge([(i, 0, [p.model_copy(deep=True)]) for i, p in enumerate(papers)])
    assert len(merged) == 1
    out = ingest(merged)
    assert len(out) == 1
    return out[0]


def _via_ingest(papers: list[Paper]) -> Paper:
    out = ingest([p.model_copy(deep=True) for p in papers])
    assert len(out) == 1
    return out[0]


PATHS = [pytest.param(_via_searcher, id="searcher"), pytest.param(_via_ingest, id="ingest")]


@pytest.mark.parametrize("path", PATHS)
def test_retracted_review_on_second_copy_survives_the_merge(path):
    a = _copy("a", "europepmc")
    b = _copy("b", "pubmed", ["Retracted Publication", "Review"])
    paper = path([a, b])
    assert paper.signals.is_retracted and paper.is_review
    assert paper.source == "europepmc"
    assert set(paper.signals.publication_types) == {"Retracted Publication", "Review"}


@pytest.mark.parametrize("path", PATHS)
def test_flags_survive_every_arrival_order(path):
    copies = [
        _copy("a", "europepmc"),
        _copy("b", "biorxiv", ["Review"]),
        _copy("c", "pubmed", ["Retracted Publication", "Journal Article"]),
    ]
    results = set()
    for order in permutations(copies):
        paper = path(list(order))
        results.add((paper.is_review, paper.signals.is_retracted,
                     frozenset(paper.signals.publication_types)))
    assert results == {(True, True, frozenset({"Review", "Retracted Publication", "Journal Article"}))}


def test_flags_set_directly_are_ored_too():
    # A source may set the flags itself rather than through publication types.
    a = _copy("a", "europepmc")
    b = _copy("b", "pubmed", signals=PaperSignals(is_retracted=True), is_review=True)
    kept = a.model_copy(deep=True)
    merge_records(kept, b)
    assert kept.is_review and kept.signals.is_retracted


def test_merged_from_lists_every_copy_with_its_flags():
    a = _copy("a", "europepmc")
    b = _copy("b", "pubmed", ["Review"], pmid="42")
    c = _copy("c", "biorxiv", ["Retracted Publication"])
    alone = Paper(id="z", doi="10.9/other", title="Something else entirely", abstract="Z.")
    out = ingest([a, b, c, alone])
    assert len(out) == 2
    paper, single = out
    assert [r["id"] for r in paper.merged_from] == ["a", "b", "c"]
    by_id = {r["id"]: r for r in paper.merged_from}
    assert by_id["a"]["source"] == "europepmc" and not by_id["a"]["is_review"]
    assert by_id["b"]["is_review"] and by_id["b"]["pmid"] == "42"
    assert by_id["c"]["is_retracted"] and by_id["c"]["publication_types"] == ["Retracted Publication"]
    assert set(by_id["a"]) == {"source", "id", "doi", "pmid", "pmcid",
                               "publication_types", "is_review", "is_retracted"}
    assert single.merged_from == [] and single.signal_conflicts == []


def test_merged_from_carries_through_both_paths():
    # Merged by the searcher, then again by ingestion: every copy still listed.
    a = _copy("a", "europepmc")
    b = _copy("b", "pubmed", ["Review"])
    searched = merge([(0, 0, [a]), (2, 0, [b])])
    c = _copy("c", "drop_folder", ["Retracted Publication"])
    [paper] = ingest(searched + [c])
    assert [r["id"] for r in paper.merged_from] == ["a", "b", "c"]
    assert paper.is_review and paper.signals.is_retracted


def test_same_record_found_twice_is_not_listed_twice():
    a = _copy("a", "europepmc", found_by_queries=["q1"])
    again = _copy("a", "europepmc", found_by_queries=["q2"])
    [paper] = merge([(0, 0, [a]), (0, 1, [again])])
    assert paper.merged_from == []
    assert paper.found_by_queries == ["q1", "q2"]


def test_signal_conflicts_record_a_year_disagreement():
    a = _copy("a", "europepmc", signals=PaperSignals(year=2021, journal="J Immunol"))
    b = _copy("b", "pubmed", signals=PaperSignals(year=2022, journal="j immunol"))
    [paper] = ingest([a, b, b.model_copy(update={"id": "b2"})])
    assert paper.signals.year == 2021
    assert paper.signal_conflicts == ["year: 2021 vs 2022 (pubmed)"]


def test_identifiers_and_metadata_are_donated():
    a = _copy("a", "europepmc", signals=PaperSignals(citation_count=None))
    b = _copy("b", "pubmed", pmid="123", pmcid="PMC9", url="https://x", authors=["Smith J"],
              signals=PaperSignals(journal="Nature", year=2020, citation_count=5, open_access=True))
    [paper] = ingest([a, b])
    assert (paper.pmid, paper.pmcid, paper.url, paper.authors) == ("123", "PMC9", "https://x", ["Smith J"])
    assert (paper.signals.journal, paper.signals.year, paper.signals.citation_count) == ("Nature", 2020, 5)
    assert paper.signals.open_access
    assert paper.signal_conflicts == []


def test_every_real_paper_keeps_a_retraction_on_its_twin():
    papers = read_papers(CORPUS)
    baseline = ingest(papers)
    twins = []
    for p in papers:
        twin = p.model_copy(deep=True)
        twin.id = f"{p.id}-twin"
        twin.source = "twin"
        twin.signals.publication_types = ["Retracted Publication"]
        twins.append(twin)
    out = ingest(papers + twins)
    assert len(out) == len(baseline)
    assert all(p.signals.is_retracted for p in out)


@pytest.mark.parametrize("path", PATHS)
def test_a_doi_written_as_a_link_is_the_same_paper(path):
    a = _copy("a", "europepmc")
    b = _copy("b", "pubmed", ["Retracted Publication"], title="IL-17 in psoriasis: corrected title")
    a.doi, b.doi = "10.1/X", "https://doi.org/10.1/x"
    assert path([a, b]).signals.is_retracted


def test_a_duplicate_linked_only_through_a_third_copy_is_merged():
    """A (DOI) and B (PMID, retracted, a different title) are the same paper,
    which only C (DOI and PMID) shows. One pass of ingestion kept A and B
    apart, and A - unflagged - went on to be used as evidence."""
    a = Paper(id="A", source="europepmc", doi="10.1/y", title="IL-23 in psoriasis", abstract="A.")
    b = Paper(id="B", source="pubmed", pmid="123", title="IL-23 in psoriasis: a mouse study", abstract="A.",
              signals=PaperSignals(publication_types=["Retracted Publication"]))
    c = Paper(id="C", source="pubmed", doi="10.1/y", pmid="123", title="IL-23 in psoriasis", abstract="A.")
    out = ingest([a, b, c])
    assert len(out) == 1
    assert out[0].signals.is_retracted
