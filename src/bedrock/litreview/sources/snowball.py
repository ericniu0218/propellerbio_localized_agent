"""Citation snowballing: the papers that cite the best searched papers, and
the papers they cite. Off by default (`search.snowball.seeds: 0` in
`defaults/pipeline.yaml`) until it has been measured on the benchmark
corpora.

Why. A keyword search finds papers that use the question's exact names in
their title or abstract. A follow-up trial, or the knockout study a review
leans on, can name the target only in its full text, or name it the way the
knowledge table does not know yet. Both are one citation away from a paper
the search did find. Frontier research tools follow those links; this is the
same idea on the two hosts the searcher may already reach, so no new host
and no new place the scientist's terms can go.

How, and why each choice:

- It runs inside Stage 1's search (`LiteratureSearcher.search`), so only
  when a search was asked for, and never from a frozen set or a corpus file.
- Code chooses the seeds, before any model has seen a paper: the first
  `seeds` searched papers, in the searcher's own fixed merge order, that
  have a PubMed id, are not retracted by their own record, and name every
  required entity by one of its exact terms in their title or abstract.
  Choosing seeds after Stage 3 would make what gets collected depend on a
  model's grades.
- For each seed it asks Europe PMC (`/MED/<pmid>/references` and
  `/citations`) and PubMed (`elink`, `pubmed_pubmed_refs` and
  `pubmed_pubmed_citedin`) for the linked PubMed ids, at most `per_seed`
  per seed and direction, Europe PMC's first. The request carries only an
  identifier, never a search term.
- A linked paper the search already kept is not added again. The rest, in
  seed order and then link order, are fetched as ordinary records (Europe
  PMC first, PubMed for the ids it does not hold), up to `max_added`. Each
  records how it was found ("snowball: cited by PMID123" or "snowball:
  reference of PMID123") in `found_by_queries`.
- Nothing is let in because it is linked. Every added paper goes through
  ingestion and the exact-entity gate like any other, so a cited paper
  about a near-miss is excluded with its reason shown.
- The policy is part of the frozen set's search signature
  (`stages/s1_collect.collection_signature`), so a set collected without
  snowballing is collected again when a search asks for it, and never the
  reverse silently. With it off the signature is unchanged, so no stored
  set is re-collected for the knob existing.
- A link request that fails is recorded as a failed query, and the report
  says the search was incomplete, as for any other query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from bedrock.litreview.knowledge import TermMatcher
from bedrock.litreview.models import Paper
from bedrock.litreview.sources.base import SearchRequest
from bedrock.litreview.sources.http import SourceError

log = logging.getLogger(__name__)

#: Directions, in the order they are followed: what a seed cites (usually the
#: primary studies behind it) before what cites it.
RELATIONS = ("references", "citations")
_FOUND_BY = {"references": "reference of", "citations": "cited by"}
#: Link sources, in the searcher's merge priority.
LINK_SOURCES = ("europepmc", "pubmed")


@dataclass(frozen=True)
class SnowballPolicy:
    seeds: int = 0
    """How many searched papers to follow. 0 turns snowballing off."""
    per_seed: int = 50
    """At most this many linked papers per seed and direction."""
    max_added: int = 100
    """At most this many papers added in all, after the searched ones."""

    @property
    def on(self) -> bool:
        return self.seeds > 0 and self.per_seed > 0 and self.max_added > 0

    def signature(self) -> dict | None:
        """What goes into the collection signature, or None when off."""
        if not self.on:
            return None
        return {"seeds": self.seeds, "per_seed": self.per_seed, "max_added": self.max_added}

    @classmethod
    def from_knobs(cls, knob) -> SnowballPolicy:
        return cls(
            seeds=int(knob("search.snowball.seeds", 0)),
            per_seed=int(knob("search.snowball.per_seed", 50)),
            max_added=int(knob("search.snowball.max_added", 100)),
        )


def _required(request: SearchRequest) -> list[TermMatcher]:
    """One matcher per required entity, with its near-misses, so that the
    longer near-miss wins as it does in the gate: "psoriatic arthritis" is
    not a mention of psoriasis."""
    if request.spec is not None:
        return [TermMatcher(e.exact_terms or [e.canonical], e.near_miss_terms, e.near_miss_patterns)
                for e in request.spec.required_entities]
    return [TermMatcher(terms or [name]) for name, terms in request.exact_terms.items()]


def choose_seeds(papers: list[Paper], request: SearchRequest, n: int) -> list[Paper]:
    """The first `n` papers, in the order given, that can seed: a PubMed id,
    not retracted by their own record, and every required entity named by
    an exact term in the title or abstract. None when nothing is required
    (a search of the question's own text has nothing to check a seed by)."""
    required = _required(request)
    if not required or n <= 0:
        return []
    seeds = []
    for paper in papers:
        if not paper.pmid or paper.signals.is_retracted:
            continue
        text = f"{paper.title}\n{paper.abstract or ''}"
        if all(m.exact_count(text) for m in required):
            seeds.append(paper)
            if len(seeds) == n:
                break
    return seeds


def follow(policy: SnowballPolicy, seeds: list[Paper], clients: dict, known_pmids: set[str], report) -> list[Paper]:
    """The papers linked to `seeds` that are not in `known_pmids`, as full
    records, in seed order then link order, at most `policy.max_added`.

    `clients` maps "europepmc" and "pubmed" to objects with `linked(pmid,
    relation, limit)`; the Europe PMC one also has `records(pmids, found_by)`
    and the PubMed one `fetch(pmids, query)`. `report` is the search's
    `SearchReport`: every link request is counted as a query under
    "snowball", and a failed one is recorded as a failure."""
    found_by: dict[str, list[str]] = {}
    queries = report.queries.setdefault("snowball", [])
    for seed in seeds:
        for relation in RELATIONS:
            taken: list[str] = []
            for source in LINK_SOURCES:
                client = clients.get(source)
                if client is None:
                    continue
                label = f"{source} {relation} PMID{seed.pmid}"
                queries.append(label)
                try:
                    ids = client.linked(seed.pmid, relation, policy.per_seed)
                except (SourceError, httpx.HTTPError, ValueError) as exc:
                    report.failures[f"snowball [{label}]"] = str(exc)[:300]
                    log.warning("snowball: %s failed: %s", label, exc)
                    continue
                report.snowball_links += len(ids)
                taken.extend(i for i in ids if i not in taken)
            for pmid in taken[: policy.per_seed]:
                if pmid in known_pmids or pmid == seed.pmid:
                    continue
                found_by.setdefault(pmid, []).append(f"snowball: {_FOUND_BY[relation]} PMID{seed.pmid}")

    wanted = list(found_by)[: policy.max_added]
    if not wanted:
        return []
    papers = _records(wanted, clients, report)
    for paper in papers:
        paper.found_by_queries = found_by[paper.pmid]
    return papers


def _records(pmids: list[str], clients: dict, report) -> list[Paper]:
    """Europe PMC's record for each id where it has one (it carries the
    citation count and open-access flag), PubMed's for the rest."""
    got: dict[str, Paper] = {}
    epmc, pubmed = clients.get("europepmc"), clients.get("pubmed")
    if epmc is not None:
        report.queries["snowball"].append("europepmc records")
        try:
            for p in epmc.records(pmids, found_by="snowball"):
                got.setdefault(p.pmid, p)
        except (SourceError, httpx.HTTPError, ValueError) as exc:
            report.failures["snowball [europepmc records]"] = str(exc)[:300]
            log.warning("snowball: europepmc records failed: %s", exc)
    rest = [pmid for pmid in pmids if pmid not in got]
    if rest and pubmed is not None:
        report.queries["snowball"].append("pubmed records")
        try:
            for p in pubmed.fetch(rest, "snowball"):
                got.setdefault(p.pmid, p)
        except (SourceError, httpx.HTTPError, ValueError) as exc:
            report.failures["snowball [pubmed records]"] = str(exc)[:300]
            log.warning("snowball: pubmed records failed: %s", exc)
    return [got[pmid] for pmid in pmids if pmid in got and got[pmid].title]
