"""Citation snowballing (`sources/snowball.py`), offline: Europe PMC's
citations and references and PubMed's elink played by a mock transport.

What this proves, with no network:

- off by default: no link request is sent, and the collection signature is
  the one a set collected before snowballing existed carries
- on, code picks the seeds (a PubMed id, not retracted, every required
  entity named) before any model sees a paper, follows references before
  citations, Europe PMC before PubMed, and adds only papers the search did
  not already keep, each saying how it was found
- a link request carries an identifier, never a search term, and reaches
  only the two approved hosts
- the policy is part of the signature, so a set collected without it is
  collected again when a search asks for it
- a failed link request is a failed query: recorded, and the report says
  the search was incomplete
- an added paper passes the same gate as any other: a cited paper about a
  near-miss is excluded with its reason
"""

from __future__ import annotations

import json

import httpx
import pytest

from bedrock.litreview.coverage import acquisition_summary
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.sources.http import SOURCE_HOSTS
from bedrock.litreview.sources.literature import LiteratureSearcher
from bedrock.litreview.sources.snowball import SnowballPolicy, choose_seeds
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s1_collect import (
    collect,
    collection_signature,
    incomplete_search,
    search_request,
    terms_signature,
)
from bedrock.litreview.stages.s2_gate import gate_papers
from tests.litreview.test_sources import CLOSED, KNOCKOUT, PREPRINT, PUBMED_XML, QUESTION, REVIEW, Servers, _hit

ON = SnowballPolicy(seeds=1, per_seed=50, max_added=100)

# What the seed, PMID 111, cites and is cited by.
FOLLOW_UP = _hit("888", "Anti-IL-23 therapy prevents psoriasis relapse: a randomised trial",
                 abstract="Psoriasis relapse was prevented by IL-23 blockade in a randomised trial.")
FOUNDATION = _hit("777", "IL-23 drives psoriasis-like skin inflammation",
                  abstract="Intradermal IL-23 induced psoriasis-like lesions.")
NEAR_MISS = _hit("666", "IL-23 blockade in psoriatic arthritis",
                 abstract="Patients with psoriatic arthritis improved on IL-23 blockade.")
PUBMED_ONLY_XML = """<?xml version="1.0"?>
<PubmedArticleSet><PubmedArticle>
  <MedlineCitation><PMID>999</PMID><Article>
   <Journal><Title>Br J Dermatol</Title><JournalIssue><PubDate><Year>2015</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>IL-23 in psoriatic plaques.</ArticleTitle>
   <Abstract><AbstractText>IL-23 was raised in psoriasis plaques.</AbstractText></Abstract>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
  </Article></MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">999</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle></PubmedArticleSet>"""


class LinkServers(Servers):
    """The search servers of `test_sources`, plus the citation lists."""

    def __init__(self, *, elink_status: int = 200, **kw) -> None:
        super().__init__(**kw)
        self.elink_status = elink_status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.url.host == "www.ebi.ac.uk":
            if path.endswith("/MED/111/references"):
                self.requests.append(request)
                return httpx.Response(200, json={"referenceList": {"reference": [
                    {"id": "777", "source": "MED"}, {"id": "222", "source": "MED"},
                    {"id": "X1", "source": "AGR"}, {"title": "a reference Europe PMC could not match"},
                ]}})
            if path.endswith("/MED/111/citations"):
                self.requests.append(request)
                return httpx.Response(200, json={"citationList": {"citation": [
                    {"id": "888", "source": "MED"}, {"id": "666", "source": "MED"},
                ]}})
            if path.endswith(("/references", "/citations")):
                self.requests.append(request)
                return httpx.Response(200, json={})
            if "EXT_ID:" in request.url.params.get("query", ""):
                self.requests.append(request)
                query = request.url.params["query"]
                hits = [h for h in (FOUNDATION, FOLLOW_UP, NEAR_MISS) if f"EXT_ID:{h['pmid']}" in query]
                return httpx.Response(200, json={"resultList": {"result": hits}, "nextCursorMark": "*"})
        if request.url.host == "eutils.ncbi.nlm.nih.gov" and path.endswith("elink.fcgi"):
            self.requests.append(request)
            if self.elink_status != 200:
                return httpx.Response(self.elink_status)
            name = request.url.params["linkname"]
            links = {"pubmed_pubmed_refs": ["777", "999"], "pubmed_pubmed_citedin": ["888"]}[name]
            return httpx.Response(200, json={"linksets": [{"dbfrom": "pubmed", "ids": [request.url.params["id"]],
                                                           "linksetdbs": [{"linkname": name, "links": links}]}]})
        if request.url.host == "eutils.ncbi.nlm.nih.gov" and path.endswith("efetch.fcgi"):
            ids = (request.content or b"").decode()
            if "999" in ids and "111" not in ids:
                self.requests.append(request)
                return httpx.Response(200, text=PUBMED_ONLY_XML)
        return super().__call__(request)

    def links(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith(("/references", "/citations", "elink.fcgi"))]


def searcher(servers: Servers, snowball: SnowballPolicy | None = None) -> LiteratureSearcher:
    return LiteratureSearcher(transport=httpx.MockTransport(servers), sleep=lambda s: None, snowball=snowball)


@pytest.fixture()
def spec():
    return parse_question(QUESTION, None, model=None, knowledge=load_knowledge())


def by_pmid(papers):
    return {p.pmid: p for p in papers}


# --- off -------------------------------------------------------------------------


def test_off_by_default_nothing_is_followed_and_the_signature_is_unchanged(spec):
    servers = LinkServers()
    s = searcher(servers)
    papers = s.search(search_request(spec))
    assert servers.links() == []
    assert sorted(p.pmid for p in papers if p.pmid) == ["111", "222", "444", "555"]
    assert s.last_report.snowball_seeds == [] and "snowball" not in s.last_report.queries
    assert collection_signature(spec, s) == terms_signature(spec)
    assert "snowball" not in acquisition_summary(s.last_report, len(papers))


def test_the_shipped_knob_is_off():
    from bedrock.litreview.config import load_settings

    settings = load_settings(model_set="laptop", env={}, check_endpoint=False)
    assert LiteratureSearcher.from_settings(settings).snowball.on is False


# --- on --------------------------------------------------------------------------


def test_on_adds_linked_papers_after_the_searched_ones_saying_how_each_was_found(spec):
    servers = LinkServers()
    s = searcher(servers, ON)
    papers = s.search(search_request(spec))
    assert {"111", "222", "444", "555"} <= {p.pmid for p in papers}

    added = [p for p in papers if p.found_by_queries and p.found_by_queries[0].startswith("snowball:")]
    # References before citations, Europe PMC's links before PubMed's; 222 was
    # already searched, and nothing linked is added twice.
    assert [p.pmid for p in added] == ["777", "999", "888", "666"]
    assert papers[-len(added):] == added, "added papers come after the searched ones"
    got = by_pmid(added)
    assert got["777"].found_by_queries == ["snowball: reference of PMID111"]
    assert got["888"].found_by_queries == ["snowball: cited by PMID111"]
    # Europe PMC's record where it has one, PubMed's for the rest.
    assert got["777"].source == "europepmc" and got["999"].source == "pubmed"

    report = s.last_report
    assert report.snowball_seeds == ["PMID111"]
    assert report.snowball_added == 4
    assert report.snowball_links == 2 + 2 + 2 + 1
    assert any("snowball: followed 1 papers" in line for line in report.lines())
    summary = acquisition_summary(report, len(papers))
    assert summary["snowball"] == {"seeds": ["PMID111"], "links_returned": 7, "papers_added": 4}


def test_a_link_request_carries_an_identifier_never_a_search_term(spec):
    servers = LinkServers()
    searcher(servers, ON).search(search_request(spec))
    assert servers.links()
    terms = {t.lower() for e in spec.entities for t in e.exact_terms}
    for request in servers.links():
        url = str(request.url).lower()
        assert not any(term in url for term in terms), url
    assert {r.url.host for r in servers.requests} <= SOURCE_HOSTS


def test_limits_hold_per_seed_and_in_all(spec):
    papers = searcher(LinkServers(), SnowballPolicy(seeds=1, per_seed=1, max_added=100)).search(search_request(spec))
    added = [p.pmid for p in papers if p.found_by_queries and p.found_by_queries[0].startswith("snowball:")]
    assert added == ["777", "888"], "one per seed and direction"
    papers = searcher(LinkServers(), SnowballPolicy(seeds=1, per_seed=50, max_added=2)).search(search_request(spec))
    added = [p.pmid for p in papers if p.found_by_queries and p.found_by_queries[0].startswith("snowball:")]
    assert added == ["777", "999"]


# --- seeds -----------------------------------------------------------------------


def test_seeds_are_chosen_by_code_from_the_searched_order(spec):
    from bedrock.litreview.sources.europepmc import paper_from_hit

    request = search_request(spec)
    unrelated = paper_from_hit(_hit("1", "IL-23 in Crohn's disease", abstract="Crohn's disease patients."), "q")
    near_miss = paper_from_hit(NEAR_MISS, "q")  # "psoriatic arthritis" only: not a mention of psoriasis
    retracted = paper_from_hit(_hit("2", "Psoriasis and IL-23"), "q")
    retracted.signals.is_retracted = True
    preprint = paper_from_hit(PREPRINT, "q")  # names psoriasis, but has no PubMed id
    first = paper_from_hit(KNOCKOUT, "q")
    second = paper_from_hit(CLOSED, "q")
    papers = [unrelated, near_miss, retracted, preprint, first, second]
    assert [p.pmid for p in choose_seeds(papers, request, 1)] == ["111"]
    assert [p.pmid for p in choose_seeds(papers, request, 5)] == ["111", "444"]
    assert choose_seeds(papers, request, 0) == []


# --- the signature ---------------------------------------------------------------


def test_the_policy_is_part_of_the_signature(spec):
    on = collection_signature(spec, searcher(LinkServers(), ON))
    off = collection_signature(spec, searcher(LinkServers()))
    assert on != off
    body = json.loads(on)
    assert body["snowball"] == {"seeds": 1, "per_seed": 50, "max_added": 100}
    assert {k: v for k, v in body.items() if k != "snowball"} == json.loads(off)


def test_a_set_collected_without_snowballing_is_collected_again_when_a_search_asks_for_it(tmp_path):
    """The runner's rule: a search whose signature differs from the one the
    frozen set was recorded under collects again."""
    from bedrock.litreview.config import load_settings
    from bedrock.litreview.db import Database
    from bedrock.litreview.runner import answer_question

    settings = load_settings(model_set="laptop", env={}, check_endpoint=False)
    settings.cache_dir, settings.out_dir = tmp_path / "cache", tmp_path / "out"
    db = Database(tmp_path / "db.sqlite3")
    first = answer_question(QUESTION, settings=settings, use_model=False, searcher=searcher(LinkServers()), db=db)
    assert db.collected_for(first.spec.id) == terms_signature(first.spec)
    reused = LinkServers()
    answer_question(QUESTION, settings=settings, use_model=False, searcher=searcher(reused), db=db)
    assert reused.requests == [], "the same search, unchanged, reuses the frozen set"

    again = LinkServers()
    result = answer_question(QUESTION, settings=settings, use_model=False, searcher=searcher(again, ON), db=db)
    assert again.links(), "the set was reused instead of collected again"
    assert json.loads(db.collected_for(first.spec.id))["snowball"]["seeds"] == 1
    assert "PMID888" in {p.id for p in result.papers}


# --- failure ---------------------------------------------------------------------


def test_a_failed_link_request_is_a_failed_query(spec):
    s = searcher(LinkServers(elink_status=400), ON)
    papers = s.search(search_request(spec))
    report = s.last_report
    assert any(k.startswith("snowball [pubmed references PMID111]") for k in report.failures)
    # Europe PMC's links still arrived.
    assert {"777", "888"} <= {p.pmid for p in papers}
    note = incomplete_search(report)
    assert note and "snowball" in note


# --- the gate --------------------------------------------------------------------


def test_an_added_paper_passes_the_same_gate_as_any_other(spec):
    papers = collect(spec, searcher=searcher(LinkServers(), ON))
    results = gate_papers(papers, spec)
    ids = by_pmid(papers)
    assert results[ids["888"].id].status == "pass"
    near = results[ids["666"].id]
    assert near.status != "pass" and "psoriatic arthritis" in (near.reason or "")
    assert results[ids["222"].id].status == "review"
