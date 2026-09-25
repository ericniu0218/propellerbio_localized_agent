"""The paper searcher, offline: Europe PMC and PubMed played by a mock transport.

What this proves, with no network:

- queries are built from the question's exact terms and config focus words,
  and nothing else
- each source's records map onto `Paper` with the signals downstream needs:
  reviews and retractions recognisable, preprints flagged, the article's own
  DOI (not one from its reference list)
- duplicates across sources merge, keeping Europe PMC's record and every
  query that found the paper
- open-access full text is fetched only where it can exist, and comes out of
  Stage 1 split into sections
- one source failing is tolerated and reported; every source failing raises
- the searcher drives the whole pipeline, through the runner, the CLI and the
  HTTP app
"""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pytest

from bedrock.litreview import cli
from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.runner import answer_question
from bedrock.litreview.sources.base import PaperSearcher, SearchRequest
from bedrock.litreview.sources.corpus import read_papers
from bedrock.litreview.sources.europepmc import paper_from_hit
from bedrock.litreview.sources.http import SourceError, SourceHTTP
from bedrock.litreview.sources.literature import LiteratureSearcher, SearchFailed, merge
from bedrock.litreview.sources.pubmed import parse_articles
from bedrock.litreview.sources.query import build_queries
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s1_collect import collect, search_request

QUESTION = "If I want to prevent psoriasis, what should I target?"
FOCUS = {"target_identification": ["knockout", "blockade"], "prevent": ["prevention"]}


# --- what the servers say ------------------------------------------------------

def _hit(pmid, title, *, pmcid=None, doi=None, oa="N", types=("Journal Article",), source="MED",
         cited=None, abstract="Psoriasis abstract."):
    return {
        "id": pmid or "PPR1", "source": source, "pmid": pmid, "pmcid": pmcid, "doi": doi,
        "title": title, "abstractText": abstract, "isOpenAccess": oa, "citedByCount": cited,
        "pubYear": "2021", "firstPublicationDate": "2021-03-04",
        "pubTypeList": {"pubType": list(types)},
        "journalInfo": {"journal": {"title": "J Invest Dermatol"}},
        "authorList": {"author": [{"fullName": "Smith J"}]},
    }


KNOCKOUT = _hit("111", "IL-23 knockout prevents psoriasis-like dermatitis in mice", pmcid="PMC111",
                doi="10.1000/a", oa="Y", cited=12,
                abstract="IL-23 knockout mice were protected from imiquimod-induced psoriasis-like dermatitis.")
REVIEW = _hit("222", "Psoriasis: a review of IL-17 pathways", doi="10.1000/b", types=("Review",))
CLOSED = _hit("444", "Psoriasis flare after IL-6 blockade", pmcid="PMC444", doi="10.1000/d")
PREPRINT = _hit(None, "TYK2 loss-of-function protects against psoriasis", doi="10.1101/2024.01.01.1",
                source="PPR", types=("Preprint",))

PUBMED_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation><PMID>111</PMID><Article>
   <Journal><Title>J Invest Dermatol</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
   <ArticleTitle>IL-23 knockout prevents psoriasis-like dermatitis in mice.</ArticleTitle>
   <Abstract><AbstractText>IL-23 knockout mice were protected.</AbstractText></Abstract>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
  </Article></MedlineCitation>
  <PubmedData><ArticleIdList>
   <ArticleId IdType="pubmed">111</ArticleId><ArticleId IdType="doi">10.1000/A</ArticleId>
   <ArticleId IdType="pmc">PMC111</ArticleId>
  </ArticleIdList></PubmedData>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation><PMID>555</PMID><Article>
   <Journal><Title>Exp Dermatol</Title><JournalIssue><PubDate><MedlineDate>2019 Mar-Apr</MedlineDate></PubDate></JournalIssue></Journal>
   <ArticleTitle>IL-36 receptor <i>blockade</i> in psoriasis.</ArticleTitle>
   <ELocationID EIdType="doi">10.1000/e</ELocationID>
   <Abstract>
    <AbstractText Label="BACKGROUND">IL-36 drives psoriasis.</AbstractText>
    <AbstractText Label="RESULTS">Blockade reduced PASI by 40% (n=30).</AbstractText>
   </Abstract>
   <AuthorList><Author><LastName>Lee</LastName><Initials>K</Initials></Author>
    <Author><CollectiveName>PSO Study Group</CollectiveName></Author></AuthorList>
   <PublicationTypeList><PublicationType>Journal Article</PublicationType>
    <PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
   <ArticleDate DateType="Electronic"><Year>2019</Year><Month>03</Month><Day>05</Day></ArticleDate>
  </Article></MedlineCitation>
  <PubmedData>
   <ArticleIdList><ArticleId IdType="pubmed">555</ArticleId></ArticleIdList>
   <ReferenceList><Reference><ArticleIdList><ArticleId IdType="doi">10.9999/cited-paper</ArticleId></ArticleIdList></Reference></ReferenceList>
  </PubmedData>
 </PubmedArticle>
</PubmedArticleSet>"""

JATS = """<?xml version="1.0"?>
<article><front><article-meta><abstract><p>IL-23 knockout mice were protected.</p></abstract></article-meta></front>
<body>
 <sec sec-type="methods"><title>Methods</title><p>Female C57BL/6 mice (n=10 per group) received imiquimod.</p></sec>
 <sec sec-type="results"><title>Results</title><p>IL-23 knockout reduced ear thickness by 55% compared with wild-type mice.</p></sec>
 <sec sec-type="discussion"><title>Discussion</title><p>IL-23 is required for psoriasis-like inflammation.</p></sec>
</body></article>"""


class Servers:
    """A mock transport for Europe PMC and PubMed, recording every request."""

    def __init__(self, *, pubmed_status: int = 200, epmc_status: int = 200, down: bool = False) -> None:
        self.pubmed_status = pubmed_status
        self.epmc_status = epmc_status
        self.down = down
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.down:
            raise httpx.ConnectError("name resolution failed", request=request)
        path = request.url.path
        if request.url.host == "www.ebi.ac.uk":
            if path.endswith("/fullTextXML"):
                if "PMC111" in path:
                    return httpx.Response(200, text=JATS)
                return httpx.Response(500, text="no full text")
            if self.epmc_status != 200:
                return httpx.Response(self.epmc_status)
            query = request.url.params["query"]
            hits = [PREPRINT] if "SRC:PPR" in query else [KNOCKOUT, REVIEW, CLOSED]
            return httpx.Response(200, json={"resultList": {"result": hits}, "nextCursorMark": "*"})
        if request.url.host == "eutils.ncbi.nlm.nih.gov":
            if self.pubmed_status != 200:
                return httpx.Response(self.pubmed_status)
            if path.endswith("esearch.fcgi"):
                return httpx.Response(200, json={"esearchresult": {"idlist": ["111", "555"]}})
            if path.endswith("efetch.fcgi"):
                return httpx.Response(200, text=PUBMED_XML)
        return httpx.Response(404)

    def queries(self, host: str) -> list[str]:
        out = []
        for r in self.requests:
            if r.url.host != host:
                continue
            if "query" in r.url.params:
                out.append(r.url.params["query"])
            elif "term" in r.url.params:
                out.append(r.url.params["term"])
        return out


def searcher(servers: Servers, **kwargs) -> LiteratureSearcher:
    kwargs.setdefault("focus", FOCUS)
    return LiteratureSearcher(transport=httpx.MockTransport(servers), sleep=lambda s: None, **kwargs)


@pytest.fixture()
def spec():
    return parse_question(QUESTION, None, model=None, knowledge=load_knowledge())


# --- queries -------------------------------------------------------------------

def test_queries_use_only_the_exact_terms_and_configured_focus_words(spec):
    request = search_request(spec)
    queries = build_queries(request, FOCUS)
    assert [q.label for q in queries] == ["entities", "focus:target_identification", "focus:prevent"]

    exact = {t.lower() for terms in request.exact_terms.values() for t in terms}
    focus = {w for words in FOCUS.values() for w in words}
    servers = Servers()
    searcher(servers).search(request)
    sent = servers.queries("www.ebi.ac.uk") + servers.queries("eutils.ncbi.nlm.nih.gov")
    assert sent
    for query in sent:
        for term in re.findall(r'"([^"]+)"', query):
            assert term.lower() in exact | focus, f"{term!r} was never approved for searching"
    # The plain entity query is always sent; focus queries add to it.
    assert any("knockout" not in q and "prevention" not in q for q in sent)


def test_a_question_with_no_known_entity_is_searched_as_typed():
    queries = build_queries(SearchRequest(question='What is "zzx" doing?'))
    assert [q.label for q in queries] == ["question_text"]
    assert queries[0].free_text == "What is zzx doing?"


# --- record mapping ------------------------------------------------------------

def test_pubmed_records_carry_the_signals_the_pipeline_needs():
    first, second = parse_articles(PUBMED_XML, "q")
    assert first.id == "PMID111" and first.pmcid == "PMC111" and first.doi == "10.1000/A"
    assert first.signals.year == 2021 and first.published is None  # a year alone is not a date

    assert second.doi == "10.1000/e", "the DOI must be the article's own, not a cited paper's"
    assert second.title == "IL-36 receptor blockade in psoriasis."
    assert second.abstract == "BACKGROUND: IL-36 drives psoriasis. RESULTS: Blockade reduced PASI by 40% (n=30)."
    assert "Retracted Publication" in second.signals.publication_types
    assert second.signals.year == 2019 and str(second.published) == "2019-03-05"
    assert second.authors == ["Lee K", "PSO Study Group"]
    assert second.source == "pubmed" and second.url == "https://pubmed.ncbi.nlm.nih.gov/555/"


def test_pubmed_xml_that_does_not_parse_is_a_source_error():
    with pytest.raises(SourceError):
        parse_articles("<PubmedArticleSet><oops", "q")


def test_europe_pmc_flags_preprints_and_open_access():
    preprint = paper_from_hit(PREPRINT, "q")
    assert preprint.signals.is_preprint and preprint.id == "EPMCPPR1"
    paper = paper_from_hit(KNOCKOUT, "q")
    assert paper.signals.open_access and paper.signals.citation_count == 12 and not paper.signals.is_preprint


# --- merging -------------------------------------------------------------------

def test_duplicates_merge_on_doi_and_keep_the_europe_pmc_record(spec):
    servers = Servers()
    s = searcher(servers)
    papers = s.search(search_request(spec))
    by_pmid = {p.pmid: p for p in papers if p.pmid}
    knockout = by_pmid["111"]
    assert knockout.source == "europepmc" and knockout.signals.citation_count == 12
    assert any(q.startswith("pubmed [entities]") for q in knockout.found_by_queries)
    assert any(q.startswith("europepmc [focus:prevent]") for q in knockout.found_by_queries)
    assert len([p for p in papers if p.pmid == "111"]) == 1
    preprint = next(p for p in papers if p.doi == PREPRINT["doi"])
    assert preprint.source == "biorxiv" and preprint.signals.is_preprint
    assert s.last_report.hits == {"europepmc": 9, "biorxiv": 3, "pubmed": 6}
    assert s.last_report.unique == len(papers) == 5


def test_merge_interleaves_in_proportion_and_is_deterministic():
    long = [paper_from_hit(_hit(str(i), f"Long list paper {i}"), "a") for i in range(100, 108)]
    short = [paper_from_hit(_hit(str(i), f"Short list paper {i}"), "b") for i in range(200, 202)]
    order = [p.pmid for p in merge([(2, 0, short), (0, 0, long)])]
    assert order == [p.pmid for p in merge([(0, 0, long), (2, 0, short)])]
    # Two short-list papers spread across eight long-list ones: not both first.
    assert order.index("200") == 2 and order.index("201") == 7


def test_max_results_is_respected(spec):
    request = search_request(spec, max_results=2)
    assert len(searcher(Servers()).search(request)) == 2


# --- full text -----------------------------------------------------------------

def test_open_access_full_text_is_fetched_and_split_into_sections_by_stage_1(spec):
    servers = Servers()
    s = searcher(servers)
    papers = collect(spec, searcher=s)
    knockout = next(p for p in papers if p.pmid == "111")
    assert knockout.text_source == "europepmc_xml" and knockout.has_full_text
    assert "55%" in knockout.sections["results"]
    assert "n=10" in knockout.sections["methods"]

    fetched = [r.url.path for r in servers.requests if r.url.path.endswith("/fullTextXML")]
    assert "/europepmc/webservices/rest/PMC444/fullTextXML" not in fetched, \
        "Europe PMC said PMC444 is not open access; asking anyway only costs retries"
    assert s.last_report.full_text == 1

    review = next(p for p in papers if p.pmid == "222")
    retracted = next(p for p in papers if p.pmid == "555")
    assert review.is_review and review.text_source == "abstract_only"
    assert retracted.signals.is_retracted


def test_full_text_can_be_switched_off(spec):
    servers = Servers()
    papers = searcher(servers, full_text_max=0).search(search_request(spec))
    assert not any(p.full_text for p in papers)
    assert not any(r.url.path.endswith("/fullTextXML") for r in servers.requests)


# --- failure -------------------------------------------------------------------

def test_one_source_failing_is_tolerated_and_reported(spec, caplog):
    s = searcher(Servers(pubmed_status=503))
    papers = s.search(search_request(spec))
    assert papers and all(p.source in ("europepmc", "biorxiv") for p in papers)
    assert set(s.last_report.failures) == {"pubmed [entities]", "pubmed [focus:target_identification]",
                                           "pubmed [focus:prevent]"}
    assert any("FAILED pubmed" in line for line in s.last_report.lines())


def test_every_source_failing_raises_instead_of_returning_nothing(spec):
    s = searcher(Servers(down=True))
    with pytest.raises(SearchFailed, match="every source failed"):
        s.search(search_request(spec))
    assert s.last_report.failures


def test_a_bad_request_is_not_retried():
    servers = Servers(epmc_status=400)
    http = SourceHTTP(transport=httpx.MockTransport(servers), sleep=lambda s: None)
    assert http.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=x").status_code == 400
    assert len(servers.requests) == 1


def test_retries_are_spent_then_the_error_is_raised():
    slept: list[float] = []
    servers = Servers(epmc_status=503)
    http = SourceHTTP(transport=httpx.MockTransport(servers), sleep=slept.append, retries=3)
    with pytest.raises(SourceError, match="HTTP 503"):
        http.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=x")
    assert len(servers.requests) == 3 and slept == [1.0, 2.0]


def test_unknown_sources_are_refused():
    with pytest.raises(ValueError, match="sci-hub"):
        LiteratureSearcher(sources=("europepmc", "sci-hub"))


# --- wiring --------------------------------------------------------------------

def test_it_is_a_paper_searcher_and_reads_its_settings():
    settings = load_settings(model_set="laptop", env={"NCBI_API_KEY": "k", "SOURCES_CONTACT_EMAIL": "a@b.c"},
                             check_endpoint=False)
    s = LiteratureSearcher.from_settings(settings)
    assert isinstance(s, PaperSearcher)
    assert s.ncbi_api_key == "k" and s.contact_email == "a@b.c"
    assert s.sources == ("europepmc", "biorxiv", "pubmed")
    assert "target_identification" in s.focus


def test_pubmed_is_sent_the_key_tool_and_email(spec):
    servers = Servers()
    searcher(servers, ncbi_api_key="k", contact_email="a@b.c", sources=("pubmed",)).search(search_request(spec))
    esearch = next(r for r in servers.requests if r.url.path.endswith("esearch.fcgi"))
    assert esearch.url.params["api_key"] == "k" and esearch.url.params["email"] == "a@b.c"
    assert esearch.url.params["tool"] == "litreview"
    # Nothing sent identifies the client: its name beside its search terms
    # in a public server's logs would say what it is working on.
    assert all(r.headers["User-Agent"] == "litreview/0.2" for r in servers.requests)


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    return s


def test_the_searcher_drives_the_pipeline_end_to_end(settings, tmp_path):
    result = answer_question(
        QUESTION, settings=settings, use_model=False, searcher=searcher(Servers()),
        db=Database(settings.db_path),
    )
    assert {p.pmid for p in result.papers if p.pmid} == {"111", "222", "444", "555"}
    assert result.gates["PMID222"].status == "review"
    assert result.gates["PMID555"].status == "retracted"
    assert result.gates["PMID111"].status == "pass"
    assert result.paths["html"].exists()

    # The paper set is frozen for the question: asking again does not search.
    servers = Servers()
    again = answer_question(QUESTION, settings=settings, use_model=False, searcher=searcher(servers),
                            db=Database(settings.db_path))
    assert servers.requests == [] and len(again.papers) == len(result.papers)


class _MockedSearcher(LiteratureSearcher):
    servers = Servers()

    def __init__(self, **kwargs):
        super().__init__(transport=httpx.MockTransport(self.servers), sleep=lambda s: None, **kwargs)


def test_cli_search_writes_the_handoff_file_and_ask_reads_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "LiteratureSearcher", _MockedSearcher)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("OUT_DIR", str(tmp_path / "out"))
    out = tmp_path / "papers.jsonl"
    assert cli.main(["search", QUESTION, "--out", str(out)]) == 0
    papers = read_papers(out)
    assert len(papers) == 5 and any(p.has_full_text for p in papers)

    assert cli.main(["ask", QUESTION, "--corpus", str(out), "--no-model", "--models", "laptop"]) == 0
    assert Path(capsys.readouterr().out.strip().splitlines()[-1]).exists()


def test_cli_reports_a_failed_search_and_a_missing_paper_set(tmp_path, monkeypatch, capsys):
    class Down(_MockedSearcher):
        servers = Servers(down=True)

    monkeypatch.setattr(cli, "LiteratureSearcher", Down)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    assert cli.main(["search", QUESTION, "--out", str(tmp_path / "p.jsonl")]) == 1
    assert "every source failed" in capsys.readouterr().err
    assert not (tmp_path / "p.jsonl").exists()

    assert cli.main(["ask", QUESTION, "--no-model", "--models", "laptop"]) == 1
    assert "pass --corpus FILE, or --search" in capsys.readouterr().err


def test_http_ask_searches_only_when_asked(settings, monkeypatch):
    testclient = pytest.importorskip("fastapi.testclient")
    from bedrock.litreview.api import create_app

    lanes = [Servers(down=True), Servers()]
    made = []

    def from_settings(cls, s, **kw):
        made.append(1)
        return searcher(lanes[len(made) - 1])

    monkeypatch.setattr(LiteratureSearcher, "from_settings", classmethod(from_settings))
    settings.backend = "fake"
    client = testclient.TestClient(create_app(settings))

    resp = client.post("/ask", json={"question": QUESTION})
    assert resp.status_code == 422 and "search=true" in resp.json()["detail"]
    assert made == []

    resp = client.post("/ask", json={"question": QUESTION, "search": True})
    assert resp.status_code == 502 and "every source failed" in resp.json()["detail"]

    resp = client.post("/ask", json={"question": QUESTION, "search": True})
    assert resp.status_code == 200, resp.text
    report_id = resp.json()["report_id"]
    assert client.get(f"/reports/{report_id}").status_code == 200

    resp = client.post(f"/reports/{report_id}/ratings", json={"score": 4, "rater": "jd"})
    assert resp.status_code == 200 and resp.json()["score"] == 4
    assert client.post("/reports/nope/ratings", json={"score": 4, "rater": "jd"}).status_code == 404


def test_a_benchmark_corpus_can_come_from_the_searcher(tmp_path):
    from bedrock.litreview.bench.corpus import build_corpus
    from bedrock.litreview.bench.models import BenchQuestion, CorpusSpec

    bq = BenchQuestion(id="q", question=QUESTION, corpus=CorpusSpec(queries=[]))
    manifest = build_corpus(bq, tmp_path, searcher=searcher(Servers()), progress=lambda m: None)
    assert manifest["searcher"] == "literature" and manifest["papers"] == 5 and manifest["full_text"] == 1
    assert manifest["queries"] and all("psoriasis" in q for q in manifest["queries"])
    assert len(read_papers(tmp_path / "papers.jsonl")) == 5
