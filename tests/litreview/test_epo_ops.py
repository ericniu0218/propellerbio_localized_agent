"""EPO OPS patent sourcing, offline: a mock transport plays the OPS API.

What this proves, with no network and no real account:

- OAuth2 client-credentials: the token endpoint is called with the right
  Basic auth, the token is cached, refreshed proactively before its ~20-minute
  expiry, and refreshed once more (not looped) on a 401
- the CQL query is built from the question's exact terms and nothing else —
  same discipline `sources/query.py` already enforces for papers
- a search response's <exchange-document> maps onto PatentRecord: family id,
  docdb id, kind code, applicant, dates, abstract — exactly the shapes quoted
  from the OPS reference guide in epo_ops.py's own docstring
- a restricted-record placeholder is skipped, not a crash
- family id merges cross-jurisdiction duplicates
- claim 1 is fetched only for the papers within `claims_max`
- the X-Throttling-Control header is read on every response and a degraded
  colour slows the *next* request in that category
- a fair-use 403 is a clean error, not a retry loop
- the generated "Competitor landscape" section groups by assignee and never
  makes a model call
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from bedrock.litreview.models import PatentRecord
from bedrock.litreview.sources.epo_auth import TOKEN_URL, AuthError, TokenStore
from bedrock.litreview.sources.epo_ops import (
    EpoOps,
    PatentSearchFailed,
    _docdb_ref,
    cql_query,
    parse_claim_1,
    parse_exchange_document,
    parse_search_response,
    phrase,
)
from bedrock.litreview.sources.epo_throttle import EpoThrottle, parse
from bedrock.litreview.sources.http import SourceHTTP
from bedrock.litreview.sources.patents_base import PatentSearchRequest
from bedrock.litreview.sources.query import Query
from bedrock.litreview.stages.s7_write import _competitors

KEY, SECRET = "test-key", "test-secret"

# --- fixtures: response bodies, shaped exactly as quoted in epo_ops.py's own
# docstring from the primary reference document -----------------------------

TOKEN_OK = {
    "access_token": "tok-1", "expires_in": "1199", "token_type": "Bearer",
    "issued_at": "1364247843353", "scope": "", "status": "approved",
}


def search_xml(*docs: str, total: int | None = None) -> str:
    total = total if total is not None else len(docs)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<ops:world-patent-data xmlns:ops="http://ops.epo.org" xmlns="http://www.epo.org/exchange">'
        f'<ops:biblio-search total-result-count="{total}">'
        f'<ops:query syntax="CQL">test</ops:query>'
        f'<ops:range begin="1" end="{max(len(docs), 1)}"/>'
        f'<ops:search-result><exchange-documents>{"".join(docs)}</exchange-documents></ops:search-result>'
        f'</ops:biblio-search></ops:world-patent-data>'
    )


def exchange_document(
    *, family="37717388", country="EP", number="3000000", kind="A1",
    title="ANTI-IL-23 ANTIBODY AND METHODS OF USE", applicant="Genentech, Inc.",
    pub_date="20200101", priority_date="20180601", abstract="An antibody that binds IL-23.",
) -> str:
    return f"""
    <exchange-document system="ops.epo.org" family-id="{family}" country="{country}"
                        doc-number="{number}" kind="{kind}">
      <bibliographic-data>
        <publication-reference>
          <document-id document-id-type="docdb">
            <country>{country}</country><doc-number>{number}</doc-number>
            <kind>{kind}</kind><date>{pub_date}</date>
          </document-id>
        </publication-reference>
        <priority-claim kind="national">
          <document-id document-id-type="docdb"><date>{priority_date}</date></document-id>
        </priority-claim>
        <parties><applicants>
          <applicant sequence="1" data-format="epodoc">
            <applicant-name><name>{applicant}</name></applicant-name>
          </applicant>
          <applicant sequence="1" data-format="original">
            <applicant-name><name>{applicant.upper()}</name></applicant-name>
          </applicant>
        </applicants></parties>
        <invention-title lang="en">{title}</invention-title>
      </bibliographic-data>
      <abstract lang="en"><p>{abstract}</p></abstract>
    </exchange-document>
    """


RESTRICTED_PLACEHOLDER = """
<exchange-document system="ops.epo.org" family-id="1" country="JP" doc-number="1" kind="A">
  This record is not available through OPS
</exchange-document>
"""

CLAIMS_XML = """<?xml version="1.0"?>
<ops:world-patent-data xmlns:ops="http://ops.epo.org" xmlns="http://www.epo.org/fulltext">
  <ftxt:claims xmlns:ftxt="http://www.epo.org/fulltext" lang="EN">
    <claim id="c-en-0001" num="0001"><claim-text>1. An isolated antibody that binds human IL-23.</claim-text></claim>
    <claim id="c-en-0002" num="0002"><claim-text>2. The antibody of claim 1, wherein...</claim-text></claim>
  </ftxt:claims>
</ops:world-patent-data>"""


# --- TokenStore ----------------------------------------------------------------


class Clock:
    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def token_transport(calls: list, *, status=200, body=None):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url == TOKEN_URL
        assert request.method == "POST"
        expected = "Basic " + base64.b64encode(f"{KEY}:{SECRET}".encode()).decode()
        assert request.headers["Authorization"] == expected
        assert request.content == b"grant_type=client_credentials"
        return httpx.Response(status, json=body if body is not None else TOKEN_OK)
    return httpx.MockTransport(handler)


def test_token_is_fetched_with_basic_auth_and_cached():
    calls: list = []
    store = TokenStore(KEY, SECRET, http=_http(token_transport(calls)))
    assert store.header() == {"Authorization": "Bearer tok-1"}
    assert store.header() == {"Authorization": "Bearer tok-1"}
    assert len(calls) == 1, "a cached token must not be re-fetched"


def test_token_refreshes_before_it_expires():
    clock = Clock()
    calls: list = []
    store = TokenStore(KEY, SECRET, http=_http(token_transport(calls)), now=clock)
    store.header()  # expires_at = 1000 + 1199 = 2199
    clock.t = 2199 - 30  # 30s before real expiry: inside the 60s margin
    store.header()
    assert len(calls) == 2


def test_token_is_not_refreshed_well_before_expiry():
    clock = Clock()
    calls: list = []
    store = TokenStore(KEY, SECRET, http=_http(token_transport(calls)), now=clock)
    store.header()
    clock.t += 500
    store.header()
    assert len(calls) == 1


def test_invalidate_forces_a_fresh_token():
    calls: list = []
    store = TokenStore(KEY, SECRET, http=_http(token_transport(calls)))
    store.header()
    store.invalidate()
    store.header()
    assert len(calls) == 2


def test_bad_credentials_raise_a_clear_error():
    store = TokenStore(KEY, SECRET, http=_http(token_transport([], status=401, body={"error": "invalid_client"})))
    with pytest.raises(AuthError):
        store.header()


# --- throttle --------------------------------------------------------------------


def test_parses_the_reference_guides_own_example_header():
    # Quoted verbatim from §2.3.3 of the reference guide.
    status = parse("idle (retrieval=green:200, search=yellow:20, "
                    "inpadoc=red:30, images=green:200, other=green:1000)")
    assert status.system_state == "idle"
    assert status.colour("search") == "yellow"
    assert status.colour("inpadoc") == "red"
    assert status.services["retrieval"] == ("green", 200)


def test_a_missing_or_malformed_header_is_not_an_error():
    assert parse("") is None
    assert parse("not this shape at all") is None


def test_throttle_sleeps_longer_as_the_colour_degrades():
    slept = []
    throttle = EpoThrottle(sleep=slept.append)
    throttle.observe("idle (search=green:30)")
    throttle.wait_before("search")
    throttle.observe("busy (search=yellow:10)")
    throttle.wait_before("search")
    throttle.observe("overloaded (search=red:5)")
    throttle.wait_before("search")
    throttle.observe("overloaded (search=black:0)")
    throttle.wait_before("search")
    assert slept == [1.5, 8.0, 60.0]  # green produced no sleep call at all


def test_an_unrelated_category_being_red_does_not_slow_this_one():
    slept = []
    throttle = EpoThrottle(sleep=slept.append)
    throttle.observe("busy (search=green:20, inpadoc=red:5)")
    throttle.wait_before("search")
    assert slept == []


# --- CQL -------------------------------------------------------------------------


def test_a_single_word_term_is_not_quoted():
    assert phrase("psoriasis") == "ta=psoriasis"


def test_a_hyphenated_single_token_is_not_quoted():
    # CQL's own quoting rule is "wrap it if it contains a space" (§4.2, rule
    # 6-7) — a hyphen is not a token break, so this must NOT be quoted. An
    # earlier draft of this test asserted the opposite from an unfounded
    # assumption; this pins the behaviour the spec actually documents.
    assert phrase("IL-23") == "ta=IL-23"


def test_a_multi_word_term_is_quoted():
    assert phrase("IL-23 antibody") == 'ta="IL-23 antibody"'


def test_query_groups_render_as_cql():
    q = Query(label="entities", groups=(("IL-23", "interleukin 23"), ("psoriasis",)))
    assert cql_query(q) == '(ta=IL-23 OR ta="interleukin 23") AND ta=psoriasis'


# --- XML parsing, against shapes confirmed in the primary document -----------------


def test_parses_a_real_shaped_exchange_document():
    root = _fromstring(search_xml(exchange_document()))
    doc = next(_iter_local(root, "exchange-document"))
    record = parse_exchange_document(doc, found_by="ta=IL-23")
    assert record.id == "EP3000000A1"
    assert record.family_id == "37717388"
    assert record.jurisdiction == "EP"
    assert record.kind_code == "A1"
    assert record.title == "ANTI-IL-23 ANTIBODY AND METHODS OF USE"
    assert record.abstract == "An antibody that binds IL-23."
    assert record.assignee == "Genentech, Inc."  # the epodoc-format name, not the ALL-CAPS original
    from datetime import date
    assert record.publication_date == date(2020, 1, 1)
    assert record.priority_date == date(2018, 6, 1)
    assert record.found_by_queries == ["ta=IL-23"]


def test_a_restricted_placeholder_is_skipped_not_a_crash():
    records, total = parse_search_response(search_xml(exchange_document(), RESTRICTED_PLACEHOLDER).encode(), found_by="q")
    assert len(records) == 1
    assert total == 2


def test_total_result_count_is_read_even_when_the_page_is_partial():
    _, total = parse_search_response(search_xml(exchange_document(), total=4321).encode(), found_by="q")
    assert total == 4321


def test_claim_1_is_the_first_claim_in_document_order():
    assert parse_claim_1(CLAIMS_XML.encode()) == ("An isolated antibody that binds human IL-23.", False)


def _claims_xml(text: str) -> bytes:
    return (f'<?xml version="1.0"?><ops:world-patent-data xmlns:ops="http://ops.epo.org" '
            f'xmlns="http://www.epo.org/fulltext"><claims><claim><claim-text>{text}</claim-text>'
            f'</claim></claims></ops:world-patent-data>').encode()


def claim_1(text: str):
    found = parse_claim_1(_claims_xml(text))
    return found[0] if found else None


@pytest.mark.parametrize("raw", [
    "CLAIMS 1. A method for treating UC",
    "CLAIMS: 1. A method for treating UC",
    "1) Claims 1. A method for treating UC",
    "1. claims 1. A method for treating UC",
    "1. CLAIMS 1. A method for treating UC",              # seen live: a page number, then the heading
    "1. 2. CLAIMS WHAT IS CLAIMED IS:3. 1. A method for treating UC",  # stacked numbers and headings
    "WO2026012345A1CLAIMS 1 . A method for treating UC",   # publication number glued on, spaced dot
    "A method for treating UC",                           # no numbering at all
])
def test_the_heading_and_stacked_numbers_are_removed_in_each_shape_they_come_in(raw):
    assert claim_1(raw) == "A method for treating UC"


@pytest.mark.parametrize("raw", [
    "1. Claims processing apparatus comprising a CPU",  # the word, but no claim number after it
    "A claim for a composition comprising an antibody",
])
def test_a_claim_that_merely_uses_the_word_is_left_alone(raw):
    assert claim_1(raw) == raw.removeprefix("1. ")


def test_a_bare_heading_is_not_a_claim():
    assert claim_1("CLAIMS") is None


def test_an_unrecognised_claims_shape_returns_none_not_a_crash(caplog):
    assert parse_claim_1(b"<root><surprise>text</surprise></root>") is None


# --- claim 1 out of the whole claim set -------------------------------------------
#
# What the claims endpoint returns is ONE <claim> holding every claim as flat
# text, and the first version of this parser returned all of it as "claim 1" -
# 5,000 to 55,000 characters, which the report footer called "never the full
# claim set". Claim 1 is cut out here, from real captured responses: the first
# 9,000 characters of the claims of 14 real WO, EP and CA filings from a live
# search (tests/litreview/fixtures/epo_claims_real.json), unchanged.

REAL_CLAIMS = json.loads((Path(__file__).parent / "fixtures" / "epo_claims_real.json").read_text())


@pytest.mark.parametrize("patent", sorted(REAL_CLAIMS))
def test_claim_1_of_a_real_claim_set_is_one_claim_and_not_the_next_ones(patent):
    from bedrock.litreview.sources.epo_ops import MAX_CLAIM_1_CHARS

    found = parse_claim_1(_claims_xml(REAL_CLAIMS[patent]["text"].replace("&", "&amp;").replace("<", "&lt;")))
    assert found is not None
    claim, cut = found
    assert 0 < len(claim) <= MAX_CLAIM_1_CHARS
    assert not re.match(r"^\s*(?:\d+\s*[.)]\s*)+", claim), f"numbering left on the front: {claim[:30]!r}"
    assert "<img" not in claim and "EMIRef" not in claim
    # It never runs into the next claim: no dependent claim's opening in it.
    assert not re.search(r"(?i)\b(?:according to|of|as claimed in|as defined in)\s+claims?\s+1\b", claim)
    assert not re.search(r"\u6839\u636e\u6743\u5229\u8981\u6c42\s*1", claim)
    if cut:
        assert len(claim) > MAX_CLAIM_1_CHARS - 100, "only a long claim is ever cut"


def test_the_numbers_glued_to_the_previous_word_do_not_hide_claim_2():
    """EP and CA text runs claims together: '... salt or solvate thereof.2. The
    compound according to claim 1'. Claim 2 is found by what it says."""
    claim = claim_1("1. A compound of formula (I), or a salt thereof.2. The compound according to claim 1, "
                    "wherein R is H.3. The compound of claim 2.")
    assert claim == "A compound of formula (I), or a salt thereof."


def test_stacked_page_and_paragraph_numbers_before_claim_2_are_not_left_on_claim_1():
    claim = claim_1("1. A method of treating psoriasis comprising dosing a patient.13. 3. The method of claim 1, "
                    "wherein the dose is oral.")
    assert claim == "A method of treating psoriasis comprising dosing a patient."


def test_a_number_that_ends_a_real_sentence_in_claim_1_is_kept():
    assert claim_1("1. A formulation wherein the pH is 2. 2. The formulation of claim 1.") == \
        "A formulation wherein the pH is 2."


def test_a_long_claim_is_cut_and_says_so():
    from bedrock.litreview.sources.epo_ops import MAX_CLAIM_1_CHARS

    found = parse_claim_1(_claims_xml("1. A compound selected from " + "alanine, " * 1000 + "valine."))
    assert found is not None and found[1] is True and len(found[0]) <= MAX_CLAIM_1_CHARS


def test_docdb_ref_strips_jurisdiction_and_kind_for_the_claims_url():
    record = PatentRecord(id="EP3000000A1", title="x", jurisdiction="EP", kind_code="A1")
    assert _docdb_ref(record) == "EP.3000000.A1"


# --- the client, end to end --------------------------------------------------------


def _fromstring(text: str):
    from lxml import etree
    return etree.fromstring(text.encode())


def _iter_local(el, name: str):
    for child in el.iter():
        if child.tag.rsplit("}", 1)[-1] == name:
            yield child


def _http(transport: httpx.MockTransport) -> SourceHTTP:
    return SourceHTTP(transport=transport, sleep=lambda s: None, timeout=10.0)


class OpsServer:
    """Plays the token endpoint plus search/claims, recording every request."""

    def __init__(self, docs: list[str], *, throttle_header: str | None = None,
                 fail_after_auth: int | None = None, claims: dict[str, bytes] | None = None) -> None:
        self.docs = docs
        self.throttle_header = throttle_header
        self.fail_after_auth = fail_after_auth
        self.claims = claims or {}
        self.requests: list[httpx.Request] = []
        self._token_calls = 0
        self._search_calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url == TOKEN_URL:
            self._token_calls += 1
            token = f"tok-{self._token_calls}"
            return httpx.Response(200, json={**TOKEN_OK, "access_token": token})

        headers = {"X-Throttling-Control": self.throttle_header} if self.throttle_header else {}
        if request.url.path.endswith("/claims"):
            number = request.url.path.split("/")[-2]
            body = self.claims.get(number, CLAIMS_XML.encode())
            return httpx.Response(200, content=body, headers={**headers, "Content-Type": "application/fulltext+xml"})

        if self.fail_after_auth is not None and self._search_calls >= self.fail_after_auth:
            return httpx.Response(403, headers={**headers, "X-Rejection-Reason": "RegisteredQuotaPerWeek"})

        self._search_calls += 1
        begin = int(request.headers["X-OPS-Range"].split("-")[0])
        page = self.docs[begin - 1: begin - 1 + 100]
        return httpx.Response(
            200, content=search_xml(*page, total=len(self.docs)).encode(),
            headers={**headers, "Content-Type": "application/xml"},
        )


def make_ops(server: OpsServer, **kwargs) -> EpoOps:
    return EpoOps(KEY, SECRET, http=_http(httpx.MockTransport(server)), **kwargs)


def test_search_satisfies_patentsearcher_and_reads_biblio_and_abstract_in_one_call():
    server = OpsServer([exchange_document()])
    ops = make_ops(server)
    request = PatentSearchRequest(question="q", exact_terms={"IL23A": ["IL-23"]}, max_results=10)
    records = ops.search(request)
    assert len(records) == 1
    assert records[0].abstract  # biblio,abstract constituent — one round trip, not two
    search_requests = [r for r in server.requests if r.url != TOKEN_URL and not r.url.path.endswith("/claims")]
    assert all("search/biblio,abstract" in str(r.url) for r in search_requests)


def test_family_id_merges_cross_jurisdiction_duplicates():
    server = OpsServer([
        exchange_document(family="999", country="EP", number="1"),
        exchange_document(family="999", country="US", number="2"),
        exchange_document(family="888", country="JP", number="3"),
    ])
    ops = make_ops(server)
    records = ops.search(PatentSearchRequest(question="q", exact_terms={"x": ["term"]}))
    assert {r.family_id for r in records} == {"999", "888"}
    merged = next(r for r in records if r.family_id == "999")
    assert len(merged.found_by_queries) >= 1


def test_claim_1_is_fetched_only_up_to_claims_max():
    docs = [exchange_document(family=str(i), number=str(i)) for i in range(5)]
    server = OpsServer(docs)
    ops = make_ops(server, claims_max=2)
    records = ops.search(PatentSearchRequest(question="q", exact_terms={"x": ["term"]}, max_results=10))
    assert sum(1 for r in records if r.claim_1) == 2
    assert sum(1 for r in records if r.claim_1 is None) == 3


def test_a_single_page_cannot_exceed_the_documented_limit():
    ops = make_ops(OpsServer([]))
    with pytest.raises(ValueError):
        ops.search_page("ta=x", 1, 101)


def test_401_refreshes_the_token_and_retries_once_preserving_headers():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == TOKEN_URL:
            calls["n"] += 1
            return httpx.Response(200, json={**TOKEN_OK, "access_token": f"tok-{calls['n']}"})
        if request.url.path.endswith("/claims"):
            if request.headers["Authorization"] == "Bearer tok-1":
                return httpx.Response(401)
            # The retry must still carry the caller's own header, not just the token.
            assert request.headers["Accept"] == "application/fulltext+xml"
            return httpx.Response(200, content=CLAIMS_XML.encode())
        return httpx.Response(404)

    ops = EpoOps(KEY, SECRET, http=_http(httpx.MockTransport(handler)))
    record = PatentRecord(id="EP1A1", title="x", jurisdiction="EP", kind_code="A1")
    assert ops.claim_1(record) is not None
    assert calls["n"] == 2  # refreshed once, not looped


def test_a_fair_use_403_is_a_clean_error_not_a_retry_loop():
    server = OpsServer([exchange_document()], fail_after_auth=0)
    ops = make_ops(server)
    with pytest.raises(PatentSearchFailed, match="fair-use quota exhausted"):
        ops.search_page("ta=x", 1, 10)


def test_no_results_is_a_404_not_an_empty_page():
    """OPS answers a genuine zero-hit CQL query with 404 (confirmed behaviour
    quoted from the reference guide), which must not be treated as failure."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == TOKEN_URL:
            return httpx.Response(200, json=TOKEN_OK)
        return httpx.Response(404)

    ops = EpoOps(KEY, SECRET, http=_http(httpx.MockTransport(handler)))
    records, total = ops.search_page("ta=nonexistent-term-xyz", 1, 25)
    assert records == [] and total == 0


def test_missing_credentials_are_refused_up_front():
    with pytest.raises(ValueError):
        EpoOps("", "")


# --- the report's competitor-landscape section --------------------------------------


def test_no_patents_says_how_to_add_them():
    body = _competitors(SimpleNamespace(patents=[], patent_search_failed=False))
    assert "Also search patents" in body and "bedrock-litreview patents" in body


def test_a_search_that_failed_is_not_reported_as_one_nobody_asked_for():
    """The section is written from `patents` alone, so an empty list after a
    failed search would read "No patent search was run" - a false statement in a
    report that did ask for one."""
    body = _competitors(SimpleNamespace(patents=[], patent_search_failed=True))
    assert "could not answer it" in body
    assert "No patent search was run" not in body
    assert "never evidence" in body, "and the reader is told nothing else depended on it"


def test_competitors_are_grouped_by_assignee_one_row_per_family():
    patents = [
        PatentRecord(id="EP1A1", family_id="f1", title="Anti-IL-23 antibody", assignee="Genentech, Inc.",
                     kind_code="A1", priority_date="2018-01-01"),
        PatentRecord(id="US1A1", family_id="f1", title="Anti-IL-23 antibody", assignee="Genentech, Inc.",
                     kind_code="A1", priority_date="2018-01-01"),
        PatentRecord(id="EP2B1", family_id="f2", title="IL-23 receptor blocker", assignee="Amgen Inc.",
                     kind_code="B1", priority_date="2019-06-01"),
    ]
    body = _competitors(SimpleNamespace(patents=patents))
    assert "Genentech, Inc." in body and "Amgen Inc." in body
    assert "2 patent famil" in body  # f1 and f2 — the US filing merged into f1's row, not a third row
    assert "3 filing" in body  # but the filing count still counts all three records
    assert body.index("2018-01-01") < body.index("2019-06-01")  # earliest priority sorts first


def test_granted_vs_pending_is_distinguished():
    patents = [PatentRecord(id="EP1B1", family_id="f1", title="x", assignee="A", kind_code="B1"),
               PatentRecord(id="EP2A1", family_id="f2", title="y", assignee="A", kind_code="A1")]
    body = _competitors(SimpleNamespace(patents=patents))
    assert "1 granted, 1 pending" in body


def test_no_model_call_is_possible_for_this_section():
    """A generated section takes no client/model at all — there is nothing
    here for `report/guard.py` to check, by construction."""
    import inspect
    assert "client" not in inspect.signature(_competitors).parameters


# --- claim availability is learned, not listed -----------------------------------


class NoClaimsServer(OpsServer):
    """Like EPO's claims endpoint: only some offices have claim text to give."""

    def __init__(self, docs, *, serves=("WO", "EP"), **kw):
        super().__init__(docs, **kw)
        self.serves = serves
        self.claim_requests: list[str] = []

    def __call__(self, request):
        if request.url.path.endswith("/claims"):
            office = request.url.path.split("/")[-2].split(".")[0]
            self.claim_requests.append(office)
            if office not in self.serves:
                self.requests.append(request)
                return httpx.Response(404, text='<fault xmlns="http://ops.epo.org"><code>CLIENT.InvalidCountryCode'
                                                '</code><message>unsupported country code</message></fault>')
        return super().__call__(request)


def test_an_office_with_no_claim_text_is_asked_once_and_then_left_alone():
    """A live search spent 40 claim requests, about half a minute, learning that
    CN, US, JP, KR, DE and ZA filings have no claim text through OPS."""
    docs = [exchange_document(family=str(i), country="CN", number=str(i)) for i in range(6)]
    server = NoClaimsServer(docs)
    ops = make_ops(server, claims_max=40)
    records = ops.search(PatentSearchRequest(question="q", exact_terms={"x": ["term"]}, max_results=10))
    assert len(records) == 6 and not any(r.claim_1 for r in records)
    assert server.claim_requests == ["CN"], "one question to the office, not six"


def test_claims_skipped_for_a_known_office_do_not_use_up_the_claim_budget():
    """The budget is for calls that can answer: 3 CN filings first must not leave
    a budget of 2 spent before the WO ones that behind them can be read."""
    docs = ([exchange_document(family=f"c{i}", country="CN", number=f"c{i}") for i in range(3)]
            + [exchange_document(family=f"w{i}", country="WO", number=f"w{i}") for i in range(4)])
    server = NoClaimsServer(docs)
    ops = make_ops(server, claims_max=2)
    records = ops.search(PatentSearchRequest(question="q", exact_terms={"x": ["term"]}, max_results=10))
    assert [r.jurisdiction for r in records if r.claim_1] == ["WO", "WO"]


def test_a_failure_that_is_not_the_country_is_not_remembered_against_the_office():
    docs = [exchange_document(family=str(i), country="WO", number=str(i)) for i in range(3)]

    class Flaky(NoClaimsServer):
        def __call__(self, request):
            if request.url.path.endswith("/claims") and len(self.claim_requests) == 0:
                self.claim_requests.append("WO")
                return httpx.Response(404, text="<fault><code>SERVER.EntityNotFound</code></fault>")
            return super().__call__(request)

    server = Flaky(docs, serves=("WO",))
    records = make_ops(server, claims_max=10).search(
        PatentSearchRequest(question="q", exact_terms={"x": ["term"]}, max_results=10))
    assert sum(1 for r in records if r.claim_1) == 2, "one filing's missing claims is that filing's problem"


def test_the_account_can_be_checked_with_a_token_request_and_nothing_else():
    server = OpsServer([])
    make_ops(server).check()
    assert [str(r.url) for r in server.requests] == [TOKEN_URL]


def test_a_refused_account_fails_the_check_with_a_clear_error():
    ops = EpoOps(KEY, SECRET, http=_http(httpx.MockTransport(lambda r: httpx.Response(401, text="bad"))))
    with pytest.raises(AuthError, match="rejected"):
        ops.check()


# --- search per target -------------------------------------------------------------


def _searches(server) -> list[str]:
    return [dict(r.url.params)["q"] for r in server.requests
            if "search/biblio" in str(r.url)]


def test_each_target_is_searched_together_with_the_questions_own_entities():
    """Searching 'psoriasis' alone finds mostly filings that list it among dozens
    of diseases (5 of the first 100 hits mention IL-23); 'IL-23 AND psoriasis'
    finds 26 and all 26 do."""
    server = OpsServer([exchange_document()])
    make_ops(server).search(PatentSearchRequest(
        question="q", exact_terms={"psoriasis": ["psoriasis"]},
        targets={"IL23A": ["IL-23", "interleukin-23"], "IL17A": ["IL-17A"]}))
    assert _searches(server) == ["ta=psoriasis AND (ta=IL-23 OR ta=interleukin-23)",
                                 "ta=psoriasis AND ta=IL-17A"]


def test_a_target_that_is_the_questions_own_entity_is_not_searched_twice():
    server = OpsServer([exchange_document()])
    make_ops(server).search(PatentSearchRequest(
        question="q", exact_terms={"IL23A": ["IL-23"], "psoriasis": ["psoriasis"]}, targets={"IL23A": ["IL-23"]}))
    assert _searches(server) == ["ta=psoriasis AND ta=IL-23"]


def test_without_targets_the_search_is_the_questions_own_terms_as_before():
    server = OpsServer([exchange_document()])
    make_ops(server).search(PatentSearchRequest(question="q", exact_terms={"psoriasis": ["psoriasis"]}))
    assert _searches(server) == ["ta=psoriasis"]


def test_a_filing_remembers_every_target_that_found_it():
    server = OpsServer([exchange_document(family="999", country="EP", number="1")])
    records = make_ops(server).search(PatentSearchRequest(
        question="q", exact_terms={"psoriasis": ["psoriasis"]},
        targets={"IL23A": ["IL-23"], "IL17A": ["IL-17A"]}))
    assert len(records) == 1, "the same family under two targets is one family"
    assert records[0].matched_targets == ["IL23A", "IL17A"]


def test_one_crowded_target_cannot_use_every_slot():
    docs = [exchange_document(family=str(i), number=str(i)) for i in range(30)]
    server = OpsServer(docs)
    records = make_ops(server).search(PatentSearchRequest(
        question="q", exact_terms={"psoriasis": ["psoriasis"]}, targets={"IL23A": ["IL-23"]},
        per_target=5, max_results=100))
    assert len(records) == 5
