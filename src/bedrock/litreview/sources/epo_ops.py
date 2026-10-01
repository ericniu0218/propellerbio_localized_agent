"""EPO Open Patent Services (OPS) — patent search, for the competitor
landscape lane only. Never a source of biological evidence; see
`sources/patents_base.py` for the line this draws and why.

    searcher = EpoOps(consumer_key=..., consumer_secret=...)
    records = searcher.search(PatentSearchRequest(...))    # satisfies PatentSearcher

Everything here is checked against the EPO's own OPS RESTful Web Services
Reference Guide v1.3.19 (the 154-page PDF at
https://link.epo.org/web/ops_v3.2_documentation_-_version_1.3.19_en.pdf),
read directly for this module — not secondhand. Confidence is marked per
piece below, because one part (the claims fulltext XML) could not be: the
guide shows it as a screenshot (its Figure 25), not extractable text.

CONFIRMED from the primary document, with page references:
  - OAuth2 token endpoint, request/response shape (p.35-36) — epo_auth.py
  - Base URL, search endpoint, CQL syntax and field codes (§3.1.1, §4.2)
  - biblio+abstract combined in one search call: `/search/biblio,abstract` (p.58)
  - <exchange-document family-id="..." country=".." doc-number=".." kind="..">
    and, when the abstract constituent is requested, a sibling
    <abstract lang="en"><p>...</p></abstract> (p.58, p.75 verbatim)
  - Range paging: max 100/page, max 2000 total results per query (p.60)
  - X-Throttling-Control, X-IndividualQuotaPerHour-Used and the
    idle/busy/overloaded, green/yellow/red/black scheme (§2.3.3) — epo_throttle.py
  - Claims endpoint URL: /published-data/publication/{type}/{number}/claims,
    returns Accept: application/fulltext+xml (p.66)

CONFIRMED LIVE (a real account, a search for "interleukin-23"): the claims
endpoint returns real claim text, and `claim_1()` below finds it. It was NOT
confirmed against the primary document, because the guide's own claims example
(its Figure 25) is an image the text extraction could not read, so the exact
element shape is still only what this parser tolerates: it matches by local
name ("claim" or "claim-text"), logs a warning and returns None rather than
raising if nothing recognisable is found. The one thing the live run showed it
getting wrong - a leading "CLAIMS" heading, which is in the register's own text
for WO filings - is removed by `_without_claims_heading`.
`scripts/litreview/epo_ops_selftest.py` repeats the live check.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from lxml import etree

from bedrock.litreview.models import PatentRecord
from bedrock.litreview.sources.epo_auth import TokenStore
from bedrock.litreview.sources.epo_throttle import EpoThrottle
from bedrock.litreview.sources.http import SourceError, SourceHTTP
from bedrock.litreview.sources.patents_base import PatentSearchRequest
from bedrock.litreview.sources.query import Query, build_queries

log = logging.getLogger(__name__)

BASE = "https://ops.epo.org/3.2/rest-services"

# EPO's own hard ceilings (p.60), not a policy choice of ours. A query that
# would need more than this has to be narrowed with more exact terms, the
# same shape problem `defaults/pipeline.yaml: search.per_query` already
# solves for the literature searcher.
MAX_PAGE = 100
MAX_TOTAL = 2000


class PatentSearchFailed(SourceError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


# --- CQL -----------------------------------------------------------------------

# CQL treats a bare space as a token break; anything with one, or an OPS
# operator keyword, must be quoted. Simpler than the literature searcher's
# `phrase()` needs to be, because titleandabstract is the only field used
# here and OPS's own quoting rule is just "wrap it if it isn't one word".
_NEEDS_QUOTES = re.compile(r'[\s"()]|^(and|or|not)$', re.I)


def phrase(term: str) -> str:
    """One exact term as a CQL titleandabstract (`ta`) clause."""
    term = term.replace('"', "")
    if _NEEDS_QUOTES.search(term):
        return f'ta="{term}"'
    return f"ta={term}"


def cql_query(query: Query) -> str:
    return query.render(phrase)


# --- XML helpers -----------------------------------------------------------------
#
# OPS mixes a default namespace (the exchange-documents schema) with an
# "ops:" prefix (search/family/legal wrappers) in the same document, and the
# exact prefix binding is not something a caller should have to know. Every
# lookup below matches by local-name so it is robust to either.


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _find(el, *names: str):
    for child in el.iter():
        if _local(child.tag) in names:
            yield child


def _first_text(el, *names: str) -> str | None:
    for child in _find(el, *names):
        text = "".join(child.itertext()).strip()
        if text:
            return text
    return None


def _docdb_date(text: str | None) -> date | None:
    """OPS dates are "yyyymmdd". A partial or missing date is common (an
    application with no publication date yet) and is not an error."""
    if not text or len(text) != 8 or not text.isdigit():
        return None
    try:
        return date(int(text[:4]), int(text[4:6]), int(text[6:8]))
    except ValueError:
        return None


def parse_exchange_document(doc, *, found_by: str) -> PatentRecord | None:
    """One <exchange-document> -> one PatentRecord. Returns None for a
    placeholder ("This record is not available through OPS" — p.59) rather
    than raising, so one restricted record does not lose the rest of a page."""
    pub_ref = next(_find(doc, "publication-reference"), None)
    if pub_ref is None:
        return None
    docdb = next((d for d in _find(pub_ref, "document-id")
                  if d.get("document-id-type") == "docdb"), None)
    if docdb is None:
        return None
    country = _first_text(docdb, "country")
    number = _first_text(docdb, "doc-number")
    kind = _first_text(docdb, "kind")
    if not country or not number:
        return None

    title = _first_text(doc, "invention-title")
    if not title:
        return None

    applicant = None
    for name_el in _find(doc, "applicant-name"):
        # "epodoc" format ("IBM [US]") is the normalised name OPS itself
        # prefers; fall back to whatever format is present rather than
        # requiring epodoc specifically, since not every office supplies it.
        parent = name_el.getparent()
        applicant = _first_text(name_el, "name")
        if applicant and (parent is None or parent.get("data-format") == "epodoc"):
            break

    # Not shown in the reference guide's own search example (p.58-59; it was
    # trimmed with "<!-- classifications removed -->"), but priority-claim is
    # standard DOCDB bibliographic-data, and it does appear in the guide's
    # legal-service example (p.119) with this same shape. `pr`/`spr` being
    # indexable in the CQL catalogue implies it is populated on search hits
    # too; matched by local name so a nesting difference does not break it.
    priority_ref = next(_find(doc, "priority-claim"), None)
    priority_date = _docdb_date(_first_text(priority_ref, "date")) if priority_ref is not None else None

    return PatentRecord(
        id=f"{country}{number}{kind or ''}",
        family_id=doc.get("family-id"),
        title=re.sub(r"\s+", " ", title).strip(),
        abstract=_first_text(doc, "abstract"),
        assignee=applicant,
        priority_date=priority_date,
        publication_date=_docdb_date(_first_text(docdb, "date")),
        kind_code=kind,
        jurisdiction=country,
        found_by_queries=[found_by],
    )


def parse_search_response(xml_bytes: bytes, *, found_by: str) -> tuple[list[PatentRecord], int]:
    """-> (records, total_result_count). One malformed document is logged and
    skipped, exactly as `europepmc.paper_from_hit` skips a bad hit."""
    root = etree.fromstring(xml_bytes)
    search = next(_find(root, "biblio-search"), None)
    total = int(search.get("total-result-count", "0")) if search is not None else 0
    records = []
    for doc in _find(root, "exchange-document"):
        try:
            record = parse_exchange_document(doc, found_by=found_by)
        except (ValueError, TypeError) as exc:
            log.warning("epo_ops: skipping malformed record: %s", exc)
            continue
        if record is not None:
            records.append(record)
    return records, total


# A WO filing's claims often open with the word CLAIMS as a heading, and the
# register's text keeps it: a live search for "interleukin-23" returned
# '1. CLAIMS 1. A method for treating ulcerative colitis ...' for claim 1. The
# heading is in the patent's own text, not something this parser added, so it
# is removed here - and only in the exact shape seen: an optional leading
# number, the word CLAIM(S), then the claim's own number. A claim that merely
# begins with the word ("1. Claims processing ...") has no number after it and
# is left alone.
_CLAIMS_HEADING = re.compile(r"^\s*(?:\d+\s*[.)]\s*)?claims?\s*:?\s*(?=\d+\s*[.)]\s)", re.I)


def _without_claims_heading(text: str) -> str:
    return _CLAIMS_HEADING.sub("", text, count=1).strip() or text


def parse_claim_1(xml_bytes: bytes) -> str | None:
    """First claim's text, or None if nothing recognisable is found.

    Deliberately loose (see the module docstring: this shape is not
    independently confirmed). Matches any element locally named "claim" or
    "claim-text", in document order, and returns the first non-empty one.
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as exc:
        log.warning("epo_ops: claims response did not parse as XML: %s", exc)
        return None
    for el in root.iter():
        if _local(el.tag) in ("claim", "claim-text"):
            text = re.sub(r"\s+", " ", "".join(el.itertext())).strip()
            if text:
                return _without_claims_heading(text)
    log.warning("epo_ops: no <claim>/<claim-text> element recognised in the claims "
                "response — this shape was not confirmed against the primary doc; "
                "see the module docstring and run scripts/litreview/epo_ops_selftest.py")
    return None


# --- the client --------------------------------------------------------------


class EpoOps:
    """`PatentSearcher`. One `TokenStore`, one `EpoThrottle`, shared across
    every call this instance makes — both are keyed to the consumer key,
    not to one request."""

    name = "epo_ops"

    def __init__(
        self,
        consumer_key: str,
        consumer_secret: str,
        *,
        http: SourceHTTP | None = None,
        tokens: TokenStore | None = None,
        throttle: EpoThrottle | None = None,
        claims_max: int = 40,
    ) -> None:
        if not consumer_key or not consumer_secret:
            raise ValueError("EPO OPS needs a consumer key and secret (EPO_OPS_KEY / EPO_OPS_SECRET)")
        self.http = http or SourceHTTP(timeout=60.0, min_interval=0.0)
        self.tokens = tokens or TokenStore(consumer_key, consumer_secret, http=self.http)
        self.throttle = throttle or EpoThrottle()
        self.claims_max = claims_max
        self._own_http = http is None

    def close(self) -> None:
        """Close the HTTP connection, if this instance opened one. A caller that
        handed in its own `http` keeps ownership of it, so this leaves it alone."""
        if self._own_http:
            self.http.close()

    # -- one authenticated, throttle-aware request -----------------------------

    def _request(self, service: str, method: str, path: str, *, headers: dict | None = None, **kwargs) -> etree._Element:
        """`service` is the X-Throttling-Control category this call falls
        under ("search" or "retrieval") — used only to decide how long to
        wait before the call, never to change what is sent.

        `headers` is kept separate from `**kwargs` and never consumed by the
        first attempt, specifically so the 401 retry below can resend it —
        merging it into kwargs and popping it on the first call was an
        earlier version of this method, and it silently dropped a caller's
        own headers (e.g. claims' `Accept: application/fulltext+xml`) on
        retry. Caught before this shipped; kept as the reason for the split.
        """
        caller_headers = dict(headers or {})
        self.throttle.wait_before(service)
        resp = self.http.request(method, f"{BASE}/{path}", headers={**caller_headers, **self.tokens.header()}, **kwargs)
        self.throttle.observe(resp.headers.get("X-Throttling-Control"))

        if resp.status_code == 401:
            # The cached token was accepted at issue time but the server no
            # longer honours it — refresh once and retry once, never loop.
            self.tokens.invalidate()
            resp = self.http.request(method, f"{BASE}/{path}", headers={**caller_headers, **self.tokens.header()}, **kwargs)
            self.throttle.observe(resp.headers.get("X-Throttling-Control"))

        if resp.status_code == 403 and resp.headers.get("X-Rejection-Reason"):
            raise PatentSearchFailed(
                f"EPO OPS fair-use quota exhausted ({resp.headers['X-Rejection-Reason']}); "
                "see https://www.epo.org/service-support/ordering/fair-use.html",
                status_code=403,
            )
        if resp.status_code != 200:
            raise PatentSearchFailed(
                f"EPO OPS {method} {path} -> HTTP {resp.status_code}: {resp.text[:300]}",
                status_code=resp.status_code,
            )
        try:
            return etree.fromstring(resp.content)
        except etree.XMLSyntaxError as exc:
            raise PatentSearchFailed(f"EPO OPS returned unparseable XML for {path}: {exc}") from None

    # -- search -------------------------------------------------------------------

    def search_page(self, cql: str, begin: int, end: int) -> tuple[list[PatentRecord], int]:
        if end - begin + 1 > MAX_PAGE:
            raise ValueError(f"a single page cannot exceed {MAX_PAGE} results (p.60 of the reference guide)")
        try:
            root = self._request(
                "search", "GET", "published-data/search/biblio,abstract",
                params={"q": cql}, headers={"X-OPS-Range": f"{begin}-{end}"},
            )
        except PatentSearchFailed as exc:
            if exc.status_code == 404:
                return [], 0  # OPS answers a zero-result CQL query with 404, not an empty list
            raise
        return parse_search_response(etree.tostring(root), found_by=cql)

    def search_all(self, cql: str, limit: int) -> list[PatentRecord]:
        limit = min(limit, MAX_TOTAL)
        records: list[PatentRecord] = []
        begin = 1
        while len(records) < limit:
            end = min(begin + MAX_PAGE - 1, limit)
            page, total = self.search_page(cql, begin, end)
            if not page:
                break
            records.extend(page)
            if total and end >= min(total, MAX_TOTAL):
                break
            begin = end + 1
        return records

    # -- claim 1, bounded --------------------------------------------------------

    def claim_1(self, record: PatentRecord) -> str | None:
        try:
            root = self._request(
                "retrieval", "GET",
                f"published-data/publication/docdb/{_docdb_ref(record)}/claims",
                headers={"Accept": "application/fulltext+xml"},
            )
        except PatentSearchFailed as exc:
            log.warning("epo_ops: could not fetch claim 1 for %s: %s", record.id, exc)
            return None
        return parse_claim_1(etree.tostring(root))

    # -- PatentSearcher -----------------------------------------------------------

    def search(self, request: PatentSearchRequest) -> list[PatentRecord]:
        queries = build_queries(request)  # same code path/discipline as the literature searcher
        by_family: dict[str, PatentRecord] = {}
        without_family: list[PatentRecord] = []
        for query in queries:
            cql = cql_query(query)
            try:
                for record in self.search_all(cql, request.max_results):
                    key = record.family_id
                    if key and key in by_family:
                        by_family[key].found_by_queries.append(cql)
                        continue
                    if key:
                        by_family[key] = record
                    else:
                        without_family.append(record)
            except PatentSearchFailed as exc:
                log.warning("epo_ops: query %r failed: %s", cql, exc)

        records = list(by_family.values()) + without_family
        # Claim 1 is a per-document call (p.66); bounded the same way the
        # literature searcher bounds open-access full text
        # (defaults/pipeline.yaml: search.full_text_max) — relevance decides
        # which patents are worth the extra round trip, not "all of them".
        for record in records[: self.claims_max]:
            record.claim_1 = self.claim_1(record)
        return records[: request.max_results]


def _docdb_ref(record: PatentRecord) -> str:
    """"EP", "1000000", "A1" -> "EP.1000000.A1", the docdb input format
    (p.118's own example: `legal/publication/docdb/EP.1000000.A1`)."""
    number = record.id
    if record.jurisdiction and number.startswith(record.jurisdiction):
        number = number[len(record.jurisdiction):]
    if record.kind_code and number.endswith(record.kind_code):
        number = number[: -len(record.kind_code)]
    parts = [record.jurisdiction or "", number]
    if record.kind_code:
        parts.append(record.kind_code)
    return ".".join(p for p in parts if p)
