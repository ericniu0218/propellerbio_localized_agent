"""Ask Europe PMC, PubMed and Crossref about retractions, concerns and corrections.

This is the network half of `publication_integrity`, and the only code that
refreshes integrity status. It runs in two ways:

  automatic  at the start of each new review, when the app's retraction-check
             setting is `auto` (the default; `/reviews settings`,
             `preferences.py`). `check_papers` asks PubMed about papers with a
             PMID and Crossref about papers with only a DOI, skips any paper
             checked successfully in the last `integrity.stale_after_days`,
             stops at once if the network is not there, and keeps to a time
             budget. A review never fails because a check could not run: the
             report says which papers were not checked.
  by hand    `bedrock-litreview integrity refresh --yes`, which also asks
             Europe PMC and shows what it will send before sending it.

Never on `verify-run`, a resume, or opening a report.

What leaves the machine: validated publication identifiers (PMID, DOI) and
nothing else - never a question, a report or a finding - to `www.ebi.ac.uk`,
`eutils.ncbi.nlm.nih.gov` and `api.crossref.org` (`INTEGRITY_HOSTS`),
redirects included. Crossref was approved for this by the project owner on 26
September 2026; the literature search itself still cannot reach it.
Identifiers do say what someone is reading, which is why the setting exists.

Only request shapes seen to work are sent, each captured in
`tests/litreview/integrity_fixtures/`: Europe PMC's core search for one PMID
(`EXT_ID:<pmid> AND SRC:MED`), PubMed efetch with a comma-separated PMID list,
Crossref `works/<doi>`, and Europe PMC `fullTextXML` for an open-access
notice's own text. Europe PMC's documented `POST /status-update` returned 404
when probed (2026-09-26) and is not used.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable
from urllib.parse import quote

from bedrock.litreview.publication_integrity import (
    PublicationIntegrityObservation,
    bundle_line,
    derive_state,
    is_normalized_id,
    observe,
    paper_ids,
)
from bedrock.litreview.sources.europepmc import EPMC
from bedrock.litreview.sources.http import INTEGRITY_HOSTS, SourceError, SourceHTTP
from bedrock.litreview.sources.pubmed import EUTILS, TOOL

CROSSREF = "https://api.crossref.org/works"
MAX_IDS = 400
"""The most identifiers one refresh may send."""
MAX_RESPONSE_BYTES = 5_000_000
PUBMED_BATCH = 100
PROVIDERS = ("europepmc", "pubmed", "crossref")
AUTO_BUDGET_SECONDS = 90.0
NOTICE_STATUSES = ("expression_of_concern", "retracted", "withdrawn", "reinstated")


class RefreshRefused(ValueError):
    """The request would send something it must not."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate(ids: Iterable[str]) -> list[str]:
    """The identifiers, de-duplicated and in order, or an error naming the
    first one that is not exactly a normalised PMID, PMCID or DOI."""
    out: list[str] = []
    for i in ids:
        if not is_normalized_id(i):
            raise RefreshRefused(f"not a well-formed identifier: {i!r}")
        if i not in out:
            out.append(i)
    if len(out) > MAX_IDS:
        raise RefreshRefused(f"{len(out)} identifiers; one refresh may send at most {MAX_IDS}")
    return out


def checkable(ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """(identifiers a refresh can ask about, those it cannot): PMIDs and DOIs.
    A PMCID alone is not sent: no validated request shape looks one up."""
    ids = list(ids)
    return [i for i in ids if i.startswith(("PMID:", "DOI:"))], [i for i in ids if i.startswith("PMCID:")]


def plan(ids: list[str], providers: Iterable[str] = PROVIDERS) -> list[tuple[str, list[str]]]:
    """(provider, identifiers) for each request a refresh would send: one
    Europe PMC request per PMID, PubMed efetch in batches of PMIDs, one
    Crossref request per DOI."""
    ids = validate(ids)
    pmids = [i for i in ids if i.startswith("PMID:")]
    dois = [i for i in ids if i.startswith("DOI:")]
    out: list[tuple[str, list[str]]] = []
    for provider in providers:
        if provider not in PROVIDERS:
            raise RefreshRefused(f"unknown or unapproved provider {provider!r}; allowed: {', '.join(PROVIDERS)}")
        if provider == "europepmc":
            out += [(provider, [i]) for i in pmids]
        elif provider == "pubmed":
            out += [(provider, pmids[s:s + PUBMED_BATCH]) for s in range(0, len(pmids), PUBMED_BATCH)]
        else:
            out += [(provider, [i]) for i in dois]
    return out


_BODY = re.compile(r"<body[^>]*>(.*?)</body>", re.S)
_TAGS = re.compile(r"<[^>]+>")


class IntegrityRefresher:
    def __init__(
        self, *, http: SourceHTTP | None = None, api_key: str = "", email: str = "",
        clock: Callable[[], str] = _now, timeout: float = 30.0, retries: int = 3,
    ) -> None:
        self.http = http or SourceHTTP(min_interval=0.34, timeout=timeout, retries=retries, hosts=INTEGRITY_HOSTS)
        self.api_key = api_key
        self.email = email
        self.clock = clock

    def _request(self, provider: str, ids: list[str]) -> tuple[str, dict]:
        if provider == "europepmc":
            [one] = ids
            return f"{EPMC}/search", {"query": f"EXT_ID:{one.split(':', 1)[1]} AND SRC:MED", "format": "json",
                                      "resultType": "core"}
        if provider == "crossref":
            [one] = ids
            # The DOI goes in the path, percent-encoded apart from its slashes,
            # so nothing in it can add a query parameter or change the host.
            return f"{CROSSREF}/{quote(one.split(':', 1)[1], safe='/')}", (
                {"mailto": self.email} if self.email else {})
        params = {"db": "pubmed", "id": ",".join(i.split(":", 1)[1] for i in ids), "retmode": "xml", "tool": TOOL}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return f"{EUTILS}/efetch.fcgi", params

    def _failed(self, provider, ids, retrieved, error, status=None, uri=None):
        if self.api_key:
            error = error.replace(self.api_key, "***")
        return PublicationIntegrityObservation(
            provider=provider, requested_ids=sorted(ids), retrieved_at=retrieved, outcome="error",
            http_status=status, source_uri=uri, error=error)

    def fetch(self, provider: str, ids: list[str]) -> tuple[PublicationIntegrityObservation, bytes | None, dict]:
        """One request. Returns the observation, the raw body (None on a
        transport failure) and a bundle line for carrying it elsewhere. The
        recorded URI never holds the query string, so never a key."""
        url, params = self._request(provider, ids)
        recorded_uri = url
        retrieved = self.clock()
        try:
            resp = self.http.get(url, params=params)
        except SourceError as exc:
            return self._failed(provider, ids, retrieved, str(exc), uri=recorded_uri), None, {}
        body = resp.content
        if len(body) > MAX_RESPONSE_BYTES:
            return self._failed(provider, ids, retrieved, f"response of {len(body)} bytes exceeds the "
                                f"{MAX_RESPONSE_BYTES}-byte limit", resp.status_code, recorded_uri), None, {}
        obs = observe(provider, body, requested_ids=ids, retrieved_at=retrieved, http_status=resp.status_code,
                      source_uri=recorded_uri, purpose="refresh")
        line = bundle_line(provider, body, requested_ids=ids, captured_at=retrieved, source_uri=recorded_uri,
                           http_status=resp.status_code)
        return obs, body, line

    def notice_text(self, notice_id: str, pmcid: str) -> tuple[PublicationIntegrityObservation, bytes | None]:
        """The opening of an open-access notice's own text, from Europe PMC:
        what an expression of concern says the concern is. Recorded as an
        observation about the notice, so a report can quote it."""
        retrieved = self.clock()
        url = f"{EPMC}/{pmcid}/fullTextXML"
        try:
            resp = self.http.get(url)
        except SourceError as exc:
            return self._failed("europepmc", [notice_id], retrieved, str(exc), uri=url), None
        body = resp.content[:MAX_RESPONSE_BYTES]
        text = ""
        if resp.status_code == 200:
            m = _BODY.search(body.decode("utf-8", errors="replace"))
            text = re.sub(r"\s+", " ", _TAGS.sub(" ", m.group(1))).strip() if m else ""
        return PublicationIntegrityObservation(
            provider="europepmc", requested_ids=[notice_id], resolved_ids=[notice_id] if text else [],
            retrieved_at=retrieved, outcome="success" if text else "not_found", http_status=resp.status_code,
            raw_sha256=hashlib.sha256(body).hexdigest(), source_uri=url,
            record_details={notice_id: {"title": "", "excerpt": text[:500], "types": ["notice full text"]}}
            if text else {},
        ), body

    def close(self) -> None:
        self.http.close()


# --- the automatic check at the start of a review ------------------------------------------------------


def check_papers(
    db, papers, refresher: IntegrityRefresher, *, now: datetime, ttl: timedelta,
    budget_seconds: float = AUTO_BUDGET_SECONDS, monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Check a review's papers, record what comes back, and say what happened.

    Papers checked successfully within `ttl` are not asked about again. PubMed
    is asked about PMIDs (in batches); Crossref about papers that have a DOI
    but no PMID. When a new expression of concern, retraction or reinstatement
    turns up, the notice itself is looked up, so the report can say what it
    is about. Stops at the first sign of no network, or when the time budget
    is spent; everything not asked is counted, never guessed.
    """
    started = monotonic()
    due_pmids: list[str] = []
    due_dois: list[str] = []
    fresh = 0
    for paper in papers:
        ids = paper_ids(paper)
        if not ids:
            continue
        state = derive_state(ids, db.integrity_observations(ids), now=now, ttl=ttl)
        if state.freshness == "fresh":
            fresh += 1
            continue
        pmid = next((i for i in ids if i.startswith("PMID:")), None)
        doi = next((i for i in ids if i.startswith("DOI:")), None)
        if pmid:
            due_pmids.append(pmid)
        elif doi:
            due_dois.append(doi)
    requests = [("pubmed", due_pmids[s:s + PUBMED_BATCH]) for s in range(0, len(due_pmids), PUBMED_BATCH)]
    requests += [("crossref", [d]) for d in due_dois]
    summary = {"fresh_skipped": fresh, "asked": 0, "errors": 0, "not_asked": 0, "notices": [],
               "stopped": None}
    found: list[PublicationIntegrityObservation] = []
    for n, (provider, batch) in enumerate(requests):
        if monotonic() - started > budget_seconds:
            summary["stopped"] = "time budget spent"
            summary["not_asked"] = sum(len(b) for _, b in requests[n:])
            break
        obs, raw, _ = refresher.fetch(provider, batch)
        db.add_integrity_observation(obs, raw)
        summary["asked"] += len(batch)
        if obs.outcome == "error":
            summary["errors"] += len(batch)
            if raw is None and obs.http_status is None:
                # No answer at all: most likely no network. Do not spend the
                # review waiting on every other request to fail the same way.
                summary["stopped"] = "the services could not be reached"
                summary["not_asked"] = sum(len(b) for _, b in requests[n + 1:])
                break
        found.append(obs)
    _look_up_notices(db, refresher, found, summary, started, budget_seconds, monotonic)
    return summary


def _look_up_notices(db, refresher, found, summary, started, budget_seconds, monotonic) -> None:
    """Fetch the notices behind new concerns, retractions and reinstatements."""
    notices = sorted({e.notice_id for obs in found for e in obs.events
                      if e.status in NOTICE_STATUSES and e.notice_id})
    for notice in notices:
        summary["notices"].append(notice)
        if monotonic() - started > budget_seconds or summary["stopped"]:
            continue
        known = {k for o in db.integrity_observations([notice]) for k, v in o.record_details.items()
                 if k == notice and v.get("excerpt")}
        if known:
            continue
        if notice.startswith("PMID:"):
            obs, raw, _ = refresher.fetch("europepmc", [notice])
            db.add_integrity_observation(obs, raw)
            pmcid = next((i.split(":", 1)[1] for i in obs.resolved_ids if i.startswith("PMCID:")), None)
            detail = obs.record_details.get(notice) or {}
            if pmcid and not detail.get("excerpt"):
                text_obs, body = refresher.notice_text(notice, pmcid)
                db.add_integrity_observation(text_obs, body)
        elif notice.startswith("DOI:"):
            obs, raw, _ = refresher.fetch("crossref", [notice])
            db.add_integrity_observation(obs, raw)


def summary_line(summary: dict[str, Any]) -> str:
    """One line for a review's progress: what the automatic check did."""
    parts = [f"retraction check: {summary['asked']} paper(s) asked about"]
    if summary["fresh_skipped"]:
        parts.append(f"{summary['fresh_skipped']} checked recently")
    if summary["notices"]:
        parts.append(f"{len(summary['notices'])} notice(s) found")
    if summary["errors"] or summary["not_asked"]:
        parts.append(f"{summary['errors'] + summary['not_asked']} could not be checked"
                     + (f" ({summary['stopped']})" if summary["stopped"] else ""))
    return "; ".join(parts)


def automatic_check_for(settings, *, http: SourceHTTP | None = None):
    """The check `service.run_review` gives a new review when the setting is
    `auto`, else None. It sends publication identifiers - never the question -
    skips papers checked recently, and never fails a review: without a
    network it records that nothing could be checked, and the report says so."""
    if getattr(settings, "integrity_checks", "auto") != "auto":
        return None

    def check(db, papers, *, now, ttl):
        refresher = IntegrityRefresher(http=http, api_key=settings.ncbi_api_key, email=settings.contact_email,
                                       timeout=10.0, retries=1)
        try:
            return check_papers(db, papers, refresher, now=now, ttl=ttl)
        finally:
            refresher.close()

    return check
