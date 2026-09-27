"""Retractions, expressions of concern and corrections: observed, dated, never overwritten.

A paper's integrity status is not a property of the paper. It is something a
provider said about it at a time: Europe PMC's `commentCorrectionList`,
PubMed's `CommentsCorrections`, Crossref's `update-to`/`updated-by`. So it is
stored as *observations* - who was asked, about which identifiers, when, what
came back (by hash), and the relationships parsed out of it - appended to a
table and never edited. The current status of a paper is derived from them
on demand, with its own freshness clock.

Two things this keeps apart, because confusing them is how integrity data
goes wrong:

  direction   Europe PMC writes "Retraction in" on the retracted article and
              "Retraction of" on the notice. The first means *this* paper is
              retracted; the second means this record *is* a retraction notice
              about another paper. A notice is not a retracted experiment.
  status and  an event says what happened (retracted, concern, corrected,
  freshness   updated, republished, withdrawn, reinstated); freshness says when
              anyone last successfully looked. A failed lookup never erases an
              earlier adverse event and never counts as a check. "No notice
              found" is scoped to the providers asked, and is never proof that
              a paper was not retracted.

What happens with it: a retracted paper is excluded from new runs (as the
gate always did for retractions it could see); a concern or a correction is
flagged, never silently excluded - a person decides what it means for the
finding quoted. Old runs are never rewritten: a refresh records a new
observation and `verify-run` reports it as an advisory beside the untouched
report.

Nothing here goes to the network. The explicit refresh command
(`bedrock-litreview integrity refresh`) does, on the two existing source hosts
only, with identifiers and never question text; Crossref is parsed from
offline bundles, never fetched.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = 1
PARSER_VERSION = "publication_integrity.v1"
SNAPSHOT_ARTIFACT = "publication_integrity_snapshot.v1"

Status = Literal[
    "retracted", "expression_of_concern", "corrected", "updated", "republished", "withdrawn", "reinstated",
    "unknown_relation",
]
Outcome = Literal["success", "not_found", "error", "not_checked"]
Provider = Literal["europepmc", "pubmed", "crossref"]
ADVERSE = ("retracted", "withdrawn", "expression_of_concern")


# --- identifiers ----------------------------------------------------------------------------

_PMID = re.compile(r"^\d{1,9}$")
_PMCID = re.compile(r"^PMC\d{1,9}$", re.I)
_DOI = re.compile(r"^10\.\d{4,9}/[^\s\"<>\\]{1,200}$")


def normalize_id(kind: str, value: str | None) -> str | None:
    """`PMID:123`, `PMCID:PMC123` or `DOI:10.x/y` (lower-cased), or None when
    the value is not a well-formed identifier of that kind. Everything the
    refresh sends is one of these; nothing else is ever put in a request."""
    if value is None:
        return None
    v = str(value).strip()
    if kind == "pmid" and _PMID.match(v):
        return f"PMID:{int(v)}"
    if kind == "pmcid" and _PMCID.match(v):
        return f"PMCID:{v.upper()}"
    if kind == "doi":
        v = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", v, flags=re.I)
        if _DOI.match(v):
            return f"DOI:{v.lower()}"
    return None


def is_normalized_id(value: object) -> bool:
    """True for exactly what `normalize_id` produces: `PMID:1`, `PMCID:PMC1`, `DOI:10.x/y`."""
    if not isinstance(value, str) or ":" not in value:
        return False
    prefix, rest = value.split(":", 1)
    kind = {"PMID": "pmid", "PMCID": "pmcid", "DOI": "doi"}.get(prefix)
    return kind is not None and normalize_id(kind, rest) == value


def paper_ids(paper) -> list[str]:
    """The normalised identifiers of a paper, in a fixed order."""
    out = [normalize_id("pmid", paper.pmid), normalize_id("pmcid", paper.pmcid), normalize_id("doi", paper.doi)]
    return [x for x in out if x]


# --- models -----------------------------------------------------------------------------------


class IntegrityAssertion(BaseModel):
    """Who said it: a provider, the field, and the provider's own record id."""

    provider: Provider
    field: str
    source: str | None = None
    """Crossref's `source`: `publisher` or `retraction-watch`."""
    record_id: str | None = None
    date: str | None = None


class PublicationIntegrityEvent(BaseModel):
    """One relationship: `notice_id` says `status` about `subject_id`."""

    subject_id: str | None
    """The article the event is about (the original). None when the provider
    gave no identifier for it."""
    notice_id: str | None
    """The notice, erratum or updated version; None when not identified."""
    status: Status
    raw_type: str
    asserted_on: Literal["subject_record", "notice_record"]
    """Which record carried the relationship: the original's own record
    ("Retraction in") or the notice's ("Retraction of")."""
    reference: str | None = None
    assertions: list[IntegrityAssertion] = Field(default_factory=list)
    subject_aliases: list[str] = Field(default_factory=list)
    """The subject's other identifiers, when the relationship was on the
    subject's own record (which lists its PMID, PMCID and DOI together). A
    notice names its subject by one identifier only, and nothing is matched
    on a title: no status moves between papers by resemblance."""

    def key(self) -> tuple:
        return (self.subject_id, self.notice_id, self.status)


class PublicationIntegrityObservation(BaseModel):
    """One lookup: what was asked, of whom, when, and what came back."""

    schema_version: int = SCHEMA_VERSION
    parser_version: str = PARSER_VERSION
    provider: Provider
    request_purpose: Literal["refresh", "offline_import", "search"] = "refresh"
    requested_ids: list[str]
    resolved_ids: list[str] = Field(default_factory=list)
    """The identifiers of the record(s) the provider actually returned."""
    source_uri: str | None = None
    retrieved_at: str
    provider_date: str | None = None
    outcome: Outcome
    http_status: int | None = None
    raw_sha256: str | None = None
    events: list[PublicationIntegrityEvent] = Field(default_factory=list)
    identity_conflicts: list[str] = Field(default_factory=list)
    other_relations: list[str] = Field(default_factory=list)
    """Relationships that are not integrity events ("Comment in"), kept so
    it is visible they were seen and deliberately not counted."""
    error: str | None = None
    record_details: dict[str, dict[str, Any]] = Field(default_factory=dict)
    """Identifier -> the returned record's own title and opening text (at most
    `EXCERPT_CHARS`) and publication types. For a notice, this is its reason:
    an expression of concern says in its text what the concern is."""
    import_provenance: dict[str, Any] | None = None
    """For an offline import: the bundle's hash and who/where it says it came
    from. A hash proves the bytes are the bytes, not who produced them."""

    @property
    def observation_id(self) -> str:
        body = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(body.encode("utf-8")).hexdigest()[:32]


# --- relationship vocabularies ------------------------------------------------------------------

# (status, which record carries it: True = the subject's own record)
_EPMC_TYPES: dict[str, tuple[Status | None, bool]] = {
    "retraction in": ("retracted", True), "retraction of": ("retracted", False),
    "partial retraction in": ("unknown_relation", True), "partial retraction of": ("unknown_relation", False),
    "expression of concern in": ("expression_of_concern", True),
    "expression of concern for": ("expression_of_concern", False),
    "erratum in": ("corrected", True), "erratum for": ("corrected", False),
    "update in": ("updated", True), "update of": ("updated", False),
    "republished in": ("republished", True), "republished from": ("republished", False),
    "retracted and republished in": ("retracted", True), "retracted and republished from": ("retracted", False),
    "comment in": (None, True), "comment on": (None, False), "cites": (None, False),
    "summary for patients in": (None, True), "original report in": (None, True),
    "reprint in": (None, True), "reprint of": (None, False),
}
_PUBMED_TYPES: dict[str, str] = {
    "RetractionIn": "retraction in", "RetractionOf": "retraction of",
    "PartialRetractionIn": "partial retraction in", "PartialRetractionOf": "partial retraction of",
    "ExpressionOfConcernIn": "expression of concern in", "ExpressionOfConcernFor": "expression of concern for",
    "ErratumIn": "erratum in", "ErratumFor": "erratum for", "UpdateIn": "update in", "UpdateOf": "update of",
    "RepublishedIn": "republished in", "RepublishedFrom": "republished from",
    "RetractedandRepublishedIn": "retracted and republished in",
    "RetractedandRepublishedFrom": "retracted and republished from",
    "CommentIn": "comment in", "CommentOn": "comment on", "Cites": "cites",
    "SummaryForPatientsIn": "summary for patients in", "OriginalReportIn": "original report in",
    "ReprintIn": "reprint in", "ReprintOf": "reprint of",
}
_CROSSREF_TYPES: dict[str, Status] = {
    "retraction": "retracted", "removal": "withdrawn", "withdrawal": "withdrawn",
    "expression_of_concern": "expression_of_concern", "expression-of-concern": "expression_of_concern",
    "correction": "corrected", "corrigendum": "corrected", "erratum": "corrected",
    "addendum": "updated", "clarification": "updated", "new_version": "updated", "new_edition": "updated",
    "reinstatement": "reinstated", "partial_retraction": "unknown_relation",
}


def _event_from_relation(
    record_id: str | None, other_id: str | None, raw: str, *, provider: Provider, field: str,
    reference: str | None = None, record: str | None = None,
) -> tuple[PublicationIntegrityEvent | None, str | None]:
    """(event, None) for an integrity relationship, (None, raw) for one that
    is not (a comment), and an `unknown_relation` event for one not known."""
    status, on_subject = _EPMC_TYPES.get(raw.strip().lower(), ("unknown_relation", True))
    if status is None:
        return None, raw
    subject, notice = (record_id, other_id) if on_subject else (other_id, record_id)
    return PublicationIntegrityEvent(
        subject_id=subject, notice_id=notice, status=status, raw_type=raw,
        asserted_on="subject_record" if on_subject else "notice_record", reference=reference,
        assertions=[IntegrityAssertion(provider=provider, field=field, record_id=record)],
    ), None


# --- parsers ------------------------------------------------------------------------------------


def _epmc_record_id(hit: dict) -> str | None:
    src, ext = str(hit.get("source") or ""), str(hit.get("id") or "")
    candidates = [normalize_id("pmid", ext) if src == "MED" else None,
                  normalize_id("pmcid", ext) if src == "PMC" else None,
                  normalize_id("pmid", hit.get("pmid")), normalize_id("pmcid", hit.get("pmcid")),
                  normalize_id("doi", hit.get("doi"))]
    return next((c for c in candidates if c), None)


def _with_aliases(events: list[PublicationIntegrityEvent], own: list[str]) -> list[PublicationIntegrityEvent]:
    return [e.model_copy(update={"subject_aliases": sorted(set(own) - {e.subject_id})})
            if e.asserted_on == "subject_record" else e for e in events]


def _epmc_other_id(item: dict) -> str | None:
    src, ext = str(item.get("source") or ""), str(item.get("id") or "")
    if src == "MED":
        return normalize_id("pmid", ext)
    if src == "PMC":
        return normalize_id("pmcid", ext)
    return None


def parse_europepmc_hit(hit: dict) -> tuple[list[str], list[PublicationIntegrityEvent], list[str]]:
    """(the record's own ids, its integrity events, its other relations)."""
    own = [x for x in (normalize_id("pmid", hit.get("pmid")), normalize_id("pmcid", hit.get("pmcid")),
                       normalize_id("doi", hit.get("doi"))) if x]
    record = _epmc_record_id(hit)
    items = ((hit.get("commentCorrectionList") or {}).get("commentCorrection")) or []
    if isinstance(items, dict):
        items = [items]
    events, others = [], []
    for item in items:
        if not isinstance(item, dict) or not item.get("type"):
            continue
        event, other = _event_from_relation(
            record, _epmc_other_id(item), str(item["type"]), provider="europepmc",
            field="commentCorrectionList", reference=item.get("reference"), record=str(hit.get("id") or ""),
        )
        if event:
            events.append(event)
        elif other:
            others.append(f"{other}: {item.get('source')}/{item.get('id')}")
    return own, _with_aliases(events, own), others


def parse_pubmed_article(article: ET.Element) -> tuple[list[str], list[PublicationIntegrityEvent], list[str]]:
    pmid = normalize_id("pmid", (article.findtext("MedlineCitation/PMID") or "").strip())
    own = [pmid] if pmid else []
    for el in article.findall("PubmedData/ArticleIdList/ArticleId"):
        kind = {"pmc": "pmcid", "doi": "doi"}.get(el.get("IdType", ""))
        if kind:
            value = normalize_id(kind, el.text)
            if value and value not in own:
                own.append(value)
    events, others = [], []
    for cc in article.findall("MedlineCitation/CommentsCorrectionsList/CommentsCorrections"):
        ref_type = cc.get("RefType", "")
        raw = _PUBMED_TYPES.get(ref_type, ref_type)
        other = normalize_id("pmid", (cc.findtext("PMID") or "").strip())
        event, other_rel = _event_from_relation(
            pmid, other, raw, provider="pubmed", field="CommentsCorrectionsList",
            reference=(cc.findtext("RefSource") or "").strip() or None, record=pmid,
        )
        if event:
            events.append(event.model_copy(update={"raw_type": ref_type}))
        elif other_rel:
            others.append(f"{ref_type}: {other or 'no PMID'}")
    return own, _with_aliases(events, own), others


def parse_pubmed_xml(text: str) -> list[tuple[list[str], list[PublicationIntegrityEvent], list[str]]]:
    root = ET.fromstring(text)
    return [parse_pubmed_article(a) for a in root.findall("PubmedArticle")]


def _crossref_date(item: dict) -> str | None:
    parts = ((item.get("updated") or {}).get("date-parts") or [[None]])[0]
    return "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts) if p is not None) or None


def parse_crossref_work(message: dict) -> tuple[list[str], list[PublicationIntegrityEvent], list[str]]:
    """Crossref: a notice carries `update-to` (subject = the DOI it updates);
    an original carries `updated-by` (notice = the DOI that updated it). Each
    row's `source` - publisher or retraction-watch - is kept as an assertion;
    two rows about one retraction are one event with two assertions."""
    own_doi = normalize_id("doi", message.get("DOI"))
    own = [own_doi] if own_doi else []
    events: list[PublicationIntegrityEvent] = []
    for field, on_subject in (("update-to", False), ("updated-by", True)):
        for item in message.get(field) or []:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("type") or "")
            status = _CROSSREF_TYPES.get(raw.lower(), "unknown_relation")
            other = normalize_id("doi", item.get("DOI"))
            subject, notice = (own_doi, other) if on_subject else (other, own_doi)
            events.append(PublicationIntegrityEvent(
                subject_id=subject, notice_id=notice, status=status, raw_type=raw,
                asserted_on="subject_record" if on_subject else "notice_record",
                assertions=[IntegrityAssertion(provider="crossref", field=field, source=item.get("source"),
                                               record_id=str(item.get("record-id")) if item.get("record-id") else None,
                                               date=_crossref_date(item))],
            ))
    return own, merge_events(events), []


def merge_events(events: Iterable[PublicationIntegrityEvent]) -> list[PublicationIntegrityEvent]:
    """One event per (subject, notice, status), every assertion kept. Two
    providers reporting one retraction are one retraction, twice attested."""
    merged: dict[tuple, PublicationIntegrityEvent] = {}
    for e in events:
        k = e.key()
        if k in merged:
            existing = merged[k]
            seen = {a.model_dump_json() for a in existing.assertions}
            extra = [a for a in e.assertions if a.model_dump_json() not in seen]
            merged[k] = existing.model_copy(update={"assertions": [*existing.assertions, *extra]})
        else:
            merged[k] = e
    return sorted(merged.values(), key=lambda e: (e.subject_id or "", e.notice_id or "", e.status))


# --- observations from raw provider responses --------------------------------------------------------


def observe(
    provider: Provider, raw: bytes, *, requested_ids: list[str], retrieved_at: str, http_status: int | None,
    source_uri: str | None, purpose: str = "refresh", import_provenance: dict | None = None,
) -> PublicationIntegrityObservation:
    """Parse one raw response into one observation. A body that cannot be
    parsed is an `error` observation, kept - never a silent success."""
    sha = hashlib.sha256(raw).hexdigest()
    base = dict(provider=provider, request_purpose=purpose, requested_ids=sorted(set(requested_ids)),
                retrieved_at=retrieved_at, http_status=http_status, raw_sha256=sha, source_uri=source_uri,
                import_provenance=import_provenance)
    if http_status is not None and http_status >= 400:
        outcome = "not_found" if http_status == 404 else "error"
        return PublicationIntegrityObservation(**base, outcome=outcome, error=f"HTTP {http_status}")
    try:
        if provider == "europepmc":
            data = json.loads(raw.decode("utf-8"))
            records = [parse_europepmc_hit(h) for h in ((data.get("resultList") or {}).get("result") or [])]
        elif provider == "pubmed":
            records = parse_pubmed_xml(raw.decode("utf-8"))
        else:
            data = json.loads(raw.decode("utf-8"))
            message = data.get("message") if isinstance(data, dict) else None
            records = [parse_crossref_work(message)] if isinstance(message, dict) else []
    except (ValueError, ET.ParseError, UnicodeDecodeError) as exc:
        return PublicationIntegrityObservation(**base, outcome="error", error=f"unparseable response: {exc}")

    try:
        details = _record_details(provider, raw)
    except (ValueError, ET.ParseError, UnicodeDecodeError, AttributeError, TypeError):
        details = {}
    resolved: list[str] = []
    events: list[PublicationIntegrityEvent] = []
    others: list[str] = []
    conflicts: list[str] = []
    wanted = set(requested_ids)
    for own, record_events, record_others in records:
        resolved.extend(x for x in own if x not in resolved)
        if wanted and not (set(own) & wanted):
            # The provider answered with a record we did not ask about: its
            # relationships are not attributed to anything we asked about.
            conflicts.append(f"returned record {', '.join(own) or '(no identifiers)'} matches no requested id")
            continue
        events.extend(record_events)
        others.extend(record_others)
    missing = sorted(wanted - set(resolved))
    outcome: Outcome = "success" if records and not (wanted and set(missing) == wanted) else "not_found"
    if missing and outcome == "success":
        conflicts.append(f"no record returned for {', '.join(missing)}")
    return PublicationIntegrityObservation(
        **base, outcome=outcome, resolved_ids=resolved, events=merge_events(events),
        identity_conflicts=conflicts, other_relations=sorted(set(others)), record_details=details,
    )


EXCERPT_CHARS = 500
_TAGS = re.compile(r"<[^>]+>")


def _plain(text: object) -> str:
    return re.sub(r"\s+", " ", _TAGS.sub(" ", str(text or ""))).strip()


def _record_details(provider: str, raw: bytes) -> dict[str, dict[str, Any]]:
    """Each returned record's title, opening text and types, by its identifiers."""
    out: dict[str, dict[str, Any]] = {}

    def add(ids, title, text, types):
        entry = {"title": _plain(title)[:300], "excerpt": _plain(text)[:EXCERPT_CHARS], "types": list(types)}
        for i in ids:
            out[i] = entry

    if provider == "europepmc":
        for hit in (json.loads(raw.decode("utf-8")).get("resultList") or {}).get("result") or []:
            own, _, _ = parse_europepmc_hit(hit)
            add(own, hit.get("title"), hit.get("abstractText"), (hit.get("pubTypeList") or {}).get("pubType") or [])
    elif provider == "pubmed":
        for article in ET.fromstring(raw.decode("utf-8")).findall("PubmedArticle"):
            own, _, _ = parse_pubmed_article(article)
            title = "".join(article.find("MedlineCitation/Article/ArticleTitle").itertext()) \
                if article.find("MedlineCitation/Article/ArticleTitle") is not None else ""
            text = " ".join("".join(el.itertext()) for el in article.findall(
                "MedlineCitation/Article/Abstract/AbstractText"))
            types = [el.text or "" for el in article.findall(
                "MedlineCitation/Article/PublicationTypeList/PublicationType")]
            add(own, title, text, types)
    else:
        message = json.loads(raw.decode("utf-8")).get("message") or {}
        own, _, _ = parse_crossref_work(message)
        title = (message.get("title") or [""])[0] if isinstance(message.get("title"), list) else message.get("title")
        add(own, title, message.get("abstract"), [message.get("type")] if message.get("type") else [])
    return out


# --- the derived, current view --------------------------------------------------------------------


class IntegrityState(BaseModel):
    """What the local observations say about one paper now. Rebuildable."""

    subject_ids: list[str]
    statuses: list[Status] = Field(default_factory=list)
    events: list[PublicationIntegrityEvent] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    last_successful_check: dict[str, str] = Field(default_factory=dict)
    """Provider -> when it last answered successfully about this paper."""
    freshness: Literal["fresh", "stale", "never_checked"] = "never_checked"
    failures_since_success: int = 0

    @property
    def adverse(self) -> list[str]:
        return [s for s in self.statuses if s in ADVERSE]

    @property
    def retracted(self) -> bool:
        return "retracted" in self.statuses or "withdrawn" in self.statuses


def _about(event: PublicationIntegrityEvent, ids: set[str]) -> bool:
    return event.subject_id in ids or bool(ids & set(event.subject_aliases))


def _parse_time(value: str) -> datetime:
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def derive_state(
    subject_ids: list[str],
    observations: list[PublicationIntegrityObservation],
    *,
    now: datetime,
    ttl: timedelta = timedelta(days=7),
) -> IntegrityState:
    """The paper's status from every observation that mentions it.

    Events come from successful observations only, and accumulate: a later
    success that no longer lists a notice does not erase an earlier one (a
    provider dropping a relationship is not a reinstatement), and a later
    failure changes nothing but the failure count. Freshness is separate:
    `fresh` when some provider answered successfully within `ttl` of `now`.
    The seven-day default is an operating choice for the owner to set, not a
    safety guarantee; an adverse event stays visible however stale.
    """
    ids = set(subject_ids)
    events: list[PublicationIntegrityEvent] = []
    used: list[str] = []
    last: dict[str, str] = {}
    failures = 0
    for obs in sorted(observations, key=lambda o: (o.retrieved_at, o.observation_id)):
        # About this paper: asked about it, answered about it, or - a notice
        # looked up on its own - naming it as the article it concerns. Only the
        # first two can make it fresh; a notice found elsewhere says nothing
        # about what the paper's own record currently shows.
        names_it = any(_about(e, ids) for e in obs.events)
        if not (ids & set(obs.requested_ids) or ids & set(obs.resolved_ids) or names_it):
            continue
        used.append(obs.observation_id)
        if obs.outcome != "success":
            if ids & set(obs.requested_ids):
                failures += 1
            continue
        failures = 0
        if ids & set(obs.resolved_ids):
            last[obs.provider] = max(last.get(obs.provider, ""), obs.retrieved_at)
        events.extend(e for e in obs.events if _about(e, ids))
    merged = merge_events(events)
    statuses = sorted({e.status for e in merged})
    newest = max((_parse_time(v) for v in last.values()), default=None)
    freshness = "never_checked" if newest is None else ("fresh" if now - newest <= ttl else "stale")
    return IntegrityState(
        subject_ids=sorted(ids), statuses=statuses, events=merged, observation_ids=used,
        last_successful_check=dict(sorted(last.items())), freshness=freshness, failures_since_success=failures,
    )


def notice_label(state: IntegrityState) -> str | None:
    """A short caveat for a reference, from adverse or correcting events.
    Identifiers only - no dates - so the same known state gives the same text."""
    parts = []
    names = {"retracted": "retracted", "withdrawn": "withdrawn", "expression_of_concern": "expression of concern",
             "corrected": "correction published", "updated": "update published", "republished": "republished",
             "reinstated": "reinstatement recorded", "unknown_relation": "unclassified notice"}
    for status in ("retracted", "withdrawn", "expression_of_concern", "corrected", "updated", "republished",
                   "reinstated", "unknown_relation"):
        notices = sorted({e.notice_id or "unidentified notice" for e in state.events if e.status == status})
        if notices:
            parts.append(f"{names[status]} ({', '.join(notices)})")
    return "; ".join(parts) or None


# --- offline bundles --------------------------------------------------------------------------------


def bundle_line(provider: Provider, raw: bytes, *, requested_ids: list[str], captured_at: str,
                source_uri: str | None, http_status: int | None) -> dict[str, Any]:
    """One entry of an offline bundle: the raw response and what binds it."""
    return {
        "bundle_version": 1, "parser_version": PARSER_VERSION, "provider": provider,
        "requested_ids": sorted(set(requested_ids)), "captured_at": captured_at, "source_uri": source_uri,
        "http_status": http_status, "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_b64": base64.b64encode(raw).decode("ascii"),
    }


class BundleError(ValueError):
    """An offline bundle entry that must not be imported."""


def observation_from_bundle(entry: dict[str, Any], *, bundle_sha256: str, line: int) -> tuple[
        PublicationIntegrityObservation, bytes]:
    """Check one bundle entry and turn it into an observation.

    Refused: an unknown provider or parser version, identifiers that are not
    well-formed, bytes that do not hash to the recorded digest, and a response
    whose records match none of the identifiers it claims to answer. What an
    accepted entry proves is only that these bytes were captured as stated by
    whoever made the bundle: nothing here authenticates the outside machine,
    and the observation says so in its provenance.
    """
    if entry.get("bundle_version") != 1 or entry.get("parser_version") != PARSER_VERSION:
        raise BundleError(f"line {line}: unsupported bundle or parser version")
    provider = entry.get("provider")
    if provider not in ("europepmc", "pubmed", "crossref"):
        raise BundleError(f"line {line}: unknown provider {provider!r}")
    ids = entry.get("requested_ids") or []
    if not ids or not all(is_normalized_id(i) for i in ids):
        raise BundleError(f"line {line}: requested identifiers are missing or malformed")
    try:
        raw = base64.b64decode(entry.get("raw_b64") or "", validate=True)
    except ValueError as exc:
        raise BundleError(f"line {line}: raw response is not valid base64") from exc
    if hashlib.sha256(raw).hexdigest() != entry.get("raw_sha256"):
        raise BundleError(f"line {line}: raw response does not match its sha256 (tampered or truncated)")
    captured = entry.get("captured_at")
    try:
        _parse_time(str(captured))
    except ValueError as exc:
        raise BundleError(f"line {line}: captured_at is not a timestamp") from exc
    obs = observe(
        provider, raw, requested_ids=ids, retrieved_at=str(captured), http_status=entry.get("http_status"),
        source_uri=entry.get("source_uri"), purpose="offline_import",
        import_provenance={"bundle_sha256": bundle_sha256, "line": line, "authenticated": False},
    )
    if obs.resolved_ids and not (set(ids) & set(obs.resolved_ids)):
        raise BundleError(f"line {line}: the response answers none of the identifiers it claims")
    return obs, raw


# --- what a run knew -----------------------------------------------------------------------------------


def build_snapshot(db, papers, *, now: datetime, ttl: timedelta) -> dict[str, Any]:
    """The run's `publication_integrity_snapshot.v1`: for each paper, what the
    local observations said when the run started, and which observations
    those were. Local reads only - a run never refreshes. A paper nobody has
    looked up is counted, not listed, and is `never_checked`, not clean."""
    rows: dict[str, Any] = {}
    unobserved = 0
    for paper in papers:
        ids = paper_ids(paper)
        observations = db.integrity_observations(ids) if ids else []
        if not observations:
            unobserved += 1
            continue
        state = derive_state(ids, observations, now=now, ttl=ttl)
        notices = [_notice_detail(db, e) for e in state.events]
        decision = db.integrity_decision(ids) if "reinstated" in state.statuses else None
        rows[paper.id] = {
            "notices": notices,
            "decision": decision,
            "subject_ids": state.subject_ids,
            "statuses": state.statuses,
            "events": [e.model_dump() for e in state.events],
            "observation_ids": state.observation_ids,
            "freshness": state.freshness,
            "last_successful_check": state.last_successful_check,
            "notice_label": notice_label(state),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "taken_at": now.isoformat(timespec="seconds"),
        "source": "local observations only; nothing was fetched for this run",
        "stale_after_days": ttl.days,
        "papers": rows,
        "never_checked": unobserved,
    }


def notice_link(notice_id: str | None) -> str | None:
    if not notice_id:
        return None
    kind, _, value = notice_id.partition(":")
    return {"PMID": f"https://pubmed.ncbi.nlm.nih.gov/{value}/", "DOI": f"https://doi.org/{value}",
            "PMCID": f"https://europepmc.org/article/PMC/{value}"}.get(kind)


def _notice_detail(db, event: PublicationIntegrityEvent) -> dict[str, Any]:
    """The notice behind an event, as recorded: its own title and opening
    words, where a lookup fetched them, and a link to read the rest."""
    title = excerpt = ""
    if event.notice_id:
        for obs in db.integrity_observations([event.notice_id]):
            detail = obs.record_details.get(event.notice_id) or {}
            title = title or detail.get("title", "")
            excerpt = excerpt or detail.get("excerpt", "")
    return {"status": event.status, "notice_id": event.notice_id, "title": title, "excerpt": excerpt,
            "link": notice_link(event.notice_id),
            "asserted_by": sorted({f"{a.provider}{'/' + a.source if a.source else ''}" for a in event.assertions})}


def concern_papers(snapshot: dict[str, Any]) -> set[str]:
    """Papers with a recorded expression of concern: kept, shown, ranked
    after all other evidence and never used in a report's Answer."""
    return {pid for pid, row in (snapshot.get("papers") or {}).items()
            if "expression_of_concern" in (row.get("statuses") or [])}


_STATUS_WORDS = {"retracted": "Retracted", "withdrawn": "Withdrawn", "expression_of_concern": "Expression of concern",
                 "corrected": "Correction published", "updated": "Update published", "republished": "Republished",
                 "reinstated": "Reinstated after retraction", "unknown_relation": "Unclassified notice"}


def notice_texts(snapshot: dict[str, Any]) -> dict[str, str]:
    """paper id -> what a reader is told next to the paper: each notice, its
    own title and opening words, and where to read it. The notice's words are
    shown as they were recorded; nothing here interprets them."""
    out: dict[str, str] = {}
    for pid, row in (snapshot.get("papers") or {}).items():
        parts = []
        for n in row.get("notices") or []:
            words = _STATUS_WORDS.get(n["status"], n["status"])
            ident = n.get("notice_id") or "notice not identified"
            text = f"{words} ({ident})"
            if n.get("title"):
                text += f": “{n['title']}”"
            if n.get("excerpt"):
                text += f" — the notice says: “{n['excerpt'][:300].rstrip()}…”"
            if n.get("link"):
                text += f" <{n['link']}>"
            parts.append(text)
        if parts:
            out[pid] = "; ".join(parts)
    return out


def _added_back(row: dict[str, Any]) -> bool:
    return "reinstated" in (row.get("statuses") or []) and (row.get("decision") or {}).get("decision") == "include"


def checked_in(snapshot: dict[str, Any]) -> set[str] | None:
    """Papers with a successful lookup on record when the run started, or None
    for a snapshot that did not record it."""
    if snapshot.get("status") == "not_recorded" or "papers" not in snapshot:
        return None
    return {pid for pid, row in snapshot["papers"].items() if row.get("freshness") in ("fresh", "stale")}


def retracted_in(snapshot: dict[str, Any]) -> dict[str, str]:
    """paper id -> why, for papers the snapshot shows retracted or withdrawn -
    except a reinstated one that a person has added back. A reinstatement on
    its own never lets a paper back in."""
    return {pid: row["notice_label"] or "retraction recorded" for pid, row in (snapshot.get("papers") or {}).items()
            if {"retracted", "withdrawn"} & set(row.get("statuses") or []) and not _added_back(row)}


def added_back_in(snapshot: dict[str, Any]) -> dict[str, dict]:
    """paper id -> the decision, for reinstated papers a person added back."""
    return {pid: row["decision"] for pid, row in (snapshot.get("papers") or {}).items() if _added_back(row)}


def reinstated_waiting_in(snapshot: dict[str, Any]) -> dict[str, str]:
    """paper id -> notices, for reinstated papers still left out, waiting for
    a person to decide whether to add them back."""
    return {pid: row["notice_label"] or "reinstatement recorded" for pid, row in (snapshot.get("papers") or {}).items()
            if "reinstated" in (row.get("statuses") or []) and not _added_back(row)}


def notices_in(snapshot: dict[str, Any]) -> dict[str, str]:
    """paper id -> caveat text, for every paper with any recorded notice."""
    return {pid: row["notice_label"] for pid, row in (snapshot.get("papers") or {}).items() if row.get("notice_label")}


def advisories_for_run(db, papers, snapshot: dict[str, Any] | None, *, now: datetime, ttl: timedelta) -> list[str]:
    """What is known *now* about the papers a stored report cites, set beside
    what the run knew. Local reads only; never a refresh. Nothing here changes
    the stored report: a notice found later is a reason for a person to look
    again at the findings that cite that paper, not a rewrite of history."""
    out: list[str] = []
    if snapshot is None or snapshot.get("status") == "not_recorded":
        out.append("publication status when this run was made: not recorded (the run predates integrity snapshots)")
        snapshot = None
    then = (snapshot or {}).get("papers") or {}
    never = stale = 0
    for paper in papers:
        ids = paper_ids(paper)
        state = derive_state(ids, db.integrity_observations(ids), now=now, ttl=ttl) if ids else None
        if state is None or state.freshness == "never_checked":
            never += 1
            if state is None or not state.statuses:
                continue
        elif state.freshness == "stale":
            stale += 1
        known = set((then.get(paper.id) or {}).get("statuses") or [])
        new = [s for s in state.statuses if s not in known]
        label = notice_label(state) or ""
        if new and snapshot is not None:
            out.append(f"{paper.id}: now recorded as {', '.join(new)} ({label}); not known when this run was made - "
                       "a person should reconsider the findings that cite it")
        elif new:
            out.append(f"{paper.id}: recorded now as {', '.join(state.statuses)} ({label}); status at run time unknown")
        elif state.statuses:
            out.append(f"{paper.id}: {', '.join(state.statuses)} ({label}), already known when the run was made")
    if papers:
        out.append(f"of {len(papers)} cited paper(s): {never} never checked for notices, {stale} checked more than "
                   f"{ttl.days} day(s) ago; `bedrock-litreview integrity plan --run <id>` shows what a refresh would send")
    return out


# --- a person's decision about a reinstated paper ------------------------------------------------------


def parse_identifier(text: str) -> str | None:
    """What a person types, as a normalised identifier: `PMID:123`, `PMID123`
    (a paper id as reports show it), `PMC123`, `DOI:10.x/y` or a bare `10.x/y`."""
    t = text.strip()
    if ":" in t and not t.lower().startswith("10."):
        kind, value = t.split(":", 1)
        return normalize_id(kind.strip().lower(), value.strip())
    if t.upper().startswith("PMID"):
        return normalize_id("pmid", t[4:])
    if t.upper().startswith("PMC"):
        return normalize_id("pmcid", t)
    if t.startswith("10."):
        return normalize_id("doi", t)
    return normalize_id("pmid", t)


def decide_reinstated(db, text: str, *, include: bool, by: str, note: str | None, now: datetime) -> str:
    """Record a scientist adding a reinstated paper back (or taking it out
    again). Refused unless this machine has recorded that the paper was
    reinstated: a retracted paper is never added back on anyone's say-so
    alone. Returns a sentence saying what was recorded."""
    subject = parse_identifier(text)
    if subject is None:
        raise ValueError(f"{text!r} is not a PMID, PMCID or DOI")
    state = derive_state([subject], db.integrity_observations([subject]), now=now)
    if "reinstated" not in state.statuses:
        shown = ", ".join(state.statuses) or "no notice"
        raise ValueError(f"{subject} has no recorded reinstatement ({shown}); only a paper that was retracted and "
                         "later reinstated can be added back")
    if not by.strip():
        raise ValueError("say who is deciding: a name is recorded with the decision")
    db.add_integrity_decision(subject, "include" if include else "exclude", by=by.strip(),
                              at=now.isoformat(timespec="seconds"), note=note)
    if include:
        return (f"{subject} will be included in new reviews again, recorded as your decision ({by.strip()}). "
                "Reports already written are not changed.")
    return f"{subject} will be left out of new reviews again, recorded as your decision ({by.strip()})."
