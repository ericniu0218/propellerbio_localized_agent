"""Retractions, concerns and corrections: parsed with direction, stored as observations.

The real responses in `integrity_fixtures/` are public records about one
retracted Lancet paper (PMID 32450107), its expression of concern (32504543)
and its retraction notices, plus two Crossref works - so the parsers are held
to the providers' actual contracts. The synthetic cases cover what those do
not: collisions, cycles, failures after success, reinstatement.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from bedrock.litreview.db import Database
from bedrock.litreview.ingest import prepare
from bedrock.litreview.publication_integrity import (
    BundleError,
    PublicationIntegrityEvent,
    bundle_line,
    derive_state,
    is_normalized_id,
    merge_events,
    normalize_id,
    notice_label,
    observation_from_bundle,
    observe,
)
from bedrock.litreview.sources.europepmc import paper_from_hit
from bedrock.litreview.sources.pubmed import parse_articles

FIX = Path(__file__).parent / "integrity_fixtures"
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)
ORIGINAL, CONCERN, RETRACTION_1, RETRACTION_2, ERRATUM = (
    "PMID:32450107", "PMID:32504543", "PMID:32511943", "PMID:32653079", "PMID:32485145")


def raw(name: str) -> bytes:
    return (FIX / name).read_bytes()


def test_the_fixtures_are_the_bytes_the_audit_captured():
    readme = (FIX / "README.md").read_text()
    for path in FIX.iterdir():
        if path.suffix in (".json", ".xml"):
            assert hashlib.sha256(path.read_bytes()).hexdigest() in readme, path.name


# --- parsing, with direction ---------------------------------------------------------------------


def test_europe_pmc_original_record_carries_its_own_retraction_concern_and_erratum():
    obs = observe("europepmc", raw("epmc-original.json"), requested_ids=[ORIGINAL], retrieved_at="2026-09-27T00:10:23Z",
                  http_status=200, source_uri="fixture")
    assert obs.outcome == "success" and ORIGINAL in obs.resolved_ids and not obs.identity_conflicts
    by = {(e.status, e.notice_id): e for e in obs.events}
    for notice in (RETRACTION_1, RETRACTION_2):
        e = by[("retracted", notice)]
        assert e.subject_id == ORIGINAL and e.asserted_on == "subject_record" and e.raw_type == "Retraction in"
    assert by[("expression_of_concern", CONCERN)].subject_id == ORIGINAL
    assert by[("corrected", ERRATUM)].subject_id == ORIGINAL
    assert any(r.startswith("Comment in") for r in obs.other_relations), "comments are seen and not counted"
    assert not any(e.raw_type.lower().startswith("comment") for e in obs.events)


def test_the_concern_notice_points_back_at_the_original():
    obs = observe("europepmc", raw("epmc-eoc-notice.json"), requested_ids=[CONCERN], retrieved_at="2026-09-27T00:10Z",
                  http_status=200, source_uri="fixture")
    [event] = obs.events
    assert (event.subject_id, event.notice_id, event.status) == (ORIGINAL, CONCERN, "expression_of_concern")
    assert event.asserted_on == "notice_record" and event.raw_type == "Expression of concern for"


def test_an_erratum_is_a_correction_of_another_record_not_a_retraction():
    obs = observe("europepmc", raw("epmc-correction-sample.json"), requested_ids=[], retrieved_at="2026-09-27T00:10Z",
                  http_status=200, source_uri="fixture")
    assert {e.status for e in obs.events} == {"corrected"}
    assert all(e.asserted_on == "notice_record" for e in obs.events)


def test_pubmed_reftypes_match_europe_pmc():
    obs = observe("pubmed", raw("pubmed-pair.xml"), requested_ids=[ORIGINAL, CONCERN], retrieved_at="2026-09-27T00:10Z",
                  http_status=200, source_uri="fixture")
    keys = {(e.subject_id, e.notice_id, e.status, e.raw_type) for e in obs.events}
    assert (ORIGINAL, RETRACTION_1, "retracted", "RetractionIn") in keys
    assert (ORIGINAL, CONCERN, "expression_of_concern", "ExpressionOfConcernIn") in keys
    assert (ORIGINAL, CONCERN, "expression_of_concern", "ExpressionOfConcernFor") not in keys, \
        "the reciprocal link on the notice is the same event, merged"
    reciprocal = next(e for e in obs.events if e.status == "expression_of_concern")
    assert len(reciprocal.assertions) == 1 or {a.record_id for a in reciprocal.assertions} >= {ORIGINAL}
    assert {"DOI:10.1016/s0140-6736(20)31180-6", "PMCID:PMC7255293"} <= set(obs.resolved_ids)


def test_crossref_original_is_updated_by_a_retraction_watch_assertion():
    obs = observe("crossref", raw("crossref-original.json"), requested_ids=["DOI:10.1021/am300292v"],
                  retrieved_at="2026-09-27T00:10Z", http_status=200, source_uri="fixture")
    [event] = obs.events
    assert (event.subject_id, event.notice_id, event.status) == (
        "DOI:10.1021/am300292v", "DOI:10.1021/acsami.9b11759", "retracted")
    assert [(a.source, a.record_id) for a in event.assertions] == [("retraction-watch", "20946")]


def test_two_crossref_rows_about_one_retraction_are_one_event_twice_attested():
    obs = observe("crossref", raw("crossref-notice.json"), requested_ids=["DOI:10.1177/17588359231172420"],
                  retrieved_at="2026-09-27T00:10Z", http_status=200, source_uri="fixture")
    [event] = obs.events
    assert event.subject_id == "DOI:10.1177/1758835920922055" and event.asserted_on == "notice_record"
    assert sorted(a.source for a in event.assertions) == ["publisher", "retraction-watch"]


def test_the_same_event_from_two_providers_is_merged_and_conflicting_ones_are_kept():
    a = PublicationIntegrityEvent(subject_id="PMID:1", notice_id="PMID:2", status="retracted", raw_type="Retraction in",
                                  asserted_on="subject_record")
    b = a.model_copy(update={"raw_type": "RetractionIn"})
    c = a.model_copy(update={"status": "corrected", "raw_type": "correction"})
    merged = merge_events([a, b, c])
    assert [e.status for e in merged] == ["corrected", "retracted"], "a disagreement is two events, not a vote"


# --- identity ------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind, value, expected", [
    ("pmid", "32450107", "PMID:32450107"), ("pmid", "32450107 OR 1", None), ("pmid", "", None),
    ("pmcid", "pmc7255293", "PMCID:PMC7255293"), ("doi", "https://doi.org/10.1016/S0140-6736(20)31180-6",
                                                  "DOI:10.1016/s0140-6736(20)31180-6"),
    ("doi", "10.1000/a b", None), ("doi", 'x"&query=evil', None), ("doi", "not-a-doi", None),
])
def test_only_well_formed_identifiers_are_accepted(kind, value, expected):
    assert normalize_id(kind, value) == expected
    if expected:
        assert is_normalized_id(expected)


def test_a_response_about_another_record_is_not_attributed():
    obs = observe("europepmc", raw("epmc-original.json"), requested_ids=["PMID:99999"],
                  retrieved_at="2026-09-27T00:10Z", http_status=200, source_uri="fixture")
    assert obs.events == [] and obs.identity_conflicts and obs.outcome == "not_found"


def test_errors_are_observations_not_silence():
    assert observe("pubmed", b"<not xml", requested_ids=[ORIGINAL], retrieved_at="t", http_status=200,
                   source_uri=None).outcome == "error"
    assert observe("europepmc", b"", requested_ids=[ORIGINAL], retrieved_at="t", http_status=503,
                   source_uri=None).outcome == "error"
    assert observe("europepmc", b"{}", requested_ids=[ORIGINAL], retrieved_at="t", http_status=404,
                   source_uri=None).outcome == "not_found"


# --- the derived state -----------------------------------------------------------------------------


def _ok(retrieved_at: str, events=(), ids=(ORIGINAL,), provider="europepmc"):
    body = json.dumps({"resultList": {"result": [{"id": i.split(":")[1], "source": "MED", "pmid": i.split(":")[1]}
                                                 for i in ids]}}).encode()
    obs = observe(provider, body, requested_ids=list(ids), retrieved_at=retrieved_at, http_status=200,
                  source_uri=None)
    return obs.model_copy(update={"events": list(events)})


def _event(status, notice="PMID:2", subject=ORIGINAL):
    return PublicationIntegrityEvent(subject_id=subject, notice_id=notice, status=status, raw_type=status,
                                     asserted_on="subject_record")


def _failed(retrieved_at: str):
    return observe("europepmc", b"", requested_ids=[ORIGINAL], retrieved_at=retrieved_at, http_status=503,
                   source_uri=None)


def test_a_failure_after_an_adverse_event_erases_nothing_and_is_not_a_check():
    ok = _ok("2026-09-20T12:00:00+00:00", [_event("retracted")])
    state = derive_state([ORIGINAL], [ok, _failed("2026-09-27T11:00:00+00:00")], now=NOW)
    assert state.retracted and state.failures_since_success == 1
    assert state.last_successful_check == {"europepmc": "2026-09-20T12:00:00+00:00"}


def test_an_error_then_a_success_counts_the_success():
    state = derive_state([ORIGINAL], [_failed("2026-09-26T00:00:00+00:00"), _ok("2026-09-27T00:00:00+00:00")], now=NOW)
    assert state.freshness == "fresh" and state.failures_since_success == 0 and state.statuses == []


def test_a_later_success_without_the_notice_is_not_a_reinstatement():
    first = _ok("2026-09-01T00:00:00+00:00", [_event("expression_of_concern")])
    later = _ok("2026-09-27T00:00:00+00:00")
    assert derive_state([ORIGINAL], [first, later], now=NOW).statuses == ["expression_of_concern"]


def test_concern_then_retraction_then_reinstatement_are_all_kept():
    events = [_event("expression_of_concern", "PMID:2"), _event("retracted", "PMID:3"), _event("reinstated", "PMID:4")]
    state = derive_state([ORIGINAL], [_ok("2026-09-27T00:00:00+00:00", events)], now=NOW)
    assert state.statuses == ["expression_of_concern", "reinstated", "retracted"]
    assert state.retracted, "reinstatement is recorded, never applied automatically"


def test_a_correction_is_flagged_not_excluded():
    state = derive_state([ORIGINAL], [_ok("2026-09-27T00:00:00+00:00", [_event("corrected")])], now=NOW)
    assert not state.retracted and state.adverse == [] and "correction published" in notice_label(state)


def test_a_notice_found_on_its_own_counts_but_does_not_make_the_original_fresh():
    notice_lookup = observe("europepmc", raw("epmc-eoc-notice.json"), requested_ids=[CONCERN],
                            retrieved_at="2026-09-27T00:00:00+00:00", http_status=200, source_uri="fixture")
    state = derive_state([ORIGINAL], [notice_lookup], now=NOW)
    assert state.statuses == ["expression_of_concern"] and state.freshness == "never_checked"


def test_cycles_do_not_loop():
    a = _event("retracted", notice="PMID:2", subject="PMID:1")
    b = _event("retracted", notice="PMID:1", subject="PMID:2")
    state = derive_state(["PMID:1"], [_ok("2026-09-27T00:00:00+00:00", [a, b], ids=("PMID:1",))], now=NOW)
    assert [e.notice_id for e in state.events] == ["PMID:2"]


@pytest.mark.parametrize("age, expected", [
    (timedelta(days=7), "fresh"), (timedelta(days=7, seconds=1), "stale"),
])
def test_freshness_boundary(age, expected):
    checked = (NOW - age).isoformat()
    assert derive_state([ORIGINAL], [_ok(checked)], now=NOW).freshness == expected


def test_never_checked_is_not_clean():
    state = derive_state([ORIGINAL], [], now=NOW)
    assert state.freshness == "never_checked" and state.statuses == []


# --- storage ------------------------------------------------------------------------------------------


def test_observations_are_appended_never_replaced(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = _ok("2026-09-20T00:00:00+00:00", [_event("retracted")])
    second = _failed("2026-09-27T00:00:00+00:00")
    db.add_integrity_observation(first, b"{}")
    db.add_integrity_observation(first, b"{}")
    db.add_integrity_observation(second)
    stored = db.integrity_observations([ORIGINAL])
    assert [o.outcome for o in stored] == ["success", "error"]
    assert db.integrity_observation(first.observation_id) == first
    assert db.conn.execute("SELECT COUNT(*) FROM integrity_observations").fetchone()[0] == 2


# --- search-time flag ---------------------------------------------------------------------------------


def test_a_retracted_original_found_by_search_is_flagged_from_its_own_record():
    hit = json.loads(raw("epmc-original.json"))["resultList"]["result"][0]
    hit = {**hit, "pubTypeList": {"pubType": ["Journal Article"]}, "title": "Synthetic primary study"}
    assert prepare(paper_from_hit(hit, "q")).signals.is_retracted, "relation-only retraction is caught"


def test_the_notices_themselves_are_not_flagged_by_their_relations():
    notice = json.loads(raw("epmc-eoc-notice.json"))["resultList"]["result"][0]
    assert not paper_from_hit(notice, "q").signals.is_retracted
    papers = {p.pmid: p for p in parse_articles(raw("pubmed-pair.xml").decode(), "q")}
    assert papers["32450107"].signals.is_retracted
    assert not papers["32504543"].signals.is_retracted


# --- offline bundles ----------------------------------------------------------------------------------


def _bundle(body: bytes = None, **changes):
    body = body if body is not None else raw("epmc-original.json")
    line = bundle_line("europepmc", body, requested_ids=[ORIGINAL], captured_at="2026-09-27T00:10:23+00:00",
                       source_uri="https://www.ebi.ac.uk/europepmc/webservices/rest/search", http_status=200)
    line.update(changes)
    return line


def test_a_bundle_entry_imports_with_its_provenance():
    obs, body = observation_from_bundle(_bundle(), bundle_sha256="b" * 64, line=1)
    assert obs.request_purpose == "offline_import" and obs.import_provenance["authenticated"] is False
    assert obs.raw_sha256 == hashlib.sha256(body).hexdigest() and any(e.status == "retracted" for e in obs.events)


@pytest.mark.parametrize("changes, message", [
    ({"raw_sha256": "0" * 64}, "does not match its sha256"),
    ({"requested_ids": ["PMID:32450107 OR 1"]}, "malformed"),
    ({"requested_ids": []}, "malformed"),
    ({"parser_version": "publication_integrity.v0"}, "unsupported"),
    ({"provider": "somewhere"}, "unknown provider"),
    ({"captured_at": "yesterday"}, "not a timestamp"),
    ({"requested_ids": ["PMID:1"]}, "answers none"),
])
def test_a_bad_bundle_entry_is_refused(changes, message):
    with pytest.raises(BundleError, match=message):
        observation_from_bundle(_bundle(**changes), bundle_sha256="b" * 64, line=1)
