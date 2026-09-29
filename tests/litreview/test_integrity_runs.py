"""Publication notices in runs: excluded, flagged, or advised - and history left alone.

A retraction recorded on this machine keeps the paper out of a new run; a
concern or correction keeps it in and marks it wherever it is cited; and a
notice recorded *after* a run leaves that run's report byte-for-byte as it
was, and appears only as an advisory when the run is verified. Refreshing is
an explicit command that shows what it would send and sends nothing without
`--yes`. No test here reaches the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import httpx
import pytest

from bedrock.litreview.cli import _integrity, main
from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.publication_integrity import SNAPSHOT_ARTIFACT, bundle_line, observe
from bedrock.litreview.report.invariants import assess_run, verify_run
from bedrock.litreview.runner import answer_question
from bedrock.litreview.sources.http import SourceHTTP
from bedrock.litreview.sources.integrity import IntegrityRefresher
from tests.litreview.fixtures import QUESTION, corpus, fake_client

P1 = "DOI:10.1000/p1"


def epmc_body(doi: str, relation: str, notice_pmid: str = "999") -> bytes:
    return json.dumps({"resultList": {"result": [{
        "id": "123", "source": "MED", "pmid": "123", "doi": doi,
        "commentCorrectionList": {"commentCorrection": [{"source": "MED", "id": notice_pmid, "type": relation}]},
    }]}}).encode()


def record(db, relation: str, when: str = "2026-09-26T00:00:00+00:00"):
    body = epmc_body("10.1000/p1", relation)
    obs = observe("europepmc", body, requested_ids=[P1], retrieved_at=when, http_status=200, source_uri="test")
    db.add_integrity_observation(obs, body)
    return obs


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    monkeypatch.setenv("BEDROCK_LITREVIEW_DATA_DIR", str(tmp_path))
    return s


def run(settings, db):
    return answer_question(QUESTION, settings=settings, client=fake_client(), db=db, papers=corpus())


def test_a_run_pins_what_it_knew_even_when_it_knew_nothing(settings):
    db = Database(settings.db_path)
    r = run(settings, db)
    snap = json.loads(db.get_artifact(r.run.id, SNAPSHOT_ARTIFACT))
    assert snap["papers"] == {} and snap["never_checked"] == len(r.papers)
    assert snap["source"].startswith("local observations only")


def test_a_recorded_retraction_keeps_the_paper_out_of_a_new_run(settings):
    db = Database(settings.db_path)
    record(db, "Retraction in")
    r = run(settings, db)
    assert r.gates["P1"].status == "retracted" and "PMID:999" in r.gates["P1"].reason
    assert "P1" not in r.plan.paper_ids and all(u.paper_id != "P1" for u in r.report.evidence)
    assert r.report.integrity == []


def record_notice_text(db, notice="PMID:999", title="Expression of concern: IL-23 blockade in mice",
                       excerpt="The editors have concerns about image duplication in figure 2."):
    from bedrock.litreview.publication_integrity import PublicationIntegrityObservation

    db.add_integrity_observation(PublicationIntegrityObservation(
        provider="europepmc", requested_ids=[notice], resolved_ids=[notice], retrieved_at="2026-09-26T00:00:00+00:00",
        outcome="success", record_details={notice: {"title": title, "excerpt": excerpt, "types": []}}))


def test_a_recorded_concern_is_flagged_ranked_last_and_kept_out_of_the_answer(settings):
    """The owner's rule: do not recommend it, do not hide it, say why, let
    the scientist decide."""
    db = Database(settings.db_path)
    record(db, "Expression of concern in")
    record_notice_text(db)
    r = run(settings, db)
    assert r.gates["P1"].status == "pass" and any(u.paper_id == "P1" for u in r.report.evidence), "never dropped"
    ref = next(x for x in r.report.references if x.paper_id == "P1")
    assert "Expression of concern (PMID:999): “Expression of concern: IL-23 blockade in mice”" in ref.citation
    assert "the notice says: “The editors have concerns about image duplication" in ref.citation
    assert "<https://pubmed.ncbi.nlm.nih.gov/999/>" in ref.citation
    assert "not used in the Answer; the scientist makes the final call" in ref.citation

    answer = next(sec for sec in r.report.sections if sec.id == "answer")
    by_short = {v: k for k, v in r.report.short_ids.items()}
    by_id = {u.id: u for u in r.report.evidence}
    assert not any(by_id[by_short[c]].paper_id == "P1" for x in answer.sentences for c in x.citations)
    assert (answer.note or "").startswith("Not used in this answer: IL17A, IL23A")
    targets = next(sec for sec in r.report.sections if sec.id == "targets")
    assert targets.subsections, "the flagged evidence is still shown under Candidate targets"
    table = next(sec for sec in r.report.sections if sec.id == "evidence_table").body_markdown
    assert "⚠ concern" in table
    gaps = next(sec for sec in r.report.sections if sec.id == "gaps").body_markdown
    assert "not used in the Answer" in gaps and "recorded publication notice" in gaps
    assert r.report.integrity == []


def test_a_correction_is_flagged_but_not_ranked_down(settings):
    db = Database(settings.db_path)
    record(db, "Erratum in")
    r = run(settings, db)
    ref = next(x for x in r.report.references if x.paper_id == "P1")
    assert "Correction published (PMID:999)" in ref.citation and "not used in the Answer" not in ref.citation
    table = next(sec for sec in r.report.sections if sec.id == "evidence_table").body_markdown
    assert "⚠ concern" not in table


def test_a_notice_recorded_after_a_run_leaves_its_report_alone_and_is_advised(settings):
    db = Database(settings.db_path)
    r = run(settings, db)
    before = db.get_artifact(r.run.id, "report")
    manifest = db.get_artifact(r.run.id, "papers_manifest")

    record(db, "Expression of concern in", when="2026-09-27T00:00:00+00:00")
    assert db.get_artifact(r.run.id, "report") == before
    assert db.get_artifact(r.run.id, "papers_manifest") == manifest
    assert verify_run(db, r.run.id) == []
    a = assess_run(db, r.run.id, now=datetime(2026, 9, 27, 12, tzinfo=timezone.utc))
    assert a.violations == [] and a.exit_code() == 0
    [advice] = [x for x in a.advisories if x.startswith("P1:")]
    assert "not known when this run was made" in advice and "expression_of_concern" in advice


def test_a_run_from_before_snapshots_says_its_run_time_status_is_unknown(settings):
    db = Database(settings.db_path)
    r = run(settings, db)
    with db.conn:
        db.conn.execute("DELETE FROM artifacts WHERE run_id = ? AND name = ?", (r.run.id, SNAPSHOT_ARTIFACT))
    a = assess_run(db, r.run.id)
    assert a.advisories[0].startswith("publication status when this run was made: not recorded")


def test_nothing_in_a_run_or_a_verification_reaches_the_network(settings, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no network")

    monkeypatch.setattr(httpx.Client, "send", refuse)
    db = Database(settings.db_path)
    r = run(settings, db)
    assess_run(db, r.run.id)


# --- the command ---------------------------------------------------------------------------------


def _args(*argv):
    """Parse with the real command line, then hand the namespace to `_integrity`
    directly, so a test can give it a refresher on a mock transport."""
    from bedrock.litreview import cli

    captured = {}

    def fake(args, **kwargs):
        captured["args"] = args
        return 0

    real = cli._integrity
    cli._integrity = fake
    try:
        cli.main(list(argv))
    finally:
        cli._integrity = real
    return captured["args"]


def _refresher(handler):
    http = SourceHTTP(transport=httpx.MockTransport(handler), sleep=lambda s: None, retries=1)
    return IntegrityRefresher(http=http, clock=lambda: "2026-09-27T00:00:00+00:00")


def test_plan_lists_what_would_be_sent_and_sends_nothing(settings, capsys, monkeypatch):
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sent")))
    code = main(["integrity", "plan", "--ids", "PMID:32450107", "DOI:10.1000/P1", "--db", str(settings.db_path)])
    out = capsys.readouterr().out
    assert code == 0 and "PMID:32450107" in out and "no question" in out
    assert "crossref (api.crossref.org): DOI:10.1000/p1" in out


def test_refresh_without_yes_sends_nothing(settings, capsys):
    calls = []
    args = _args("integrity", "refresh", "--ids", "PMID:32450107", "--db", str(settings.db_path))
    code = _integrity(args, refresher=_refresher(lambda r: calls.append(r) or httpx.Response(200, json={})))
    assert code == 2 and calls == []
    assert Database(settings.db_path).conn.execute("SELECT COUNT(*) FROM integrity_observations").fetchone()[0] == 0


def test_refresh_with_yes_records_observations_and_writes_a_bundle(settings, tmp_path, capsys):
    from tests.litreview.test_publication_integrity import raw

    sent = []

    def handler(request):
        sent.append(request)
        if request.url.host == "www.ebi.ac.uk":
            return httpx.Response(200, content=raw("epmc-original.json"))
        return httpx.Response(200, content=raw("pubmed-pair.xml"))

    bundle = tmp_path / "bundle.jsonl"
    args = _args("integrity", "refresh", "--ids", "PMID:32450107", "--yes", "--bundle-out", str(bundle),
                 "--providers", "europepmc,pubmed", "--db", str(settings.db_path))
    assert _integrity(args, refresher=_refresher(handler)) == 0
    assert {r.url.host for r in sent} == {"www.ebi.ac.uk", "eutils.ncbi.nlm.nih.gov"}
    assert all("32450107" in str(r.url) for r in sent) and len(sent) == 2
    epmc = next(r for r in sent if r.url.host == "www.ebi.ac.uk")
    assert epmc.url.params["query"] == "EXT_ID:32450107 AND SRC:MED", "only the audited request shape is sent"
    stored = Database(settings.db_path).integrity_observations(["PMID:32450107"])
    assert {o.provider for o in stored} == {"europepmc", "pubmed"}
    assert all(any(e.status == "retracted" for e in o.events) for o in stored)
    assert len(bundle.read_text().splitlines()) == 2


def test_a_redirect_off_the_allowlist_is_refused_and_recorded_as_an_error(settings, capsys):
    def handler(request):
        if request.url.host == "evil.example":
            raise AssertionError("followed a redirect off the allowlist")
        return httpx.Response(302, headers={"Location": "https://evil.example/steal"})

    args = _args("integrity", "refresh", "--ids", "PMID:32450107", "--yes", "--providers", "europepmc",
                 "--db", str(settings.db_path))
    assert _integrity(args, refresher=_refresher(handler)) == 1
    [obs] = Database(settings.db_path).integrity_observations(["PMID:32450107"])
    assert obs.outcome == "error" and obs.events == []


def test_malformed_identifiers_and_unknown_providers_are_refused_before_anything_is_sent(settings, capsys):
    assert main(["integrity", "plan", "--ids", "PMID:1 OR 2", "--db", str(settings.db_path)]) == 1
    assert main(["integrity", "plan", "--ids", "PMID:1", "--providers", "scopus", "--db",
                 str(settings.db_path)]) == 1
    assert "unapproved provider" in capsys.readouterr().err


def test_import_accepts_a_good_bundle_and_refuses_a_tampered_one(settings, tmp_path, capsys):
    from tests.litreview.test_publication_integrity import raw

    good = bundle_line("europepmc", raw("epmc-original.json"), requested_ids=["PMID:32450107"],
                       captured_at="2026-09-27T00:10:23+00:00", source_uri="x", http_status=200)
    path = tmp_path / "b.jsonl"
    path.write_text(json.dumps(good) + "\n")
    assert main(["integrity", "import", str(path), "--db", str(settings.db_path)]) == 0
    assert "not authenticated" in capsys.readouterr().out

    bad = dict(good, raw_b64=good["raw_b64"][:-8] + "AAAAAAA=")
    path.write_text(json.dumps(bad) + "\n")
    assert main(["integrity", "import", str(path), "--db", str(settings.db_path)]) == 1
    assert "tampered" in capsys.readouterr().err
    assert Database(settings.db_path).conn.execute("SELECT COUNT(*) FROM integrity_observations").fetchone()[0] == 1


def test_status_reads_local_records_only(settings, capsys, monkeypatch):
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("sent")))
    db = Database(settings.db_path)
    record(db, "Retraction in")
    assert main(["integrity", "status", "--ids", "DOI:10.1000/p1", "PMID:5", "--db", str(settings.db_path)]) == 0
    out = capsys.readouterr().out
    assert "DOI:10.1000/p1: retracted" in out and "PMID:5: no notice recorded [never_checked]" in out


def test_a_legacy_run_resumed_after_its_gate_does_not_learn_todays_notices(settings):
    db = Database(settings.db_path)
    r = run(settings, db)
    with db.conn:
        db.conn.execute("DELETE FROM artifacts WHERE run_id = ? AND name = ?", (r.run.id, SNAPSHOT_ARTIFACT))
    record(db, "Expression of concern in")
    again = answer_question(QUESTION, settings=settings, client=fake_client(), db=db, resume_run_id=r.run.id)
    assert json.loads(db.get_artifact(r.run.id, SNAPSHOT_ARTIFACT))["status"] == "not_recorded"
    assert not any("Publication notice" in ref.citation for ref in again.report.references)
    assert "⚠ concern" not in json.dumps(again.report.model_dump())
    assert assess_run(db, r.run.id).advisories[0].startswith("publication status when this run was made: not recorded")
