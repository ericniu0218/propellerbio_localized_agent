"""The owner's decisions of 26 September 2026, each held to its wording.

1. Retraction checks run automatically by default; the app has a setting to
   make them manual, explained in plain words.
2. Crossref may be asked directly, by the retraction check only.
3. An expression of concern: kept, flagged with the notice's own words,
   ranked after all other evidence, never used in the Answer.
4. A reinstated paper is never added back automatically: the scientist is
   told, and adds it back themselves.
5. Reviews are shown in their own section, with links and their own words,
   and counted as evidence nowhere.

No test here reaches the network: the providers are `httpx.MockTransport`.
"""

from __future__ import annotations

import json
import os
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from bedrock.litreview import preferences
from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import Paper
from bedrock.litreview.publication_integrity import (
    build_snapshot,
    decide_reinstated,
    observe,
)
from bedrock.litreview.report.invariants import check_report, verify_run
from bedrock.litreview.runner import answer_question
from bedrock.litreview.sources.http import INTEGRITY_HOSTS, SOURCE_HOSTS, SourceError, SourceHTTP
from bedrock.litreview.sources.integrity import IntegrityRefresher, automatic_check_for, check_papers
from bedrock.litreview.stages.s6_score import best_units
from tests.litreview.fixtures import QUESTION, corpus, fake_client

FIX = Path(__file__).parent / "integrity_fixtures"
NOW = datetime(2026, 9, 27, 12, tzinfo=timezone.utc)
WEEK = timedelta(days=7)


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path, s.data_dir = tmp_path / "c", tmp_path / "o", tmp_path / "db.sqlite3", tmp_path
    return s


def run(settings, db, papers=None, **kwargs):
    return answer_question(QUESTION, settings=settings, client=fake_client(), db=db,
                           papers=papers if papers is not None else corpus(), **kwargs)


# --- 1. the setting -----------------------------------------------------------------------------------


def test_retraction_checks_are_automatic_unless_someone_chooses_manual(tmp_path):
    assert preferences.integrity_mode(tmp_path, env={}) == "auto"
    assert preferences.integrity_mode(tmp_path, {"integrity_checks": "manual"}, env={}) == "manual", "site default"
    preferences.save_integrity_mode(tmp_path, "manual")
    assert preferences.integrity_mode(tmp_path, {"integrity_checks": "auto"}, env={}) == "manual", \
        "the choice saved in the app wins over the site default"
    assert preferences.integrity_mode(tmp_path, env={"BEDROCK_LITREVIEW_INTEGRITY_CHECKS": "auto"}) == "auto"
    assert stat.S_IMODE(os.stat(tmp_path / "preferences.json").st_mode) == 0o600


def test_a_bad_setting_is_a_config_error_not_a_silent_default(tmp_path):
    (tmp_path / "preferences.json").write_text(json.dumps({"integrity_checks": "sometimes"}))
    with pytest.raises(ConfigError, match="auto' or 'manual"):
        load_settings(model_set="laptop", env={"BEDROCK_LITREVIEW_DIR": str(tmp_path)}, check_endpoint=False)
    with pytest.raises(preferences.PreferenceError):
        preferences.save_integrity_mode(tmp_path, "sometimes")


def test_manual_means_no_automatic_check(settings):
    settings.integrity_checks = "manual"
    assert automatic_check_for(settings) is None
    settings.integrity_checks = "auto"
    assert callable(automatic_check_for(settings))


def test_the_explanations_name_the_trade_offs_in_plain_words():
    auto, manual = preferences.INTEGRITY_EXPLAINED["auto"], preferences.INTEGRITY_EXPLAINED["manual"]
    assert "ID numbers" in auto and "never your question" in auto and "Trade-off" in auto
    assert "Trade-off" in manual and "without a" in manual


async def test_the_app_shows_and_changes_the_setting(settings):
    from bedrock.ui.review_command import run_reviews_command

    notices: list[tuple[str, str]] = []

    async def notify(text, level):
        notices.append((text, level))

    await run_reviews_command("settings", settings_for=lambda: settings, notify=notify)
    text, level = notices[-1]
    assert level == "info" and "Retraction checks: Automatic" in text and "← current" in text
    assert "**Automatic (recommended).**" in text and "**Manual.**" in text and "/reviews settings manual" in text

    await run_reviews_command("settings manual", settings_for=lambda: settings, notify=notify)
    assert "Retraction checks: Manual** — saved." in notices[-1][0]
    assert preferences.integrity_mode(settings.data_dir, env={}) == "manual"

    await run_reviews_command("settings sometimes", settings_for=lambda: settings, notify=notify)
    assert notices[-1][1] == "error"


# --- 2. the automatic check, including Crossref ------------------------------------------------------------


def provider(requests, *, fail=False):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if fail:
            raise httpx.ConnectError("no network")
        host, path = request.url.host, request.url.path
        if host == "eutils.ncbi.nlm.nih.gov":
            return httpx.Response(200, content=(FIX / "pubmed-pair.xml").read_bytes())
        if host == "api.crossref.org":
            return httpx.Response(200, content=(FIX / "crossref-original.json").read_bytes())
        if path.endswith("/fullTextXML"):
            return httpx.Response(200, text="<article><body><p>The Lancet editors are aware that serious "
                                            "scientific questions have been raised about the data.</p></body></article>")
        if "32504543" in request.url.params.get("query", ""):
            return httpx.Response(200, content=(FIX / "epmc-eoc-notice.json").read_bytes())
        return httpx.Response(200, json={"resultList": {"result": []}})
    return handler


def refresher(requests, **kwargs):
    http = SourceHTTP(transport=httpx.MockTransport(provider(requests, **kwargs)), sleep=lambda s: None, retries=1,
                      hosts=INTEGRITY_HOSTS)
    return IntegrityRefresher(http=http, clock=lambda: "2026-09-27T11:00:00+00:00")


def papers_to_check():
    return [
        Paper(id="PMID32450107", pmid="32450107", title="Registry analysis"),
        Paper(id="DOIONLY", doi="10.1021/am300292v", title="A paper known only by its DOI"),
        Paper(id="PMID5", pmid="5", title="Checked yesterday"),
    ]


def test_the_automatic_check_asks_pubmed_and_crossref_and_skips_recent_checks(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    recent = observe("pubmed", b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>5</PMID></MedlineCitation>"
                                b"</PubmedArticle></PubmedArticleSet>",
                     requested_ids=["PMID:5"], retrieved_at="2026-09-26T12:00:00+00:00", http_status=200,
                     source_uri=None)
    db.add_integrity_observation(recent)
    sent: list[httpx.Request] = []
    summary = check_papers(db, papers_to_check(), refresher(sent), now=NOW, ttl=WEEK)

    hosts = [r.url.host for r in sent]
    assert hosts.count("eutils.ncbi.nlm.nih.gov") == 1
    pubmed = next(r for r in sent if r.url.host == "eutils.ncbi.nlm.nih.gov")
    assert pubmed.url.params["id"] == "32450107", "the paper checked yesterday is not asked about again"
    crossref = [r.url.path for r in sent if r.url.host == "api.crossref.org"]
    assert crossref == ["/works/10.1021/am300292v", "/works/10.1021/acsami.9b11759"], \
        "the paper, then the retraction notice Crossref named"
    assert set(hosts) <= INTEGRITY_HOSTS
    assert summary["fresh_skipped"] == 1 and summary["asked"] == 2 and summary["errors"] == 0
    assert "PMID:32504543" in summary["notices"]

    snapshot = build_snapshot(db, papers_to_check(), now=NOW, ttl=WEEK)
    concern = next(n for n in snapshot["papers"]["PMID32450107"]["notices"] if n["status"] == "expression_of_concern")
    assert concern["title"].startswith("Expression of concern: Hydroxychloroquine")
    assert concern["excerpt"].startswith("The Lancet editors are aware"), "the notice's own words are the reason"
    assert concern["link"] == "https://pubmed.ncbi.nlm.nih.gov/32504543/"
    assert "retracted" in snapshot["papers"]["DOIONLY"]["statuses"], "Crossref's retraction-watch assertion"


def test_no_network_stops_the_check_at_once_and_says_so(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    sent: list[httpx.Request] = []
    summary = check_papers(db, papers_to_check(), refresher(sent, fail=True), now=NOW, ttl=WEEK)
    assert len(sent) == 1, "one failed request, not one per paper"
    assert summary["stopped"] == "the services could not be reached" and summary["not_asked"] >= 1


def test_the_time_budget_is_kept(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    ticks = iter([0.0, 0.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0])
    summary = check_papers(db, papers_to_check(), refresher([]), now=NOW, ttl=WEEK, budget_seconds=90,
                           monotonic=lambda: next(ticks))
    assert summary["stopped"] == "time budget spent" and summary["not_asked"] == 1


def test_a_review_with_the_automatic_check_records_it_and_reports_unreachable_services(settings):
    db = Database(settings.db_path)

    def unreachable(db, papers, *, now, ttl):
        return {"asked": 0, "fresh_skipped": 0, "errors": 1, "not_asked": 3, "notices": [],
                "stopped": "the services could not be reached"}

    r = run(settings, db, integrity_check=unreachable)
    snap = json.loads(db.get_artifact(r.run.id, "publication_integrity_snapshot.v1"))
    assert snap["automatic_check"]["stopped"] == "the services could not be reached"
    gaps = next(s for s in r.report.sections if s.id == "gaps").body_markdown
    assert "stopped early: the services could not be reached" in gaps
    assert "have never been checked for retractions" in gaps


def test_crossref_is_reachable_by_the_retraction_check_only():
    assert "api.crossref.org" in INTEGRITY_HOSTS and "api.crossref.org" not in SOURCE_HOSTS
    search = SourceHTTP(transport=httpx.MockTransport(lambda r: httpx.Response(200)), retries=1)
    with pytest.raises(SourceError, match="may only reach"):
        search.get("https://api.crossref.org/works/10.1000/x")
    with pytest.raises(ValueError, match="outside the approved lists"):
        SourceHTTP(hosts=frozenset({"example.com"}))


# --- 3. expressions of concern ---------------------------------------------------------------------------


def test_findings_from_a_questioned_paper_come_after_all_others(settings):
    p1 = [u for u in run(settings, Database(settings.db_path)).report.evidence if u.paper_id == "P1"]
    other = p1[0].model_copy(update={"id": "other", "paper_id": "P9"})
    ranks = {"P1": 1, "P9": 50}
    assert best_units([*p1, other], ranks, 3)[0].paper_id == "P1"
    assert best_units([*p1, other], ranks, 3, demoted={"P1"})[0].paper_id == "P9"


# --- 4. reinstatement ---------------------------------------------------------------------------------------


def reinstated(db, doi="10.1000/p1"):
    body = json.dumps({"status": "ok", "message": {"DOI": doi, "type": "journal-article", "updated-by": [
        {"DOI": "10.1000/retraction", "type": "retraction", "source": "publisher"},
        {"DOI": "10.1000/reinstatement", "type": "reinstatement", "source": "publisher"},
    ]}}).encode()
    db.add_integrity_observation(observe("crossref", body, requested_ids=[f"DOI:{doi}"],
                                         retrieved_at="2026-09-26T00:00:00+00:00", http_status=200, source_uri=None))


def test_a_reinstated_paper_stays_out_and_the_scientist_is_told(settings):
    db = Database(settings.db_path)
    reinstated(db)
    said: list[str] = []
    r = run(settings, db, progress=said.append)
    assert r.gates["P1"].status == "retracted"
    assert all(u.paper_id != "P1" for u in r.report.evidence)
    gaps = next(s for s in r.report.sections if s.id == "gaps").body_markdown
    assert "was retracted and later **reinstated**" in gaps and "/reviews reinstate P1" in gaps
    assert any("retracted and later reinstated" in m for m in said)


def test_the_scientist_adds_it_back_and_it_counts_again(settings):
    db = Database(settings.db_path)
    reinstated(db)
    message = decide_reinstated(db, "DOI:10.1000/p1", include=True, by="Dr Rivera", note="read the notice",
                                now=NOW)
    assert "included in new reviews again" in message
    r = run(settings, db)
    assert r.gates["P1"].status == "pass" and "added back by Dr Rivera" in r.gates["P1"].reason
    assert any(u.paper_id == "P1" for u in r.report.evidence)
    gaps = next(s for s in r.report.sections if s.id == "gaps").body_markdown
    assert "added back by Dr Rivera" in gaps
    assert r.report.integrity == [] and verify_run(db, r.run.id) == []

    decide_reinstated(db, "DOI:10.1000/p1", include=False, by="Dr Rivera", note=None, now=NOW)
    assert run(settings, db).gates["P1"].status == "retracted"


def test_a_paper_flagged_retracted_in_its_own_record_can_be_added_back_after_reinstatement(settings):
    db = Database(settings.db_path)
    reinstated(db)
    decide_reinstated(db, "10.1000/p1", include=True, by="Dr Rivera", note=None, now=NOW)
    papers = [p.model_copy(update={"signals": p.signals.model_copy(update={"is_retracted": True})})
              if p.id == "P1" else p for p in corpus()]
    r = run(settings, db, papers=papers)
    assert r.gates["P1"].status == "pass" and r.report.integrity == []
    assert any("flagged retracted" in v for v in check_report(r.report, r.papers, r.spec, load_knowledge())), \
        "without the recorded decision the same findings are a violation"


def test_nothing_is_added_back_without_a_recorded_reinstatement(settings):
    db = Database(settings.db_path)
    with pytest.raises(ValueError, match="no recorded reinstatement"):
        decide_reinstated(db, "PMID:1", include=True, by="Dr Rivera", note=None, now=NOW)
    reinstated(db)
    with pytest.raises(ValueError, match="say who"):
        decide_reinstated(db, "DOI:10.1000/p1", include=True, by=" ", note=None, now=NOW)


async def test_the_app_command_records_the_decision_as_the_user(settings, monkeypatch):
    from bedrock.ui.review_command import run_reviews_command

    db = Database(settings.db_path)
    reinstated(db)
    monkeypatch.setenv("BEDROCK_USER", "jdoe")
    notices = []

    async def notify(text, level):
        notices.append((text, level))

    await run_reviews_command("reinstate PMID:1", settings_for=lambda: settings, notify=notify)
    assert notices[-1][1] == "error" and "no recorded reinstatement" in notices[-1][0]
    await run_reviews_command("reinstate DOI:10.1000/p1 --note checked", settings_for=lambda: settings, notify=notify)
    assert notices[-1][1] == "info" and "(jdoe)" in notices[-1][0]
    assert Database(settings.db_path).integrity_decision(["DOI:10.1000/p1"])["decided_by"] == "jdoe"


# --- 5. background from reviews ------------------------------------------------------------------------------


def test_reviews_are_shown_with_their_own_words_and_a_link(settings):
    r = run(settings, Database(settings.db_path))
    background = next(s for s in r.report.sections if s.id == "background").body_markdown
    assert "**They are not evidence in this report**" in background
    assert "*Therapeutic targets in psoriasis: a review* · review" in background
    assert "> We review IL-23, IL-17 and TNF as targets in psoriasis." in background
    assert r.report.integrity == []


def test_reviews_change_nothing_else_in_the_report(settings):
    with_review = run(settings, Database(settings.db_path))
    settings.db_path = settings.db_path.with_name("without.sqlite3")
    without = run(settings, Database(settings.db_path), papers=[p for p in corpus() if p.id != "P4"])
    assert [u.id for u in with_review.report.evidence] == [u.id for u in without.report.evidence]
    assert [t.target for t in with_review.summary.targets] == [t.target for t in without.summary.targets]
    assert "P4" not in {ref.paper_id for ref in with_review.report.references}


def test_a_background_quote_that_is_not_the_reviews_text_is_caught(settings):
    r = run(settings, Database(settings.db_path))
    sections = [s.model_copy(update={"body_markdown": (s.body_markdown or "").replace(
        "We review IL-23", "We recommend IL-23")}) if s.id == "background" else s for s in r.report.sections]
    report = r.report.model_copy(update={"sections": sections})
    assert any(v.startswith("quote: [background]") for v in check_report(report, r.papers, r.spec, load_knowledge()))


def test_a_review_says_at_the_start_what_the_retraction_check_will_send(settings):
    from bedrock.litreview.service import ReviewRequest
    from bedrock.ui.review_command import start_notice

    request = ReviewRequest(question=QUESTION)
    assert "ID numbers (not your question) go to PubMed and Crossref" in start_notice(request, settings)
    settings.integrity_checks = "manual"
    assert "manual — nothing is sent" in start_notice(request, settings)


def test_a_review_says_at_the_start_that_a_patent_search_sends_its_terms_out(settings):
    """A patent search puts the question's own exact terms in a URL to the EPO.
    The notice says so before the run, as the literature search's does, and says
    patents are never evidence - the one thing a reader could get wrong."""
    from bedrock.litreview.service import ReviewRequest
    from bedrock.ui.review_command import start_notice

    quiet = start_notice(ReviewRequest(question=QUESTION), settings)
    assert "Patents" not in quiet, "a review nobody asked for patents in says nothing about them"

    searched = start_notice(ReviewRequest(question=QUESTION, patent_search=True), settings)
    assert "European Patent Office" in searched and "never evidence" in searched

    from_file = start_notice(ReviewRequest(question=QUESTION, patents=[]), settings)
    assert "records given with `--patents`" in from_file and "European Patent Office" not in from_file
