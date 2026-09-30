"""The integrity check: a sound run passes, and each kind of breakage is caught.

Each test below breaks one guarantee by hand — a quote that is not the
paper's text, a sample size nothing states, a sentence resting on nothing, a
lost retraction flag, a paper body changed after the run — and checks that
`report/invariants.py` names it. A check that never fires proves nothing.
"""

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import Paper, PaperSignals, Sentence
from bedrock.litreview.report.invariants import check_flags, check_report, verify_run
from bedrock.litreview.runner import answer_question
from bedrock.litreview.stages.s8_render import to_html, to_markdown
from tests.litreview.fixtures import QUESTION, corpus, fake_client

KB = load_knowledge()


@pytest.fixture()
def result(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    db = Database(s.db_path)
    r = answer_question(QUESTION, settings=s, client=fake_client(), db=db, papers=corpus())
    return r, db


def _check(r, report=None, papers=None):
    return check_report(report or r.report, papers or r.papers, r.spec, KB)


def test_a_sound_run_passes_at_run_time_and_when_re_verified(result):
    r, db = result
    assert r.report.evidence and r.report.kept_sentences()
    assert r.report.integrity == []
    assert verify_run(db, r.run.id) == []


def test_a_stored_report_without_the_new_disagreements_section_still_verifies(result):
    from bedrock.litreview.models import Report
    from bedrock.litreview.stages.s8_render import to_json

    r, db = result
    legacy = r.report.model_copy(update={"sections": [s for s in r.report.sections if s.id != "disagreements"]})
    body = to_json(legacy)
    assert to_json(Report.model_validate_json(body)) == body
    assert _check(r, legacy) == []
    # Persist in the same form as a run written before this section existed,
    # including its pinned hash. Verification must not add the new section.
    import hashlib
    db.put_artifact(r.run.id, "report", body)
    run = db.get_run(r.run.id)
    run.report_sha256 = hashlib.sha256(body.encode()).hexdigest()
    db.save_run(run)
    assert verify_run(db, r.run.id) == []


def test_a_quote_that_is_not_the_papers_text_is_caught(result):
    r, _ = result
    units = [u.model_copy(update={"quote": u.quote.replace("mice", "patients")}) if i == 0 else u
             for i, u in enumerate(r.report.evidence)]
    assert any(v.startswith("quote:") for v in _check(r, r.report.model_copy(update={"evidence": units})))


def test_a_sample_size_nothing_states_is_caught(result):
    r, _ = result
    units = [u.model_copy(update={"n": 999}) if i == 0 else u for i, u in enumerate(r.report.evidence)]
    assert any("sample size 999" in v for v in _check(r, r.report.model_copy(update={"evidence": units})))


def test_a_sentence_resting_on_nothing_is_caught(result):
    r, _ = result
    section = r.report.sections[0].model_copy(deep=True)
    cited = next(s.citations for _, s in r.report.kept_sentences())
    section.sentences.append(Sentence(text="The effect was seen in 4,812 patients.", citations=cited))
    report = r.report.model_copy(update={"sections": [section, *r.report.sections[1:]]})
    assert any(v.startswith("sentences:") for v in _check(r, report))


def test_a_changed_paper_is_caught(result):
    r, _ = result
    unit = r.report.evidence[0]
    papers = [p.model_copy(update={"sections": {**p.sections, unit.section: "rewritten " + p.sections[unit.section]}})
              if p.id == unit.paper_id else p for p in r.papers]
    assert any(v.startswith("quote:") for v in _check(r, papers=papers))


def test_a_changed_paper_version_in_the_database_is_caught(result):
    r, db = result
    with db.conn:
        db.conn.execute("UPDATE paper_versions SET body = replace(body, 'mice', 'rats')")
    assert any(v.startswith("pinned:") for v in verify_run(db, r.run.id))


def test_a_flag_lost_in_a_merge_is_caught():
    p = Paper(id="p", title="t", signals=PaperSignals(publication_types=["Journal Article"]), merged_from=[
        {"id": "a", "publication_types": ["Journal Article"], "is_review": False, "is_retracted": False},
        {"id": "b", "publication_types": ["Retracted Publication"], "is_review": False, "is_retracted": True},
    ])
    found = check_flags([p])
    assert any("Retracted Publication" in v for v in found) and any("was retracted" in v for v in found)


def test_evidence_from_a_retracted_paper_is_caught(result):
    r, _ = result
    pid = r.report.evidence[0].paper_id
    papers = [p.model_copy(update={"signals": p.signals.model_copy(update={"is_retracted": True})})
              if p.id == pid else p for p in r.papers]
    assert any("flagged retracted" in v for v in _check(r, papers=papers))


def test_a_failed_report_opens_with_a_warning(result):
    r, _ = result
    report = r.report.model_copy(update={"integrity": ["quote: finding x: quote is not the paper's text"]})
    md, page = to_markdown(report), to_html(report)
    assert "failed its integrity check" in md.split("##")[0]
    assert 'class="integrity"' in page and "failed its integrity check" in page
    assert "integrity check" not in to_markdown(r.report).split("##")[0]


def test_verify_run_all_reports_a_run_it_cannot_verify(result, capsys):
    """A missing pinned paper is exactly what the check exists to report; it
    must not be skipped as "not a finished run"."""
    from bedrock.litreview.cli import main

    r, db = result
    with db.conn:
        db.conn.execute("DELETE FROM paper_versions WHERE paper_id = ?", (r.report.evidence[0].paper_id,))
    db_path = db.conn.execute("PRAGMA database_list").fetchone()[2]
    assert main(["verify-run", "--all", "--db", db_path]) == 1
    assert "cannot be verified" in capsys.readouterr().out


def test_the_most_relevant_sources_quote_and_reason_are_checked(result):
    r, _ = result
    assert r.report.best_source_id, "the fixture run names a most relevant source"
    sections = []
    for sec in r.report.sections:
        body = sec.body_markdown or ""
        if "> " in body:
            body = body.replace("\n> ", "\n> Blocking TNF cured every patient. ", 1)
            body += "\nWhy it ranks first: it tested 400 patients in a phase 3 trial."
        sections.append(sec.model_copy(update={"body_markdown": body}))
    found = _check(r, r.report.model_copy(update={"sections": sections}))
    assert any("most relevant source's quote" in v for v in found)
    assert any("ranking reason" in v for v in found)


@pytest.mark.parametrize("reason, flagged", [
    ("Blocking IL-17A had a significant effect on epidermal thickness.", True),
    ("Blocking IL-17A had no significant effect on epidermal thickness.", False),
], ids=["dropped-negation", "faithful"])
def test_a_ranking_reason_that_drops_the_abstracts_negation_is_caught(result, reason, flagged):
    """The fixture's best source says IL-17A blockade had no significant
    effect; a reason saying it had one restates the opposite finding."""
    r, _ = result
    assert r.report.best_source_id == "P1"
    first, *rest = r.report.sections
    body = (first.body_markdown or "") + f"\nWhy it ranks first: {reason}"
    report = r.report.model_copy(update={"sections": [first.model_copy(update={"body_markdown": body}), *rest]})
    assert any("ranking reason" in v for v in _check(r, report)) is flagged
