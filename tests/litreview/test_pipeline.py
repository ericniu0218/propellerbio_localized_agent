"""End to end, with a scripted fake model: question in, report out.

What this proves, with no GPU and no network:

- the stages run in order and hand each other the right shapes
- a fabricated quote and a near-miss finding are destroyed by code
- the citation guard removes a sentence with an invented number and one
  naming a drug that is not in its cited evidence
- the report is byte-identical when the same question is asked again
"""

import json

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.runner import answer_question
from tests.litreview.fixtures import QUESTION, corpus, fake_client


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir = tmp_path / "cache"
    s.out_dir = tmp_path / "out"
    s.db_path = tmp_path / "db.sqlite3"
    return s


def run(settings, tmp_path, **kwargs):
    return answer_question(
        QUESTION, settings=settings, client=fake_client(), papers=corpus(),
        db=Database(tmp_path / "db.sqlite3"), **kwargs,
    )


def test_the_question_is_read_and_invented_entities_are_discarded(settings, tmp_path):
    result = run(settings, tmp_path)
    names = [e.canonical for e in result.spec.entities]
    assert names == ["psoriasis"], "eczema was not written in the question and must not steer the search"
    assert result.spec.question_type == "target_identification"
    assert result.spec.goal == "prevent"


def test_the_gate_excludes_near_misses_reviews_and_passing_mentions(settings, tmp_path):
    gates = run(settings, tmp_path).gates
    assert gates["P1"].status == "pass"
    assert gates["P2"].status == "pass"
    assert gates["P3"].status == "near_miss", "atopic dermatitis is not psoriasis"
    assert gates["P4"].status == "review"
    assert gates["P5"].status == "absent"
    assert gates["P6"].status == "absent", "one passing mention in a results section is not enough"


def test_the_ranking_puts_graded_primary_papers_first(settings, tmp_path):
    ranked = run(settings, tmp_path).ranked
    assert [r.paper_id for r in ranked[:2]] == ["P1", "P2"] or [r.paper_id for r in ranked[:2]] == ["P2", "P1"]
    assert all(r.gate != "pass" for r in ranked[2:])


def test_fabricated_quotes_and_near_miss_findings_never_become_evidence(settings, tmp_path):
    result = run(settings, tmp_path)
    quotes = [u.quote for u in result.summary.units]
    assert not any("cured every patient" in q for q in quotes)
    assert not any(u.disease_raw == "atopic dermatitis" for u in result.summary.units)
    assert result.extract_stats["quote_failed"] >= 1
    assert result.extract_stats.get("near_miss", 0) + result.extract_stats.get("other_disease", 0) >= 1
    assert any(u.outcome == "no_effect" for u in result.summary.units), "negative results are first-class"


def test_the_guard_removes_invented_numbers_and_names(settings, tmp_path):
    report = run(settings, tmp_path).report
    kept = [s.text for _, s in report.kept_sentences()]
    assert kept, "the grounded sentence survives"
    assert not any("99%" in t for t in kept)
    assert not any("Secukinumab" in t for t in kept)
    dropped = [s for sec in report.sections for s in sec.sentences if not s.kept]
    assert any("number 99" in (s.drop_reason or "") for s in dropped)


def test_every_kept_sentence_cites_existing_evidence(settings, tmp_path):
    report = run(settings, tmp_path).report
    known = set(report.short_ids.values())
    for _, sentence in report.kept_sentences():
        assert sentence.citations and set(sentence.citations) <= known


def test_the_same_question_gives_byte_identical_reports(settings, tmp_path):
    first = run(settings, tmp_path)
    second = run(settings, tmp_path)
    assert first.paths["json"].read_bytes() == second.paths["json"].read_bytes()
    assert first.paths["md"].read_bytes() == second.paths["md"].read_bytes()
    assert first.report_id == second.report_id


def test_renderers_write_all_three_formats(settings, tmp_path):
    result = run(settings, tmp_path)
    md = result.paths["md"].read_text()
    assert "## Most relevant source" in md and "## References" in md
    assert "<html" in result.paths["html"].read_text()
    assert json.loads(result.paths["json"].read_text())["question"] == QUESTION


def test_the_report_files_are_readable_only_by_this_account(settings, tmp_path):
    """A report quotes unpublished papers, and which target a company is
    considering is itself the strategy. `--out` may point at a shared drive, so
    the mode goes on the files rather than resting on the parent directory."""
    result = run(settings, tmp_path)
    for name, path in result.paths.items():
        assert path.stat().st_mode & 0o777 == 0o600, f"{name} is world-readable"
    run_json = result.paths["json"].parent / "run.json"
    assert run_json.stat().st_mode & 0o777 == 0o600
    assert result.paths["json"].parent.stat().st_mode & 0o777 == 0o700


def test_without_a_model_the_pipeline_still_ranks_and_plans(settings, tmp_path):
    result = answer_question(
        QUESTION, settings=settings, use_model=False, papers=corpus(), db=Database(tmp_path / "db2.sqlite3")
    )
    assert result.ranked[0].gate == "pass"
    assert result.plan.jobs, "lexical ranking and heuristic facets still choose papers to read"
    assert not result.summary.units


def test_a_crashed_run_resumes_from_its_stage(settings, tmp_path):
    first = run(settings, tmp_path)
    resumed = answer_question(
        QUESTION, settings=settings, client=fake_client(), db=Database(tmp_path / "db.sqlite3"),
        resume_run_id=first.run.id,
    )
    assert resumed.report_id == first.report_id


def test_a_finished_run_reopens_exactly_as_written(settings, tmp_path):
    from bedrock.litreview.runner import load_result

    first = run(settings, tmp_path)
    again = load_result(Database(tmp_path / "db.sqlite3"), first.run.id)
    assert again.report_id == first.report_id
    assert again.report == first.report and [u.id for u in again.summary.units] == [u.id for u in first.summary.units]


def test_fragments_and_repeats_never_reach_the_report():
    """MedGemma emitted a bare 'A.' twice, and the same sentence four times
    in a row. Neither states anything a scientist can use."""
    from bedrock.litreview.stages.s7_write import MIN_SENTENCE_CHARS, REPEAT_OVERLAP, _content_words, _overlap

    assert len("A.") < MIN_SENTENCE_CHARS
    a = _content_words("UBE2L3 is downregulated in the epidermis of psoriatic skin.")
    assert _overlap(a, _content_words("UBE2L3 is downregulated in the epidermis of psoriatic skin, not the dermis.")) \
        >= REPEAT_OVERLAP


def test_the_plan_spends_its_paper_budget():
    """The per-facet maxima fill before the paper budget does, because one
    paper answers several needs. What is left over should still be read."""
    from bedrock.litreview.facets import facets_for
    from bedrock.litreview.models import Paper, RankedPaper
    from bedrock.litreview.stages.s0_question import parse_question
    from bedrock.litreview.stages.s4_plan import build_plan

    spec = parse_question(QUESTION)
    papers, ranked = {}, []
    for i in range(30):
        pid = f"P{i}"
        papers[pid] = Paper(
            id=pid, title=f"Blocking IL-{i} in psoriasis",
            abstract="Knockout mice were protected from psoriasis; patients in a randomised trial improved.",
        )
        ranked.append(RankedPaper(paper_id=pid, rank=i + 1, gate="pass", grade=3, study_kind="primary_research",
                                  facets=["causal", "clinical"]))
    plan = build_plan(spec, facets_for(spec), ranked, papers, max_papers=16)
    assert len(plan.jobs) == 16, "the budget is the limit, not the facet caps"
