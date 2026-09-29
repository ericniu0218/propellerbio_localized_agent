"""The coverage ledger accounts for every paper, and changes nothing it records.

Each test sets up one way a paper can fall out of a report - excluded by the
gate, never graded, graded and not read, read with nothing verified, verified
and never cited - and checks the ledger names it, with counts that reconcile.
The first tests check the ledger is bookkeeping only: the stages return the
same thing, and the model is asked the same questions, with it and without.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.coverage import ARTIFACT, DENOMINATORS
from bedrock.litreview.db import Database
from bedrock.litreview.facets import facets_for
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import Paper
from bedrock.litreview.runner import answer_question
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s2_gate import gate_papers
from bedrock.litreview.stages.s3_rank import rank_papers
from bedrock.litreview.stages.s4_plan import build_plan
from bedrock.litreview.stages.s5_extract import extract_plan
from tests.litreview.fixtures import QUESTION, corpus, fake_client, responder


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    return s


def _run(settings, *, client=None, papers=None, use_model=True, db=None, **kwargs):
    db = db or Database(settings.db_path)
    r = answer_question(QUESTION, settings=settings, client=client if client is not None else fake_client(),
                        papers=papers if papers is not None else corpus(), db=db, use_model=use_model, **kwargs)
    return r, db, json.loads(db.get_artifact(r.run.id, ARTIFACT))


# --- bookkeeping only ------------------------------------------------------------------------


def test_the_collectors_change_no_stage_output_and_no_model_call():
    papers = corpus()
    by_id = {p.id: p for p in papers}
    spec = parse_question(QUESTION, None, model=None)
    facets = facets_for(spec)
    gates = gate_papers(papers, spec)

    plain, watched = fake_client(), fake_client()
    ranked = rank_papers(papers, gates, spec, facets, plain, model="m")
    reasons: dict = {}
    ranked_watched = rank_papers(papers, gates, spec, facets, watched, model="m", selection=reasons)
    assert ranked == ranked_watched and plain.prompts == watched.prompts and reasons

    plan = build_plan(spec, facets, ranked, by_id)
    read: dict = {}
    assert build_plan(spec, facets, ranked, by_id, reasons=read) == plan
    assert set(read) == set(plan.paper_ids)

    a, b = fake_client(), fake_client()
    units, stats = extract_plan(plan, by_id, spec, facets, a, model="m")
    traces: dict = {}
    units_w, stats_w = extract_plan(plan, by_id, spec, facets, b, model="m", traces=traces)
    assert (units, stats) == (units_w, stats_w) and a.prompts == b.prompts
    assert set(traces) == set(plan.paper_ids)


def test_the_ledger_is_beside_the_report_not_in_it(settings):
    r, db, ledger = _run(settings)
    body = db.get_artifact(r.run.id, "report")
    assert "coverage" not in body and "disposition" not in body
    assert ledger["schema_version"] == 1 and ledger["run_id"] == r.run.id
    assert ledger["corpus_manifest_hash"] == r.run.papers_sha256
    assert len(ledger["selection_policy"]["sha256"]) == 64


# --- counts reconcile ---------------------------------------------------------------------------


def test_every_count_reconciles_to_named_papers_and_units(settings):
    r, _, ledger = _run(settings)
    counts, papers = ledger["counts"], ledger["papers"]
    assert set(counts) == set(DENOMINATORS), "every count says what it counts"
    assert sum(ledger["dispositions"].values()) == counts["papers_collected"] == len(papers)
    assert sum(counts["gate"].values()) == counts["papers_collected"]
    assert counts["graded"] + counts["grading_failed"] == counts["grading_selected"]
    assert counts["read_selected"] + counts["unread_qualified"] == counts["qualified"]
    assert counts["units_accepted"] == sum(len(p["unit_ids"]) for p in papers.values()) == len(r.summary.units)
    assert counts["chunks_attempted"] == sum(len((p["extraction"] or {}).get("chunks") or [])
                                             for p in papers.values())
    assert counts["units_cited_after_guard"] <= counts["units_sent_to_synthesis"] <= counts["units_accepted"]
    cited = {u for p in papers.values() for u in p["cited_unit_ids"]}
    assert len(cited) == counts["units_cited_after_guard"]
    assert set(ledger["targets"]["answer"]) <= set(ledger["targets"]["narrative"]) \
        <= set(ledger["targets"]["evidence_table"])


def test_each_paper_names_the_stage_it_stopped_at(settings):
    _, _, ledger = _run(settings)
    papers = ledger["papers"]
    assert papers["P3"]["disposition"] == "gate_near_miss"
    assert papers["P1"]["disposition"] == "cited"
    assert papers["P1"]["read"]["reason"] == "best_source"
    assert papers["P1"]["grading"]["selected_reason"] == "lexical_top_40"


def test_a_read_paper_that_yields_nothing_verified_is_named_with_why(settings):
    """The guselkumab trial's abstract never names IL-23, so the finding's
    target is found nowhere and dropped - visible here, not silently absent."""
    _, _, ledger = _run(settings)
    p2 = ledger["papers"]["P2"]
    assert p2["disposition"] == "read_no_verified_units"
    [chunk] = p2["extraction"]["chunks"]
    assert chunk["status"] == "completed" and chunk["findings"] > 0
    assert chunk["rejected"], "each rejected finding has a reason"


def test_a_rejected_quote_is_told_apart_from_no_findings(settings):
    _, _, ledger = _run(settings)
    [abstract, *_] = ledger["papers"]["P1"]["extraction"]["chunks"]
    assert abstract["rejected"].get("quote_failed") == 1

    def silent(prompt, schema):
        if (schema or {}).get("title") == "ExtractionResult":
            return json.dumps({"findings": [], "is_review": False})
        return responder(prompt, schema)

    settings.db_path = settings.db_path.with_name("quiet.sqlite3")
    _, _, quiet = _run(settings, client=FakeLLMClient(responder=silent))
    chunk = quiet["papers"]["P1"]["extraction"]["chunks"][0]
    assert chunk == {"id": "abstract:0", "status": "completed", "findings": 0, "rejected": {}}
    assert quiet["papers"]["P1"]["disposition"] == "read_no_verified_units"


def test_planned_facets_are_checked_against_verified_units(settings):
    """A facet the plan called covered because tagged papers were selected is
    realized only when verified units for it exist."""
    _, _, ledger = _run(settings)
    facets = ledger["facets"]
    assert facets["prevention"]["planned_papers"] == 2 and facets["prevention"]["realized_papers"] == 1
    assert facets["causal"]["planned_status"] == "thin" and facets["causal"]["realized_status"] == "gap"
    assert "causal" in ledger["unmet_requirements"]
    assert "negative" not in ledger["unmet_requirements"], "a facet with min 0 is never unmet"


# --- the ways a paper can fall out ---------------------------------------------------------------


def test_a_paper_past_the_grading_budget_is_named_as_such(settings):
    settings.pipeline.setdefault("rank", {})["max_graded"] = 1
    _, _, ledger = _run(settings)
    reasons = {pid: (p["grading"] or {}).get("selected_reason") for pid, p in ledger["papers"].items()}
    assert sorted(v for v in reasons.values() if v) == ["below_lexical_cutoff_1", "lexical_top_1"]
    assert "not_graded_budget" in ledger["dispositions"]
    assert ledger["counts"]["grading_selected"] == 1


def test_a_grading_call_that_fails_is_counted_as_a_failure(settings):
    def flaky(prompt, schema):
        if (schema or {}).get("title") == "RelevanceJudgement" and "Guselkumab" in prompt:
            return "not json"
        return responder(prompt, schema)

    _, _, ledger = _run(settings, client=FakeLLMClient(responder=flaky))
    assert ledger["papers"]["P2"]["disposition"] == "grading_failed"
    assert ledger["counts"]["grading_failed"] == 1


def test_a_title_only_paper_is_recorded_as_title_only(settings):
    papers = [*corpus(), Paper(id="T1", title="Psoriasis and IL-23: a title with no abstract")]
    _, _, ledger = _run(settings, papers=papers)
    assert ledger["papers"]["T1"]["text"] == "title_only"


def test_a_target_named_only_in_an_unread_paper_is_not_counted(settings):
    """A name in a paper nobody read is a lead, not coverage."""
    papers = [*corpus(), Paper(id="U1", title="Psoriasis cohort", abstract="Psoriasis patients had high TYK2 levels.")]
    settings.pipeline.setdefault("plan", {})["max_papers"] = 1
    _, _, ledger = _run(settings, papers=papers)
    assert "TYK2" not in json.dumps(ledger["targets"])
    assert ledger["papers"]["U1"]["disposition"] in ("qualified_not_read", "not_graded", "graded_below_min")


def test_a_unit_cited_only_by_a_sentence_the_guard_removed_is_not_cited(settings):
    def overreach(prompt, schema):
        if (schema or {}).get("title") == "DraftSection":
            ids = schema["properties"]["sentences"]["items"]["properties"]["citations"]["items"]["enum"]
            return json.dumps({"sentences": [
                {"citations": [ids[-1]], "text": "This reduced disease activity by 99% in every model tested."}]})
        return responder(prompt, schema)

    _, _, ledger = _run(settings, client=FakeLLMClient(responder=overreach))
    counts = ledger["counts"]
    assert counts["units_cited_after_guard"] == 0 and counts["units_cited_only_by_dropped"] >= 1
    assert ledger["targets"]["answer"] == [] and ledger["targets"]["narrative"] == []
    assert ledger["targets"]["evidence_table"], "the evidence is still in the table"
    assert Counter(p["disposition"] for p in ledger["papers"].values())["units_not_cited"] >= 1


# --- empty, model-free and legacy states ---------------------------------------------------------


def test_a_run_without_a_model_says_what_it_did_not_do(settings):
    _, _, ledger = _run(settings, use_model=False)
    counts = ledger["counts"]
    assert counts["graded"] == 0 and counts["units_accepted"] == 0
    assert counts["chunks_attempted"] is None and counts["findings_proposed"] is None
    assert all(row["realized_status"] is None for row in ledger["facets"].values())
    assert ledger["unmet_requirements"] is None
    assert "read_not_extracted" in ledger["dispositions"]


def test_a_corpus_where_nothing_passes_the_gate_still_has_a_ledger(settings):
    off_topic = [p for p in corpus() if p.id in ("P3",)]
    _, _, ledger = _run(settings, papers=off_topic)
    assert ledger["counts"]["qualified"] == 0 and ledger["counts"]["read_selected"] == 0
    assert ledger["dispositions"] == {"gate_near_miss": 1}
    assert ledger["acquisition"]["status"] == "not_recorded"


def test_a_resumed_run_without_ledger_records_leaves_them_null(settings):
    """A run whose early stages ran before the ledger existed: nothing is reconstructed."""
    r, db, _ = _run(settings)
    with db.conn:
        db.conn.execute("DELETE FROM artifacts WHERE run_id = ? AND name = 'coverage_inputs'", (r.run.id,))
    again = answer_question(QUESTION, settings=settings, client=fake_client(), db=db, resume_run_id=r.run.id)
    ledger = json.loads(db.get_artifact(again.run.id, ARTIFACT))
    p1 = ledger["papers"]["P1"]
    assert p1["grading"]["selected_reason"] is None and p1["read"]["reason"] is None
    assert p1["extraction"] is None and ledger["counts"]["chunks_attempted"] is None
    assert ledger["counts"]["grading_selected"] is None
    assert again.report_id == r.report_id


def test_the_report_says_what_was_verified_not_what_was_planned(settings):
    """The plan tagged one paper for causal evidence and none of its findings
    survived; the report must not say that paper "supplied" it."""
    r, _, ledger = _run(settings)
    gaps = next(s for s in r.report.sections if s.id == "gaps").body_markdown
    plan = next(s for s in r.report.sections if s.id == "plan").body_markdown
    assert ledger["facets"]["causal"]["planned_papers"] == 1 and ledger["facets"]["causal"]["realized_papers"] == 0
    assert "1 paper(s) were read for **targets with causal evidence**, but no verified finding was extracted" in gaps
    assert "supplied **targets with causal evidence**" not in gaps
    causal_row = next(line for line in plan.splitlines() if "Targets with causal evidence" in line)
    assert causal_row.rstrip().endswith("| 1 | 0 | 0 | gap |")


def test_a_report_without_a_model_does_not_claim_facets_were_covered(settings):
    r, _, _ = _run(settings, use_model=False)
    gaps = next(s for s in r.report.sections if s.id == "gaps").body_markdown
    plan = next(s for s in r.report.sections if s.id == "plan").body_markdown
    assert "assembled without a model" in gaps
    assert "| covered |" not in plan and "not extracted" in plan


def test_a_live_search_records_its_acquisition_counts_not_its_queries(settings):
    from bedrock.litreview.sources.literature import SearchReport

    class Searcher:
        name = "literature"

        def __init__(self):
            self.last_report = None

        def search(self, request):
            self.last_report = SearchReport(
                queries={"europepmc": ['TITLE_ABS:"psoriasis"', 'TITLE_ABS:"psoriasis" AND prevention']},
                hits={"europepmc": 9}, failures={"pubmed [entity]": "HTTP 503"}, unique=8, returned=6,
                full_text_tried=4, full_text=2, title_only=1)
            return corpus()

    r = answer_question(QUESTION, settings=settings, client=fake_client(), searcher=Searcher(),
                        db=Database(settings.db_path))
    ledger = json.loads(Database(settings.db_path).get_artifact(r.run.id, ARTIFACT))
    acq = ledger["acquisition"]
    assert acq["status"] == "recorded" and acq["queries_sent"] == {"europepmc": 2}
    assert acq["records_returned"] == {"europepmc": 9} and acq["truncated_by_cap"] == 2
    assert acq["failed_queries"] == ["pubmed [entity]"] and acq["collected_after_ingest"] == len(r.papers)
    assert "psoriasis" not in json.dumps(acq), "the search terms themselves are not copied into the ledger"
