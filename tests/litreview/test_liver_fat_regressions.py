"""Regressions from a real review: "Does tirzepatide reduce liver fat in adults
with obesity?" (qwen3:8b, 28 September 2026, the first report a scientist ran on
this machine).

Three separate losses turned 200 collected papers into a two-paper report whose
only numbers were p-values:

1. The first run had no model, so the question read as "tirzepatide, obesity";
   its search froze 200 papers that were never searched for liver fat. The
   second run read "liver fat" too, and silently reused that set.
2. "liver fat" was not in the knowledge table, so it matched verbatim only:
   papers about MASLD, NAFLD or MRI-PDFF were "not about the subject".
3. A Results finding filed under "obesity with MASLD" was dropped as being about
   "another disease" - obesity, which the question itself names. That dropped
   every finding with an effect size in it.

Plus one filter hole: a case report's background sentence ("has been also
reported to be beneficial") was cited as the evidence.
"""

import json

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.extract.verify import other_work_reason
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import ExtractedFinding
from bedrock.litreview.runner import answer_question
from bedrock.litreview.stages.s0_question import parse_question
from bedrock.litreview.stages.s1_collect import stale_collection, terms_signature
from bedrock.litreview.stages.s5_extract import entity_check
from tests.litreview.fixtures import QUESTION, corpus, fake_client

KB = load_knowledge()
LIVER = "Does tirzepatide reduce liver fat in adults with obesity?"


# --- 2. the knowledge table ------------------------------------------------------


def test_liver_fat_is_read_as_the_steatosis_concept_with_or_without_a_model():
    spec = parse_question(LIVER, None, knowledge=KB)
    by_name = {e.canonical: e for e in spec.entities}
    assert set(by_name) == {"hepatic steatosis", "obesity", "tirzepatide"}
    liver = by_name["hepatic steatosis"]
    assert liver.required and liver.resolved_from == "knowledge_table"
    for name in ("liver fat", "MASLD", "NAFLD", "hepatic steatosis", "MRI-PDFF", "intrahepatic lipid"):
        assert name in liver.exact_terms
    # Steatohepatitis passes or fails on its own words, never as a near-miss.
    assert not {"MASH", "NASH"} & set(liver.near_miss_terms)


# --- 3. the question's own diseases are never "another" disease ---------------------


def _finding(**kw) -> ExtractedFinding:
    base = dict(quote="x" * 30, target="tirzepatide", direction="block", modality="other",
                intervention="tirzepatide", disease="obesity with MASLD", evidence_type="human_clinical",
                species="patients", outcome="improved", setting="treatment", effect_detail="", n=None,
                combo_with=None, target_is_primary=True, facet="clinical")
    base.update(kw)
    return ExtractedFinding(**base)


def test_a_finding_filed_under_the_questions_own_population_is_kept():
    spec = parse_question(LIVER, None, knowledge=KB)
    quote = ("Regarding hepatic and lipid profiles, significant improvements were noted in AST, which "
             "decreased from 46.3 ± 18.5 U/L to 29.8 ± 10.1 U/L (−35.6%, p < 0.001).")
    drop, _ = entity_check(_finding(quote=quote), spec, KB)
    assert drop is None


def test_a_disease_the_question_does_not_name_is_still_another_disease():
    spec = parse_question(LIVER, None, knowledge=KB)
    drop, _ = entity_check(_finding(disease="type 2 diabetes"), spec, KB)
    assert drop == "other_disease:type 2 diabetes"


# --- the case report's background sentence ------------------------------------------


def test_someone_elses_reported_result_is_not_this_papers_finding():
    e8 = "Tirzepatide has been also reported to be beneficial in reducing liver fat content."
    assert other_work_reason(e8, "abstract") == "other_work:lead_in"
    assert other_work_reason("The p65/p50 heterodimer has been described to activate transcription.", "results")


def test_the_papers_own_work_in_the_present_perfect_is_kept():
    assert other_work_reason("We have shown that tirzepatide reduces liver fat.", "results") is None
    assert other_work_reason("In this study, liver fat has been reduced by 8.1%.", "results") is None
    assert other_work_reason("Body weight has been reduced by 12% after 3 months.", "results") is None


# --- 1. a frozen set is for the reading of the question it was searched for ---------


class _Searcher:
    name = "fake-search"
    last_report = None

    def __init__(self):
        self.calls = 0

    def search(self, request):
        self.calls += 1
        return corpus()


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    return s


def _ask(settings, db, **kw):
    return answer_question(QUESTION, settings=settings, client=fake_client(), db=db, **kw)


def test_the_same_reading_reuses_its_frozen_set(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    searcher = _Searcher()
    first = _ask(settings, db, searcher=searcher)
    again = _ask(settings, db, searcher=searcher)
    assert searcher.calls == 1
    assert again.run.papers_sha256 == first.run.papers_sha256
    assert json.loads(db.collected_for(first.spec.id))["kind"] == "search"


def test_a_different_reading_searches_again_when_a_search_is_asked_for(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    searcher = _Searcher()
    first = _ask(settings, db, searcher=searcher)
    subject = first.spec.required_entities[0].canonical
    _ask(settings, db, searcher=searcher, approved_terms={subject: ["an extra approved name"]})
    assert searcher.calls == 2


def test_a_different_reading_without_a_search_says_so_in_the_report(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = _ask(settings, db, searcher=_Searcher())
    subject = first.spec.required_entities[0].canonical
    second = _ask(settings, db, approved_terms={subject: ["an extra approved name"]})
    gaps = next(s for s in second.report.sections if s.id == "gaps")
    assert "Incomplete search" in gaps.body_markdown and "--search --refresh" in gaps.body_markdown
    # A resume of that run keeps saying it.
    resumed = answer_question(QUESTION, settings=settings, client=fake_client(), db=db, resume_run_id=second.run.id)
    assert "Incomplete search" in next(s for s in resumed.report.sections if s.id == "gaps").body_markdown
    # The same reading as the search said nothing of the sort.
    plain = next(s for s in first.report.sections if s.id == "gaps")
    assert "Incomplete search" not in plain.body_markdown


def test_a_set_frozen_before_signatures_were_recorded_is_traced_to_the_run_that_collected_it(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    # Papers handed in record no signature, like a database from before this change.
    first = _ask(settings, db, papers=corpus())
    assert db.collected_for(first.spec.id) is None
    subject = first.spec.required_entities[0].canonical
    second = _ask(settings, db, approved_terms={subject: ["an extra approved name"]})
    assert "fewer names" in next(s for s in second.report.sections if s.id == "gaps").body_markdown


def test_stale_collection_names_the_entity_the_old_search_never_looked_for():
    narrow = parse_question("Does tirzepatide reduce body weight in adults with obesity?", None, knowledge=KB)
    wide = parse_question(LIVER, None, knowledge=KB)
    note = stale_collection(terms_signature(narrow), wide)
    assert note and "**hepatic steatosis**" in note
    assert stale_collection(terms_signature(wide), wide) is None
    assert stale_collection(None, wide) is None


# --- the population says who, not what ---------------------------------------------


def test_a_disease_named_only_in_the_population_does_not_narrow_the_search():
    """Requiring "obesity" dropped SURPASS-3 MRI - the randomised trial that
    measured liver fat by MRI-PDFF, in people with type 2 diabetes - and the
    SYNERGY-NASH papers, from the search itself."""
    from bedrock.litreview.sources.base import SearchRequest
    from bedrock.litreview.sources.query import build_queries
    from bedrock.litreview.stages.s1_collect import search_request

    spec = parse_question(LIVER, None, knowledge=KB)
    assert spec.population == "adults with obesity"
    assert [e.canonical for e in spec.required_entities] == ["hepatic steatosis", "tirzepatide"]
    rendered = build_queries(search_request(spec))[0].render(lambda t: f'"{t}"')
    assert '"obesity"' not in rendered and '"MASLD"' in rendered and '"tirzepatide"' in rendered
    assert isinstance(search_request(spec), SearchRequest)


def test_the_population_disease_stays_required_when_it_is_the_only_disease():
    spec = parse_question("Does tirzepatide reduce body weight in adults with obesity?", None, knowledge=KB)
    assert {e.canonical for e in spec.required_entities} == {"obesity", "tirzepatide"}


def test_a_question_without_a_population_is_unchanged():
    spec = parse_question("If I want to prevent psoriasis, what should I target?", None, knowledge=KB)
    assert spec.population is None and [e.canonical for e in spec.required_entities] == ["psoriasis"]
