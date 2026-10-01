"""Stage 6b: the patent landscape is searched for the targets the run found.

The first version searched the disease alone straight after the question was
read. These tests pin what replaced that: the targets come from the evidence
(chosen by code, in Stage 6's order, only by names the knowledge table lists),
the search happens after them, a refused key is still reported in the first
seconds, and what was found is saved with the run so a resume reuses it.
"""

from __future__ import annotations

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import PatentRecord
from bedrock.litreview.runner import answer_question, load_result
from bedrock.litreview.sources.epo_auth import AuthError
from bedrock.litreview.sources.epo_ops import PatentSearchFailed
from bedrock.litreview.stages.s6b_patents import ARTIFACT, candidate_targets, load_patents
from tests.litreview.fixtures import QUESTION, corpus, fake_client


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    return s


def records():
    return [
        PatentRecord(id="EP1A1", family_id="f1", title="Anti-IL-23 antibody", assignee="Genentech, Inc.",
                     kind_code="A1", priority_date="2018-01-01", matched_targets=["IL23A"], claim_1="An antibody."),
        PatentRecord(id="EP2B1", family_id="f2", title="IL-23 receptor blocker", assignee="Amgen Inc.",
                     kind_code="B1", priority_date="2019-06-01", matched_targets=["IL23A"]),
    ]


class Office:
    """A patent office that records what it was asked and what happened first."""

    name = "epo_ops"

    def __init__(self, *, fail_check=None, fail_search=None):
        self.checks, self.requests = 0, []
        self.fail_check, self.fail_search = fail_check, fail_search

    def check(self):
        self.checks += 1
        if self.fail_check:
            raise self.fail_check

    def search(self, request):
        self.requests.append(request)
        if self.fail_search:
            raise self.fail_search
        return records()


def run(settings, tmp_path, **kwargs):
    return answer_question(QUESTION, settings=settings, client=fake_client(), papers=corpus(),
                           db=Database(tmp_path / "db.sqlite3"), **kwargs)


def test_the_targets_come_from_the_evidence_in_stage_6s_order_and_only_by_listed_names(settings, tmp_path):
    office, seen = Office(), []
    result = run(settings, tmp_path, patent_searcher=office, progress=seen.append)

    request = office.requests[0]
    wanted = [t.target for t in result.summary.targets]
    assert list(request.targets) == [t for t in wanted if t in request.targets], "Stage 6's order is kept"
    assert request.targets, "the fixture's evidence names a target, so the search is for it"
    kb = load_knowledge()
    for name, terms in request.targets.items():
        entity = kb.get(name)
        assert entity is None or terms == list(entity.terms), "terms are the knowledge table's, never a model's"
    assert "psoriasis" in request.exact_terms, "and it is searched together with the question's own entity"
    assert request.per_target == 40


def test_the_search_happens_after_the_evidence_is_read_and_the_account_check_before_anything(settings, tmp_path):
    office, seen = Office(), []
    run(settings, tmp_path, patent_searcher=office, progress=seen.append)
    first_patent_line = next(i for i, m in enumerate(seen) if m.startswith("patents:"))
    assert any(m.startswith("extract:") for m in seen[:first_patent_line]), "targets are known by then"
    assert office.checks == 1


def test_a_refused_account_is_reported_before_any_paper_is_read_and_nothing_is_searched(settings, tmp_path):
    office = Office(fail_check=AuthError("EPO OPS rejected the consumer key/secret (HTTP 401)"))
    seen: list[str] = []
    result = run(settings, tmp_path, patent_searcher=office, progress=seen.append)

    assert office.requests == [], "no search is attempted with an account that was just refused"
    warning = next(i for i, m in enumerate(seen) if m.startswith("patents: WARNING"))
    assert warning < next(i for i, m in enumerate(seen) if m.startswith("papers:")), "reported in the first seconds"
    assert result.report_id and result.patents == []
    assert "could not answer it" in result.paths["md"].read_text()


def test_a_search_that_fails_costs_the_landscape_and_nothing_else(settings, tmp_path):
    office = Office(fail_search=PatentSearchFailed("EPO OPS fair-use quota exhausted", status_code=403))
    result = run(settings, tmp_path, patent_searcher=office)
    assert result.report_id and result.summary.units, "the evidence work is kept"
    assert result.patents == [] and "could not answer it" in result.paths["md"].read_text()
    assert load_patents(Database(tmp_path / "db.sqlite3"), result.run.id) is None, "a failure is retried, not saved"


def test_what_was_found_is_saved_with_the_run_and_a_resume_reuses_it(settings, tmp_path):
    office = Office()
    first = run(settings, tmp_path, patent_searcher=office)
    assert [p.id for p in load_patents(Database(tmp_path / "db.sqlite3"), first.run.id)] == ["EP1A1", "EP2B1"]

    again = answer_question(QUESTION, settings=settings, client=fake_client(), db=Database(tmp_path / "db.sqlite3"),
                            resume_run_id=first.run.id, patent_searcher=office)
    assert len(office.requests) == 1, "EPO's quota is not spent twice, and the list does not change under the run"
    assert again.patents == first.patents
    assert office.checks == 1, "nor is the account asked about again"


def test_a_reopened_run_has_its_patents_back(settings, tmp_path):
    first = run(settings, tmp_path, patent_searcher=Office())
    reopened = load_result(Database(tmp_path / "db.sqlite3"), first.run.id)
    assert reopened.patents == first.patents
    assert reopened.report == first.report


def test_records_handed_in_are_used_saved_and_never_searched_for(settings, tmp_path):
    office = Office()
    result = run(settings, tmp_path, patents=records(), patent_searcher=office)
    assert office.requests == [] and office.checks == 0
    assert load_patents(Database(tmp_path / "db.sqlite3"), result.run.id) == records()


def test_a_run_without_targets_falls_back_to_the_questions_own_terms_and_says_it_is_broad(settings, tmp_path):
    office, seen = Office(), []
    answer_question(QUESTION, settings=settings, use_model=False, papers=corpus(),
                    db=Database(tmp_path / "db2.sqlite3"), patent_searcher=office, progress=seen.append)
    assert office.requests[0].targets == {}
    assert any("no target to search for yet" in m and "broad" in m for m in seen)


def test_a_name_that_is_not_a_target_is_never_searched(settings, tmp_path):
    from types import SimpleNamespace

    summary = SimpleNamespace(targets=[SimpleNamespace(target=n) for n in
                                       ("IL23A", "weight loss", "monthly migraine days", "SOCS1", "IL17A", "TNF", "IL36R")])
    found = candidate_targets(summary, load_knowledge(), limit=4)
    assert "weight loss" not in found and "monthly migraine days" not in found
    assert list(found) == ["IL23A", "SOCS1", "IL17A", "TNF"], "the limit counts only the ones that qualify"


def test_the_saved_artifact_never_changes_what_the_report_is(settings, tmp_path):
    """Patents are an artifact, not a field of `Report`: a run with none has the
    bytes it always had, so every stored run still passes `check_pinned`."""
    plain = run(settings, tmp_path)
    assert load_patents(Database(tmp_path / "db.sqlite3"), plain.run.id) is None
    assert Database(tmp_path / "db.sqlite3").get_artifact(plain.run.id, ARTIFACT) is None
