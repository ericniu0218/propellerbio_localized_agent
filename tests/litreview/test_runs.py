"""A saved run is pinned to the exact paper versions it read.

Paper records are replaced in place when a question's papers are refreshed.
These tests prove that a finished run reopens, and a stopped run resumes, on
the papers it was built from — not on whatever the question holds now — and
that a corpus file edited after it was written is refused.
"""

import hashlib
import json
import logging
import re
import sqlite3

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import _SCHEMA, Database, paper_sha256
from bedrock.litreview.models import Paper
from bedrock.litreview.runner import answer_question, load_result
from bedrock.litreview.sources.base import SearchRequest
from bedrock.litreview.sources.corpus import CorpusSearcher, sidecar_path, write_papers
from tests.litreview.fixtures import QUESTION, corpus, fake_client


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir = tmp_path / "cache"
    s.out_dir = tmp_path / "out"
    s.db_path = tmp_path / "db.sqlite3"
    return s


def run(settings, db, **kwargs):
    kwargs.setdefault("papers", corpus())
    return answer_question(QUESTION, settings=settings, client=fake_client(), db=db, **kwargs)


def edited_corpus() -> list[Paper]:
    """The same papers, but P1 now reports a different number."""
    papers = corpus()
    p1 = papers[0]
    p1.abstract = p1.abstract.replace("by 60%", "by 35%")
    p1.sections = {k: v.replace("by 60%", "by 35%") for k, v in p1.sections.items()}
    assert "35%" in p1.abstract
    return papers


def section_text(paper: Paper, section: str) -> str:
    text = paper.abstract if section == "abstract" else paper.sections.get(section, "")
    return re.sub(r"\s+", " ", text or "")


def test_a_finished_run_reopens_on_its_own_papers_after_a_refresh(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = run(settings, db)
    assert any("60%" in u.quote for u in first.summary.units)

    run(settings, db, papers=edited_corpus(), refresh_papers=True)
    current = {p.id: p for p in db.get_papers(first.spec.id)}
    assert "35%" in current["P1"].abstract, "the question's current set did change"

    again = load_result(db, first.run.id)
    by_id = {p.id: p for p in again.papers}
    assert "60%" in by_id["P1"].abstract and "35%" not in by_id["P1"].abstract
    assert [p.id for p in again.papers] == [p.id for p in first.papers]
    for unit in again.summary.units:
        assert re.sub(r"\s+", " ", unit.quote) in section_text(by_id[unit.paper_id], unit.section)


def test_a_resumed_run_uses_its_pinned_papers_not_the_refreshed_set(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = run(settings, db)

    # Stop it after Stage 3, then refresh the question's papers underneath it.
    stopped = db.get_run(first.run.id)
    stopped.stage_reached = 3
    db.save_run(stopped)
    db.save_papers(first.spec.id, edited_corpus())

    resumed = answer_question(
        QUESTION, settings=settings, client=fake_client(), db=db, resume_run_id=first.run.id,
    )
    p1 = {p.id: p for p in resumed.papers}["P1"]
    assert "60%" in p1.abstract and "35%" not in p1.abstract
    assert resumed.run.papers_sha256 == first.run.papers_sha256
    assert resumed.report_id == first.report_id


def test_a_run_without_a_manifest_warns_and_falls_back(settings, tmp_path, caplog):
    db = Database(tmp_path / "db.sqlite3")
    first = run(settings, db)
    with db.conn:
        db.conn.execute("DELETE FROM artifacts WHERE run_id = ? AND name = 'papers_manifest'", (first.run.id,))
    db.save_papers(first.spec.id, edited_corpus())
    with caplog.at_level(logging.WARNING, logger="bedrock.litreview.runner"):
        again = load_result(db, first.run.id)
    assert "unpinned" in caplog.text
    assert "35%" in {p.id: p for p in again.papers}["P1"].abstract


def test_identical_runs_pin_identical_papers_and_record_their_inputs(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = run(settings, db)
    second = run(settings, db)
    assert first.run.papers_sha256 and first.run.papers_sha256 == second.run.papers_sha256
    assert first.run.id != second.run.id

    manifest = db.get_artifact(first.run.id, "papers_manifest")
    assert hashlib.sha256(manifest.encode("utf-8")).hexdigest() == first.run.papers_sha256
    assert json.loads(manifest) == [[p.id, paper_sha256(p)] for p in first.papers]

    hashes = first.run.input_hashes
    assert "knowledge/entities.yaml" in hashes
    assert {"defaults/pipeline.yaml", "defaults/report.yaml", "defaults/models.yaml"} <= set(hashes)
    assert hashes == second.run.input_hashes
    # Provenance goes on the Run, never into the report.
    assert first.report_id == second.report_id
    saved = json.loads((first.paths["json"].parent / "run.json").read_text())
    assert saved["papers_sha256"] == first.run.papers_sha256


def test_reusing_the_frozen_set_pins_it_too(settings, tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    first = run(settings, db)
    reused = answer_question(QUESTION, settings=settings, client=fake_client(), db=db)
    assert reused.run.papers_sha256 == first.run.papers_sha256
    assert db.get_artifact(reused.run.id, "papers_manifest") == db.get_artifact(first.run.id, "papers_manifest")


def test_paper_versions_are_kept_and_a_missing_one_is_named(tmp_path):
    db = Database(tmp_path / "db.sqlite3")
    original, edited = corpus(), edited_corpus()
    m1 = db.save_papers("q", original)
    m2 = db.save_papers("q", edited)
    assert m1[0][0] == m2[0][0] == "P1" and m1[0][1] != m2[0][1]
    assert m1[1:] == m2[1:], "unchanged papers keep their hash"
    assert db.get_paper_versions(m1)[0].abstract == original[0].abstract
    assert db.get_paper_versions(m2)[0].abstract == edited[0].abstract
    assert [paper_sha256(p) for p in db.get_paper_versions(m1)] == [sha for _, sha in m1]
    with pytest.raises(ValueError, match="P1"):
        db.get_paper_versions([("P1", "0" * 64)])


def test_an_old_database_is_backfilled_on_open(tmp_path):
    path = tmp_path / "old.sqlite3"
    old_schema = re.sub(r"CREATE TABLE IF NOT EXISTS paper_versions \(.*?\);\n", "", _SCHEMA, flags=re.S)
    assert "paper_versions" not in old_schema
    paper = corpus()[0]
    body = paper.model_dump_json()
    conn = sqlite3.connect(str(path))
    conn.executescript(old_schema)
    with conn:
        conn.execute("INSERT INTO papers (id, body) VALUES (?, ?)", (paper.id, body))
    conn.close()

    db = Database(path)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    assert db.get_paper_versions([(paper.id, sha)])[0] == paper
    db.close()
    rows = sqlite3.connect(str(path)).execute("SELECT COUNT(*) FROM paper_versions").fetchone()[0]
    assert rows == 1, "a second open does not duplicate the backfill"
    Database(path).close()
    assert sqlite3.connect(str(path)).execute("SELECT COUNT(*) FROM paper_versions").fetchone()[0] == 1


def test_a_corpus_file_carries_its_hash_and_an_edit_is_refused(tmp_path):
    path = tmp_path / "papers.jsonl"
    write_papers(corpus(), path)
    sidecar = sidecar_path(path)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sidecar.read_text().strip() == expected

    searcher = CorpusSearcher(path)
    assert searcher.sha256 == expected
    assert [p.id for p in searcher.search(SearchRequest(question="q", max_results=100))] == [p.id for p in corpus()]

    path.write_text(path.read_text().replace("60%", "35%"))
    with pytest.raises(ValueError, match="corpus file changed since it was written"):
        CorpusSearcher(path)

    sidecar.unlink()
    assert CorpusSearcher(path).sha256 == hashlib.sha256(path.read_bytes()).hexdigest(), \
        "a corpus with no sidecar is read as it is"


def test_a_run_from_a_corpus_file_records_its_hash(settings, tmp_path):
    path = tmp_path / "papers.jsonl"
    write_papers(corpus(), path)
    result = answer_question(
        QUESTION, settings=settings, client=fake_client(), db=Database(tmp_path / "db.sqlite3"),
        searcher=CorpusSearcher(path),
    )
    assert result.run.input_hashes[f"corpus:{path}"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_a_resume_keeps_the_hashes_its_earlier_stages_read():
    """The first hash of each input is what stages 0-5 read; a change seen on
    resume is recorded beside it, never over it."""
    from bedrock.litreview.models import Run
    from bedrock.litreview.runner import _record_inputs

    run = Run(id="r", question_id="q", started="t")
    _record_inputs(run)
    key = next(k for k in run.input_hashes if k.endswith("entities.yaml"))
    run.input_hashes[key] = "an-older-version"
    _record_inputs(run)
    assert run.input_hashes[key] == "an-older-version"
    assert run.input_hashes[f"resumed:{key}"] != "an-older-version"


def test_read_papers_refuses_an_edited_corpus_file(tmp_path):
    """The benchmark reads corpora with `read_papers`, not `CorpusSearcher`."""
    from bedrock.litreview.sources.corpus import read_papers

    path = tmp_path / "papers.jsonl"
    write_papers(corpus()[:2], path)
    assert len(read_papers(path)) == 2
    path.write_text(path.read_text().replace("psoriasis", "eczema"))
    with pytest.raises(ValueError, match="changed since it was written"):
        read_papers(path)


def test_two_identical_runs_each_report_the_report_they_wrote(settings, tmp_path):
    """A report id is the hash of the report, so both runs share one row.

    Looking the report up by run id therefore called the earlier of two
    identical runs unfinished, which is what `/reviews` and `bedrock-litreview
    runs` show.
    """
    db = Database(tmp_path / "db.sqlite3")
    first = run(settings, db)
    second = run(settings, db)
    assert first.run.id != second.run.id
    assert first.report_id == second.report_id

    assert db.report_id_for(first.run.id) == first.report_id
    assert db.report_id_for(second.run.id) == second.report_id

    # Both are listed. Their order is not asserted: two runs started in the same
    # second tie on `started`, and recent_runs then orders them by id.
    assert {r.id for r in db.recent_runs()} == {first.run.id, second.run.id}


def test_a_mistyped_resume_id_is_refused_not_started_fresh(settings, tmp_path):
    from bedrock.litreview.stages.s0_question import QuestionError

    db = Database(tmp_path / "db.sqlite3")
    with pytest.raises(QuestionError, match="no run 'nope' to resume"):
        run(settings, db, resume_run_id="nope")
    assert db.conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 0


def test_a_search_that_finds_nothing_is_refused_not_reported(settings, tmp_path):
    from bedrock.litreview.stages.s1_collect import NothingFound

    with pytest.raises(NothingFound, match="no papers were found"):
        run(settings, Database(tmp_path / "db.sqlite3"), papers=[])
