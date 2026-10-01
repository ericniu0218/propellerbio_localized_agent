"""Deleting one review run's output, and what must survive it.

`/forget` for a chat deletes everything about that conversation. A review
cannot work that way, because a run is not self-contained: the papers it read
are shared with every other run of the same question, and `paper_versions` is
append-only precisely so that any past run can be re-proved against the exact
bytes it saw.

So the two halves of this file are "the run really is gone" and "nothing else
is". The second is the one worth having: the failure it guards against is
silent. Deleting a run and taking its papers with it would leave a *different*
run looking finished and verifiable right up until somebody ran `verify-run` on
it, possibly months later.
"""

from __future__ import annotations

import pytest

from bedrock.litreview.api import forget_run
from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.models import Rating
from bedrock.litreview.report.invariants import verify_run
from bedrock.litreview.runner import answer_question
from tests.litreview.fixtures import QUESTION, corpus, fake_client

OTHER_QUESTION = "If I want to prevent psoriasis, what should I target?"


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.data_dir = tmp_path / "data"
    s.cache_dir = tmp_path / "data" / "cache"
    s.out_dir = tmp_path / "data" / "reports"
    s.db_path = tmp_path / "data" / "litreview.sqlite3"
    return s


def ask(settings, db, question=QUESTION, **kwargs):
    return answer_question(
        question, settings=settings, client=fake_client(), papers=corpus(), db=db, **kwargs
    )


@pytest.fixture()
def db(settings):
    return Database(settings.db_path)


def test_the_run_its_artifacts_and_its_report_are_all_gone(settings, db):
    result = ask(settings, db)
    run_id, report_id = result.run.id, result.report_id
    assert db.get_artifact(run_id, "report") is not None

    forgotten = forget_run(run_id, db=db, settings=settings)

    assert forgotten is not None
    assert db.get_run(run_id) is None
    assert db.get_report(report_id) is None
    assert db.get_artifact(run_id, "report") is None
    assert forgotten.deleted.artifacts > 0, "a finished run stores one artifact per stage"
    assert forgotten.deleted.report_deleted


def test_the_rendered_files_and_their_directory_go_with_it(settings, db):
    result = ask(settings, db)
    directory = result.paths["json"].parent
    assert (directory / "run.json").exists()

    forgotten = forget_run(result.run.id, db=db, settings=settings)

    assert {p.name for p in forgotten.files_removed} == {
        "report.json", "report.md", "report.html", "run.json",
    }
    assert not directory.exists(), "the empty directory should not be left behind"


def test_a_file_somebody_else_put_there_is_left_and_keeps_the_directory(settings, db):
    """The path is built from values out of the database, so this deletes files
    it knows the names of rather than the directory tree. A scientist's own note
    beside a report is not ours to remove."""
    result = ask(settings, db)
    directory = result.paths["json"].parent
    note = directory / "my-notes.txt"
    note.write_text("read this one again")

    forget_run(result.run.id, db=db, settings=settings)

    assert note.exists()
    assert directory.exists()
    assert not (directory / "report.html").exists()


def test_the_question_and_its_papers_survive_so_it_can_be_asked_again(settings, db):
    result = ask(settings, db)
    question_id = result.run.question_id

    forget_run(result.run.id, db=db, settings=settings)

    assert db.get_question(question_id) is not None
    assert db.get_papers(question_id), "the question's frozen paper set is not this run's to delete"

    again = ask(settings, db)
    assert again.report_id, "the same question still answers"


def test_the_papers_another_run_pinned_are_never_removed(settings, db):
    """The silent failure this file exists for. Two runs of the same question
    read the same paper versions; deleting one run must leave the other
    re-provable, byte for byte."""
    kept = ask(settings, db)
    doomed = ask(settings, db)
    assert kept.run.id != doomed.run.id

    manifest = db.get_run(kept.run.id).papers_sha256
    forget_run(doomed.run.id, db=db, settings=settings)

    assert db.get_run(kept.run.id).papers_sha256 == manifest
    assert verify_run(db, kept.run.id) == [], "the surviving run must still re-prove"


def test_a_report_two_runs_share_is_kept_for_the_other_one(settings, db):
    """A report id is the hash of the report body, and an identical re-run is
    *designed* to produce identical bytes — so both runs point at one row and one
    directory. Deleting it with the first run would leave the second finished and
    reportless."""
    first = ask(settings, db)
    second = ask(settings, db)
    assert first.report_id == second.report_id, "identical runs are supposed to match"
    assert first.run.id != second.run.id

    forgotten = forget_run(first.run.id, db=db, settings=settings)

    assert not forgotten.deleted.report_deleted
    assert forgotten.deleted.report_shared_with == (second.run.id,)
    assert forgotten.files_removed == ()
    assert db.get_report(first.report_id) is not None
    assert second.paths["html"].exists(), "the other run's report is still on disk"
    assert db.get_run(second.run.id) is not None

    # ...and deleting the second one now does remove it, since nothing needs it.
    after = forget_run(second.run.id, db=db, settings=settings)
    assert after.deleted.report_deleted
    assert db.get_report(second.report_id) is None
    assert not second.paths["html"].exists()


def test_another_questions_run_is_untouched(settings, db):
    mine = ask(settings, db)
    theirs = ask(settings, db, question=OTHER_QUESTION)

    forget_run(mine.run.id, db=db, settings=settings)

    assert db.get_run(theirs.run.id) is not None
    assert db.get_report(theirs.report_id) is not None
    assert theirs.paths["html"].exists()
    assert verify_run(db, theirs.run.id) == []


def test_ratings_of_a_deleted_report_go_with_it(settings, db):
    """A rating of a report that no longer exists says nothing about anything."""
    result = ask(settings, db)
    db.add_rating(Rating(
        report_id=result.report_id, rater="jd", score=4, created="2026-09-26T00:00:00+00:00",
    ))
    assert len(db.ratings(result.report_id)) == 1

    forgotten = forget_run(result.run.id, db=db, settings=settings)

    assert forgotten.deleted.ratings == 1
    assert db.ratings(result.report_id) == []


def test_an_unfinished_run_has_no_report_to_delete(settings, db):
    from bedrock.litreview.models import Run

    db.save_run(Run(id="halfway", question_id="q", started="2026-09-26T00:00:00+00:00", stage_reached=3))
    db.put_artifact("halfway", "gates", "[]")

    forgotten = forget_run("halfway", db=db, settings=settings)

    assert forgotten.deleted.report_id is None
    assert not forgotten.deleted.report_deleted
    assert forgotten.files_removed == ()
    assert db.get_run("halfway") is None
    assert db.get_artifact("halfway", "gates") is None


def test_an_unknown_run_id_is_reported_and_nothing_is_deleted(settings, db):
    result = ask(settings, db)

    assert forget_run("not-a-run-id", db=db, settings=settings) is None

    assert db.get_run(result.run.id) is not None
    assert result.paths["html"].exists()


def test_runs_for_report_reads_the_runs_not_the_reports_row(settings, db):
    """`save_report` is INSERT OR REPLACE, so the row's own `run_id` names
    whichever run wrote it last. Asking the row would therefore miss the earlier
    run — which is exactly the sibling that must not lose its report."""
    first = ask(settings, db)
    second = ask(settings, db)
    assert db.runs_for_report(first.report_id) == sorted([first.run.id, second.run.id])
    assert db.runs_for_report("no-such-report") == []
