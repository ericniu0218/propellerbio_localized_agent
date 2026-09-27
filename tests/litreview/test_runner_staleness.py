"""`stale_reasons` — a note, not a violation, about the rules moving since a run.

`verify-run` re-proves that a stored report is still faithful to the papers
it pinned. It says nothing about whether the *code* that would write a report
today, or the knowledge tables and config it would read, have since changed —
so a report that is perfectly sound can still be out of date with today's
rules. These tests check that `stale_reasons` notices exactly that change,
and only that: a fresh run must come back clean, a run predating the field
must stay silent rather than guess, and a changed `code_version` or a changed
entry in `input_hashes` must each produce a human-readable note.
"""

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.runner import answer_question, stale_reasons
from tests.litreview.fixtures import QUESTION, corpus, fake_client


def _run(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    db = Database(s.db_path)
    r = answer_question(QUESTION, settings=s, client=fake_client(), db=db, papers=corpus())
    return r, db


def test_a_fresh_run_has_no_stale_reasons(tmp_path):
    r, _db = _run(tmp_path)
    assert r.run.code_version is not None, "a fresh run always records the code version it ran under"
    assert stale_reasons(r.run) == []


def test_a_run_predating_the_field_stays_silent(tmp_path):
    # A run written before this check existed has no code_version recorded at
    # all. It must not be treated as "everything changed" - that would be a
    # false alarm on every old run in the store, forever.
    r, _db = _run(tmp_path)
    r.run.code_version = None
    assert stale_reasons(r.run) == []


def test_a_changed_code_version_is_reported(tmp_path):
    r, _db = _run(tmp_path)
    r.run.code_version = "not-the-real-version"
    reasons = stale_reasons(r.run)
    assert any("code changed" in reason for reason in reasons)


def test_a_changed_input_hash_is_reported(tmp_path):
    r, _db = _run(tmp_path)
    assert r.run.input_hashes, "a fresh run records hashes of the config and knowledge files it read"
    key = next(iter(r.run.input_hashes))
    r.run.input_hashes[key] = "0" * 64
    reasons = stale_reasons(r.run)
    assert any(key in reason and "changed since this run" in reason for reason in reasons)
