"""`verify-run` says which rules it checked with, not only whether a run passed.

A stored run is re-proved with the code and tables installed now. These tests
change one part of that environment at a time and check that the assessment
names it - and that a changed environment is reported *beside* an intact
report, never as a violation of it. Tampered bytes and failed checks stay
violations, exactly as `verify_run` has always returned them.
"""

from __future__ import annotations

import json
import shutil

import pytest

from bedrock.litreview import runner
from bedrock.litreview.config import DEFAULTS_DIR, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.knowledge import Knowledge
from bedrock.litreview.report import verification_context as vc
from bedrock.litreview.report.invariants import assess_run, verify_run
from bedrock.litreview.runner import answer_question
from tests.litreview.fixtures import QUESTION, corpus, fake_client

CLEAN = vc.CodeIdentity("abc1234", "git-describe", False)


@pytest.fixture(autouse=True)
def clean_code(monkeypatch):
    """A developer's dirty checkout would make every comparison `not_exact`;
    the tests pin a clean identity and change it only where that is the point."""
    monkeypatch.setattr(vc, "code_identity", lambda: CLEAN)
    monkeypatch.setattr(runner, "code_identity", lambda: CLEAN)


@pytest.fixture()
def stored(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir, s.out_dir, s.db_path = tmp_path / "cache", tmp_path / "out", tmp_path / "db.sqlite3"
    db = Database(s.db_path)
    r = answer_question(QUESTION, settings=s, client=fake_client(), db=db, papers=corpus())
    return r, db, s


def _override_dir(tmp_path, **edits: str):
    """An override directory holding edited copies of packaged config files."""
    d = tmp_path / "override"
    d.mkdir(exist_ok=True)
    for name, extra in edits.items():
        shutil.copy(DEFAULTS_DIR / name.replace("_", "."), d / name.replace("_", "."))
        with open(d / name.replace("_", "."), "a", encoding="utf-8") as fh:
            fh.write(extra)
    return {"BEDROCK_LITREVIEW_CONFIG_DIR": str(d)}


def _contexts(db, run_id):
    return json.loads(db.get_artifact(run_id, vc.ARTIFACT))


def _rewrite(db, run_id, contexts):
    db.put_artifact(run_id, vc.ARTIFACT, json.dumps(contexts))


def _rows(db):
    return [db.conn.execute(f"SELECT * FROM {t} ORDER BY 1, 2").fetchall() for t in ("runs", "artifacts", "reports")]


# --- what a run records ------------------------------------------------------------------


def test_a_run_records_its_context_without_paths_or_a_different_report(stored):
    r, db, _ = stored
    [ctx] = _contexts(db, r.run.id)
    assert ctx["schema_version"] == 1 and ctx["stage_at_start"] == -1 and ctx["stage_at_end"] == 8
    assert set(ctx["inputs"]) >= {"config/models.yaml", "config/pipeline.yaml", "config/report.yaml",
                                  "knowledge/entities.yaml", "knowledge/meaning_words.yaml"}
    assert set(ctx["prompts"]) == set(vc.PROMPTS) and all(len(p["sha256"]) == 64 for p in ctx["prompts"].values())
    assert ctx["code"] == {"version": "abc1234", "method": "git-describe", "dirty": False}
    assert ctx["verifier_sha256"] and set(ctx["schemas"]) == {"schema/Report", "schema/EvidenceUnit",
                                                               "schema/Paper", "schema/Run"}
    # No absolute path from this machine goes into the record.
    assert str(DEFAULTS_DIR.parent) not in db.get_artifact(r.run.id, vc.ARTIFACT)
    # The report itself is unchanged by any of this: provenance lives beside it.
    assert "verification_context" not in db.get_artifact(r.run.id, "report")


def test_a_resumed_run_appends_its_context_and_keeps_the_first(stored, tmp_path):
    r, db, s = stored
    first = _contexts(db, r.run.id)[0]
    again = answer_question(QUESTION, settings=s, client=fake_client(), db=db, resume_run_id=r.run.id)
    contexts = _contexts(db, again.run.id)
    assert len(contexts) == 2 and contexts[0] == first
    assert contexts[1]["stage_at_start"] == 8 and contexts[1]["stage_at_end"] == 8


# --- the comparison, one change at a time ----------------------------------------------------


def test_an_unchanged_environment_is_said_to_be_unchanged(stored):
    r, db, _ = stored
    a = assess_run(db, r.run.id)
    assert (a.artifact_integrity, a.current_rule_check, a.environment.status) == ("pass", "pass", "unchanged")
    assert a.environment.differences == [] and a.violations == []
    assert a.summary().startswith("Pinned artifacts intact; current-rule checks pass; environment unchanged")
    assert a.exit_code() == 0 and a.exit_code(require_same_rules=True) == 0


def test_one_changed_knowledge_file_is_named_and_is_not_a_violation(stored, monkeypatch):
    r, db, _ = stored
    from bedrock.litreview.extract import verify

    monkeypatch.setattr(verify, "_meaning_words_bytes", lambda: b"meaning: {}\n# edited\n")
    a = assess_run(db, r.run.id)
    assert a.environment.status == "changed"
    assert a.environment.changed_roles == ["knowledge/meaning_words.yaml"]
    [d] = a.environment.differences
    assert d.kind == "changed" and d.affects == "generation+verification"
    assert a.violations == [] and a.current_rule_check == "pass"
    assert "environment changed in knowledge/meaning_words.yaml" in a.summary()
    assert a.exit_code() == 0 and a.exit_code(require_same_rules=True) == 1


def test_several_changed_config_files_are_each_named(stored, tmp_path):
    r, db, _ = stored
    env = _override_dir(tmp_path, pipeline_yaml="\n# tuned\n", report_yaml="\n# reworded\n")
    a = assess_run(db, r.run.id, env=env)
    by_role = {d.role: d for d in a.environment.differences}
    assert set(by_role) == {"config/pipeline.yaml", "config/report.yaml"}
    assert all(d.kind == "source_changed" and d.affects == "generation" for d in by_role.values())
    assert "not substituted" in by_role["config/pipeline.yaml"].reason


def test_an_override_identical_to_the_default_is_not_a_difference(stored, tmp_path):
    r, db, _ = stored
    env = _override_dir(tmp_path, pipeline_yaml="")
    assert assess_run(db, r.run.id, env=env).environment.status == "unchanged"


def test_a_prompt_edited_without_a_version_bump_is_caught(stored):
    r, db, _ = stored
    contexts = _contexts(db, r.run.id)
    contexts[-1]["prompts"]["extract"]["sha256"] = "0" * 64
    _rewrite(db, r.run.id, contexts)
    [d] = assess_run(db, r.run.id).environment.differences
    assert d.role == "prompt/extract" and d.kind == "changed"
    assert d.reason == "content changed without a version bump"


def test_different_code_is_a_different_environment_not_a_different_verdict(stored, monkeypatch):
    r, db, _ = stored
    monkeypatch.setattr(vc, "code_identity", lambda: vc.CodeIdentity("def5678", "git-describe", False))
    a = assess_run(db, r.run.id)
    [d] = a.environment.differences
    assert d.role == "code" and d.kind == "changed" and "not a different verdict" in d.reason
    assert a.violations == []


@pytest.mark.parametrize("identity, kind, status", [
    (vc.CodeIdentity(None, "unknown", None), "unknown_now", "incomplete"),
    (vc.CodeIdentity("abc1234-dirty", "git-describe", True), "changed", "changed"),
])
def test_unknown_and_dirty_code_identities_are_not_called_equal(stored, monkeypatch, identity, kind, status):
    r, db, _ = stored
    monkeypatch.setattr(vc, "code_identity", lambda: identity)
    a = assess_run(db, r.run.id)
    assert [d.kind for d in a.environment.differences] == [kind] and a.environment.status == status


def test_a_dirty_tree_that_matches_its_own_label_is_still_not_exact(stored, monkeypatch):
    r, db, _ = stored
    contexts = _contexts(db, r.run.id)
    contexts[-1]["code"] = {"version": "abc1234-dirty", "method": "git-describe", "dirty": True}
    _rewrite(db, r.run.id, contexts)
    monkeypatch.setattr(vc, "code_identity", lambda: vc.CodeIdentity("abc1234-dirty", "git-describe", True))
    a = assess_run(db, r.run.id)
    assert [d.kind for d in a.environment.differences] == ["not_exact"]
    assert a.environment.status == "incomplete" and a.exit_code(require_same_rules=True) == 1


def test_package_metadata_identity_compares_like_any_other(stored, monkeypatch):
    r, db, _ = stored
    contexts = _contexts(db, r.run.id)
    contexts[-1]["code"] = {"version": "0.1.0", "method": "package-metadata", "dirty": None}
    _rewrite(db, r.run.id, contexts)
    monkeypatch.setattr(vc, "code_identity", lambda: vc.CodeIdentity("0.1.0", "package-metadata", None))
    assert assess_run(db, r.run.id).environment.status == "unchanged"


def test_a_role_the_run_did_not_record_makes_the_comparison_incomplete(stored):
    r, db, _ = stored
    contexts = _contexts(db, r.run.id)
    del contexts[-1]["inputs"]["knowledge/meaning_words.yaml"]
    _rewrite(db, r.run.id, contexts)
    a = assess_run(db, r.run.id)
    [d] = a.environment.differences
    assert (d.role, d.kind) == ("knowledge/meaning_words.yaml", "not_recorded")
    assert a.environment.status == "incomplete" and a.exit_code() == 0 and a.exit_code(require_same_rules=True) == 1


def test_a_caller_supplied_knowledge_table_is_not_compared_with_the_default(stored):
    r, db, _ = stored
    from bedrock.litreview.knowledge import load_knowledge

    custom = Knowledge(list(load_knowledge().entities))  # same entries, built in memory: no fingerprint
    a = assess_run(db, r.run.id, kb=custom)
    [d] = a.environment.differences
    assert (d.role, d.kind) == ("knowledge/entities.yaml", "unknown_now") and "in memory" in d.reason
    assert a.current_rule_check == "pass"


def test_contexts_that_differ_between_invocations_are_noted(stored):
    r, db, _ = stored
    [ctx] = _contexts(db, r.run.id)
    older = json.loads(json.dumps(ctx))
    older["inputs"]["config/pipeline.yaml"]["sha256"] = "1" * 64
    older["stage_at_end"] = 5
    _rewrite(db, r.run.id, [older, ctx])
    a = assess_run(db, r.run.id)
    assert a.environment.status == "unchanged", "the last invocation, which wrote the report, matches"
    assert any("2 different environments" in n for n in a.environment.notes)


# --- legacy runs --------------------------------------------------------------------------------


def _legacy(db, run, **fields):
    """A run as code before `verification_context.v1` stored it."""
    with db.conn:
        db.conn.execute("DELETE FROM artifacts WHERE run_id = ? AND name = ?", (run.id, vc.ARTIFACT))
    db.save_run(run.model_copy(update=fields))


def test_a_legacy_run_is_compared_from_what_it_recorded(stored):
    r, db, _ = stored
    _legacy(db, db.get_run(r.run.id))
    a = assess_run(db, r.run.id)
    assert a.environment.legacy and a.environment.status == "incomplete"
    assert a.environment.differences == [], "its hashes, labels and code version all match"
    assert "prompt content not recorded in this legacy run" in a.summary()
    assert a.exit_code() == 0 and a.exit_code(require_same_rules=True) == 1


def test_a_legacy_run_with_no_hashes_says_so(stored):
    r, db, _ = stored
    _legacy(db, db.get_run(r.run.id), input_hashes={}, prompt_versions={}, code_version=None)
    a = assess_run(db, r.run.id)
    assert a.environment.status == "incomplete"
    assert "the run recorded no input hashes" in a.environment.notes
    assert {d.kind for d in a.environment.differences} == {"not_recorded"}


def test_a_legacy_override_that_is_gone_is_not_reopened_or_substituted(stored):
    r, db, _ = stored
    run = db.get_run(r.run.id)
    hashes = dict(run.input_hashes)
    del hashes["defaults/pipeline.yaml"]
    hashes["/home/someone/.config/bedrock/litreview/pipeline.yaml"] = "2" * 64
    hashes["corpus:/data/papers.jsonl"] = "3" * 64
    hashes["resumed:knowledge/entities.yaml"] = "4" * 64
    hashes["mystery-input"] = "5" * 64
    _legacy(db, run, input_hashes=hashes)
    by_role = {d.role: d for d in assess_run(db, r.run.id).environment.differences}
    assert by_role["config/pipeline.yaml"].kind == "source_changed"
    assert by_role["knowledge/entities.yaml"].kind == "changed", "the later (resumed) hash is compared"
    assert by_role["mystery-input"].kind == "unrecognised"
    assert not any(role.startswith("corpus") for role in by_role), "a corpus is pinned, not verifier policy"


def test_a_malformed_context_falls_back_to_the_legacy_fields_and_says_so(stored):
    r, db, _ = stored
    for body in ("not json", "[1, 2]", "[]"):
        db.put_artifact(r.run.id, vc.ARTIFACT, body)
        a = assess_run(db, r.run.id)
        assert a.environment.status == "incomplete" and a.violations == []
        assert "verification_context.v1 artifact" in a.environment.notes[0]


# --- the verdict is unchanged -------------------------------------------------------------------


def test_tampering_is_still_a_violation_whatever_the_environment(stored):
    r, db, _ = stored
    with db.conn:
        db.conn.execute("UPDATE paper_versions SET body = replace(body, 'mice', 'rats')")
    a = assess_run(db, r.run.id)
    assert a.artifact_integrity == "fail" and a.exit_code() == 1
    assert a.environment.status == "unchanged"
    assert a.violations == verify_run(db, r.run.id)


def test_a_tampered_report_is_caught(stored):
    r, db, _ = stored
    body = db.get_artifact(r.run.id, "report")
    db.put_artifact(r.run.id, "report", body.replace("mice", "rats", 1))
    a = assess_run(db, r.run.id)
    assert a.artifact_integrity == "fail" and any("report" in v for v in a.artifact_errors)


def test_a_run_whose_papers_are_missing_is_unavailable_not_passed(stored):
    r, db, _ = stored
    with db.conn:
        db.conn.execute("DELETE FROM paper_versions WHERE paper_id = ?", (r.report.evidence[0].paper_id,))
    a = assess_run(db, r.run.id)
    assert a.current_rule_check == "unavailable" and a.exit_code() == 1
    assert any(v.startswith("cannot be verified") for v in a.violations)


def test_the_list_wrapper_is_unchanged(stored):
    r, db, _ = stored
    assert verify_run(db, r.run.id) == [] == assess_run(db, r.run.id).violations


def test_assessing_reads_only(stored, monkeypatch):
    r, db, _ = stored
    import httpx

    def no_network(*args, **kwargs):
        raise AssertionError("verification must not reach the network")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(httpx.AsyncClient, "send", no_network)
    before = _rows(db)
    assess_run(db, r.run.id)
    assert _rows(db) == before


# --- the two front ends agree -------------------------------------------------------------------


def test_cli_exits_zero_on_drift_by_default_and_non_zero_when_strict(stored, capsys, monkeypatch):
    from bedrock.litreview.cli import main
    from bedrock.litreview.extract import verify

    r, db, _ = stored
    db_path = db.conn.execute("PRAGMA database_list").fetchone()[2]
    monkeypatch.setattr(verify, "_meaning_words_bytes", lambda: b"meaning: {}\n")
    assert main(["verify-run", r.run.id, "--db", db_path]) == 0
    out = capsys.readouterr().out
    assert f"{r.run.id}: OK" in out and "environment changed in knowledge/meaning_words.yaml" in out
    assert main(["verify-run", r.run.id, "--db", db_path, "--require-same-rules"]) == 1


def test_the_terminal_command_shows_the_same_answer(stored, monkeypatch):
    from bedrock.litreview.extract import verify
    from bedrock.ui.review_command import _verify_runs

    r, db, s = stored
    monkeypatch.setattr(verify, "_meaning_words_bytes", lambda: b"meaning: {}\n")
    [(run_id, violations, summary, changes, advisories)] = _verify_runs(s, r.run.id, False)
    a = assess_run(db, r.run.id)
    assert (run_id, violations, summary) == (r.run.id, a.violations, a.summary())
    assert changes == [d.line() for d in a.environment.differences]
    assert advisories == a.advisories
