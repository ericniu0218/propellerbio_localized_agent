from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from .reference import Journal, artifact_path, low_regions, pae_overview, plddt_scores, selected_model


def test_journal_survives_restart_and_distinguishes_reingests(tmp_path):
    root = tmp_path / "history"
    journal = Journal(root)
    first = journal.append("op1", actor_id="alice", action="digest.ingest", state="intent")
    assert journal.append("op1", actor_id="alice", action="digest.ingest", state="intent") == first
    journal.append("op1", actor_id="alice", action="digest.ingest", state="succeeded",
                   subject_id="same-artifact")
    journal = Journal(root)
    journal.append("op2", actor_id="alice", action="digest.ingest", state="intent")
    journal.append("op2", actor_id="alice", action="digest.ingest", state="succeeded",
                   subject_id="same-artifact")
    assert len(journal.events()) == 4
    assert len(journal.events(after=first)) == 3
    assert root.stat().st_mode & 0o777 == 0o700
    assert journal.path.stat().st_mode & 0o777 == 0o600


def test_journal_rejects_changed_retries_and_orphaned_outcomes(tmp_path):
    journal = Journal(tmp_path)
    with pytest.raises(ValueError, match="matching"):
        journal.append("op", actor_id="a", action="approval", state="succeeded")
    journal.append("op", actor_id="a", action="approval", state="intent")
    journal.append("op", actor_id="a", action="approval", state="unknown")
    with pytest.raises(ValueError, match="different"):
        journal.append("op", actor_id="a", action="approval", state="succeeded")
    with pytest.raises(ValueError, match="different"):
        journal.append("op", actor_id="b", action="approval", state="intent")
    assert len(journal.events()) == 2


def test_journal_concurrent_duplicate_intent_records_once(tmp_path):
    journal = Journal(tmp_path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(lambda _: journal.append("op", actor_id="a", action="fold",
                                                    state="intent"), range(16)))
    assert len(set(ids)) == 1
    assert len(journal.events()) == 1


@pytest.mark.parametrize("details", [{"prompt": "private"}, {"source_bytes": []},
                                      {"error_code": "x" * 9000}])
def test_journal_rejects_unexpected_or_oversized_metadata(tmp_path, details):
    with pytest.raises(ValueError):
        Journal(tmp_path).append("op", actor_id="a", action="fold", state="intent", details=details)


def test_path_checks_hash_traversal_and_symlink_escape(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "file").write_text("hello")
    digest = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    assert artifact_path(run, "file", digest) == run / "file"
    with pytest.raises(ValueError, match="hash"):
        artifact_path(run, "file", "0" * 64)
    outside = tmp_path / "secret"
    outside.write_text("private")
    (run / "link").symlink_to(outside)
    for path in ("../secret", "link", str(outside)):
        with pytest.raises(ValueError):
            artifact_path(run, path)


def test_model_mapping_uses_rank_order():
    assert selected_model("ranked_0.cif", {"order": ["model_3_ptm", "model_1_ptm"]}) == "model_3_ptm"
    assert selected_model("ranked_1.pdb", {"order": ["model_3_ptm", "model_1_ptm"]}) == "model_1_ptm"
    assert selected_model("relaxed_model_2_ptm.pdb", {}) == "model_2_ptm"


@pytest.mark.parametrize("name,ranking", [("ranked_0.pdb", {}), ("ranked_1.cif", {"order": ["model_1"]}),
                                          ("ranked_0.pdb", {"order": ["../secret"]})])
def test_ambiguous_model_mapping_is_rejected(name, ranking):
    with pytest.raises(ValueError):
        selected_model(name, ranking)


def test_plddt_ranges_use_input_positions_and_correct_threshold_edges():
    scores = plddt_scores({"residueNumber": list(range(1, 7)),
                          "confidenceScore": [49.9, 50, 69.9, 70, 90, 10]}, 6)
    assert low_regions(scores) == ((1, 3), (6, 6))
    assert low_regions(scores, 50) == ((1, 1), (6, 6))


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [True], [-1], [101], ["95"], []])
def test_invalid_plddt_rejected(scores):
    with pytest.raises(ValueError):
        plddt_scores({"residueNumber": [1], "confidenceScore": scores}, 1)


def test_residue_number_mismatch_rejected():
    with pytest.raises(ValueError):
        plddt_scores({"residueNumber": [2], "confidenceScore": [90]}, 1)


def test_pae_preserves_asymmetry_and_small_high_error_regions():
    raw = [[0, 1, 2, 3, 4], [5, 0, 7, 8, 9], [10, 11, 0, 13, 14],
           [15, 16, 17, 0, 19], [20, 21, 22, 23, 0]]
    result = pae_overview([{"predicted_aligned_error": raw, "max_predicted_aligned_error": 30}], 5, 2)
    assert result["ranges"] == ((1, 2), (3, 5))
    assert result["values"] == ((5.0, 9.0), (21.0, 23.0))
    assert result["rows"] == "aligned residue"
    assert result["columns"] == "scored residue"


def test_pae_accepts_official_rounding_at_maximum():
    result = pae_overview([{"predicted_aligned_error": [[31.8]],
                            "max_predicted_aligned_error": 31.75}], 1)
    assert result["values"] == ((31.8,),)


@pytest.mark.parametrize("matrix", [[[0, 1]], [[0, -1], [0, 1]],
                                    [[0, float("nan")], [0, 1]], [[0, True], [0, 1]],
                                    [[0, 1000], [0, 1]]])
def test_invalid_pae_rejected(matrix):
    with pytest.raises(ValueError):
        pae_overview({"predicted_aligned_error": matrix, "max_predicted_aligned_error": 30}, 2)
