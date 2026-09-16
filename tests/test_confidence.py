from __future__ import annotations

import hashlib
import json
from pathlib import Path

from bedrock.provenance import HistoryStore, RunCatalog


def _entry(path: Path, run_dir: Path) -> dict:
    return {
        "path": str(path.relative_to(run_dir)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def _confidence_run(
    tmp_path: Path,
    *,
    scores: list = [49.9, 50, 69.9, 70, 90],
    recorded_ranking: bool = True,
) -> tuple[RunCatalog, Path, dict]:
    work = tmp_path / "biotech"
    run_dir = work / "runs" / "run-confidence"
    output_dir = run_dir / "alphafold2" / "input"
    output_dir.mkdir(parents=True)
    structure = output_dir / "ranked_0.pdb"
    structure.write_text("ATOM\n")
    ranking = output_dir / "ranking_debug.json"
    ranking.write_text(json.dumps({"order": ["model_3_ptm", "model_1_ptm"]}))
    plddt = output_dir / "confidence_model_3_ptm.json"
    plddt.write_text(
        json.dumps(
            {
                "residueNumber": list(range(1, len(scores) + 1)),
                "confidenceScore": scores,
            }
        )
    )
    pae = output_dir / "pae_model_3_ptm.json"
    matrix = [[float((row * 5 + column) % 24) for column in range(5)] for row in range(5)]
    pae.write_text(
        json.dumps(
            [{"predicted_aligned_error": matrix, "max_predicted_aligned_error": 30.0}]
        )
    )
    source = run_dir / "source.fasta"
    source.write_text(">candidate\nACDEF\n")
    outputs = {
        "structure": _entry(structure, run_dir),
        "confidence": [_entry(plddt, run_dir), _entry(pae, run_dir)],
    }
    if recorded_ranking:
        outputs["ranking"] = _entry(ranking, run_dir)
    manifest = {
        "schema_version": 2 if recorded_ranking else 1,
        "run_id": "run-confidence",
        "created_at": "2026-07-01T00:00:00+00:00",
        "status": "completed",
        "source": {
            "snapshot": "source.fasta",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "residue_count": 5,
        },
        "alphafold2": {"selected_model": "model_3_ptm"} if recorded_ranking else {},
        "outputs": outputs,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    return RunCatalog(work, HistoryStore(tmp_path / "history")), run_dir, manifest


def test_confidence_pairs_ranked_structure_with_selected_model_and_preserves_pae_direction(
    tmp_path: Path,
):
    catalog, run_dir, _ = _confidence_run(tmp_path)

    report = catalog.confidence("run-confidence")

    assert report.model_id == "model_3_ptm"
    assert report.plddt == (49.9, 50.0, 69.9, 70.0, 90.0)
    assert report.low_regions == ((1, 3),)
    assert report.very_low_regions == ((1, 1),)
    assert report.pae is not None
    assert report.pae.values[0][1] == 1.0
    assert report.pae.values[1][0] == 5.0
    assert report.pae.ranges == ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5))
    assert report.pae.source_path == run_dir / "alphafold2/input/pae_model_3_ptm.json"
    assert report.notices == ()


def test_legacy_unrecorded_ranking_is_disclosed_without_rewriting_manifest(tmp_path: Path):
    catalog, run_dir, _ = _confidence_run(tmp_path, recorded_ranking=False)
    before = (run_dir / "manifest.json").read_bytes()

    report = catalog.confidence("run-confidence")

    assert report.model_id == "model_3_ptm"
    assert report.plddt is not None
    assert any("unrecorded local ranking" in notice for notice in report.notices)
    assert (run_dir / "manifest.json").read_bytes() == before


def test_changed_plddt_hash_does_not_hide_independently_valid_pae(tmp_path: Path):
    catalog, run_dir, _ = _confidence_run(tmp_path)
    (run_dir / "alphafold2/input/confidence_model_3_ptm.json").write_text(
        '{"residueNumber":[1,2,3,4,5],"confidenceScore":[99,99,99,99,99]}'
    )

    report = catalog.confidence("run-confidence")

    assert report.plddt is None
    assert report.pae is not None
    assert any("pLDDT" in notice and "hash" in notice for notice in report.notices)


def test_missing_or_ambiguous_model_pairing_suppresses_both_metrics(tmp_path: Path):
    catalog, run_dir, _ = _confidence_run(tmp_path, recorded_ranking=False)
    (run_dir / "alphafold2/input/ranking_debug.json").unlink()

    report = catalog.confidence("run-confidence")

    assert report.model_id is None
    assert report.plddt is None
    assert report.pae is None
    assert any("pair" in notice.lower() for notice in report.notices)


def test_invalid_plddt_keeps_valid_pae_available(tmp_path: Path):
    catalog, run_dir, manifest = _confidence_run(tmp_path, scores=[95, float("nan"), 80, 75, 60])
    plddt = run_dir / "alphafold2/input/confidence_model_3_ptm.json"
    manifest["outputs"]["confidence"][0] = _entry(plddt, run_dir)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    report = catalog.confidence("run-confidence")

    assert report.plddt is None
    assert report.pae is not None
    assert any("pLDDT" in notice and "invalid" in notice for notice in report.notices)


def test_invalid_pae_shape_keeps_valid_plddt_available(tmp_path: Path):
    catalog, run_dir, manifest = _confidence_run(tmp_path)
    pae = run_dir / "alphafold2/input/pae_model_3_ptm.json"
    pae.write_text(
        json.dumps(
            {"predicted_aligned_error": [[0, 1]], "max_predicted_aligned_error": 30}
        )
    )
    manifest["outputs"]["confidence"][1] = _entry(pae, run_dir)
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    report = catalog.confidence("run-confidence")

    assert report.plddt is not None
    assert report.pae is None
    assert any("PAE" in notice and "invalid" in notice for notice in report.notices)


def test_pae_block_maximum_keeps_sparse_high_error_visible(tmp_path: Path):
    catalog, run_dir, manifest = _confidence_run(tmp_path)
    count = 33
    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    matrix[32][32] = 29.0
    pae = run_dir / "alphafold2/input/pae_model_3_ptm.json"
    pae.write_text(
        json.dumps(
            {"predicted_aligned_error": matrix, "max_predicted_aligned_error": 31.75}
        )
    )
    plddt = run_dir / "alphafold2/input/confidence_model_3_ptm.json"
    plddt.write_text(
        json.dumps(
            {
                "residueNumber": list(range(1, count + 1)),
                "confidenceScore": [90] * count,
            }
        )
    )
    manifest["source"]["residue_count"] = count
    manifest["outputs"]["confidence"] = [_entry(plddt, run_dir), _entry(pae, run_dir)]
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    report = catalog.confidence("run-confidence")

    assert report.pae is not None
    assert len(report.pae.values) == 32
    assert report.pae.ranges[-1] == (32, 33)
    assert report.pae.values[-1][-1] == 29.0


def test_confidence_respects_the_run_recorded_sequence_limit(tmp_path: Path):
    catalog, run_dir, manifest = _confidence_run(tmp_path)
    manifest["limits"] = {"max_sequence_length": 4}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    report = catalog.confidence("run-confidence")

    assert report.model_id is None
    assert report.plddt is None
    assert any("between 1 and 4" in notice for notice in report.notices)
