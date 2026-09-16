"""Deterministic AlphaFold confidence normalization.

This module pairs a chosen coordinate file with the model that produced it,
then validates the official pLDDT and PAE JSON. It never deserializes AlphaFold
pickle files and never treats arbitrary coordinate B-factors as confidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


MAX_CONFIDENCE_JSON_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_RESIDUES = 2500
MODEL_PATTERN = re.compile(r"^model_[A-Za-z0-9_]+$")


@dataclass(frozen=True)
class PaeOverview:
    values: tuple[tuple[float, ...], ...]
    ranges: tuple[tuple[int, int], ...]
    maximum_angstrom: float
    source_path: Path


@dataclass(frozen=True)
class ConfidenceReport:
    run_id: str
    model_id: str | None
    residue_count: int
    plddt: tuple[float, ...] | None
    low_regions: tuple[tuple[int, int], ...]
    very_low_regions: tuple[tuple[int, int], ...]
    pae: PaeOverview | None
    notices: tuple[str, ...]


def load_confidence_report(
    run_id: str,
    run_dir: Path,
    manifest: dict,
    *,
    max_residue_count: int = DEFAULT_MAX_RESIDUES,
) -> ConfidenceReport:
    """Read one run's confidence without changing its stored artifacts."""
    notices: list[str] = []
    residue_count = _residue_count(manifest)
    if residue_count < 1 or residue_count > max_residue_count:
        notices.append(
            f"Confidence is unavailable: residue count must be between 1 and {max_residue_count}."
        )
        return _empty_report(run_id, max(residue_count, 0), notices)

    recorded = {
        entry["path"]: entry
        for entry in _artifact_entries(manifest.get("outputs"))
        if isinstance(entry.get("path"), str)
    }
    structure_entry = _structure_entry(manifest)
    if structure_entry is None:
        notices.append("Confidence is unavailable: the run has no recorded structure.")
        return _empty_report(run_id, residue_count, notices)
    structure_path, structure_problem = _verified_path(
        run_dir, structure_entry["path"], structure_entry.get("sha256"), require_hash=True
    )
    if structure_problem is not None:
        notices.append(f"Confidence is unavailable: structure {structure_problem}.")
        return _empty_report(run_id, residue_count, notices)

    assert structure_path is not None
    model_id = _model_for_structure(
        structure_path, run_dir, manifest, recorded, notices
    )
    if model_id is None:
        notices.append("Confidence is unavailable because the structure could not be paired to a model.")
        return _empty_report(run_id, residue_count, notices)

    plddt: tuple[float, ...] | None = None
    low: tuple[tuple[int, int], ...] = ()
    very_low: tuple[tuple[int, int], ...] = ()
    plddt_path = structure_path.parent / f"confidence_{model_id}.json"
    plddt_payload, problem = _recorded_json(run_dir, plddt_path, recorded, label="pLDDT")
    if problem is not None:
        notices.append(problem)
    else:
        try:
            if not isinstance(plddt_payload, dict):
                raise ValueError("root must be an object")
            plddt = _plddt_scores(plddt_payload, residue_count)
            low = _low_regions(plddt, 70.0)
            very_low = _low_regions(plddt, 50.0)
        except ValueError as exc:
            notices.append(f"pLDDT data is invalid: {exc}.")

    pae: PaeOverview | None = None
    pae_path = structure_path.parent / f"pae_{model_id}.json"
    pae_payload, problem = _recorded_json(run_dir, pae_path, recorded, label="PAE")
    if problem is not None:
        notices.append(problem)
    else:
        try:
            pae = _pae_overview(pae_payload, residue_count, pae_path.resolve())
        except ValueError as exc:
            notices.append(f"PAE data is invalid: {exc}.")

    return ConfidenceReport(
        run_id=run_id,
        model_id=model_id,
        residue_count=residue_count,
        plddt=plddt,
        low_regions=low,
        very_low_regions=very_low,
        pae=pae,
        notices=tuple(notices),
    )


def summary_document(report: ConfidenceReport, manifest: dict) -> dict:
    """Return the small, versioned confidence artifact written for new runs."""
    sources: dict[str, dict] = {}
    for name, prefix in (("ranking", "ranking_debug"), ("plddt", "confidence_"), ("pae", "pae_")):
        for entry in _artifact_entries(manifest.get("outputs")):
            path = entry.get("path")
            if isinstance(path, str) and Path(path).name.startswith(prefix):
                if name == "ranking" or report.model_id is None or report.model_id in Path(path).name:
                    sources[name] = {
                        key: entry[key]
                        for key in ("path", "sha256", "bytes")
                        if key in entry
                    }
                    break
    pae = None
    if report.pae is not None:
        pae = {
            "values": [list(row) for row in report.pae.values],
            "ranges": [list(interval) for interval in report.pae.ranges],
            "maximum_angstrom": report.pae.maximum_angstrom,
            "aggregation": "block_max",
            "rows": "aligned residue",
            "columns": "scored residue",
        }
    return {
        "schema_version": 1,
        "run_id": report.run_id,
        "model_id": report.model_id,
        "residue_count": report.residue_count,
        "plddt": list(report.plddt) if report.plddt is not None else None,
        "low_regions": [list(interval) for interval in report.low_regions],
        "very_low_regions": [list(interval) for interval in report.very_low_regions],
        "pae": pae,
        "notices": list(report.notices),
        "sources": sources,
    }


def _empty_report(run_id: str, residue_count: int, notices: list[str]) -> ConfidenceReport:
    return ConfidenceReport(run_id, None, residue_count, None, (), (), None, tuple(notices))


def _residue_count(manifest: dict) -> int:
    source = manifest.get("source")
    value = source.get("residue_count") if isinstance(source, dict) else None
    return value if type(value) is int else 0


def _structure_entry(manifest: dict) -> dict | None:
    outputs = manifest.get("outputs")
    structure = outputs.get("structure") if isinstance(outputs, dict) else None
    if isinstance(structure, dict) and isinstance(structure.get("path"), str):
        return structure
    return None


def _model_for_structure(
    structure_path: Path,
    run_dir: Path,
    manifest: dict,
    recorded: dict[str, dict],
    notices: list[str],
) -> str | None:
    direct = re.fullmatch(
        r"(?:unrelaxed|relaxed)_(model_[A-Za-z0-9_]+)\.(?:pdb|cif|mmcif)",
        structure_path.name,
    )
    if direct:
        return direct.group(1)

    ranked = re.fullmatch(r"ranked_(\d+)\.(?:pdb|cif|mmcif)", structure_path.name)
    if ranked is None:
        return None
    ranking_path = structure_path.parent / "ranking_debug.json"
    try:
        ranking_relative = str(ranking_path.resolve().relative_to(run_dir.resolve()))
    except (OSError, ValueError):
        return None
    ranking_entry = recorded.get(ranking_relative)
    if ranking_entry is None:
        if not ranking_path.is_file() or ranking_path.is_symlink():
            return None
        notices.append("Model pairing was derived from unrecorded local ranking data.")
        try:
            ranking = _read_json(ranking_path)
        except ValueError:
            return None
    else:
        verified, problem = _verified_path(
            run_dir, ranking_relative, ranking_entry.get("sha256"), require_hash=True
        )
        if problem is not None or verified is None:
            notices.append(f"Ranking data could not be verified: {problem}.")
            return None
        try:
            ranking = _read_json(verified)
        except ValueError:
            return None

    if not isinstance(ranking, dict) or not isinstance(ranking.get("order"), list):
        return None
    index = int(ranked.group(1))
    order = ranking["order"]
    if index >= len(order) or not isinstance(order[index], str):
        return None
    selected = order[index]
    if MODEL_PATTERN.fullmatch(selected) is None:
        return None
    declared = manifest.get("alphafold2", {}).get("selected_model")
    if declared is not None and declared != selected:
        notices.append("Declared selected model disagrees with the recorded ranking.")
        return None
    return selected


def _recorded_json(
    run_dir: Path,
    candidate: Path,
    recorded: dict[str, dict],
    *,
    label: str,
) -> tuple[Any | None, str | None]:
    try:
        relative = str(candidate.resolve().relative_to(run_dir.resolve()))
    except (OSError, ValueError):
        return None, f"{label} data path is invalid"
    entry = recorded.get(relative)
    if entry is None:
        return None, f"{label} data is missing or has no recorded hash"
    path, problem = _verified_path(run_dir, relative, entry.get("sha256"), require_hash=True)
    if problem is not None or path is None:
        return None, f"{label} data {problem}"
    try:
        return _read_json(path), None
    except ValueError as exc:
        return None, f"{label} data is invalid: {exc}"


def _verified_path(
    run_dir: Path,
    relative: str,
    expected_sha256: Any,
    *,
    require_hash: bool,
) -> tuple[Path | None, str | None]:
    try:
        candidate = Path(relative)
        if candidate.is_absolute():
            return None, "path is absolute"
        unresolved = run_dir / candidate
        if unresolved.is_symlink():
            return None, "path is a symlink"
        resolved = unresolved.resolve(strict=True)
        resolved.relative_to(run_dir.resolve())
        if not resolved.is_file():
            return None, "is not a regular file"
        if require_hash and (
            not isinstance(expected_sha256, str)
            or re.fullmatch(r"[a-f0-9]{64}", expected_sha256) is None
        ):
            return None, "has no valid recorded hash"
        if expected_sha256 is not None:
            with resolved.open("rb") as source:
                actual = hashlib.file_digest(source, "sha256").hexdigest()
            if actual != expected_sha256:
                return None, "hash differs from the manifest"
        return resolved, None
    except (OSError, ValueError):
        return None, "path is missing or escapes the run directory"


def _read_json(path: Path) -> Any:
    try:
        size = path.stat().st_size
        if size > MAX_CONFIDENCE_JSON_BYTES:
            raise ValueError("file exceeds the 128 MiB limit")
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("file is invalid JSON") from exc
    except UnicodeDecodeError as exc:
        raise ValueError("file is not UTF-8") from exc
    except OSError as exc:
        raise ValueError("file is unreadable") from exc


def _plddt_scores(payload: dict, residue_count: int) -> tuple[float, ...]:
    positions = payload.get("residueNumber")
    if (
        not isinstance(positions, list)
        or any(type(position) is not int for position in positions)
        or positions != list(range(1, residue_count + 1))
    ):
        raise ValueError("residue numbers do not match the input sequence")
    scores = payload.get("confidenceScore")
    if not isinstance(scores, list) or len(scores) != residue_count:
        raise ValueError("score count does not match the input sequence")
    return tuple(_number(score, 0.0, 100.0) for score in scores)


def _low_regions(scores: tuple[float, ...], threshold: float) -> tuple[tuple[int, int], ...]:
    regions: list[tuple[int, int]] = []
    start: int | None = None
    for position, score in enumerate((*scores, 100.0), start=1):
        if score < threshold and start is None:
            start = position
        elif score >= threshold and start is not None:
            regions.append((start, position - 1))
            start = None
    return tuple(regions)


def _pae_overview(
    payload: Any,
    residue_count: int,
    source_path: Path,
    *,
    bins: int = 32,
) -> PaeOverview:
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("expected one monomer PAE record")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("PAE root must be an object")
    matrix = payload.get("predicted_aligned_error")
    maximum = _number(payload.get("max_predicted_aligned_error"), 0.01, 1000.0)
    if not isinstance(matrix, list) or len(matrix) != residue_count:
        raise ValueError("matrix height does not match the input sequence")
    size = min(bins, residue_count)
    edges = [index * residue_count // size for index in range(size + 1)]
    assignment = [block for block in range(size) for _ in range(edges[block], edges[block + 1])]
    grid = [[0.0] * size for _ in range(size)]
    for row_index, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != residue_count:
            raise ValueError("PAE must be a square matrix matching the sequence")
        for column_index, raw in enumerate(row):
            value = _number(raw, 0.0, maximum + 0.051)
            output_row = assignment[row_index]
            output_column = assignment[column_index]
            grid[output_row][output_column] = max(grid[output_row][output_column], value)
    ranges = tuple((edges[index] + 1, edges[index + 1]) for index in range(size))
    return PaeOverview(
        values=tuple(tuple(row) for row in grid),
        ranges=ranges,
        maximum_angstrom=maximum,
        source_path=source_path,
    )


def _number(value: Any, lower: float, upper: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("values must be finite numbers")
    if not lower <= value <= upper:
        raise ValueError("value is outside its declared range")
    return float(value)


def _artifact_entries(value: Any) -> Iterable[dict]:
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            yield value
        else:
            for nested in value.values():
                yield from _artifact_entries(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _artifact_entries(nested)
