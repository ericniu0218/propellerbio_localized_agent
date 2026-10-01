"""Tested building blocks for COMPLIANCE_MVP.md, not integrated application code.

Agent A should adapt these into provenance/ and biotech/confidence.py. These
functions deliberately do not implement the whole MVP or a compliance system.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


class Journal:
    """One connection per call supports CLI processes and asyncio.to_thread.

    Commit intent before an operation. A separate terminal event records what
    actually happened. A missing outcome is uncertainty, not evidence of failure.
    Retries of the SAME event return its sequence; changed retries are rejected.
    This is local history: the OS-account owner can alter or delete the database.
    """

    ACTIONS = {
        "fold", "digest.ingest", "digest.forget", "approval", "run.tag",
        "litreview.review", "litreview.search",
    }
    STATES = {"intent", "succeeded", "failed", "cancelled", "unknown"}
    FIELDS = {
        "artifact_id", "run_id", "project_id", "session_id", "task_id",
        "decision", "error_code", "manifest_sha256", "source_sha256",
        "source_kind", "source_bytes", "actor_source",
    }

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)  # Must be the dedicated history directory.
        self.path = root / "history.sqlite3"
        if self.path.is_symlink():
            raise ValueError("History database must not be a symlink")
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(fd)
        self.path.chmod(0o600)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                operation_id TEXT NOT NULL,
                slot TEXT NOT NULL CHECK(slot IN ('intent', 'outcome')),
                recorded_at TEXT NOT NULL,
                body TEXT NOT NULL,
                UNIQUE(operation_id, slot)
            )""")

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.execute("PRAGMA synchronous=FULL")
        return db

    def append(self, operation_id: str, *, actor_id: str, action: str,
               state: str, subject_id: str | None = None,
               details: dict | None = None) -> int:
        if action not in self.ACTIONS or state not in self.STATES:
            raise ValueError("Unsupported action or state")
        for value in (operation_id, actor_id):
            if not isinstance(value, str) or not value or len(value) > 256:
                raise ValueError("Invalid operation or actor ID")
        if subject_id is not None and (
            not isinstance(subject_id, str) or len(subject_id) > 256
        ):
            raise ValueError("Invalid subject ID")
        details = dict(details or {})
        if details.keys() - self.FIELDS:
            raise ValueError("Only allowlisted metadata may enter local history")
        if any(type(v) not in (str, int, bool, type(None)) for v in details.values()):
            raise ValueError("History metadata must contain scalar values")
        if details.get("decision") not in (None, "approve", "reject"):
            raise ValueError("Invalid approval decision")
        body = json.dumps(dict(actor_id=actor_id, action=action, state=state,
                               subject_id=subject_id, details=details),
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(body.encode()) > 8192:
            raise ValueError("History event exceeds 8 KiB")
        slot = "intent" if state == "intent" else "outcome"
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT sequence, body FROM events WHERE operation_id=? AND slot=?",
                (operation_id, slot),
            ).fetchone()
            if old:
                if old[1] != body:
                    raise ValueError("Operation ID reused with different event content")
                return old[0]
            if slot == "outcome":
                intent = db.execute(
                    "SELECT body FROM events WHERE operation_id=? AND slot='intent'",
                    (operation_id,),
                ).fetchone()
                prior = json.loads(intent[0]) if intent else {}
                if prior.get("actor_id") != actor_id or prior.get("action") != action:
                    raise ValueError("Outcome requires the matching committed intent")
            cursor = db.execute(
                "INSERT INTO events(operation_id,slot,recorded_at,body) VALUES(?,?,?,?)",
                (operation_id, slot, datetime.now(timezone.utc).isoformat(), body),
            )
            return int(cursor.lastrowid)

    def events(self, *, after: int = 0, limit: int = 50) -> list[dict]:
        if after < 0 or not 1 <= limit <= 100:
            raise ValueError("Invalid history cursor or page size")
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT sequence, operation_id, recorded_at, body FROM events "
                "WHERE sequence>? ORDER BY sequence LIMIT ?", (after, limit),
            ).fetchall()
        return [dict(sequence=seq, operation_id=op, recorded_at=at, **json.loads(body))
                for seq, op, at, body in rows]


def artifact_path(root: Path, relative: str, expected_sha256: str | None = None) -> Path:
    """Containment and hash agreement; neither proves who produced a file."""
    if not relative or Path(relative).is_absolute():
        raise ValueError("Expected a run-relative artifact path")
    path = (root.resolve() / relative).resolve(strict=True)
    path.relative_to(root.resolve())  # Reject '..' and symlink escapes.
    if not path.is_file():
        raise ValueError("Artifact is not a regular file")
    if expected_sha256 is not None:
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected_sha256:
            raise ValueError("Artifact hash differs from the manifest")
    return path


def selected_model(structure_filename: str, ranking: dict) -> str:
    """Pair a ranked coordinate file with its model, never the first JSON file.

    The caller supplies ranking_debug.json from the selected structure's own
    directory. A missing/ambiguous ranking means confidence is unavailable.
    """
    ranked = re.fullmatch(r"ranked_(\d+)\.(?:pdb|cif|mmcif)", structure_filename)
    if ranked:
        order = ranking.get("order")
        if not isinstance(order, list) or int(ranked[1]) >= len(order):
            raise ValueError("Cannot match the ranked structure to a model")
        model = order[int(ranked[1])]
    else:
        direct = re.fullmatch(r"(?:unrelaxed|relaxed)_(model_[A-Za-z0-9_]+)\.(?:pdb|cif|mmcif)",
                              structure_filename)
        model = direct[1] if direct else None
    if not isinstance(model, str) or not re.fullmatch(r"model_[A-Za-z0-9_]+", model):
        raise ValueError("Invalid or unknown AlphaFold model ID")
    return model


def _number(value, lower: float, upper: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Confidence values must be finite numbers")
    if not lower <= value <= upper:
        raise ValueError("Confidence value outside its declared range")
    return float(value)


def plddt_scores(payload: dict, residue_count: int) -> tuple[float, ...]:
    """Support the pinned official AF2 confidence JSON for one monomer."""
    positions = payload.get("residueNumber")
    if (not isinstance(positions, list)
            or any(type(p) is not int for p in positions)
            or positions != list(range(1, residue_count + 1))):
        raise ValueError("Confidence residues do not match the input sequence")
    scores = payload.get("confidenceScore")
    if not isinstance(scores, list) or len(scores) != residue_count or residue_count < 1:
        raise ValueError("Confidence length does not match the input sequence")
    return tuple(_number(value, 0, 100) for value in scores)


def low_regions(scores: tuple[float, ...], threshold: float = 70) -> tuple[tuple[int, int], ...]:
    """Inclusive, one-based INPUT sequence positions; not PDB residue IDs."""
    _number(threshold, 0, 100)
    regions = []
    start = None
    for position, score in enumerate((*scores, 100.0), start=1):
        if score < threshold and start is None:
            start = position
        elif score >= threshold and start is not None:
            regions.append((start, position - 1))
            start = None
    return tuple(regions)


def pae_overview(payload, residue_count: int, bins: int = 32) -> dict:
    """Validate the complete matrix, retain its direction, bound terminal output.

    The caller bounds JSON bytes before parsing and limits N to the configured
    monomer maximum. Rows are aligned residues; columns are scored residues.
    Each displayed cell is the MAXIMUM in that block, not an average. This
    conservative overview keeps small high-error patches visible. Exact data
    remains in the original JSON for detailed review.
    """
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("Expected one monomer PAE record")
        payload = payload[0]
    if not isinstance(payload, dict) or residue_count < 1 or not 1 <= bins <= 64:
        raise ValueError("Invalid PAE request")
    matrix = payload.get("predicted_aligned_error")
    ceiling = _number(payload.get("max_predicted_aligned_error"), 0.01, 1000)
    if not isinstance(matrix, list) or len(matrix) != residue_count:
        raise ValueError("PAE height does not match the input sequence")
    size = min(bins, residue_count)
    # Same edges assign cells and label ranges: no skipped or duplicated residues.
    edges = [i * residue_count // size for i in range(size + 1)]
    assignment = [block for block in range(size)
                  for _ in range(edges[block], edges[block + 1])]
    grid = [[0.0] * size for _ in range(size)]
    for i, row in enumerate(matrix):
        if not isinstance(row, list) or len(row) != residue_count:
            raise ValueError("PAE must be a square matrix matching the sequence")
        for j, raw in enumerate(row):
            # AF2 rounds exported PAE to 0.1 A, while the maximum is unrounded.
            value = _number(raw, 0, ceiling + 0.051)
            a, b = assignment[i], assignment[j]
            grid[a][b] = max(grid[a][b], value)
    return dict(values=tuple(tuple(row) for row in grid),
                ranges=tuple((edges[i] + 1, edges[i + 1]) for i in range(size)),
                maximum_angstrom=ceiling, aggregation="block_max",
                rows="aligned residue", columns="scored residue")
