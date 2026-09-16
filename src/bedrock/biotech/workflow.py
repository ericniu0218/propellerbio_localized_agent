"""Sequence-to-structure-to-visualization workflow with durable provenance."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bedrock.biotech.alphafold2 import AlphaFold2Adapter, AlphaFold2Config
from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.fasta import read_monomer_fasta
from bedrock.biotech.pymol import PyMOLAdapter
from bedrock.config import Settings


@dataclass(frozen=True)
class ToolStatus:
    name: str
    ready: bool
    issues: tuple[str, ...]


@dataclass(frozen=True)
class ProteinWorkflowResult:
    run_id: str
    run_dir: Path
    structure_path: Path
    preview_path: Path | None
    session_path: Path | None
    manifest_path: Path
    warnings: tuple[str, ...] = ()


class ProteinWorkflow:
    """Deep module for the first complete scientific workflow.

    Callers provide one FASTA. The implementation validates and snapshots it,
    runs AF2, passes the ranked structure to PyMOL, and records enough inputs,
    settings, hashes, and outputs to audit or reproduce the run.
    """

    def __init__(
        self,
        work_dir: Path,
        max_sequence_length: int,
        alphafold: AlphaFold2Adapter,
        pymol: PyMOLAdapter,
    ) -> None:
        self.work_dir = work_dir.expanduser()
        self.max_sequence_length = max_sequence_length
        self.alphafold = alphafold
        self.pymol = pymol

    @classmethod
    def from_settings(cls, settings: Settings) -> "ProteinWorkflow":
        alphafold = AlphaFold2Adapter(
            AlphaFold2Config(
                launcher=settings.alphafold_launcher,
                data_dir=settings.alphafold_data_dir,
                python_executable=settings.alphafold_python,
                max_template_date=settings.alphafold_max_template_date,
                docker_image=settings.alphafold_docker_image,
                database_version=settings.alphafold_database_version,
                model_preset=settings.alphafold_model_preset,
                db_preset=settings.alphafold_db_preset,
                gpu_devices=settings.alphafold_gpu_devices,
                models_to_relax=settings.alphafold_models_to_relax,
            )
        )
        return cls(
            settings.biotech_work_dir,
            settings.biotech_max_sequence_length,
            alphafold,
            PyMOLAdapter(settings.pymol_executable),
        )

    def status(self) -> tuple[ToolStatus, ToolStatus]:
        af2_issues = self.alphafold.readiness_issues()
        pymol_issues = self.pymol.readiness_issues()
        return (
            ToolStatus("AlphaFold 2", not af2_issues, af2_issues),
            ToolStatus("open-source PyMOL", not pymol_issues, pymol_issues),
        )

    async def predict_and_render(
        self,
        source_fasta: Path,
        source_provenance: dict[str, Any] | None = None,
    ) -> ProteinWorkflowResult:
        """Validate, predict, render and record one monomer.

        ``source_provenance`` is set when the FASTA was pulled from a company
        system of record instead of being handed over as a file (see
        ``bedrock.connectors``). It goes into the manifest so the run names the
        registry record and version it was made from, not just the staging
        file the sequence happened to pass through.
        """
        record = read_monomer_fasta(source_fasta, self.max_sequence_length)
        self.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.work_dir, 0o700)
        run_id = _new_run_id()
        runs_dir = self.work_dir / "runs"
        runs_dir.mkdir(exist_ok=True, mode=0o700)
        os.chmod(runs_dir, 0o700)
        run_dir = runs_dir / run_id
        run_dir.mkdir(exist_ok=False, mode=0o700)
        source_snapshot = run_dir / "source.fasta"
        canonical_fasta = run_dir / "input.fasta"
        shutil.copyfile(source_fasta.expanduser().resolve(), source_snapshot)
        canonical_fasta.write_text(f">bedrock_target\n{_wrap(record.sequence)}\n", encoding="utf-8")
        os.chmod(source_snapshot, 0o600)
        os.chmod(canonical_fasta, 0o600)

        manifest_path = run_dir / "manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running_alphafold2",
            "created_at": _now(),
            "source": {
                "original_filename": source_fasta.name,
                "snapshot": source_snapshot.name,
                "sha256": _sha256(source_snapshot),
                "description": record.description,
                "sequence_sha256": hashlib.sha256(record.sequence.encode()).hexdigest(),
                "residue_count": len(record.sequence),
            },
            "alphafold2": {
                "model_preset": self.alphafold.config.model_preset,
                "db_preset": self.alphafold.config.db_preset,
                "max_template_date": self.alphafold.config.max_template_date,
                "docker_image": self.alphafold.config.docker_image,
                "database_version": self.alphafold.config.database_version,
                "launcher_sha256": (
                    _sha256(self.alphafold.config.launcher)
                    if self.alphafold.config.launcher and self.alphafold.config.launcher.is_file()
                    else None
                ),
                "gpu_devices": self.alphafold.config.gpu_devices,
                "models_to_relax": self.alphafold.config.models_to_relax,
            },
            "outputs": {},
            "pymol": {"executable": self.pymol.executable, "confidence_metric": "plddt"},
        }
        if source_provenance:
            manifest["source"]["system_of_record"] = dict(source_provenance)
        _write_manifest(manifest_path, manifest)

        try:
            prediction = await self.alphafold.predict(canonical_fasta, run_dir / "alphafold2")
        except asyncio.CancelledError as exc:
            manifest["status"] = "cancelled"
            manifest["finished_at"] = _now()
            manifest["error"] = str(exc)
            _write_manifest(manifest_path, manifest)
            raise
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["finished_at"] = _now()
            manifest["error"] = str(exc)
            _write_manifest(manifest_path, manifest)
            raise

        manifest["status"] = "rendering_pymol"
        manifest["alphafold2"]["command"] = list(prediction.command)
        manifest["outputs"]["structure"] = _artifact_entry(prediction.structure_path, run_dir)
        manifest["outputs"]["confidence"] = [
            _artifact_entry(path, run_dir) for path in prediction.confidence_paths
        ]
        _write_manifest(manifest_path, manifest)

        preview_path: Path | None = None
        session_path: Path | None = None
        warnings: list[str] = []
        try:
            rendered = await self.pymol.render(
                prediction.structure_path,
                run_dir / "pymol",
                confidence_metric="plddt",
            )
            preview_path = rendered.preview_path
            session_path = rendered.session_path
            manifest["outputs"]["preview"] = _artifact_entry(preview_path, run_dir)
            manifest["outputs"]["pymol_session"] = _artifact_entry(session_path, run_dir)
            manifest["status"] = "completed"
        except ToolIntegrationError as exc:
            # A rendering failure must not hide or destroy a successful, expensive prediction.
            warnings.append(str(exc))
            manifest["status"] = "completed_with_render_error"
            manifest["warnings"] = warnings

        manifest["finished_at"] = _now()
        _write_manifest(manifest_path, manifest)
        return ProteinWorkflowResult(
            run_id=run_id,
            run_dir=run_dir.resolve(),
            structure_path=prediction.structure_path,
            preview_path=preview_path,
            session_path=session_path,
            manifest_path=manifest_path.resolve(),
            warnings=tuple(warnings),
        )

    async def open_in_pymol(self, structure_path: Path | None = None) -> Path:
        target = structure_path or self.latest_structure()
        if target is None:
            raise ToolIntegrationError("No completed AlphaFold structure was found.")
        return await self.pymol.open_interactive(target)

    def latest_structure(self) -> Path | None:
        runs_dir = self.work_dir / "runs"
        if not runs_dir.is_dir():
            return None
        for manifest_path in sorted(runs_dir.glob("*/manifest.json"), reverse=True):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                relative = manifest["outputs"]["structure"]["path"]
                candidate = (manifest_path.parent / relative).resolve()
                candidate.relative_to(manifest_path.parent.resolve())
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
            if candidate.is_file():
                return candidate
        return None


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _wrap(sequence: str, width: int = 80) -> str:
    return "\n".join(sequence[index : index + width] for index in range(0, len(sequence), width))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_entry(path: Path, run_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ToolIntegrationError(f"Tool output escaped its run directory: {resolved}") from exc
    return {"path": str(relative), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)
