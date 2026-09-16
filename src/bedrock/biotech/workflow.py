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
from bedrock.biotech.confidence import load_confidence_report, summary_document
from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.fasta import read_monomer_fasta
from bedrock.biotech.pymol import PyMOLAdapter
from bedrock.config import Settings
from bedrock.provenance import HistoryError, HistoryStore, RunCatalog, RunError
from bedrock.provenance.runs import validate_project_id


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
    confidence_path: Path | None = None


class ProteinWorkflow:
    """Deep module for one auditable sequence-to-structure workflow.

    Callers provide one FASTA. The implementation snapshots first, then parses
    and runs that immutable snapshot, pairs confidence with the selected model,
    and records intent/outcome metadata without claiming local authentication.
    """

    def __init__(
        self,
        work_dir: Path,
        max_sequence_length: int,
        alphafold: AlphaFold2Adapter,
        pymol: PyMOLAdapter,
        *,
        history: HistoryStore | None = None,
        actor_id: str = "local-user",
        history_root: Path | str | None = None,
    ) -> None:
        self.work_dir = work_dir.expanduser()
        self.max_sequence_length = max_sequence_length
        self.alphafold = alphafold
        self.pymol = pymol
        self.history = history
        self._history_root = Path(history_root).expanduser() if history_root is not None else None
        self.actor_id = actor_id

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        history: HistoryStore | None = None,
    ) -> "ProteinWorkflow":
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
            history=history,
            actor_id=settings.user_id,
            history_root=None if history is not None else settings.history_dir,
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
        *,
        project_id: str | None = None,
    ) -> ProteinWorkflowResult:
        if project_id is not None:
            try:
                validate_project_id(project_id)
            except RunError as exc:
                raise ToolIntegrationError(str(exc)) from exc

        run_id = _new_run_id()
        intent_details = {
            "run_id": run_id,
            "project_id": project_id,
            "actor_source": "config",
        }
        self._append_history(
            run_id,
            state="intent",
            subject_id=run_id,
            details=intent_details,
            required=True,
        )

        run_dir: Path | None = None
        manifest_path: Path | None = None
        manifest: dict[str, Any] = {}
        warnings: list[str] = []
        result: ProteinWorkflowResult | None = None
        try:
            run_dir = self._new_run_directory(run_id)
            manifest_path = run_dir / "manifest.json"
            manifest = self._initial_manifest(run_id, source_fasta, project_id)
            _write_manifest(manifest_path, manifest)

            # Copy first. Parsing and the expensive tool both consume files under
            # this private run directory, so later source edits cannot split the
            # recorded hash from the sequence that AlphaFold actually received.
            source_snapshot = run_dir / "source.fasta"
            shutil.copyfile(source_fasta.expanduser().resolve(), source_snapshot)
            os.chmod(source_snapshot, 0o600)
            record = read_monomer_fasta(source_snapshot, self.max_sequence_length)
            canonical_fasta = run_dir / "input.fasta"
            canonical_fasta.write_text(
                f">bedrock_target\n{_wrap(record.sequence)}\n", encoding="utf-8"
            )
            os.chmod(canonical_fasta, 0o600)

            manifest["source"] = {
                "original_filename": source_fasta.name,
                "snapshot": source_snapshot.name,
                "sha256": _sha256(source_snapshot),
                "description": record.description,
                "sequence_sha256": hashlib.sha256(record.sequence.encode()).hexdigest(),
                "residue_count": len(record.sequence),
            }
            manifest["outputs"]["input"] = _artifact_entry(canonical_fasta, run_dir)
            manifest["status"] = "running_alphafold2"
            _write_manifest(manifest_path, manifest)

            prediction = await self.alphafold.predict(canonical_fasta, run_dir / "alphafold2")
            manifest["status"] = "normalizing_confidence"
            manifest["alphafold2"]["command"] = list(prediction.command)
            manifest["outputs"]["structure"] = _artifact_entry(
                prediction.structure_path, run_dir
            )
            manifest["outputs"]["confidence"] = [
                _artifact_entry(path, run_dir) for path in prediction.confidence_paths
            ]
            if prediction.ranking_path is not None:
                manifest["outputs"]["ranking"] = _artifact_entry(
                    prediction.ranking_path, run_dir
                )
            _write_manifest(manifest_path, manifest)

            confidence_path: Path | None = None
            try:
                report = load_confidence_report(
                    run_id,
                    run_dir,
                    manifest,
                    max_residue_count=self.max_sequence_length,
                )
                manifest["alphafold2"]["selected_model"] = report.model_id
                warnings.extend(report.notices)
                confidence_path = run_dir / "confidence-summary.json"
                _write_private_json(confidence_path, summary_document(report, manifest))
                manifest["outputs"]["confidence_summary"] = _artifact_entry(
                    confidence_path, run_dir
                )
            except (OSError, ValueError) as exc:
                warnings.append(f"Confidence summary could not be written: {exc}")
            _write_manifest(manifest_path, manifest)

            manifest["status"] = "rendering_pymol"
            _write_manifest(manifest_path, manifest)
            preview_path: Path | None = None
            session_path: Path | None = None
            try:
                rendered = await self.pymol.render(
                    prediction.structure_path,
                    run_dir / "pymol",
                    confidence_metric=(
                        "plddt" if manifest["alphafold2"].get("selected_model") else None
                    ),
                )
                preview_path = rendered.preview_path
                session_path = rendered.session_path
                manifest["outputs"]["preview"] = _artifact_entry(preview_path, run_dir)
                manifest["outputs"]["pymol_session"] = _artifact_entry(session_path, run_dir)
                manifest["status"] = "completed"
            except ToolIntegrationError as exc:
                warnings.append(str(exc))
                manifest["status"] = "completed_with_render_error"

            if warnings:
                manifest["warnings"] = warnings
            manifest["finished_at"] = _now()
            _write_manifest(manifest_path, manifest)
            result = ProteinWorkflowResult(
                run_id=run_id,
                run_dir=run_dir.resolve(),
                structure_path=prediction.structure_path,
                preview_path=preview_path,
                session_path=session_path,
                manifest_path=manifest_path.resolve(),
                warnings=tuple(warnings),
                confidence_path=confidence_path.resolve() if confidence_path else None,
            )
        except asyncio.CancelledError as exc:
            self._record_terminal_failure(run_id, "cancelled", exc, manifest, manifest_path)
            raise
        except Exception as exc:
            self._record_terminal_failure(run_id, "failed", exc, manifest, manifest_path)
            raise

        assert result is not None and manifest_path is not None
        outcome_details = self._outcome_details(run_id, manifest, manifest_path)
        try:
            self._append_history(
                run_id,
                state="succeeded",
                subject_id=run_id,
                details=outcome_details,
                required=True,
            )
        except ToolIntegrationError:
            warning = "Run completed, but local history could not record the outcome."
            warnings.append(warning)
            manifest["warnings"] = warnings
            _write_manifest(manifest_path, manifest)
            result = ProteinWorkflowResult(
                run_id=result.run_id,
                run_dir=result.run_dir,
                structure_path=result.structure_path,
                preview_path=result.preview_path,
                session_path=result.session_path,
                manifest_path=result.manifest_path,
                warnings=tuple(warnings),
                confidence_path=result.confidence_path,
            )
        return result

    async def open_in_pymol(self, structure_path: Path | None = None) -> Path:
        target = structure_path
        confidence_metric: str | None = None
        if target is None:
            latest = self._latest_verified_structure()
            if latest is not None:
                target, confidence_metric = latest
        if target is None:
            raise ToolIntegrationError("No completed AlphaFold structure was found.")
        return await self.pymol.open_interactive(
            target, confidence_metric=confidence_metric
        )

    def latest_structure(self) -> Path | None:
        latest = self._latest_verified_structure()
        return latest[0] if latest is not None else None

    def _latest_verified_structure(self) -> tuple[Path, str | None] | None:
        history = self._ensure_history() or _NoTagHistory()
        catalog = RunCatalog(self.work_dir, history)  # type: ignore[arg-type]
        page = catalog.list(limit=100)
        for item in page["items"]:
            run_id = item["run_id"]
            try:
                shown = catalog.show(run_id)
                manifest = shown["manifest"]
                structure = manifest.get("outputs", {}).get("structure", {})
                relative = structure.get("path")
                verified = {
                    row["path"]: row for row in catalog.verify(run_id)
                }.get(relative)
                if verified is None or verified["state"] != "match":
                    continue
                target = (shown["manifest_path"].parent / relative).resolve()
                metric = (
                    "plddt" if manifest.get("alphafold2", {}).get("selected_model") else None
                )
                return target, metric
            except (RunError, OSError, TypeError):
                continue
        return None

    def _new_run_directory(self, run_id: str) -> Path:
        self.work_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.work_dir, 0o700)
        runs_dir = self.work_dir / "runs"
        runs_dir.mkdir(exist_ok=True, mode=0o700)
        os.chmod(runs_dir, 0o700)
        run_dir = runs_dir / run_id
        run_dir.mkdir(exist_ok=False, mode=0o700)
        return run_dir

    def _initial_manifest(
        self, run_id: str, source_fasta: Path, project_id: str | None
    ) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "run_id": run_id,
            "project_id": project_id,
            "status": "preparing_input",
            "created_at": _now(),
            "limits": {"max_sequence_length": self.max_sequence_length},
            "source": {"original_filename": source_fasta.name},
            "alphafold2": {
                "model_preset": self.alphafold.config.model_preset,
                "db_preset": self.alphafold.config.db_preset,
                "max_template_date": self.alphafold.config.max_template_date,
                "docker_image": self.alphafold.config.docker_image,
                "database_version": self.alphafold.config.database_version,
                "launcher_sha256": (
                    _sha256(self.alphafold.config.launcher)
                    if self.alphafold.config.launcher
                    and self.alphafold.config.launcher.is_file()
                    else None
                ),
                "gpu_devices": self.alphafold.config.gpu_devices,
                "models_to_relax": self.alphafold.config.models_to_relax,
                "selected_model": None,
            },
            "outputs": {},
            "pymol": {"executable": self.pymol.executable, "confidence_metric": "plddt"},
        }

    def _record_terminal_failure(
        self,
        run_id: str,
        state: str,
        error: BaseException,
        manifest: dict[str, Any],
        manifest_path: Path | None,
    ) -> None:
        if manifest_path is not None and manifest:
            manifest["status"] = state
            manifest["finished_at"] = _now()
            manifest["error"] = str(error)
            try:
                _write_manifest(manifest_path, manifest)
            except (OSError, ValueError):
                pass
        details = self._outcome_details(run_id, manifest, manifest_path)
        details["error_code"] = _error_code(error)
        self._append_history(
            run_id,
            state=state,
            subject_id=run_id,
            details=details,
            required=False,
        )

    def _outcome_details(
        self,
        run_id: str,
        manifest: dict[str, Any],
        manifest_path: Path | None,
    ) -> dict[str, Any]:
        source = manifest.get("source", {})
        return {
            "run_id": run_id,
            "project_id": manifest.get("project_id"),
            "source_sha256": source.get("sha256") if isinstance(source, dict) else None,
            "manifest_sha256": _sha256_if_file(manifest_path),
            "actor_source": "config",
        }

    def _append_history(
        self,
        operation_id: str,
        *,
        state: str,
        subject_id: str,
        details: dict[str, Any],
        required: bool,
    ) -> None:
        try:
            history = self._ensure_history()
            if history is None:
                return
            history.append(
                operation_id,
                actor_id=self.actor_id,
                action="fold",
                state=state,
                subject_id=subject_id,
                details=details,
            )
        except HistoryError as exc:
            if required:
                raise ToolIntegrationError("Could not record the fold in local history.") from exc

    def _ensure_history(self) -> HistoryStore | None:
        if self.history is None and self._history_root is not None:
            self.history = HistoryStore(self._history_root)
        return self.history


class _NoTagHistory:
    def latest_project_assignments(self) -> dict[str, str]:
        return {}


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


def _sha256_if_file(path: Path | None) -> str | None:
    try:
        return _sha256(path) if path is not None and path.is_file() else None
    except OSError:
        return None


def _artifact_entry(path: Path, run_dir: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ToolIntegrationError(f"Tool output escaped its run directory: {resolved}") from exc
    return {"path": str(relative), "sha256": _sha256(resolved), "bytes": resolved.stat().st_size}


def _write_private_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    _write_private_json(path, manifest)


def _error_code(error: BaseException) -> str:
    name = type(error).__name__
    return "".join(
        character.lower() if character.isalnum() else "_" for character in name
    )[:64]
