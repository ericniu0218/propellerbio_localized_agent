"""Adapter for DeepMind's official AlphaFold 2 Docker launcher."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.process import CommandRunner, SubprocessCommandRunner


@dataclass(frozen=True)
class AlphaFold2Config:
    launcher: Path | None
    data_dir: Path | None
    python_executable: str
    max_template_date: str | None
    docker_image: str | None = None
    database_version: str | None = None
    model_preset: str = "monomer_ptm"
    db_preset: str = "full_dbs"
    gpu_devices: str = "all"
    models_to_relax: str = "best"


@dataclass(frozen=True)
class AlphaFold2Result:
    structure_path: Path
    output_dir: Path
    log_path: Path
    confidence_paths: tuple[Path, ...]
    command: tuple[str, ...]


class AlphaFold2Adapter:
    """Hide AF2 installation details behind one monomer-prediction operation."""

    def __init__(
        self,
        config: AlphaFold2Config,
        runner: CommandRunner | None = None,
    ) -> None:
        self.config = config
        self.runner = runner or SubprocessCommandRunner()

    def readiness_issues(self) -> tuple[str, ...]:
        issues: list[str] = []
        if self.config.launcher is None:
            issues.append("ALPHAFOLD_LAUNCHER is not configured")
        elif not self.config.launcher.is_file():
            issues.append(f"AlphaFold launcher is missing: {self.config.launcher}")
        if self.config.data_dir is None:
            issues.append("ALPHAFOLD_DATA_DIR is not configured")
        elif not self.config.data_dir.is_dir():
            issues.append(f"AlphaFold data directory is missing: {self.config.data_dir}")
        if not _command_exists(self.config.python_executable):
            issues.append(f"AlphaFold Python is not executable: {self.config.python_executable}")
        if not self.config.max_template_date:
            issues.append("ALPHAFOLD_MAX_TEMPLATE_DATE is not configured")
        elif not _valid_iso_date(self.config.max_template_date):
            issues.append("ALPHAFOLD_MAX_TEMPLATE_DATE must use YYYY-MM-DD")
        if not self.config.docker_image:
            issues.append("ALPHAFOLD_DOCKER_IMAGE is not configured with a pinned tag or digest")
        if not self.config.database_version:
            issues.append("ALPHAFOLD_DATABASE_VERSION is not configured")
        if self.config.model_preset not in {"monomer", "monomer_casp14", "monomer_ptm"}:
            issues.append("ALPHAFOLD_MODEL_PRESET must be a monomer preset")
        if self.config.db_preset not in {"full_dbs", "reduced_dbs"}:
            issues.append("ALPHAFOLD_DB_PRESET must be full_dbs or reduced_dbs")
        if self.config.models_to_relax not in {"best", "all", "none"}:
            issues.append("ALPHAFOLD_MODELS_TO_RELAX must be best, all, or none")
        return tuple(issues)

    async def predict(self, fasta_path: Path, output_dir: Path) -> AlphaFold2Result:
        issues = self.readiness_issues()
        if issues:
            raise ToolIntegrationError("AlphaFold 2 is not ready: " + "; ".join(issues))

        assert self.config.launcher is not None
        assert self.config.data_dir is not None
        assert self.config.max_template_date is not None
        output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        log_path = output_dir.parent / "alphafold2.log"
        command = (
            self.config.python_executable,
            str(self.config.launcher.resolve()),
            f"--fasta_paths={fasta_path.resolve()}",
            f"--data_dir={self.config.data_dir.resolve()}",
            f"--output_dir={output_dir.resolve()}",
            f"--model_preset={self.config.model_preset}",
            f"--db_preset={self.config.db_preset}",
            f"--max_template_date={self.config.max_template_date}",
            f"--docker_image_name={self.config.docker_image}",
            f"--gpu_devices={self.config.gpu_devices}",
            f"--models_to_relax={self.config.models_to_relax}",
            "--enable_gpu_relax=true",
        )
        await self.runner.run(command, log_path)

        structure = _find_ranked_structure(output_dir)
        confidence = tuple(
            sorted(
                path.resolve()
                for path in output_dir.rglob("*.json")
                if path.is_file()
                and ("confidence" in path.name.lower() or "pae" in path.name.lower())
                and _is_within(path, output_dir)
            )
        )
        return AlphaFold2Result(structure, output_dir.resolve(), log_path.resolve(), confidence, command)


def _command_exists(command: str) -> bool:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(command) is not None


def _valid_iso_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _find_ranked_structure(output_dir: Path) -> Path:
    patterns = (
        "ranked_0.cif",
        "ranked_0.mmcif",
        "ranked_0.pdb",
        "relaxed_model_*.cif",
        "relaxed_model_*.pdb",
        "unrelaxed_model_*.cif",
        "unrelaxed_model_*.pdb",
    )
    for pattern in patterns:
        candidates = sorted(path for path in output_dir.rglob(pattern) if path.is_file())
        for candidate in candidates:
            if _is_within(candidate, output_dir):
                return candidate.resolve()
    raise ToolIntegrationError(
        f"AlphaFold finished but produced no recognized structure under {output_dir}."
    )


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False
