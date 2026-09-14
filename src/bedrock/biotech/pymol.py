"""Headless rendering and desktop opening through open-source PyMOL."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.process import CommandRunner, SubprocessCommandRunner

SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}


@dataclass(frozen=True)
class PyMOLResult:
    preview_path: Path
    session_path: Path
    log_path: Path


class PyMOLAdapter:
    def __init__(self, executable: str, runner: CommandRunner | None = None) -> None:
        self.executable = executable
        self.runner = runner or SubprocessCommandRunner()

    def readiness_issues(self) -> tuple[str, ...]:
        if _command_exists(self.executable):
            return ()
        return (f"open-source PyMOL executable is not available: {self.executable}",)

    async def render(
        self,
        structure_path: Path,
        output_dir: Path,
        confidence_metric: str | None = None,
    ) -> PyMOLResult:
        structure = _validated_structure(structure_path)
        issues = self.readiness_issues()
        if issues:
            raise ToolIntegrationError("PyMOL is not ready: " + "; ".join(issues))
        if confidence_metric not in (None, "plddt"):
            raise ToolIntegrationError(f"Unsupported confidence metric: {confidence_metric}")

        output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        preview = (output_dir / "preview.png").resolve()
        session = (output_dir / "structure.pse").resolve()
        log = (output_dir / "pymol.log").resolve()
        render_script = Path(__file__).with_name("_pymol_render.py").resolve()
        environment = {
            "BEDROCK_PYMOL_INPUT": str(structure),
            "BEDROCK_PYMOL_PREVIEW": str(preview),
            "BEDROCK_PYMOL_SESSION": str(session),
            "BEDROCK_PYMOL_CONFIDENCE": confidence_metric or "",
        }
        await self.runner.run((self.executable, "-cq", "-r", str(render_script)), log, environment)
        if not preview.is_file() or not session.is_file():
            raise ToolIntegrationError("PyMOL finished without creating its preview and session files.")
        return PyMOLResult(preview, session, log)

    async def open_interactive(self, structure_path: Path) -> Path:
        structure = _validated_structure(structure_path)
        issues = self.readiness_issues()
        if issues:
            raise ToolIntegrationError("PyMOL is not ready: " + "; ".join(issues))
        try:
            await asyncio.create_subprocess_exec(
                self.executable,
                str(structure),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise ToolIntegrationError(f"Could not open PyMOL: {exc}") from exc
        return structure


def _validated_structure(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ToolIntegrationError(f"Structure file does not exist: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_STRUCTURE_SUFFIXES:
        raise ToolIntegrationError("PyMOL input must be a .pdb, .cif, or .mmcif file.")
    return resolved


def _command_exists(command: str) -> bool:
    candidate = Path(command).expanduser()
    if candidate.parent != Path(".") or candidate.is_absolute():
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(command) is not None
