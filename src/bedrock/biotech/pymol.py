"""Headless rendering and desktop opening through open-source PyMOL.

Both operations run the same fixed scripts and hand PyMOL its inputs as
environment data rather than as command text, so a structure path is never
interpolated into something PyMOL will interpret. The two paths share their
styling through ``_pymol_style``; see that module for why that matters.

``open_interactive`` takes the same ``confidence_metric`` keyword as
``render``. Passing ``"plddt"`` is a caller's declaration that this file's
B-factor column really holds AlphaFold confidence - which the workflow knows
because it produced the file. An arbitrary structure a scientist points at
gets neutral styling, because its B-factors mean something else entirely.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

from bedrock.biotech._pymol_style import SUPPORTED_METRICS
from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.process import CommandRunner, SubprocessCommandRunner

# Coordinates only. A saved ``.pse`` session is a pickle-like PyMOL program
# and is deliberately not accepted here, however convenient that would be.
SUPPORTED_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}

DetachedLauncher = Callable[[Sequence[str], Mapping[str, str]], Awaitable[None]]


@dataclass(frozen=True)
class PyMOLResult:
    preview_path: Path
    session_path: Path
    log_path: Path


class PyMOLAdapter:
    def __init__(
        self,
        executable: str,
        runner: CommandRunner | None = None,
        launcher: DetachedLauncher | None = None,
    ) -> None:
        self.executable = executable
        self.runner = runner or SubprocessCommandRunner()
        # Interactive opening cannot use ``runner``: that one waits for the
        # process and captures its output, and a desktop PyMOL window is meant
        # to outlive the command that opened it. Injectable for the same
        # reason ``runner`` is - tests must never start a real PyMOL.
        self.launcher: DetachedLauncher = launcher or self._spawn_detached

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
        _validated_metric(confidence_metric)

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

    async def open_interactive(
        self,
        structure_path: Path,
        *,
        confidence_metric: str | None = None,
    ) -> Path:
        """Open one structure in a desktop PyMOL window, styled like the preview.

        The structure is passed through the environment and loaded by the
        fixed script, not handed to PyMOL as a command-line file to guess at.
        The child process is detached deliberately: PyMOL outlives this
        command, and the employee closes it themselves.
        """
        structure = _validated_structure(structure_path)
        metric = _validated_metric(confidence_metric)
        issues = self.readiness_issues()
        if issues:
            raise ToolIntegrationError("PyMOL is not ready: " + "; ".join(issues))
        script = Path(__file__).with_name("_pymol_interactive.py").resolve()
        argv = (self.executable, "-r", str(script))
        environment = _child_environment(
            {
                "BEDROCK_PYMOL_INPUT": str(structure),
                "BEDROCK_PYMOL_CONFIDENCE": metric,
            }
        )
        try:
            await self.launcher(argv, environment)
        except OSError as exc:
            raise ToolIntegrationError(f"Could not open PyMOL: {exc}") from exc
        return structure

    async def _spawn_detached(
        self,
        argv: Sequence[str],
        environment: Mapping[str, str],
    ) -> None:
        await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            env=dict(environment),
            start_new_session=True,
        )


def _validated_metric(confidence_metric: str | None) -> str:
    if confidence_metric in (None, ""):
        return ""
    if confidence_metric not in SUPPORTED_METRICS:
        raise ToolIntegrationError(f"Unsupported confidence metric: {confidence_metric}")
    return confidence_metric


def _child_environment(extra: Mapping[str, str]) -> dict[str, str]:
    """The parent environment minus credentials, plus our own data.

    Same rule as ``SubprocessCommandRunner``: a visualisation tool has no use
    for the model server's API key, and the safest way to keep it out of a
    child process is not to hand it over in the first place.
    """
    child = os.environ.copy()
    for name in tuple(child):
        if name.upper().endswith(("_API_KEY", "_SECRET", "_TOKEN")):
            child.pop(name, None)
    child.update(extra)
    return child


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
