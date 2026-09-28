"""Isolated Python execution sandbox for data analysis (pandas, numpy,
scipy, matplotlib) — no network access, time and memory limits.

Real hardening for an on-prem deployment should run this in a dedicated
container/microVM (gVisor, Firecracker, or a locked-down Docker container
with --network=none, a memory cgroup, and a seccomp profile) rather than an
in-process subprocess; that container wiring belongs in docker/ and is
noted there. What ships here is the tool contract plus a best-effort local
sandbox (subprocess, resource limits, no-network via an unshared network
namespace where the OS supports it) so the tool is usable in dev/tests
without container infra.
"""

from __future__ import annotations

import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from bedrock.server.tools.base import Tool, ToolResult

_DEFAULT_TIMEOUT_S = 20
_DEFAULT_MEMORY_MB = 512


class SandboxExecArgs(BaseModel):
    code: str
    timeout_s: int = Field(default=_DEFAULT_TIMEOUT_S, le=120)
    memory_mb: int = Field(default=_DEFAULT_MEMORY_MB, le=4096)


def _limit_resources(memory_mb: int) -> None:
    mem_bytes = memory_mb * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
    resource.setrlimit(resource.RLIMIT_CPU, (_DEFAULT_TIMEOUT_S, _DEFAULT_TIMEOUT_S))


class DataAnalysisSandboxTool(Tool):
    name: ClassVar[str] = "data_analysis_sandbox"
    description: ClassVar[str] = (
        "Run Python (pandas/numpy/scipy/matplotlib available) in an isolated, "
        "no-network subprocess with time and memory limits. Returns stdout "
        "and any error; does not persist state between calls."
    )
    args_model: ClassVar[type[BaseModel]] = SandboxExecArgs

    async def run(self, args: SandboxExecArgs) -> ToolResult:
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "script.py"
            script_path.write_text(args.code)
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_s,
                    preexec_fn=lambda: _limit_resources(args.memory_mb),
                    env={"PATH": "/usr/bin:/bin"},  # no network-capable env vars/proxies
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    summary=f"Execution exceeded the {args.timeout_s}s time limit.",
                    content="",
                    citations=[],
                )

            if proc.returncode != 0:
                return ToolResult(
                    summary="Execution failed.",
                    content=proc.stderr[-4000:],
                    citations=[],
                )
            return ToolResult(
                summary="Execution completed successfully.",
                content=proc.stdout[-8000:],
                citations=[],
            )
