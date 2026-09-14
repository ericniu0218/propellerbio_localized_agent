"""Safe subprocess execution shared by scientific-tool adapters."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from bedrock.biotech.errors import ToolIntegrationError


class CommandRunner(Protocol):
    async def run(
        self,
        args: Sequence[str],
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None: ...


class SubprocessCommandRunner:
    """Run an argument vector without a shell and capture output in a private log."""

    async def run(
        self,
        args: Sequence[str],
        log_path: Path,
        env: Mapping[str, str] | None = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        child_env = os.environ.copy()
        # Tool processes do not need model-server or third-party credentials.
        # Keep those out even if the parent terminal happens to have them set.
        for name in tuple(child_env):
            if name.upper().endswith(("_API_KEY", "_SECRET", "_TOKEN")):
                child_env.pop(name, None)
        if env:
            child_env.update(env)

        with log_path.open("wb") as log:
            os.chmod(log_path, 0o600)
            try:
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    env=child_env,
                    start_new_session=True,
                )
            except OSError as exc:
                raise ToolIntegrationError(f"Could not start {args[0]}: {exc}") from exc

            try:
                return_code = await process.wait()
            except asyncio.CancelledError:
                await _stop_process_group(process)
                raise

        if return_code != 0:
            detail = _tail(log_path)
            suffix = f" Last log lines:\n{detail}" if detail else ""
            raise ToolIntegrationError(
                f"{Path(args[0]).name} exited with status {return_code}.{suffix}"
            )


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()


def _tail(path: Path, limit: int = 2500) -> str:
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    return data[-limit:].decode(errors="replace").strip()
