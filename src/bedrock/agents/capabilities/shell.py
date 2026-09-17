"""Running a command, which is the one capability allowlisting cannot make safe.

Every other adapter here is safe because its arguments can be checked: a path
is inside a root or it is not, a domain is on the list or it is not. A command
line has no such property - ``python -c '...'`` is every capability in this
package at once, plus the ones nobody wrote. So ``shell.run`` is not made safe
by inspecting what it is asked to do. It is made safe by the box it runs in::

    [agents.capabilities.shell]
    enabled = false                      # ships off; it is reasonable never to turn on
    workdir = "~/work/agent-scratch"
    sandbox_command = ["bwrap", "--unshare-net", "--die-with-parent",
                       "--ro-bind", "/usr", "/usr", "--bind", "{workdir}", "{workdir}"]
    allow_network = false
    timeout_seconds = 30
    max_output_bytes = 200000
    env_passthrough = ["PATH", "HOME", "LANG"]

Four properties, and the refusals that keep them true:

*No shell.* The command is split with ``shlex`` and run with ``exec``. There is
no ``/bin/sh``, so ``;`` and ``&&`` and backticks are ordinary characters in an
argument rather than instructions. Pipes and redirection do not work here, and
that is the feature.

*No network, and no pretending.* ``allow_network = false`` is only honoured if
``sandbox_command`` is configured, because nothing in the standard library can
take a network away from a child process. With no sandbox configured and no
network allowed, this capability **refuses to run**. Claiming a sandbox that
does not exist is worse than having none.

*A wall clock.* The child gets its own process group and is killed as a group
when the budget runs out, so a command that spawns children cannot leave them
behind.

*A clean environment.* Only the variables named in ``env_passthrough`` are
handed over - tokens for the registry, the calendar and the mail server are not
in scope for a subprocess.

``shell.run`` is ``CHANGES_THINGS``, so the supervisor will not call it until
somebody has approved the plan it belongs to - and the full command text
appears in that plan, because approving "run a short script" is not something
anybody can do intelligently.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bedrock.agents.capability_support import (
    CapabilityConfigError,
    as_bool,
    as_int,
    as_list,
    as_path,
    bounded,
    capability_config,
    done,
    failed,
)
from bedrock.agents.contract import Outcome, Risk
from bedrock.agents.protocol import Capability

DEFAULT_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TZ")


# ---- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class ShellConfig:
    enabled: bool = False
    workdir: Path | None = None
    sandbox_command: tuple[str, ...] = ()
    allow_network: bool = False
    timeout_seconds: int = 30
    max_output_bytes: int = 200_000
    env_passthrough: tuple[str, ...] = field(default=DEFAULT_ENV_PASSTHROUGH)

    @classmethod
    def from_settings(cls, settings: Any = None) -> "ShellConfig":
        table = capability_config("shell", settings)
        passthrough = as_list(table, "env_passthrough") or DEFAULT_ENV_PASSTHROUGH
        return cls(
            enabled=as_bool(table, "enabled", False),
            workdir=as_path(table, "workdir"),
            sandbox_command=as_list(table, "sandbox_command"),
            allow_network=as_bool(table, "allow_network", False),
            timeout_seconds=as_int(table, "timeout_seconds", 30),
            max_output_bytes=as_int(table, "max_output_bytes", 200_000),
            env_passthrough=tuple(passthrough),
        )

    def checked(self) -> "ShellConfig":
        """Every reason this must not run, checked before anything is started."""
        if not self.enabled:
            raise ShellRefused(
                "Running commands is switched off. Set `enabled = true` in "
                "[agents.capabilities.shell] if this machine is meant to allow it - "
                "it ships off, and shipping without it at all is a reasonable choice."
            )
        if self.workdir is None:
            raise CapabilityConfigError(
                "No `workdir` is configured for [agents.capabilities.shell]. A command "
                "runs in a directory somebody chose, never in whatever directory the "
                "terminal happened to be in."
            )
        if not self.workdir.is_dir():
            raise CapabilityConfigError(
                f"The configured workdir {self.workdir} is not a folder."
            )
        if not self.allow_network and not self.sandbox_command:
            raise ShellRefused(
                "This command would run with no sandbox, so its network cannot be taken "
                "away. Configure `sandbox_command` (bubblewrap on Linux, sandbox-exec on "
                "macOS), or set `allow_network = true` to say out loud that commands here "
                "reach the network."
            )
        if self.timeout_seconds <= 0:
            raise CapabilityConfigError("`timeout_seconds` must be more than zero.")
        return self

    def argv(self, command: str) -> list[str]:
        """The sandbox wrapper plus the command, as a list of arguments.

        ``{workdir}`` in the sandbox command is replaced, because a bubblewrap
        line almost always has to bind the working directory and repeating the
        path in two settings is a way to get them out of step.
        """
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ShellRefused(f"That command could not be read: {exc}.") from exc
        if not parts:
            raise ShellRefused("shell.run needs a `command` to run.")
        workdir = str(self.workdir)
        wrapper = [item.replace("{workdir}", workdir) for item in self.sandbox_command]
        return wrapper + parts

    def environment(self) -> dict[str, str]:
        kept = {
            name: os.environ[name]
            for name in self.env_passthrough
            if name in os.environ
        }
        kept.setdefault("PATH", "/usr/bin:/bin")
        return kept


class ShellRefused(Exception):
    """A command an agent asked for will not be run."""


class ShellRun:
    """Run one command inside the configured sandbox, with a wall clock on it."""

    name = "shell.run"
    risk = Risk.CHANGES_THINGS

    def __init__(self, config: ShellConfig | None = None) -> None:
        self._config = config

    @property
    def config(self) -> ShellConfig:
        if self._config is None:
            self._config = ShellConfig.from_settings()
        return self._config

    def describe(self, arguments: dict[str, str]) -> str:
        """The whole command, verbatim.

        Never summarised and never shortened: the person approving this is
        approving these exact words, and "run a short script" is not something
        anybody can refuse intelligently.
        """
        command = str(arguments.get("command", "")).strip() or "(no command given)"
        try:
            config = self.config
            network = "with network access" if config.allow_network else "with no network"
            where = config.workdir or "(no workdir configured)"
            budget = config.timeout_seconds
        except CapabilityConfigError:
            network, where, budget = "with no network", "(not configured)", 30
        return f"Run `{command}` in {where}, {network}, for at most {budget}s."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        command = str(arguments.get("command", "")).strip()
        try:
            config = self.config.checked()
            argv = config.argv(command)
        except (CapabilityConfigError, ShellRefused) as exc:
            return failed(str(exc))

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(config.workdir),
                env=config.environment(),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Its own process group, so the whole tree can be killed
                # together when the budget runs out.
                start_new_session=True,
            )
        except FileNotFoundError:
            missing = argv[0]
            return failed(
                f"`{missing}` was not found on this machine."
                + (
                    " That is the configured sandbox_command, not the command asked for."
                    if config.sandbox_command and missing == config.sandbox_command[0]
                    else ""
                ),
            )
        except PermissionError:
            return failed(f"`{argv[0]}` is not something this account may run.")
        except OSError as exc:
            return failed(f"Could not start `{argv[0]}`: {exc.strerror or exc}")

        limit = max(config.max_output_bytes, 1_000)
        try:
            out, err, code = await asyncio.wait_for(
                _collect(process, limit), timeout=config.timeout_seconds
            )
        except (asyncio.TimeoutError, TimeoutError):
            _kill_group(process)
            await process.wait()
            return failed(
                f"`{command}` was still running after {config.timeout_seconds}s and was "
                "stopped, along with anything it had started.",
            )

        lines = [f"$ {command}"]
        if out.strip():
            lines.append(bounded(out, limit))
        if err.strip():
            lines.append("stderr:\n" + bounded(err, limit))
        if code != 0:
            return failed(f"`{command}` exited with status {code}.", completed=lines)
        lines.append("Finished with status 0.")
        return done(lines)


async def _collect(
    process: asyncio.subprocess.Process, limit: int
) -> tuple[str, str, int]:
    """Drain both pipes, keeping at most ``limit`` bytes of each.

    Draining matters as much as capping: a child whose pipe fills up blocks
    forever, and a command that is blocked rather than finished looks exactly
    like a slow one until the wall clock kills it.
    """
    out, err = await asyncio.gather(
        _read_capped(process.stdout, limit), _read_capped(process.stderr, limit)
    )
    code = await process.wait()
    return out, err, code


async def _read_capped(stream: asyncio.StreamReader | None, limit: int) -> str:
    if stream is None:
        return ""
    kept = bytearray()
    while True:
        chunk = await stream.read(65_536)
        if not chunk:
            break
        if len(kept) < limit:
            kept.extend(chunk[: limit - len(kept)])
    return kept.decode("utf-8", errors="replace")


def _kill_group(process: asyncio.subprocess.Process) -> None:
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        # It finished between the timeout and the kill, or this platform has no
        # process groups. Either way there is nothing left to stop.
        try:
            process.kill()
        except ProcessLookupError:
            pass


CAPABILITY: tuple[Capability, ...] = (ShellRun(),)
"""What the roster finds in this file.

A tuple, which ``roster.discover_capability`` unpacks by name. One file
per adapter, several verbs per file where they share one set of rules.
"""

CAPABILITIES = CAPABILITY
"""Plural alias, for reading a module's contents rather than the roster's."""
