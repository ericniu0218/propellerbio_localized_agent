"""shell.run: the box, the clock, and the refusals that keep both honest.

The tests run real, harmless commands in a temporary directory. Where a test
needs the "a sandbox is configured" path, it configures ``/usr/bin/env`` as the
wrapper: it is a real program that execs whatever follows it, which is exactly
the shape of a bubblewrap or sandbox-exec line without needing either
installed.
"""

import asyncio
import sys
from pathlib import Path

import pytest

from bedrock.agents.capabilities.shell import CAPABILITIES, ShellConfig, ShellRun
from bedrock.agents.contract import Ask, AskKind, Evidence, Plan, Risk, Step
from bedrock.agents.grants import GrantStore
from bedrock.agents.supervisor import Supervisor, SupervisorError

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="these tests use POSIX process groups and /usr/bin/env"
)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    return scratch


@pytest.fixture
def sandboxed(workdir: Path) -> ShellConfig:
    """Enabled, with a wrapper standing in for a real sandbox."""
    return ShellConfig(
        enabled=True,
        workdir=workdir,
        sandbox_command=("/usr/bin/env",),
        timeout_seconds=10,
    )


class _NoJournal:
    """A journal the supervisor never gets as far as writing to."""

    def append(self, operation_id: str, **_: object) -> int:  # pragma: no cover
        raise AssertionError("a capability ran without an approved plan")


def unapproved_command(command: str) -> Plan:
    """A plan nobody has approved, for the supervisor to refuse."""
    return Plan(
        ask=Ask(
            kind=AskKind.PREPARE,
            text="run the analysis script",
            evidence=Evidence(quote="could somebody run that script"),
        ),
        steps=(
            Step(
                tool="shell.run",
                summary=f"Run {command}",
                risk=Risk.CHANGES_THINGS,
                arguments={"command": command},
            ),
        ),
    )


# ---- What the roster finds ---------------------------------------------------


def test_running_a_command_changes_things():
    assert {c.name: c.risk for c in CAPABILITIES} == {"shell.run": Risk.CHANGES_THINGS}


def test_the_whole_command_appears_in_the_description(sandboxed: ShellConfig):
    described = ShellRun(sandboxed).describe({"command": "python3 analyse.py --all"})
    assert "python3 analyse.py --all" in described
    assert "with no network" in described
    assert "at most 10s" in described


def test_describing_a_command_works_even_with_nothing_configured():
    # The roster imports this module to list what exists. That must not fail
    # just because no company has turned shell.run on.
    assert "ls" in ShellRun(ShellConfig()).describe({"command": "ls"})


# ---- Nothing runs without a grant --------------------------------------------


async def test_an_unapproved_command_is_never_started(sandboxed: ShellConfig):
    # ``protocol.Capability`` gives run() nothing but its arguments, so the gate
    # is the supervisor's: it refuses before the capability is called, and no
    # process is ever forked.
    supervisor = Supervisor(
        {"shell.run": ShellRun(sandboxed)}, GrantStore(), _NoJournal()
    )

    with pytest.raises(SupervisorError):
        await supervisor.execute(unapproved_command("/usr/bin/touch evidence"), None)
    assert not (sandboxed.workdir / "evidence").exists()


# ---- The two refusals that keep the promise ----------------------------------


async def test_running_commands_ships_switched_off(workdir: Path):
    off = ShellConfig(workdir=workdir, sandbox_command=("/usr/bin/env",))
    assert off.enabled is False
    outcome = await ShellRun(off).run(
        {"command": "/bin/echo hello"}
    )
    assert "switched off" in (outcome.failed or "")


async def test_no_sandbox_and_no_network_allowed_means_it_will_not_run(workdir: Path):
    # The standard library cannot take a network away from a child process, so
    # the only honest options are "sandbox configured" or "say out loud that
    # commands here reach the network".
    unsandboxed = ShellConfig(enabled=True, workdir=workdir)
    outcome = await ShellRun(unsandboxed).run(
        {"command": "/bin/echo hello"}
    )
    assert "would run with no sandbox" in (outcome.failed or "")


async def test_allowing_the_network_out_loud_is_enough_to_run(workdir: Path):
    open_config = ShellConfig(enabled=True, workdir=workdir, allow_network=True)
    outcome = await ShellRun(open_config).run(
        {"command": "/bin/echo hello"}
    )
    assert outcome.failed is None
    assert "hello" in outcome.completed[1]


async def test_a_missing_workdir_is_a_configuration_error(tmp_path: Path):
    config = ShellConfig(
        enabled=True, workdir=tmp_path / "gone", sandbox_command=("/usr/bin/env",)
    )
    outcome = await ShellRun(config).run(
        {"command": "/bin/echo hello"}
    )
    assert "is not a folder" in (outcome.failed or "")


# ---- Running ------------------------------------------------------------------


async def test_a_command_runs_in_the_configured_directory(sandboxed: ShellConfig):
    outcome = await ShellRun(sandboxed).run(
        {"command": "/bin/pwd"}
    )
    assert outcome.failed is None
    assert str(sandboxed.workdir.resolve()) in outcome.completed[1]


async def test_the_sandbox_wrapper_is_put_in_front_of_the_command(workdir: Path):
    # {workdir} is substituted, so a bubblewrap line can bind the working
    # directory without the path being written down twice.
    config = ShellConfig(
        enabled=True,
        workdir=workdir,
        sandbox_command=("/usr/bin/env", "SANDBOX_DIR={workdir}"),
    )
    argv = config.argv("/bin/sh -c true")
    assert argv[:2] == ["/usr/bin/env", f"SANDBOX_DIR={workdir}"]
    assert argv[2:] == ["/bin/sh", "-c", "true"]


async def test_there_is_no_shell_so_a_semicolon_is_just_an_argument(
    sandboxed: ShellConfig,
):
    outcome = await ShellRun(sandboxed).run(
        {"command": "/bin/echo one ; /usr/bin/touch owned"}
    )
    assert outcome.failed is None
    assert "one ; /usr/bin/touch owned" in outcome.completed[1]
    assert not (sandboxed.workdir / "owned").exists()


async def test_a_failing_command_reports_its_status_and_its_stderr(
    sandboxed: ShellConfig,
):
    outcome = await ShellRun(sandboxed).run(
        {"command": "/bin/sh -c 'echo trouble >&2; exit 3'"}
    )
    assert "exited with status 3" in (outcome.failed or "")
    assert any("trouble" in line for line in outcome.completed)


async def test_a_command_that_does_not_exist_says_so(sandboxed: ShellConfig):
    outcome = await ShellRun(sandboxed).run(
        {"command": "/usr/bin/definitely-not-installed"}
    )
    assert outcome.failed is not None
    assert "not found" in outcome.failed or "status" in outcome.failed


async def test_an_unreadable_command_line_is_refused_before_anything_starts(
    sandboxed: ShellConfig,
):
    outcome = await ShellRun(sandboxed).run(
        {"command": "/bin/echo 'unclosed"}
    )
    assert "could not be read" in (outcome.failed or "")


async def test_an_empty_command_is_refused(sandboxed: ShellConfig):
    outcome = await ShellRun(sandboxed).run({"command": "  "})
    assert "needs a `command`" in (outcome.failed or "")


# ---- The wall clock ----------------------------------------------------------


async def test_a_command_that_overruns_is_stopped(workdir: Path):
    config = ShellConfig(
        enabled=True, workdir=workdir, allow_network=True, timeout_seconds=1
    )
    outcome = await ShellRun(config).run(
        {"command": "/bin/sleep 30"}
    )
    assert "still running after 1s and was stopped" in (outcome.failed or "")


async def test_children_a_command_started_are_stopped_with_it(workdir: Path):
    # The child gets its own process group so the whole tree goes together:
    # a command that leaves a sleeping grandchild behind has escaped the budget.
    config = ShellConfig(
        enabled=True, workdir=workdir, allow_network=True, timeout_seconds=1
    )
    marker = workdir / "grandchild-finished"
    outcome = await ShellRun(config).run(
        {"command": f"/bin/sh -c '(sleep 2; touch {marker}) & sleep 30'"}
    )
    assert "was stopped" in (outcome.failed or "")

    await asyncio.sleep(3)
    assert not marker.exists()


# ---- The environment ----------------------------------------------------------


async def test_only_the_named_variables_are_handed_over(workdir: Path, monkeypatch):
    monkeypatch.setenv("REGISTRY_TOKEN", "t0ken")
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    config = ShellConfig(
        enabled=True,
        workdir=workdir,
        allow_network=True,
        env_passthrough=("PATH", "LANG"),
    )
    outcome = await ShellRun(config).run(
        {"command": "/usr/bin/env"}
    )
    assert outcome.failed is None
    printed = outcome.completed[1]
    assert "LANG=en_GB.UTF-8" in printed
    assert "REGISTRY_TOKEN" not in printed


def test_a_path_is_always_provided_even_if_the_config_forgot(workdir: Path):
    config = ShellConfig(enabled=True, workdir=workdir, env_passthrough=())
    assert config.environment()["PATH"]


# ---- Output limits ------------------------------------------------------------


async def test_a_command_that_prints_a_lot_is_drained_but_only_partly_kept(
    workdir: Path,
):
    # Draining matters as much as capping: a child whose pipe fills up blocks,
    # and a blocked command looks exactly like a slow one until the clock kills
    # it. This prints far more than the pipe buffer holds and still finishes.
    config = ShellConfig(
        enabled=True,
        workdir=workdir,
        allow_network=True,
        max_output_bytes=1_000,
        timeout_seconds=20,
    )
    loop = (
        "/bin/sh -c 'i=0; while [ $i -lt 20000 ]; "
        "do echo aaaaaaaaaaaaaaaaaaaa; i=$((i+1)); done'"
    )
    outcome = await ShellRun(config).run({"command": loop})
    assert outcome.failed is None
    assert 1_000 <= len(outcome.completed[1]) < 1_200
