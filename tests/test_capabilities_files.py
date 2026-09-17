"""files.search, files.read and files.write: the roots hold, nothing is replaced.

Every test builds a ``FilesConfig`` pointing at a temporary directory, so no
test here can see - or write into - a developer's real files.
"""

from pathlib import Path

import pytest

from bedrock.agents.capabilities.files import (
    CAPABILITIES,
    FilesConfig,
    FilesRead,
    FilesSearch,
    FilesWrite,
)
from bedrock.agents.contract import Ask, AskKind, Evidence, Plan, Risk, Step
from bedrock.agents.grants import GrantStore
from bedrock.agents.supervisor import Supervisor, SupervisorError


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "projects"
    (root / "protocols").mkdir(parents=True)
    (root / "protocols" / "amendment-3.md").write_text(
        "# Amendment 3\n\nEnrolment reached 412 patients by March.\n", encoding="utf-8"
    )
    (root / "notes.md").write_text("Nothing about numbers here.\n", encoding="utf-8")
    (root / "secret.key").write_text("not a served extension\n", encoding="utf-8")
    return root


@pytest.fixture
def config(workspace: Path, tmp_path: Path) -> FilesConfig:
    return FilesConfig(
        roots=(workspace,),
        output_root=tmp_path / "output",
        extensions=(".md",),
    )


def unapproved_write(**arguments: str) -> Plan:
    """A plan nobody has approved, for the supervisor to refuse."""
    return Plan(
        ask=Ask(
            kind=AskKind.CREATE,
            text="write up the enrolment numbers",
            evidence=Evidence(quote="can somebody write that up"),
        ),
        steps=(
            Step(
                tool="files.write",
                summary="Write notes/summary.md",
                risk=Risk.CHANGES_THINGS,
                arguments=arguments,
            ),
        ),
    )


# ---- What the roster finds ---------------------------------------------------


def test_the_module_offers_three_capabilities_at_two_risk_levels():
    names = {capability.name: capability.risk for capability in CAPABILITIES}
    assert names == {
        "files.search": Risk.READ_ONLY,
        "files.read": Risk.READ_ONLY,
        "files.write": Risk.CHANGES_THINGS,
    }


def test_describing_a_write_never_needs_configuration_or_a_grant():
    # The roster imports this module just to list what exists; that must not
    # read a config file, and it must certainly not write anything.
    described = FilesWrite().describe({"path": "notes/summary.md", "content": "hello"})
    assert "notes/summary.md" in described
    assert "Nothing is overwritten" in described


# ---- files.search ------------------------------------------------------------


async def test_search_matches_a_file_name_and_a_line_inside_a_file(config: FilesConfig):
    outcome = await FilesSearch(config).run({"query": "amendment"})
    assert outcome.failed is None
    assert any("amendment-3.md" in line for line in outcome.completed)

    outcome = await FilesSearch(config).run({"query": "enrolment 412"})
    assert any("412 patients" in line for line in outcome.completed)


async def test_search_skips_file_types_the_config_does_not_serve(config: FilesConfig):
    outcome = await FilesSearch(config).run({"query": "served"})
    assert outcome.failed is None
    assert not any("secret.key" in line for line in outcome.completed)


async def test_search_without_a_query_says_so(config: FilesConfig):
    outcome = await FilesSearch(config).run({})
    assert "needs a `query`" in (outcome.failed or "")


async def test_search_says_when_no_folders_are_configured():
    outcome = await FilesSearch(FilesConfig()).run({"query": "anything"})
    assert "No folders are configured" in (outcome.failed or "")


# ---- files.read --------------------------------------------------------------


async def test_read_returns_the_text_and_the_size(config: FilesConfig):
    outcome = await FilesRead(config).run({"path": "protocols/amendment-3.md"})
    assert outcome.failed is None
    assert "412 patients" in outcome.completed[1]
    assert "bytes" in outcome.completed[0]


async def test_read_refuses_a_path_that_climbs_out_of_the_root(
    config: FilesConfig, tmp_path: Path
):
    (tmp_path / "private.md").write_text("not for the agent", encoding="utf-8")
    outcome = await FilesRead(config).run({"path": "../private.md"})
    assert "outside the folders" in (outcome.failed or "")


async def test_read_refuses_an_absolute_path_outside_the_root(
    config: FilesConfig, tmp_path: Path
):
    outside = tmp_path / "private.md"
    outside.write_text("not for the agent", encoding="utf-8")
    outcome = await FilesRead(config).run({"path": str(outside)})
    assert "outside the folders" in (outcome.failed or "")


async def test_read_refuses_a_symlink_that_points_out_of_the_root(
    config: FilesConfig, workspace: Path, tmp_path: Path
):
    # The containment check runs after the path is resolved, which is the only
    # ordering that catches this: the link itself is inside the root.
    outside = tmp_path / "elsewhere.md"
    outside.write_text("not for the agent", encoding="utf-8")
    link = workspace / "shortcut.md"
    link.symlink_to(outside)

    outcome = await FilesRead(config).run({"path": "shortcut.md"})
    assert "outside the folders" in (outcome.failed or "")


async def test_read_says_plainly_when_a_file_is_not_there(config: FilesConfig):
    outcome = await FilesRead(config).run({"path": "protocols/nope.md"})
    assert "no file at" in (outcome.failed or "")


async def test_read_truncates_a_file_larger_than_the_limit(
    workspace: Path, tmp_path: Path
):
    (workspace / "big.md").write_text("x" * 5000, encoding="utf-8")
    config = FilesConfig(roots=(workspace,), output_root=tmp_path / "out", max_bytes=100)
    outcome = await FilesRead(config).run({"path": "big.md"})
    assert "first 100 bytes shown" in outcome.completed[0]
    assert len(outcome.completed[1]) == 100


# ---- files.write -------------------------------------------------------------


async def test_an_unapproved_write_never_reaches_the_capability(config: FilesConfig):
    # Where "inert without a grant" actually lives. ``protocol.Capability`` gives
    # run() nothing but its arguments, so the gate is the supervisor's: it
    # refuses before any capability is called, and no file appears.
    writer = FilesWrite(config)
    supervisor = Supervisor({"files.write": writer}, GrantStore(), _NoJournal())
    plan = unapproved_write(path="summary.md", content="hello")

    with pytest.raises(SupervisorError):
        await supervisor.execute(plan, None)
    assert not config.output_root.exists()


class _NoJournal:
    """A journal the supervisor never gets as far as writing to."""

    def append(self, operation_id: str, **_: object) -> int:  # pragma: no cover
        raise AssertionError("a capability ran without an approved plan")


async def test_write_creates_the_file_and_reports_where(config: FilesConfig):
    outcome = await FilesWrite(config).run(
        {"path": "notes/summary.md", "content": "hello"}
    )
    assert outcome.failed is None
    written = Path(outcome.artifacts[0])
    assert written.read_text(encoding="utf-8") == "hello"
    assert written == config.output_root / "notes" / "summary.md"


async def test_write_never_replaces_an_existing_file(config: FilesConfig):
    writer = FilesWrite(config)
    first = await writer.run({"path": "summary.md", "content": "first"})
    second = await writer.run({"path": "summary.md", "content": "second"})

    assert Path(first.artifacts[0]).read_text(encoding="utf-8") == "first"
    assert Path(second.artifacts[0]).name == "summary (2).md"
    assert "already existed" in second.completed[0]


async def test_write_refuses_an_absolute_path(config: FilesConfig):
    outcome = await FilesWrite(config).run(
        {"path": "/etc/passwd", "content": "no"}
    )
    assert "absolute path" in (outcome.failed or "")


async def test_write_refuses_a_path_that_climbs_out_of_the_output_root(config: FilesConfig):
    outcome = await FilesWrite(config).run(
        {"path": "../escaped.md", "content": "no"}
    )
    assert "points outside" in (outcome.failed or "")


async def test_write_cannot_reach_the_readable_roots(config: FilesConfig, workspace: Path):
    # Reading and writing have separate roots on purpose: being allowed to read
    # the protocol folder must never mean being allowed to rewrite it.
    outcome = await FilesWrite(config).run(
        {"path": str(workspace / "notes.md"), "content": "replaced"}
    )
    assert outcome.failed is not None
    unchanged = (workspace / "notes.md").read_text(encoding="utf-8")
    assert unchanged == "Nothing about numbers here.\n"
