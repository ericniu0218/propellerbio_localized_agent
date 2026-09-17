"""`/fold`, `/pymol` and `/tools` in the plain model mode.

Agent-pack mode has had these since the biotech integration; this mode used to
answer "Unknown command" while its own first-run check told people they could
use them. The wording comes from the same shared functions in
``bedrock.ui.fold_command``, so these tests check the wiring, and the one
deliberate difference: a fold here runs beside the chat instead of in its
queue, because a prediction takes hours and the model server is a different
machine from the GPU host.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from textual.widgets import Input

from bedrock.app import BedrockApp
from bedrock.biotech import ToolIntegrationError
from bedrock.config import Settings
from bedrock.ui.widgets import SystemNotice

from tests.conftest import (
    FakeBackend,
    FakeBiotechWorkflow,
    FakeRunCatalog,
    make_confidence_report,
    make_services,
)
from tests.workers import wait_for_app_workers


class BlockingWorkflow(FakeBiotechWorkflow):
    """A fold that lasts until the test says it is done, as a real one lasts hours."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def predict_and_render(self, fasta_path: Path, **kwargs):
        self.started.set()
        await self.release.wait()
        return await super().predict_and_render(fasta_path, **kwargs)


class BrokenWorkflow(FakeBiotechWorkflow):
    async def predict_and_render(self, fasta_path: Path, **kwargs):
        raise ToolIntegrationError("AlphaFold 2 is not ready: no launcher is configured")

    async def open_in_pymol(self, structure_path: Path | None = None) -> Path:
        raise ToolIntegrationError("PyMOL is not ready: pymol was not found")


def make_app(tmp_path: Path, *, backend=None, workflow=None, services=None) -> BedrockApp:
    exports = tmp_path / "registry-exports"
    exports.mkdir(exist_ok=True)
    (exports / "SEQ-00123.fasta").write_text(">SEQ-00123 kinase\nACDEFGHIK\n")
    (exports / "STRUCT-9.pdb").write_text(
        "ATOM      1  N   MET A   1      11.104  13.207  10.000  1.00 20.00           N\n"
    )
    return BedrockApp(
        settings=Settings(
            model="test-model",
            user_id="alice",
            context_dir=str(tmp_path / "context"),
            biotech_work_dir=tmp_path / "work",
            connectors={"registry": {"type": "share", "root": str(exports)}},
        ),
        backend=backend or FakeBackend(),
        biotech_workflow=workflow if workflow is not None else FakeBiotechWorkflow(tmp_path),
        services=services or make_services(),
    )


async def submit(app: BedrockApp, pilot, text: str) -> None:
    """Type a line and let everything it started finish."""
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await wait_for_app_workers(app)
    await pilot.pause()


async def type_line(app: BedrockApp, pilot, text: str) -> None:
    """Type a line without waiting for what it starts - for a fold left running."""
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await pilot.pause()


def notices(app: BedrockApp) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def until(condition, pilot, attempts: int = 200) -> None:
    for _ in range(attempts):
        if condition():
            return
        await pilot.pause()
    raise AssertionError("condition never became true")


# ---- /fold -------------------------------------------------------------------


async def test_fold_runs_and_reports_like_agent_pack_mode(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    services = make_services(
        runs=FakeRunCatalog(reports={"run_test": make_confidence_report("run_test", model_id="model_3")})
    )
    app = make_app(tmp_path, workflow=workflow, services=services)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/fold /tmp/protein.fasta")

        assert workflow.folded == [(Path("/tmp/protein.fasta"), None)]
        text = notices(app)
        assert "Started AlphaFold 2 for `/tmp/protein.fasta`" in text
        assert "Protein prediction completed" in text
        assert "ranked_0.pdb" in text and "preview.png" in text
        assert "pLDDT 90 or above" in text and "model_3" in text
        assert "/confidence run_test" in text
        assert app._scientific_job_running is False


async def test_fold_accepts_a_quoted_path_and_a_project_label(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, '/fold "/approved/my candidate.fasta" --project X')

        assert workflow.folded == [(Path("/approved/my candidate.fasta"), "X")]
        assert "for project `X`" in notices(app)


async def test_nothing_or_a_typo_never_starts_a_job(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/fold")
        await submit(app, pilot, "/fold candidate.fasta --project")

        assert workflow.folded == []
        errors = [str(n.source) for n in app.query(SystemNotice) if n.has_class("error")]
        assert any("Tell me which one-protein FASTA file to fold" in e for e in errors)
        assert any("expected one argument" in e for e in errors)
        assert app._scientific_job_running is False


async def test_fold_pulls_the_sequence_from_a_company_source_with_provenance(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/fold registry://SEQ-00123.fasta --project X")

        staged, project = workflow.folded[0]
        assert project == "X"
        assert staged.parent == tmp_path / "work" / "sources"
        assert staged.read_text().startswith(">SEQ-00123 kinase")
        assert workflow.provenance[0]["ref"] == "registry://SEQ-00123.fasta"
        assert workflow.provenance[0]["fetched_by"] == "alice"
        assert "Source: `registry://SEQ-00123.fasta`" in notices(app)


async def test_a_refused_fetch_or_a_missing_tool_is_reported_and_frees_the_slot(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/fold registry://../outside.fasta")
        assert "points outside the `registry` share" in notices(app)
        assert workflow.folded == []
        assert app._scientific_job_running is False

        # The slot really is free: the next fold starts.
        await submit(app, pilot, "/fold /tmp/protein.fasta")
        assert workflow.folded == [(Path("/tmp/protein.fasta"), None)]

    broken = make_app(tmp_path, workflow=BrokenWorkflow(tmp_path))
    async with broken.run_test() as pilot:
        await submit(broken, pilot, "/fold /tmp/protein.fasta")
        assert "AlphaFold 2 is not ready" in notices(broken)
        assert broken._scientific_job_running is False


async def test_a_running_fold_holds_its_slot_but_not_the_chat(tmp_path):
    """One fold at a time - and while it runs, the model still answers, and
    Esc (which stops a streaming reply) does not stop the fold."""
    workflow = BlockingWorkflow(tmp_path)
    backend = FakeBackend(reply="Fold jobs take a while.")
    app = make_app(tmp_path, backend=backend, workflow=workflow)
    async with app.run_test() as pilot:
        await type_line(app, pilot, "/fold /tmp/first.fasta")
        await workflow.started.wait()
        assert app._scientific_job_running is True

        await type_line(app, pilot, "/fold /tmp/second.fasta")
        assert "A scientific job is already running." in notices(app)

        await type_line(app, pilot, "how long does a fold take?")
        await until(lambda: backend.requests and app.queue_snapshot().is_idle, pilot)
        assert backend.requests[0][1][-1]["content"] == "how long does a fold take?"

        await pilot.press("escape")
        await pilot.pause()
        assert app._scientific_job_running is True

        workflow.release.set()
        await until(lambda: "Protein prediction completed" in notices(app), pilot)
        assert workflow.folded == [(Path("/tmp/first.fasta"), None)]
        assert app._scientific_job_running is False


async def test_quitting_while_a_fold_runs_closes_cleanly(tmp_path):
    """Quitting cancels the fold, and the screen is gone by the time the
    "cancelled" notice would be shown. That used to fail the worker, so the
    app ended with a traceback instead of simply closing."""
    workflow = BlockingWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await type_line(app, pilot, "/fold /tmp/first.fasta")
        await workflow.started.wait()
        await pilot.press("ctrl+c")
        await pilot.pause()
    assert app.return_code == 0
    assert app._exception is None


async def test_a_plain_english_request_still_goes_to_the_model(tmp_path):
    """Only agent-pack mode reads requests like this as jobs. Here the employee
    is talking to a model, so the words go to the model."""
    workflow = FakeBiotechWorkflow(tmp_path)
    backend = FakeBackend()
    app = make_app(tmp_path, backend=backend, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "Run AlphaFold 2 on candidate.fasta")

        assert workflow.folded == []
        assert backend.requests[0][1][-1]["content"] == "Run AlphaFold 2 on candidate.fasta"


# ---- /pymol and /tools ---------------------------------------------------------


async def test_pymol_opens_the_latest_prediction_or_a_named_file(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/pymol")
        await submit(app, pilot, '/pymol "/data/my structure.pdb"')

        assert workflow.opened == [None, Path("/data/my structure.pdb")]
        assert "Opened `/data/my structure.pdb` in PyMOL." in notices(app)


async def test_pymol_fetches_a_structure_from_a_company_source(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/pymol registry://STRUCT-9.pdb")

        (opened,) = workflow.opened
        # A staged file is named, never None: None would colour it as if it
        # were this machine's own verified prediction.
        assert opened is not None and opened.parent == tmp_path / "work" / "sources"
        assert opened.read_text().startswith("ATOM")
        assert (opened.parent / (opened.name + ".provenance.json")).is_file()
        text = notices(app)
        assert "Opened `registry://STRUCT-9.pdb` in PyMOL (version modified " in text
        assert "sha256 `" in text and opened.name in text


async def test_pymol_problems_are_reported_not_raised(tmp_path):
    workflow = FakeBiotechWorkflow(tmp_path)
    app = make_app(tmp_path, workflow=workflow)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/pymol registry://../outside.pdb")
        await submit(app, pilot, '/pymol "unterminated')
        assert workflow.opened == []
        errors = [str(n.source) for n in app.query(SystemNotice) if n.has_class("error")]
        assert any("points outside the `registry` share" in e for e in errors)
        assert any("Could not read those arguments" in e for e in errors)

    broken = make_app(tmp_path, workflow=BrokenWorkflow(tmp_path))
    async with broken.run_test() as pilot:
        await submit(broken, pilot, "/pymol")
        assert "PyMOL is not ready" in notices(broken)


async def test_tools_reports_what_is_set_up(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/tools")
        text = notices(app)
        assert "Ready: **AlphaFold 2**" in text
        assert "Ready: **open-source PyMOL**" in text


async def test_help_lists_the_scientific_commands(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/help")
        text = notices(app)
        assert "`/fold <fasta-path or source://id> [--project X]`" in text
        assert "`/pymol [structure-path or source://id]`" in text
        assert "`/tools`" in text


async def test_the_workflow_is_not_built_until_a_scientific_command_needs_it(tmp_path, monkeypatch):
    """Chatting must not depend on AlphaFold settings being readable."""
    from bedrock.biotech import ProteinWorkflow

    built = []
    monkeypatch.setattr(ProteinWorkflow, "from_settings", classmethod(lambda cls, s, **k: built.append(s) or FakeBiotechWorkflow(tmp_path)))
    app = BedrockApp(settings=Settings(model="test-model"), backend=FakeBackend(), services=make_services())
    async with app.run_test() as pilot:
        await submit(app, pilot, "hello")
        assert built == []
        await submit(app, pilot, "/tools")
        assert len(built) == 1
