"""`/sources` and ``<source>://<id>`` references, driven through the real app.

The source here is a share connector over a temporary folder, so nothing
leaves the test machine.
"""

from pathlib import Path

from textual.widgets import Input

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.ui.widgets import SystemNotice

from tests.conftest import FakeBackend

CSV = "\n".join(["well,od"] + [f"A{i},{0.1 * i:.1f}" for i in range(1, 40)])


def make_app(tmp_path: Path, backend: FakeBackend, connectors: dict) -> BedrockApp:
    return BedrockApp(
        settings=Settings(
            model="test-model",
            user_id="alice",
            context_dir=str(tmp_path / "context"),
            biotech_work_dir=tmp_path / "work",
            connectors=connectors,
        ),
        backend=backend,
    )


def exports(tmp_path: Path) -> Path:
    root = tmp_path / "lims-exports"
    (root / "plates").mkdir(parents=True)
    (root / "plates" / "P42.csv").write_text(CSV)
    return root


def notices(app: BedrockApp) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def submit(app: BedrockApp, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_digest_pulls_a_record_straight_from_a_company_source(tmp_path, fake_backend):
    app = make_app(tmp_path, fake_backend, {"lims": {"type": "share", "root": str(exports(tmp_path))}})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/digest lims://plates/P42.csv")

        text = notices(app)
        assert "Fetched `lims://plates/P42.csv`" in text
        assert "as `alice`" in text
        assert "Compressed `P42.csv`" in text
        stored = app.context_store.list()
        assert len(stored) == 1
        assert stored[0].source_kind == "tabular"


async def test_sources_lists_connectors_and_reports_broken_config(tmp_path, fake_backend):
    connectors = {
        "lims": {"type": "share", "root": str(exports(tmp_path))},
        "eln": {"type": "sharepoint"},
    }
    app = make_app(tmp_path, fake_backend, connectors)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/sources")

        text = notices(app)
        assert "`lims`" in text and "| ok |" in text
        assert "[connectors.eln]" in text


async def test_sources_search_then_fetch_leaves_a_file_with_provenance(tmp_path, fake_backend):
    app = make_app(tmp_path, fake_backend, {"lims": {"type": "share", "root": str(exports(tmp_path))}})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/sources search lims p42")
        assert "`lims://plates/P42.csv`" in notices(app)

        await submit(app, pilot, "/sources fetch lims://plates/P42.csv")
        staged = list((tmp_path / "work" / "sources").glob("*-P42.csv"))
        assert len(staged) == 1
        assert staged[0].read_text() == CSV
        assert staged[0].with_name(staged[0].name + ".provenance.json").is_file()


async def test_unknown_and_out_of_bounds_references_are_plain_messages(tmp_path, fake_backend):
    app = make_app(tmp_path, fake_backend, {"lims": {"type": "share", "root": str(exports(tmp_path))}})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/digest regsitry://SEQ-1")
        await submit(app, pilot, "/digest lims://../../etc/passwd")

        text = notices(app)
        assert "No source called `regsitry`" in text
        assert "points outside the `lims` share" in text
        assert app.context_store.list() == []


async def test_sources_with_nothing_configured_shows_how_to_add_one(tmp_path, fake_backend):
    app = make_app(tmp_path, fake_backend, {})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/sources")
        assert "[connectors.registry]" in notices(app)
