"""`/runs` driven through the real Textual apps, against the real catalogue.

These tests deliberately use Agent A's ``RunCatalog`` and ``HistoryStore``
over manifests written to a temporary directory, rather than a stub. The
question they answer is not "does the renderer format a dict" but "does an
employee who types `/runs --project X --since ...` see the right runs" - and
the filters, the sort order and the pagination arithmetic are part of that.

What they assert on is the text that ended up in the widget, because that is
what the employee reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from textual.widgets import Input

from bedrock.agentpack_app import AgentPackApp
from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.ui.widgets import SystemNotice

from tests.conftest import (
    FakeAgentPackClientUnavailable,
    FakeBackend,
    real_services,
    write_run_manifest,
)


def make_app(tmp_path, *, connected: bool = True) -> BedrockApp:
    services = real_services(tmp_path)
    return BedrockApp(
        settings=services.settings,
        backend=FakeBackend(connected=connected),
        services=services,
    )


def notices(app) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def submit(app, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


def seed_runs(app) -> Path:
    work_dir = Path(app.settings.biotech_work_dir)
    write_run_manifest(work_dir, "20260630T090000Z-aaaaaaaa", project_id="X")
    write_run_manifest(work_dir, "20260701T090000Z-bbbbbbbb", project_id="X")
    write_run_manifest(work_dir, "20260915T090000Z-cccccccc", project_id="X")
    write_run_manifest(work_dir, "20261001T090000Z-dddddddd", project_id="X")
    write_run_manifest(work_dir, "20260801T090000Z-eeeeeeee", project_id="Y")
    write_run_manifest(work_dir, "20260802T090000Z-ffffffff")  # legacy: unassigned
    return work_dir


async def test_a_quarter_of_one_project_uses_inclusive_since_and_exclusive_before(tmp_path):
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs --project X --since 2026-07-01 --before 2026-10-01")

        text = notices(app)
        assert "20260701T090000Z-bbbbbbbb" in text  # --since includes its day
        assert "20260915T090000Z-cccccccc" in text
        assert "20260630T090000Z-aaaaaaaa" not in text  # before the quarter
        assert "20261001T090000Z-dddddddd" not in text  # --before excludes its day
        assert "20260801T090000Z-eeeeeeee" not in text  # project Y
        assert "showing 1–2 of 2" in text


async def test_an_empty_result_says_so_plainly(tmp_path):
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs --project nothing-here")

        assert "none match" in notices(app)


async def test_the_next_page_hint_can_be_typed_straight_back_in(tmp_path):
    """The hint is only useful if it round-trips through the same splitting
    that read the original command."""
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs --project X --limit 2")
        hint = [
            line for line in notices(app).splitlines() if line.startswith("Next page:")
        ][0]
        await submit(app, pilot, hint.split("`")[1])

        text = notices(app)
        assert "showing 3–4 of 4" in text
        assert "20260630T090000Z-aaaaaaaa" in text


async def test_pagination_reports_totals_and_offers_the_next_page(tmp_path):
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs --limit 2")

        text = notices(app)
        assert "showing 1–2 of 6" in text
        assert "--offset 2" in text
        # Newest first, so the two most recent runs are the ones shown.
        assert "20261001T090000Z-dddddddd" in text
        assert "20260915T090000Z-cccccccc" in text
        assert "20260630T090000Z-aaaaaaaa" not in text


async def test_legacy_runs_are_listed_as_unassigned_until_tagged(tmp_path):
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs")
        assert "unassigned" in notices(app)

        await submit(app, pilot, "/runs tag 20260802T090000Z-ffffffff --project X")
        await submit(app, pilot, "/runs --project X")

        assert "20260802T090000Z-ffffffff" in notices(app)


async def test_tagging_records_history_and_never_edits_the_manifest(tmp_path):
    app = make_app(tmp_path)
    work_dir = seed_runs(app)
    manifest_path = work_dir / "runs" / "20260802T090000Z-ffffffff" / "manifest.json"
    before = manifest_path.read_bytes()

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs tag 20260802T090000Z-ffffffff --project X")

        assert manifest_path.read_bytes() == before
        assert "grants no access" in notices(app)

        await submit(app, pilot, "/audit")
        audit = notices(app)
        assert "run.tag" in audit
        assert "project_id=X" in audit


async def test_a_corrupt_manifest_warns_without_hiding_the_valid_runs(tmp_path):
    app = make_app(tmp_path)
    work_dir = seed_runs(app)
    broken = work_dir / "runs" / "20260901T090000Z-99999999"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs")

        text = notices(app)
        assert "Warning" in text and "20260901T090000Z-99999999" in text
        assert "20260915T090000Z-cccccccc" in text


async def test_detail_shows_settings_lineage_and_integrity(tmp_path):
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs show 20260915T090000Z-cccccccc")

        text = notices(app)
        assert "monomer_ptm" in text            # settings
        assert "candidate.fasta" in text        # source lineage
        assert "ranked_0.pdb" in text           # output lineage
        assert "Last recorded status" in text   # never "is running"
        assert "match" in text                  # verification ran
        assert "not proof" in text              # and is described honestly


async def test_verify_reports_a_changed_file_as_a_mismatch(tmp_path):
    app = make_app(tmp_path)
    work_dir = seed_runs(app)
    (work_dir / "runs" / "20260915T090000Z-cccccccc" / "ranked_0.pdb").write_text("ATOM CHANGED\n")

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs verify 20260915T090000Z-cccccccc")

        assert "mismatch" in notices(app)


async def test_a_quoted_project_label_survives_the_argument_split(tmp_path):
    app = make_app(tmp_path)
    work_dir = Path(app.settings.biotech_work_dir)
    write_run_manifest(work_dir, "20260915T090000Z-11111111")

    async with app.run_test() as pilot:
        await submit(app, pilot, '/runs tag 20260915T090000Z-11111111 --project "Project.X-2026"')
        await submit(app, pilot, '/runs --project "Project.X-2026"')

        assert "20260915T090000Z-11111111" in notices(app)


async def test_a_file_name_cannot_smuggle_control_characters_into_the_screen(tmp_path):
    """A run whose recorded file name contains an escape sequence must not be
    able to repaint the employee's terminal when it is listed."""
    app = make_app(tmp_path)
    work_dir = Path(app.settings.biotech_work_dir)
    run_dir = write_run_manifest(work_dir, "20260915T090000Z-22222222")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source"]["original_filename"] = "candidate\x1b[2Jevil.fasta"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs show 20260915T090000Z-22222222")

        text = notices(app)
        assert "\x1b" not in text
        assert "evil.fasta" in text


@pytest.mark.parametrize(
    "command, expected",
    [
        ("/runs --limit 500", "between 1 and 100"),
        ("/runs --since yesterday", "YYYY-MM-DD"),
        ("/runs --since 2026-10-01 --before 2026-07-01", "later than"),
        ("/runs --limit two", "invalid int"),
        ('/runs --project "unbalanced', "Could not read those arguments"),
        ("/runs show", "exactly one run ID"),
        ("/runs tag 20260915T090000Z-cccccccc", "required"),
    ],
)
async def test_bad_syntax_is_a_message_not_an_exit(tmp_path, command, expected):
    app = make_app(tmp_path)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, command)

        assert expected in notices(app)
        assert app.is_running  # still there, still usable


async def test_browsing_works_with_the_model_server_offline(tmp_path):
    app = make_app(tmp_path, connected=False)
    seed_runs(app)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs --project X")

        assert "20260915T090000Z-cccccccc" in notices(app)
        assert app.backend.requests == []  # nothing was asked of a model


async def test_browsing_works_in_agentpack_mode_with_the_ai_system_unreachable(tmp_path):
    """The agent-pack app cannot even open a session here; local browsing is
    unaffected, because it only ever reads this machine's own files."""
    services = real_services(tmp_path)
    write_run_manifest(Path(services.settings.biotech_work_dir), "20260915T090000Z-33333333")
    app = AgentPackApp(
        settings=services.settings,
        client=FakeAgentPackClientUnavailable(),
        services=services,
    )

    async with app.run_test() as pilot:
        app.query_one("#prompt", Input).value = "/runs"
        await pilot.press("enter")
        for _ in range(100):
            await pilot.pause()
            if "20260915T090000Z-33333333" in notices(app):
                break

        assert "20260915T090000Z-33333333" in notices(app)


async def test_legacy_warnings_are_preserved_on_the_detail_page(tmp_path):
    """Two different kinds of warning meet here: the catalogue's note that
    this is an older manifest, and the run's own record that its rendering
    failed. Neither may be dropped in favour of the other."""
    app = make_app(tmp_path)
    work_dir = Path(app.settings.biotech_work_dir)
    run_dir = write_run_manifest(work_dir, "20260915T090000Z-44444444")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    manifest["status"] = "completed_with_render_error"
    manifest["warnings"] = ["headless renderer unavailable"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    async with app.run_test() as pilot:
        await submit(app, pilot, "/runs show 20260915T090000Z-44444444")

        text = notices(app)
        assert "Legacy v1 manifest" in text
        assert "headless renderer unavailable" in text
        assert "completed_with_render_error" in text
