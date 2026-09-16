"""`bedrock-biotech` — the same answers, without a full-screen interface.

The point of these tests is that the command line and the app agree. They
call the real argument parser and the real command implementation, with only
``load_settings`` redirected at a temporary directory, so a mistake in the
sub-parser wiring shows up here rather than on an operator's terminal.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from bedrock.biotech import cli
from bedrock.config import Settings
from bedrock.provenance import HistoryError

from tests.conftest import write_run_manifest


@pytest.fixture
def operator(tmp_path, monkeypatch) -> Settings:
    # Rich reads COLUMNS for its width. Pinning it keeps these assertions
    # about content rather than about where a table happened to wrap.
    monkeypatch.setenv("COLUMNS", "160")
    settings = Settings(
        user_id="alice",
        action_history_dir=str(tmp_path / "history"),
        biotech_work_dir=tmp_path / "biotech",
    )
    monkeypatch.setattr(cli, "load_settings", lambda: settings)
    write_run_manifest(settings.biotech_work_dir, "20260715T090000Z-aaaaaaaa", project_id="X")
    write_run_manifest(settings.biotech_work_dir, "20261115T090000Z-bbbbbbbb", project_id="Y")
    return settings


def run(*argv: str) -> int:
    return asyncio.run(cli._run(cli.build_parser().parse_args(list(argv))))


def test_runs_lists_with_filters(operator, capsys):
    assert run("runs", "--project", "X", "--since", "2026-07-01", "--before", "2026-10-01") == 0

    out = capsys.readouterr().out
    assert "20260715T090000Z-aaaaaaaa" in out
    assert "20261115T090000Z-bbbbbbbb" not in out
    assert "bedrock-biotech confidence" in out  # the CLI's own wording, not a slash command


def test_runs_show_and_verify(operator, capsys):
    assert run("runs", "show", "20260715T090000Z-aaaaaaaa") == 0
    shown = capsys.readouterr().out
    assert "monomer_ptm" in shown
    assert "candidate.fasta" in shown

    assert run("runs", "verify", "20260715T090000Z-aaaaaaaa") == 0
    assert "match" in capsys.readouterr().out


def test_runs_tag_is_recorded_and_visible_to_the_next_listing(operator, capsys):
    assert run("runs", "tag", "20261115T090000Z-bbbbbbbb", "--project", "X") == 0
    capsys.readouterr()

    assert run("runs", "--project", "X") == 0
    assert "20261115T090000Z-bbbbbbbb" in capsys.readouterr().out


def test_audit_reads_and_exports(operator, tmp_path, capsys):
    run("runs", "tag", "20261115T090000Z-bbbbbbbb", "--project", "X")
    capsys.readouterr()

    assert run("audit") == 0
    audit = capsys.readouterr().out
    assert "run.tag" in audit
    # The command line tells the operator a command line, not a slash command.
    assert "bedrock-biotech audit --after" in audit
    assert "/audit --after" not in audit

    destination = tmp_path / "actions.jsonl"
    assert run("audit", "export", str(destination)) == 0
    assert "exported 2 event" in capsys.readouterr().out
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["state"] for row in rows] == ["intent", "succeeded"]


def test_audit_export_refuses_an_existing_file(operator, tmp_path):
    destination = tmp_path / "actions.jsonl"
    destination.write_text("existing\n")

    with pytest.raises(HistoryError):
        run("audit", "export", str(destination))

    assert destination.read_text() == "existing\n"


def test_confidence_reports_an_unavailable_run_clearly(operator, capsys):
    """This fixture's manifest has no confidence artifacts, which is exactly
    the older-installation case the plan requires to stay readable."""
    assert run("confidence", "20260715T090000Z-aaaaaaaa") == 0

    out = capsys.readouterr().out
    assert "unavailable" in out.lower()
    assert "pLDDT estimates local model confidence" in out.replace("\n", " ")


def test_the_parser_accepts_a_project_on_fold():
    args = cli.build_parser().parse_args(["fold", "/approved/candidate.fasta", "--project", "X"])

    assert args.command == "fold"
    assert args.fasta == Path("/approved/candidate.fasta")
    assert args.project == "X"


def test_browsing_needs_no_scientific_tools(operator, capsys, monkeypatch):
    """`runs`/`confidence`/`audit` must not construct a workflow: an analyst
    reviewing results has no AlphaFold installation and does not need one."""
    def explode(*args, **kwargs):
        raise AssertionError("browsing must not build a scientific workflow")

    monkeypatch.setattr(cli.ProteinWorkflow, "from_settings", explode)

    assert run("runs") == 0
    assert run("audit") == 0
    assert "20260715T090000Z-aaaaaaaa" in capsys.readouterr().out
