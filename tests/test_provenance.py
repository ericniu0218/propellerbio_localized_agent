from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from bedrock.provenance import HistoryError, HistoryStore, RunCatalog, RunError


def test_history_is_durable_idempotent_and_distinguishes_repeated_actions(tmp_path: Path):
    root = tmp_path / "history"
    history = HistoryStore(root)

    first = history.append(
        "op-1", actor_id="alice", action="digest.ingest", state="intent"
    )
    assert history.append(
        "op-1", actor_id="alice", action="digest.ingest", state="intent"
    ) == first
    history.append(
        "op-1",
        actor_id="alice",
        action="digest.ingest",
        state="succeeded",
        subject_id="same-artifact",
        details={"artifact_id": "same-artifact"},
    )

    restarted = HistoryStore(root)
    restarted.append(
        "op-2", actor_id="alice", action="digest.ingest", state="intent"
    )
    restarted.append(
        "op-2",
        actor_id="alice",
        action="digest.ingest",
        state="succeeded",
        subject_id="same-artifact",
        details={"artifact_id": "same-artifact"},
    )

    events = restarted.events()
    assert [(item["operation_id"], item["state"]) for item in events] == [
        ("op-1", "intent"),
        ("op-1", "succeeded"),
        ("op-2", "intent"),
        ("op-2", "succeeded"),
    ]
    assert all(item["schema_version"] == 1 for item in events)
    assert all(item["details"]["actor_source"] == "config" for item in events)
    assert root.stat().st_mode & 0o777 == 0o700
    assert restarted.database_path.stat().st_mode & 0o777 == 0o600


def test_history_rejects_changed_replay_and_orphaned_outcome(tmp_path: Path):
    history = HistoryStore(tmp_path / "history")

    with pytest.raises(HistoryError, match="matching committed intent"):
        history.append("op", actor_id="alice", action="approval", state="unknown")

    history.append("op", actor_id="alice", action="approval", state="intent")
    history.append("op", actor_id="alice", action="approval", state="unknown")

    with pytest.raises(HistoryError, match="different event content"):
        history.append("op", actor_id="alice", action="approval", state="succeeded")
    with pytest.raises(HistoryError, match="different event content"):
        history.append("op", actor_id="bob", action="approval", state="intent")


def test_history_serializes_concurrent_duplicate_writers(tmp_path: Path):
    history = HistoryStore(tmp_path / "history")

    with ThreadPoolExecutor(max_workers=4) as pool:
        sequences = list(
            pool.map(
                lambda _: history.append(
                    "one-operation", actor_id="alice", action="fold", state="intent"
                ),
                range(16),
            )
        )

    assert len(set(sequences)) == 1
    assert len(history.events()) == 1


def test_history_export_is_a_private_ordered_snapshot_and_never_overwrites(tmp_path: Path):
    history = HistoryStore(tmp_path / "history")
    history.append("op", actor_id="alice", action="fold", state="intent")
    history.append(
        "op", actor_id="alice", action="fold", state="failed", details={"error_code": "tool"}
    )
    destination = tmp_path / "exports" / "actions.jsonl"

    assert history.export(destination) == 2
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == [1, 2]
    assert destination.stat().st_mode & 0o777 == 0o600

    with pytest.raises(HistoryError, match="already exists"):
        history.export(destination)


@pytest.mark.parametrize(
    "details",
    [
        {"prompt": "private text"},
        {"source_bytes": []},
        {"error_code": "x" * 9000},
        {"decision": "maybe"},
    ],
)
def test_history_accepts_only_bounded_allowlisted_metadata(tmp_path: Path, details: dict):
    history = HistoryStore(tmp_path / "history")

    with pytest.raises(HistoryError):
        history.append(
            "op", actor_id="alice", action="fold", state="intent", details=details
        )


@pytest.mark.parametrize("action", ["litreview.review", "litreview.search"])
def test_history_accepts_the_literature_review_actions(tmp_path: Path, action: str):
    """Finding 14: reviews and the literature search send something off this
    machine, and `/audit` must be able to record both."""
    history = HistoryStore(tmp_path / "history")

    history.append("op", actor_id="alice", action=action, state="intent")
    history.append("op", actor_id="alice", action=action, state="succeeded")

    events = history.events()
    assert [item["state"] for item in events] == ["intent", "succeeded"]
    assert all(item["action"] == action for item in events)


def test_history_still_rejects_an_unknown_action(tmp_path: Path):
    history = HistoryStore(tmp_path / "history")

    with pytest.raises(HistoryError, match="Unsupported history action"):
        history.append("op", actor_id="alice", action="litreview.summarize", state="intent")


def test_history_actions_states_and_detail_fields_match_the_reference_copy():
    """`docs/compliance_mvp/reference.py` is a second, hand-maintained copy of
    these three sets, kept around on purpose as the frozen contract the two
    original agents built against (see COMPLIANCE_MVP.md). Nothing enforced
    the two staying equal - this is that enforcement, so adding an action
    here without updating the reference (or vice versa) fails the suite
    instead of silently drifting apart."""
    from docs.compliance_mvp import reference

    assert HistoryStore.ACTIONS == reference.Journal.ACTIONS
    assert HistoryStore.STATES == reference.Journal.STATES
    assert HistoryStore.DETAIL_FIELDS == reference.Journal.FIELDS


def _legacy_run(
    work_dir: Path,
    run_id: str,
    created_at: str,
    *,
    status: str = "completed",
    project_id: str | None = None,
) -> Path:
    run_dir = work_dir / "runs" / run_id
    run_dir.mkdir(parents=True)
    source = run_dir / "source.fasta"
    source.write_text(">candidate\nACD\n")
    structure = run_dir / "ranked_0.pdb"
    structure.write_text("ATOM\n")
    import hashlib

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": created_at,
        "status": status,
        "source": {
            "snapshot": "source.fasta",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "residue_count": 3,
        },
        "outputs": {
            "structure": {
                "path": "ranked_0.pdb",
                "sha256": hashlib.sha256(structure.read_bytes()).hexdigest(),
                "bytes": structure.stat().st_size,
            }
        },
    }
    if project_id is not None:
        manifest["project_id"] = project_id
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def test_run_catalog_filters_exact_utc_quarter_boundaries_and_warns_on_corruption(
    tmp_path: Path,
):
    work = tmp_path / "biotech"
    history = HistoryStore(tmp_path / "history")
    _legacy_run(work, "before", "2026-06-30T23:59:59+00:00", project_id="X")
    _legacy_run(work, "start", "2026-07-01T00:00:00+00:00", project_id="X")
    _legacy_run(work, "end", "2026-09-30T23:59:59+00:00", project_id="X")
    _legacy_run(work, "after", "2026-10-01T00:00:00+00:00", project_id="X")
    broken = work / "runs" / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("{not-json")

    result = RunCatalog(work, history).list(
        project_id="X", since="2026-07-01", before="2026-10-01"
    )

    assert [item["run_id"] for item in result["items"]] == ["end", "start"]
    assert result["total"] == 2
    assert result["next_offset"] is None
    assert any("broken" in warning for warning in result["warnings"])


def test_run_catalog_does_not_follow_symlinked_run_directories(tmp_path: Path):
    work = tmp_path / "biotech"
    runs = work / "runs"
    runs.mkdir(parents=True)
    outside_work = tmp_path / "outside"
    outside_manifest = _legacy_run(
        outside_work, "escaped", "2026-07-01T00:00:00+00:00"
    )
    (runs / "escaped").symlink_to(outside_manifest.parent, target_is_directory=True)

    page = RunCatalog(work, HistoryStore(tmp_path / "history")).list()

    assert page["items"] == []
    assert any("escaped" in warning for warning in page["warnings"])


def test_run_catalog_paginates_stably_and_filters_status(tmp_path: Path):
    work = tmp_path / "biotech"
    history = HistoryStore(tmp_path / "history")
    _legacy_run(work, "run-a", "2026-07-01T00:00:00+00:00")
    _legacy_run(work, "run-b", "2026-07-01T00:00:00+00:00")
    _legacy_run(work, "run-c", "2026-07-02T00:00:00+00:00", status="failed")

    first = RunCatalog(work, history).list(status="completed", limit=1)
    second = RunCatalog(work, history).list(status="completed", limit=1, offset=1)

    assert first["items"][0]["run_id"] == "run-b"
    assert first["next_offset"] == 1
    assert second["items"][0]["run_id"] == "run-a"
    assert second["next_offset"] is None


def test_run_tag_is_append_only_and_latest_success_overrides_manifest(tmp_path: Path):
    work = tmp_path / "biotech"
    history = HistoryStore(tmp_path / "history")
    manifest_path = _legacy_run(
        work, "old-run", "2026-01-01T00:00:00+00:00", project_id="original"
    )
    before = manifest_path.read_bytes()
    catalog = RunCatalog(work, history)

    catalog.tag("old-run", "project-X", actor_id="alice", operation_id="tag-1")
    catalog.tag("old-run", "project-Y", actor_id="alice", operation_id="tag-2")

    assert manifest_path.read_bytes() == before
    assert catalog.show("old-run")["project_id"] == "project-Y"
    assert catalog.list(project_id="project-Y")["total"] == 1
    assert [event["action"] for event in history.events()] == [
        "run.tag",
        "run.tag",
        "run.tag",
        "run.tag",
    ]


def test_run_verification_reports_matches_changes_and_unrecorded_legacy_files(tmp_path: Path):
    work = tmp_path / "biotech"
    history = HistoryStore(tmp_path / "history")
    manifest_path = _legacy_run(work, "run-1", "2026-01-01T00:00:00+00:00")
    run_dir = manifest_path.parent
    (run_dir / "input.fasta").write_text(">bedrock_target\nACD\n")
    (run_dir / "ranking_debug.json").write_text('{"order":["model_1"]}\n')

    first = RunCatalog(work, history).verify("run-1")
    assert {row["path"]: row["state"] for row in first} == {
        "source.fasta": "match",
        "ranked_0.pdb": "match",
        "input.fasta": "unrecorded",
        "ranking_debug.json": "unrecorded",
    }

    (run_dir / "ranked_0.pdb").write_text("CHANGED\n")
    second = RunCatalog(work, history).verify("run-1")
    structure = next(row for row in second if row["path"] == "ranked_0.pdb")
    assert structure["state"] == "mismatch"
    assert structure["expected_sha256"] != structure["actual_sha256"]


def test_run_verification_rejects_traversal_and_symlink_outputs(tmp_path: Path):
    work = tmp_path / "biotech"
    history = HistoryStore(tmp_path / "history")
    manifest_path = _legacy_run(work, "run-1", "2026-01-01T00:00:00+00:00")
    outside = tmp_path / "private.txt"
    outside.write_text("secret")
    link = manifest_path.parent / "linked.pdb"
    link.symlink_to(outside)
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["traversal"] = {"path": "../../private.txt", "sha256": "0" * 64}
    manifest["outputs"]["symlink"] = {"path": "linked.pdb", "sha256": "0" * 64}
    manifest_path.write_text(json.dumps(manifest))

    rows = RunCatalog(work, history).verify("run-1")
    states = {row["path"]: row["state"] for row in rows}

    assert states["../../private.txt"] == "invalid"
    assert states["linked.pdb"] == "invalid"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"since": "bad"}, "YYYY-MM-DD"),
        ({"since": "2026-10-01", "before": "2026-07-01"}, "before"),
        ({"limit": 101}, "between 1 and 100"),
        ({"offset": -1}, "non-negative"),
    ],
)
def test_run_catalog_rejects_invalid_queries(tmp_path: Path, kwargs: dict, message: str):
    catalog = RunCatalog(tmp_path / "biotech", HistoryStore(tmp_path / "history"))

    with pytest.raises(RunError, match=message):
        catalog.list(**kwargs)
