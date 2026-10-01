"""`/audit`, and the history that `/digest` writes into it.

The interesting cases here are all failure cases, because a journal that only
works when everything works is not worth having. Three rules are tested by
name: nothing starts if its intent cannot be recorded; a result that really
happened is kept even when its outcome cannot be recorded; and nothing the
employee does to their *screen* removes anything from history.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from textual.widgets import Input

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.context import ContextStore
from bedrock.ui.digest_command import run_digest_command
from bedrock.ui.widgets import SystemNotice

from tests.conftest import FakeBackend, FakeHistoryStore, make_services, real_services

CSV = "\n".join(["sample,ph"] + [f"S{i:04d},{6.0 + i % 20 * 0.1:.1f}" for i in range(60)])


def make_app(tmp_path, *, services=None) -> BedrockApp:
    services = services or real_services(tmp_path)
    return BedrockApp(settings=services.settings, backend=FakeBackend(), services=services)


def notices(app) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def submit(app, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def collected(messages: list[tuple[str, str]]):
    async def notify(text: str, kind: str = "info") -> None:
        messages.append((text, kind))

    return notify


# ---- What ends up in the journal -------------------------------------------


async def test_ingesting_the_same_file_twice_is_two_operations_and_one_artifact(tmp_path):
    """Content addressing makes the second ingest cheap; it does not make it
    a non-event. The employee did the thing twice and history says so."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        await submit(app, pilot, f"/digest {path}")

        assert len(app.context_store.list()) == 1
        history = await app.services.history()
        ingests = [row for row in history.events(limit=100) if row["action"] == "digest.ingest"]
        assert [row["state"] for row in ingests] == [
            "intent", "succeeded", "intent", "succeeded",
        ]
        operations = {row["operation_id"] for row in ingests}
        assert len(operations) == 2
        artifacts = {row["details"].get("artifact_id") for row in ingests if row["details"].get("artifact_id")}
        assert len(artifacts) == 1


async def test_forgetting_a_digest_and_clearing_the_chat_both_preserve_history(tmp_path):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    app = make_app(tmp_path)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        artifact_id = app.context_store.list()[0].id

        await submit(app, pilot, f"/digest forget {artifact_id}")
        assert app.context_store.list() == []

        await submit(app, pilot, "/clear")
        await submit(app, pilot, "/audit")

        text = notices(app)
        assert "digest.ingest" in text
        assert "digest.forget" in text


async def test_the_audit_page_calls_an_unconfirmed_outcome_unconfirmed(tmp_path):
    services = real_services(tmp_path)
    history = await services.history()
    history.append(
        "op-still-running",
        actor_id="alice",
        action="fold",
        state="intent",
        subject_id="20260915T090000Z-abcdabcd",
        details={"project_id": "X", "actor_source": "config"},
    )
    app = make_app(tmp_path, services=services)

    async with app.run_test() as pilot:
        await submit(app, pilot, "/audit")

        text = notices(app)
        assert "about to start" in text
        assert "never confirmed" in text
        assert "confirmed failure" not in text  # no failure was invented for it


async def test_export_writes_jsonl_to_a_quoted_path(tmp_path):
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    destination = tmp_path / "review notes"
    destination.mkdir()
    app = make_app(tmp_path)

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/digest {path}")
        await submit(app, pilot, f'/audit export "{destination / "actions.jsonl"}"')

        written = (destination / "actions.jsonl").read_text().strip().splitlines()
        assert len(written) == 2
        assert json.loads(written[0])["action"] == "digest.ingest"
        assert "Metadata only" in notices(app)


async def test_export_refuses_to_overwrite(tmp_path):
    app = make_app(tmp_path)
    destination = tmp_path / "actions.jsonl"
    destination.write_text("existing\n")

    async with app.run_test() as pilot:
        await submit(app, pilot, f"/audit export {destination}")

        assert destination.read_text() == "existing\n"
        assert "exists" in notices(app).lower()


# ---- When the journal itself fails ------------------------------------------


async def test_nothing_is_ingested_if_the_intent_cannot_be_recorded(tmp_path):
    """The one case where local history is allowed to stop the employee: if
    the intent will not commit, an untracked mutation must not begin."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    store = ContextStore(str(tmp_path / "context"))
    messages: list[tuple[str, str]] = []

    await run_digest_command(
        str(path),
        store=store,
        notify=await collected(messages),
        history=FakeHistoryStore(fail_states={"intent"}),
        actor_id="alice",
    )

    assert store.list() == []
    assert any("Nothing was started" in text for text, _ in messages)


async def test_a_successful_ingest_survives_a_failed_outcome_write(tmp_path):
    """The data is compressed and stored. Re-doing it to tidy the journal
    would be choosing the record over the result."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    store = ContextStore(str(tmp_path / "context"))
    history = FakeHistoryStore(fail_states={"succeeded"})
    messages: list[tuple[str, str]] = []

    await run_digest_command(
        str(path),
        store=store,
        notify=await collected(messages),
        history=history,
        actor_id="alice",
    )

    assert len(store.list()) == 1                   # the result stands
    assert history.states("digest.ingest") == ["intent"]
    text = "\n".join(text for text, _ in messages)
    assert "Local history has a gap" in text
    assert "NOT been repeated" in text
    assert "Compressed" in text                      # and the employee still sees the result


async def test_the_ingest_outcome_is_recorded_before_the_model_is_asked(tmp_path):
    """A slow or broken model summary must never be able to lose the record
    of an ingest that already committed, so the outcome is written first."""
    path = tmp_path / "protocol.md"
    path.write_text("# Protocol\n\n" + "Samples rest at ambient temperature. " * 30)
    store = ContextStore(str(tmp_path / "context"))
    history = FakeHistoryStore()
    observed: list[list[str]] = []

    class ObservingBackend(FakeBackend):
        async def stream_chat(self, model, messages):
            observed.append(history.states("digest.ingest"))
            yield "GIST: handling samples\n- rest at ambient"

    await run_digest_command(
        str(path),
        store=store,
        notify=await collected([]),
        backend=ObservingBackend(),
        model="test-model",
        history=history,
        actor_id="alice",
    )

    assert observed == [["intent", "succeeded"]]


async def test_a_cancelled_ingest_is_recorded_as_unknown_not_as_stopped(tmp_path, monkeypatch):
    """``asyncio.to_thread`` cannot stop a thread that is already compressing
    a file, so "cancelled" would be a claim we cannot support."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    store = ContextStore(str(tmp_path / "context"))
    history = FakeHistoryStore()
    started = asyncio.Event()

    def slow_ingest(source, *, store):
        started._loop.call_soon_threadsafe(started.set)
        import time

        time.sleep(0.2)
        raise AssertionError("not reached in this test")

    monkeypatch.setattr("bedrock.ui.digest_command.ingest", slow_ingest)
    messages: list[tuple[str, str]] = []
    task = asyncio.ensure_future(
        run_digest_command(
            str(path),
            store=store,
            notify=await collected(messages),
            history=history,
            actor_id="alice",
        )
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert history.states("digest.ingest") == ["intent", "unknown"]
    assert any("may still finish" in text for text, _ in messages)


async def test_a_missing_provenance_module_does_not_break_digest(tmp_path):
    """`/digest` predates local history and must not depend on it."""
    path = tmp_path / "assay.csv"
    path.write_text(CSV)
    store = ContextStore(str(tmp_path / "context"))
    messages: list[tuple[str, str]] = []

    await run_digest_command(
        str(path), store=store, notify=await collected(messages), history=None
    )

    assert len(store.list()) == 1


# ---- Wording now covers literature reviews ---------------------------------


async def test_the_empty_state_now_mentions_literature_reviews(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/audit")

        text = notices(app)
        assert "literature reviews" in text
        assert "Folds, digests, approvals" in text


async def test_audit_help_now_mentions_literature_reviews(tmp_path):
    app = make_app(tmp_path)
    async with app.run_test() as pilot:
        await submit(app, pilot, "/audit help")

        text = notices(app)
        assert "literature review" in text
        assert "retraction check" in text
