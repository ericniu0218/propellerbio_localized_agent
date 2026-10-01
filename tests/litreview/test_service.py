"""`service.run_review`: the one way into a review that uses a model.

The test that matters most here is
`test_the_same_papers_through_the_scheduler_give_the_same_report`: it runs the
whole pipeline twice on the same papers, once against the fake client the stage
tests use and once against a scripted model server reached through a real
`Scheduler`, and requires the two reports to be identical. That is what says the
bridge added in this change is transport and nothing else.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import AsyncIterator, Callable

import pytest

from bedrock.config import PROFILES, Settings as BedrockSettings
from bedrock.inference import InferenceError, ModelInfo, RequestOptions, Scheduler, Usage
from bedrock.litreview import service
from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.journal import ReviewNotRecorded
from bedrock.litreview.llm.scheduled import ModelServerUnavailable
from bedrock.litreview.runner import answer_question
from bedrock.litreview.service import ReviewRequest, preflight, run_review
from bedrock.litreview.sources.literature import LiteratureSearcher, SearchFailed
from bedrock.litreview.stages.s1_collect import NoPapers
from tests.conftest import FakeHistoryStore
from tests.litreview.fixtures import QUESTION, corpus, fake_client, responder

MODEL = "one-model"


@pytest.fixture()
def models_yaml(tmp_path):
    """One model name for every tier.

    Model versions are written into the report, so a run whose tiers name two
    different models could never match a `FakeLLMClient` run, which reports one.
    """
    path = tmp_path / "config"
    path.mkdir()
    (path / "models.yaml").write_text(
        f"min_context_window: 12288\n"
        f"model_sets:\n  one: {{small: {{name: {MODEL}}}, medium: {{name: {MODEL}}}}}\n"
        "jobs:\n  question: medium\n  relevance: small\n  extract: medium\n  write: medium\n"
        "defaults: {temperature: 0.0, seed: 0, max_tokens: 2048}\n"
    )
    return path


@pytest.fixture()
def settings(tmp_path, models_yaml):
    s = load_settings(
        BedrockSettings(api_base="http://localhost:11434/v1", profile=PROFILES["workstation"]),
        model_set="one",
        env={"BEDROCK_LITREVIEW_CONFIG_DIR": str(models_yaml)},
        check_endpoint=False,
    )
    s.data_dir = tmp_path / "data"
    s.cache_dir = tmp_path / "data" / "cache"
    s.out_dir = tmp_path / "data" / "reports"
    s.db_path = tmp_path / "data" / "litreview.sqlite3"
    return s


class ScriptedBackend:
    """A server that answers each request the way `fixtures.responder` would.

    The reply is yielded whole, in one piece, so nothing about fragmenting can
    change the text the pipeline sees.
    """

    def __init__(self, models: list[str] | None = None, fail_after: int | None = None) -> None:
        self.models = models if models is not None else [MODEL]
        self.fail_after = fail_after
        self.calls: list[tuple[str, list[dict[str, str]], RequestOptions | None]] = []

    async def check_connection(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m) for m in self.models]

    async def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: RequestOptions | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> AsyncIterator[str]:
        self.calls.append((model, list(messages), options))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise InferenceError("connection refused. Is the inference server running?", unavailable=True)
        yield responder(messages[-1]["content"], (options.json_schema if options else None))

    async def close(self) -> None:
        pass


def review(question=QUESTION, **kwargs) -> ReviewRequest:
    return ReviewRequest(question=question, **kwargs)


async def go(request, settings, backend, cancel=None, progress=None, history=None, actor_id=""):
    scheduler = Scheduler(backend, PROFILES["workstation"])
    return await run_review(
        request,
        settings=settings,
        scheduler=scheduler,
        progress=progress or (lambda msg: None),
        cancel=cancel or threading.Event(),
        history=history,
        actor_id=actor_id,
    )


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


async def test_a_window_too_small_is_refused_with_both_numbers_and_the_fix(settings):
    settings.context_window = 8192  # the laptop profile
    with pytest.raises(ConfigError) as exc:
        await preflight(settings, ScriptedBackend())
    message = str(exc.value)
    assert "12,288" in message and "8,192" in message
    assert "OLLAMA_CONTEXT_LENGTH=12288" in message
    assert "workstation" in message


async def test_a_model_the_jobs_need_but_the_server_lacks_names_the_pull_command(settings):
    with pytest.raises(ConfigError) as exc:
        await preflight(settings, ScriptedBackend(models=["some-other-model"]))
    assert f"ollama pull {MODEL}" in str(exc.value)
    assert "vLLM" in str(exc.value)


async def test_a_silent_server_says_so_and_names_the_address(settings):
    class Silent(ScriptedBackend):
        async def list_models(self):
            raise InferenceError("connection refused", unavailable=True)

    with pytest.raises(ConfigError, match="not answering"):
        await preflight(settings, Silent())


async def test_preflight_passes_when_the_machine_can_run_a_review(settings):
    await preflight(settings, ScriptedBackend())


async def test_a_run_without_a_model_never_touches_the_server(settings, tmp_path):
    backend = ScriptedBackend(models=[])  # would fail preflight if it ran
    result = await go(review(corpus=_corpus_file(tmp_path), use_model=False), settings, backend)
    assert result.report_id
    assert backend.calls == []


# ---------------------------------------------------------------------------
# The bridge changes nothing
# ---------------------------------------------------------------------------


async def test_the_same_papers_through_the_scheduler_give_the_same_report(settings, tmp_path):
    """Scheduler path vs. FakeLLMClient path, same papers, same bytes."""
    direct = answer_question(
        QUESTION,
        settings=settings,
        client=fake_client_named(MODEL),
        papers=corpus(),
        db=Database(tmp_path / "direct.sqlite3"),
        out_dir=tmp_path / "direct-out",
    )

    settings.db_path = tmp_path / "scheduled.sqlite3"
    scheduled = await go(
        review(papers=corpus(), out_dir=tmp_path / "scheduled-out"), settings, ScriptedBackend()
    )

    assert scheduled.report_id == direct.report_id
    assert scheduled.report.model_dump() == direct.report.model_dump()
    assert (tmp_path / "scheduled-out").exists()


async def test_every_call_goes_out_at_temperature_zero_with_the_seed_and_a_schema(settings, tmp_path):
    backend = ScriptedBackend()
    await go(review(papers=corpus()), settings, backend)

    assert backend.calls, "the review made no model calls"
    for model, messages, options in backend.calls:
        assert model == MODEL
        assert [m["role"] for m in messages] == ["user"]
        assert options.temperature == 0.0
        assert options.seed == 0
    # Structured stages constrain the reply to a schema; nothing is free-form JSON.
    assert any(options.json_schema for _, _, options in backend.calls)


async def test_the_data_directory_is_created_private(settings, tmp_path):
    await go(review(papers=corpus()), settings, ScriptedBackend())
    assert settings.data_dir.exists()
    assert oct(settings.data_dir.stat().st_mode)[-3:] == "700"


# ---------------------------------------------------------------------------
# A server that goes away mid-run
# ---------------------------------------------------------------------------


async def test_a_server_that_goes_down_mid_run_stops_the_run_and_it_resumes(settings, tmp_path):
    """A report finished while the server was down would be silently hollow."""
    db_path = tmp_path / "interrupted.sqlite3"
    settings.db_path = db_path

    with pytest.raises(ModelServerUnavailable):
        await go(review(papers=corpus()), settings, ScriptedBackend(fail_after=2))

    run_id = _last_run_id(Database(db_path))
    assert run_id, "the interrupted run was not saved, so it could not be resumed"

    # The same question again, with a working server, picks up where it stopped.
    resumed = await go(review(resume=run_id, papers=corpus()), settings, ScriptedBackend())
    assert resumed.report_id
    assert resumed.run.id == run_id


async def test_progress_messages_reach_the_caller(settings, tmp_path):
    seen: list[str] = []
    await go(review(papers=corpus()), settings, ScriptedBackend(), progress=seen.append)
    assert seen, "the pipeline reported no progress"


# ---------------------------------------------------------------------------
# standalone_scheduler
# ---------------------------------------------------------------------------


async def test_standalone_scheduler_closes_the_backend_it_opened(monkeypatch):
    backend = ScriptedBackend()
    closed: list[bool] = []

    async def close():
        closed.append(True)

    backend.close = close
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: backend)

    bedrock = BedrockSettings(api_base="http://localhost:11434/v1", request_timeout=30.0)
    async with service.standalone_scheduler(bedrock) as scheduler:
        assert scheduler.backend is backend
    assert closed == [True]


async def test_a_backend_passed_in_for_tests_is_not_closed():
    backend = ScriptedBackend()
    bedrock = BedrockSettings(api_base="http://localhost:11434/v1")
    async with service.standalone_scheduler(bedrock, backend=backend) as scheduler:
        assert scheduler.backend is backend  # and nothing raises on the way out


async def test_the_timeout_floor_is_raised_for_a_long_job(monkeypatch):
    """bedrock's 120-second chat default would fail a 4,096-token extraction."""
    seen: list[float] = []
    monkeypatch.setattr(
        "bedrock.inference.create_backend",
        lambda s: (seen.append(s.request_timeout), ScriptedBackend())[1],
    )
    bedrock = BedrockSettings(api_base="http://localhost:11434/v1", request_timeout=120.0)
    async with service.standalone_scheduler(bedrock):
        pass
    assert seen == [600.0]

    seen.clear()
    bedrock = BedrockSettings(api_base="http://localhost:11434/v1", request_timeout=900.0)
    async with service.standalone_scheduler(bedrock):
        pass
    assert seen == [900.0], "a longer configured timeout is left alone"


# ---------------------------------------------------------------------------
# Where the papers come from
# ---------------------------------------------------------------------------


async def test_the_searcher_is_built_only_when_someone_asked_for_it(settings, tmp_path, monkeypatch):
    built: list[dict] = []

    def from_settings(cls, s, **kw):
        built.append(kw)
        raise AssertionError("the searcher must not actually run here")

    from bedrock.litreview.sources.literature import LiteratureSearcher

    monkeypatch.setattr(LiteratureSearcher, "from_settings", classmethod(from_settings))

    assert service._searcher(review(papers=corpus()), settings) is None
    assert built == []

    assert service._searcher(review(corpus=_corpus_file(tmp_path)), settings).name == "corpus"
    assert built == []

    with pytest.raises(AssertionError):
        service._searcher(review(search=True, sources=("europepmc",), full_text=False), settings)
    assert built == [{"sources": ("europepmc",), "full_text_max": 0}]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fake_client_named(version: str):
    from bedrock.litreview.llm import FakeLLMClient

    return FakeLLMClient(responder=responder, version=version)


def _corpus_file(tmp_path):
    from bedrock.litreview.sources.corpus import write_papers

    path = tmp_path / "papers.jsonl"
    write_papers(corpus(), path)
    return path


def _last_run_id(db) -> str | None:
    rows = db.conn.execute("SELECT id FROM runs ORDER BY rowid DESC LIMIT 1").fetchall()
    return rows[0][0] if rows else None


# ---------------------------------------------------------------------------
# run_review: recording the run in local history (finding 14)
# ---------------------------------------------------------------------------


async def test_a_successful_run_writes_intent_then_succeeded_with_the_run_id(settings, tmp_path):
    history = FakeHistoryStore()
    settings.db_path = tmp_path / "journalled.sqlite3"

    result = await go(review(papers=corpus()), settings, ScriptedBackend(), history=history, actor_id="alice")

    rows = [r for r in history.rows if r["action"] == "litreview.review"]
    assert [r["state"] for r in rows] == ["intent", "succeeded"]
    assert {r["operation_id"] for r in rows} == {rows[0]["operation_id"]}  # one operation
    assert all(r["actor_id"] == "alice" for r in rows)
    assert rows[-1]["details"]["run_id"] == result.run.id
    assert rows[0]["subject_id"] is None  # not a resume


async def test_a_resumed_runs_intent_names_the_run_it_is_resuming(settings, tmp_path):
    settings.db_path = tmp_path / "resume-journal.sqlite3"
    with pytest.raises(ModelServerUnavailable):
        await go(review(papers=corpus()), settings, ScriptedBackend(fail_after=2))
    run_id = _last_run_id(Database(settings.db_path))

    history = FakeHistoryStore()
    await go(review(resume=run_id, papers=corpus()), settings, ScriptedBackend(), history=history, actor_id="alice")

    rows = [r for r in history.rows if r["action"] == "litreview.review"]
    assert rows[0]["subject_id"] == run_id


async def test_a_failing_run_writes_failed_with_a_bounded_error_code(settings):
    """No corpus, no search, no resume, no posted papers: `NoPapers`, cheaply
    and deterministically, without needing a model call at all."""
    history = FakeHistoryStore()

    with pytest.raises(NoPapers):
        await go(review(use_model=False), settings, ScriptedBackend(models=[]), history=history, actor_id="alice")

    rows = [r for r in history.rows if r["action"] == "litreview.review"]
    assert [r["state"] for r in rows] == ["intent", "failed"]
    assert rows[-1]["details"]["error_code"] == "no_papers"


async def test_an_interrupted_run_writes_cancelled(settings, tmp_path):
    settings.db_path = tmp_path / "cancelled-journal.sqlite3"
    history = FakeHistoryStore()

    with pytest.raises(ModelServerUnavailable):
        await go(review(papers=corpus()), settings, ScriptedBackend(fail_after=2), history=history, actor_id="alice")

    rows = [r for r in history.rows if r["action"] == "litreview.review"]
    assert [r["state"] for r in rows] == ["intent", "cancelled"]


async def test_history_none_runs_exactly_as_before(settings, tmp_path):
    """The default, and what every caller that predates this change gets."""
    settings.db_path = tmp_path / "unjournalled.sqlite3"
    result = await go(review(papers=corpus()), settings, ScriptedBackend())  # history=None
    assert result.report_id


async def test_an_intent_write_failure_stops_the_run_before_the_pipeline_runs(settings, tmp_path):
    """The one case local history is allowed to stop the employee: if the
    intent will not commit, the review must not run at all."""
    history = FakeHistoryStore(fail_states={"intent"})
    backend = ScriptedBackend()

    with pytest.raises(ReviewNotRecorded):
        await go(review(corpus=_corpus_file(tmp_path)), settings, backend, history=history, actor_id="alice")

    assert backend.calls == [], "the pipeline must never have made a model call"
    assert history.rows == [], "a failed intent write must leave nothing recorded"


async def test_an_outcome_write_failure_keeps_the_result_and_reports_the_gap(settings, tmp_path):
    """The report is written to disk either way. Re-running the pipeline to
    tidy the journal would trade a correct result for a neat record."""
    history = FakeHistoryStore(fail_states={"succeeded"})
    seen: list[str] = []
    settings.db_path = tmp_path / "gap.sqlite3"

    result = await go(
        review(papers=corpus()), settings, ScriptedBackend(),
        history=history, actor_id="alice", progress=seen.append,
    )

    assert result.report_id  # the review is not re-run
    assert history.states("litreview.review") == ["intent"]  # the outcome write failed
    assert any("Local history has a gap" in m for m in seen)
    assert any("result itself stands" in m for m in seen)


async def test_a_failed_runs_search_declarations_are_failed_too(settings, tmp_path, monkeypatch):
    """A `litreview.search` event says "authorised and began to contact this
    host" - so when the run that authorisation belonged to fails, the search
    event's own outcome must say so too, with the same error code, not sit
    forever as an unconfirmed intent for a run everyone already knows failed."""

    def boom(self, req):
        raise SearchFailed("every source failed")

    monkeypatch.setattr(LiteratureSearcher, "search", boom)
    history = FakeHistoryStore()
    settings.db_path = tmp_path / "failed-search.sqlite3"

    with pytest.raises(SearchFailed):
        await go(review(search=True), settings, ScriptedBackend(), history=history, actor_id="alice")

    assert history.states("litreview.review") == ["intent", "failed"]
    assert history.states("litreview.search") == ["intent", "failed"]
    review_code = [r for r in history.rows if r["action"] == "litreview.review"][-1]["details"]["error_code"]
    search_code = [r for r in history.rows if r["action"] == "litreview.search"][-1]["details"]["error_code"]
    assert review_code == search_code == "search_failed"


async def test_litreview_search_is_recorded_for_a_search_run(settings, tmp_path, monkeypatch):
    monkeypatch.setattr(LiteratureSearcher, "search", lambda self, req: corpus())
    history = FakeHistoryStore()
    settings.db_path = tmp_path / "search-journal.sqlite3"

    result = await go(review(search=True), settings, ScriptedBackend(), history=history, actor_id="alice")

    assert result.report_id
    rows = [r for r in history.rows if r["action"] == "litreview.search"]
    assert [r["state"] for r in rows] == ["intent", "succeeded"]
    assert all(r["details"]["source_kind"] == "literature_search" for r in rows)


async def test_litreview_search_is_recorded_for_an_automatic_retraction_check(settings, tmp_path, monkeypatch):
    """`tests/conftest.py` turns the automatic check off for the whole suite
    (see its `no_automatic_retraction_checks` fixture); this test turns it
    back on deliberately, with a fake check that never touches the network."""
    monkeypatch.setattr(service, "auto_integrity_check", lambda s: (lambda db, papers, *, now, ttl: {}))
    history = FakeHistoryStore()
    settings.db_path = tmp_path / "integrity-journal.sqlite3"

    result = await go(review(papers=corpus()), settings, ScriptedBackend(), history=history, actor_id="alice")

    assert result.report_id
    rows = [r for r in history.rows if r["action"] == "litreview.search"]
    assert [r["state"] for r in rows] == ["intent", "succeeded"]
    assert all(r["details"]["source_kind"] == "retraction_check" for r in rows)


async def test_no_litreview_search_for_a_corpus_only_run_with_manual_integrity(settings, tmp_path):
    """The automatic check is already off for the whole suite (see the fixture
    above), matching `integrity_checks = "manual"`: a corpus file contacts
    nothing, so this run must write no `litreview.search` event at all."""
    history = FakeHistoryStore()

    result = await go(
        review(corpus=_corpus_file(tmp_path)), settings, ScriptedBackend(),
        history=history, actor_id="alice",
    )

    assert result.report_id
    assert [r for r in history.rows if r["action"] == "litreview.search"] == []


async def test_the_question_text_is_never_recorded(settings, tmp_path):
    """The privacy promise `/audit` makes: metadata only. `run_id` is the join
    back to litreview's own database, which already holds the question."""
    history = FakeHistoryStore()
    # Keeps "psoriasis" so the entity-extraction guard still accepts it, and
    # adds a unique marker to search for in the recorded rows.
    question = f"{QUESTION} xyzzy19260928"
    settings.db_path = tmp_path / "no-question-text.sqlite3"

    await go(review(question=question, papers=corpus()), settings, ScriptedBackend(), history=history, actor_id="alice")

    dumped = json.dumps(history.rows)
    assert question not in dumped
    assert "xyzzy19260928" not in dumped
