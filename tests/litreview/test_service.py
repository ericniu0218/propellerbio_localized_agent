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
import threading
from typing import AsyncIterator, Callable

import pytest

from bedrock.config import PROFILES, Settings as BedrockSettings
from bedrock.inference import InferenceError, ModelInfo, RequestOptions, Scheduler, Usage
from bedrock.litreview import service
from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.llm.scheduled import ModelServerUnavailable
from bedrock.litreview.runner import answer_question
from bedrock.litreview.service import ReviewRequest, preflight, run_review
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


async def go(request, settings, backend, cancel=None, progress=None):
    scheduler = Scheduler(backend, PROFILES["workstation"])
    return await run_review(
        request,
        settings=settings,
        scheduler=scheduler,
        progress=progress or (lambda msg: None),
        cancel=cancel or threading.Event(),
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


async def test_preflight_returns_advisories_rather_than_refusing_over_them(settings):
    """A short timeout is worth saying and must not stop a run: the review
    survives it by stopping between calls and resuming."""
    settings.request_timeout = 120.0
    advisories = await preflight(settings, ScriptedBackend())
    assert [f.title for f in advisories] == ["request timeout"]
    assert not any(f.blocking for f in advisories)
    assert "request_timeout = 600" in advisories[0].detail


async def test_a_front_end_that_raised_its_own_timeout_is_not_warned_about_it(settings):
    """`bedrock-litreview` and the HTTP API build their own backend through
    `standalone_scheduler`, which floors the timeout at 600. Warning them about
    the configured 120 would be telling them to fix something already fixed —
    while `/review` in the app, which shares the chat's backend, does need it."""
    settings.request_timeout = 120.0
    assert await preflight(settings, ScriptedBackend(), timeout_floor_applied=True) == []

    floored = [f for f in service.config_findings(settings, timeout_floor_applied=True)
               if f.title == "request timeout"]
    assert floored[0].ok
    assert "raised to 600s for this process" in floored[0].detail
    # ...and it still says the app needs the config change, because that is true.
    assert "request_timeout = 600" in floored[0].detail


# ---------------------------------------------------------------------------
# findings: what `doctor` and `/reviews check` report
# ---------------------------------------------------------------------------


async def test_findings_report_everything_not_just_the_first_problem(settings):
    """`preflight` stops a run at the first blocker; a person fixing a machine
    wants the whole list, or setting it up is one error per attempt."""
    settings.context_window = 8192
    settings.request_timeout = 120.0
    found = await service.findings(settings, ScriptedBackend(models=["something-else"]))

    by_title = {f.title: f for f in found}
    assert by_title["context window"].blocking
    assert by_title["models"].blocking
    assert not by_title["request timeout"].ok and not by_title["request timeout"].blocking
    assert by_title["model server"].ok  # reachable, just without the right model
    assert "free disk" in by_title


async def test_findings_without_a_server_still_check_the_configuration(settings):
    """The normal state of a machine on the day somebody installs this."""
    found = await service.findings(settings, None)
    assert [f.title for f in found] == ["context window", "request timeout", "free disk"]
    assert all(not f.blocking for f in found)


async def test_a_missing_model_names_the_override_file_and_what_the_server_has(settings):
    """Issue 16's practical half. Nobody can be told the office server's real
    model names in advance, so the message has to say what this server calls its
    models and which file to write them into."""
    with pytest.raises(ConfigError) as exc:
        await preflight(settings, ScriptedBackend(models=["Qwen/Qwen3-30B-A3B-Instruct-2507"]))
    message = str(exc.value)
    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in message, "it must say what the server does offer"
    assert "models.yaml" in message and "--served-model-name" in message


async def test_a_nearly_full_disk_is_a_warning_not_a_refusal(settings, monkeypatch):
    """Item 21: a review keeps a model resident for hours, and on a 16 GB
    machine that has filled a disk through swap. Still a warning — the disk this
    check looks at is not what a review itself fills."""
    import shutil

    monkeypatch.setattr(
        shutil, "disk_usage", lambda _p: type("U", (), {"free": 2 * 1024**3, "total": 0, "used": 0})
    )
    disk = [f for f in service.config_findings(settings) if f.title == "free disk"][0]
    assert not disk.ok and not disk.blocking
    assert "2.0 GB free" in disk.detail
    assert await preflight(settings, ScriptedBackend()), "a warning, so the run still starts"


def test_a_company_server_name_is_explained_rather_than_assumed_obvious(settings):
    """`allowed_hosts` is the one piece of review configuration nobody finds by
    accident, and the first-run banner has been known to suggest exactly such a
    host."""
    settings.base_url = "http://ai.company.internal:8000/v1"
    settings.allowed_hosts = ("ai.company.internal",)
    note = [f for f in service.config_findings(settings) if f.title == "model server address"][0]
    assert note.ok and "allowed_hosts" in note.detail

    settings.allowed_hosts = ()
    note = [f for f in service.config_findings(settings) if f.title == "model server address"][0]
    assert 'allowed_hosts = ["ai.company.internal"]' in note.detail

    # Nothing to explain about a local address.
    settings.base_url = "http://localhost:11434/v1"
    assert not [f for f in service.config_findings(settings) if f.title == "model server address"]


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
