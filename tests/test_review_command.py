"""`/review` and `/reviews`, driving the real Textual app headlessly.

The claim being tested is that a literature review runs *beside* the chat: it
does not block a reply, it does not write over the screen, it queues behind the
employee's own model calls, and stopping it or quitting leaves a run that can be
resumed. The pipeline itself is tested in `tests/litreview/`.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import pytest

from tests.conftest import FakeBackend
from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.config import PROFILES, Settings
from bedrock.inference import RequestOptions
from bedrock.ui.argparsing import CommandUsageError
from bedrock.ui.commands import HELP_TEXT, KNOWN_COMMANDS, parse_command
from bedrock.ui.review_command import parse_review_command
from bedrock.ui.widgets import AssistantMessage, SystemNotice

from tests.litreview.fixtures import QUESTION, corpus, responder

MODEL = "test-model"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_the_question_is_taken_verbatim_so_apostrophes_work():
    """People ask "What's worth targeting?"; shell splitting would refuse it."""
    command = parse_review_command("What's worth targeting to prevent psoriasis? --no-model")
    assert command.request.question == "What's worth targeting to prevent psoriasis?"
    assert command.request.use_model is False


def test_a_quoted_path_with_spaces_survives():
    command = parse_review_command('prevent psoriasis --corpus "My Papers/psoriasis.jsonl"')
    assert command.request.corpus == Path("My Papers/psoriasis.jsonl")
    assert command.request.question == "prevent psoriasis"


def test_stop_is_the_one_verb():
    assert parse_review_command("stop").stop is True
    assert parse_review_command("STOP").stop is True
    assert parse_review_command("stop this review").stop is False


def test_resume_needs_no_question():
    command = parse_review_command("--resume abc123def")
    assert command.request.resume == "abc123def" and command.request.question == ""


def test_an_empty_argument_is_a_usage_error():
    for args in ("", "   "):
        with pytest.raises(CommandUsageError, match="Tell me what to review"):
            parse_review_command(args)


def test_a_corpus_and_a_search_together_are_refused():
    """Papers come from one place, so a run can be pinned to them."""
    with pytest.raises(CommandUsageError, match="not both"):
        parse_review_command("prevent psoriasis --corpus p.jsonl --search")


def test_an_unknown_option_is_a_usage_error_not_a_crash():
    with pytest.raises(CommandUsageError):
        parse_review_command("prevent psoriasis --korpus p.jsonl")


def test_the_options_are_read_into_the_request():
    command = parse_review_command(
        "prevent psoriasis --search --refresh --models medgemma --sources europepmc,pubmed --no-full-text"
    )
    request = command.request
    assert request.search and request.refresh
    assert request.model_set == "medgemma"
    assert request.sources == ("europepmc", "pubmed")
    assert request.full_text is False


def test_the_commands_are_known_and_documented():
    assert {"review", "litreview", "reviews"} <= KNOWN_COMMANDS
    assert "/review" in HELP_TEXT and "/reviews" in HELP_TEXT


# ---------------------------------------------------------------------------
# Helpers for the app tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def models_yaml(tmp_path, monkeypatch):
    """A models override naming the fake backend's model for every tier."""
    path = tmp_path / "litreview-config"
    path.mkdir()
    (path / "models.yaml").write_text(
        f"min_context_window: 12288\n"
        f"model_sets:\n  one: {{small: {{name: {MODEL}}}, medium: {{name: {MODEL}}}}}\n"
        "jobs:\n  question: medium\n  relevance: small\n  extract: medium\n  write: medium\n"
        "defaults: {temperature: 0.0, seed: 0, max_tokens: 2048}\n"
    )
    monkeypatch.setenv("BEDROCK_LITREVIEW_CONFIG_DIR", str(path))
    monkeypatch.setenv("BEDROCK_LITREVIEW_MODELS", "one")
    return path


@pytest.fixture()
def review_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BEDROCK_LITREVIEW_DIR", str(tmp_path / "litreview-data"))
    return tmp_path / "litreview-data"


@pytest.fixture()
def corpus_file(tmp_path):
    from bedrock.litreview.sources.corpus import write_papers

    path = tmp_path / "papers.jsonl"
    write_papers(corpus(), path)
    return path


class ReviewBackend(FakeBackend):
    """The fake chat backend, which also answers review calls.

    A review call carries a `json_schema` or comes from the pipeline's prompts;
    a chat turn does not. Keeping them apart means `requests` still holds only
    chat turns, exactly as the rest of the app tests assume.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.review_requests: list[tuple[str, list[dict[str, str]], RequestOptions | None]] = []
        self.hold_reviews: asyncio.Event | None = None

    async def stream_chat(self, model, messages, options=None, on_usage=None):
        if options is not None and options.seed is not None:
            self.review_requests.append((model, list(messages), options))
            if self.hold_reviews is not None:
                await self.hold_reviews.wait()
            yield responder(messages[-1]["content"], options.json_schema)
            return
        async for fragment in super().stream_chat(model, messages, options, on_usage):
            yield fragment


def make_app(backend, **settings_kwargs) -> BedrockApp:
    settings_kwargs.setdefault("model", MODEL)
    settings_kwargs.setdefault("profile", PROFILES["workstation"])
    return BedrockApp(settings=Settings(**settings_kwargs), backend=backend)


def notices(app) -> list[str]:
    return [str(n.source) for n in app.query(SystemNotice)]


async def run(app, text: str) -> None:
    """Drive one slash command, without typing a long question key by key."""
    await app.run_command(parse_command(text))


# ---------------------------------------------------------------------------
# A review without a model
# ---------------------------------------------------------------------------


async def test_a_review_without_a_model_finishes_and_never_calls_the_server(
    corpus_file, review_dir, models_yaml
):
    backend = ReviewBackend()
    app = make_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        text = "\n".join(notices(app))
        assert "Literature review finished" in text, text
        assert "Integrity: passed" in text
        for name in ("report.json", "report.md", "report.html"):
            assert name in text
        # ... and the notice is about files that really exist. (The notice
        # elides the middle of a long path, so the directory is checked here
        # rather than the string being read back as a path.)
        reports = list((review_dir / "reports").glob("*/*/report.html"))
        assert len(reports) == 1 and reports[0].read_text().startswith("<")

    assert backend.requests == [], "a --no-model review must not reach the model server"
    assert backend.review_requests == []


async def test_reviews_lists_the_run_and_verify_says_ok(corpus_file, review_dir, models_yaml):
    backend = ReviewBackend()
    app = make_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        run_id = _run_id(notices(app))

        await run(app, "/reviews")
        await wait_for_app_workers(app)
        await pilot.pause()
        listed = notices(app)[-1]
        assert run_id in listed and QUESTION in listed

        await run(app, f"/reviews verify {run_id}")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "OK" in notices(app)[-1]


async def test_reviews_with_nothing_recorded_says_so(review_dir, models_yaml):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/reviews")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "No literature reviews on this machine yet" in notices(app)[-1]


async def test_reviews_does_not_need_the_configured_model_set_to_exist(review_dir, models_yaml, monkeypatch):
    """Listing reads the database only. A machine whose model set is missing
    from its models.yaml must still be able to see and re-check past runs."""
    monkeypatch.setenv("BEDROCK_LITREVIEW_MODELS", "not-in-the-file")
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/reviews")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "No literature reviews on this machine yet" in notices(app)[-1], notices(app)[-1]


# ---------------------------------------------------------------------------
# Beside the chat
# ---------------------------------------------------------------------------


async def test_a_chat_message_is_answered_while_a_review_is_held_open(
    corpus_file, review_dir, models_yaml
):
    """The employee is never blocked by a review, only by one call of it.

    On a laptop the server takes one request at a time, so a chat waits for the
    review call already in flight - a few seconds - and then goes ahead of every
    remaining one. What must not happen is the chat waiting for the whole
    review, which is what talking to the server directly would have caused.
    """
    backend = ReviewBackend(models=[MODEL])
    app = make_app(backend)

    async with app.run_test() as pilot:
        await pilot.pause()
        backend.hold_reviews = asyncio.Event()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")

        # Wait until the review is genuinely inside a model call.
        for _ in range(200):
            if backend.review_requests:
                break
            await pilot.pause()
        assert backend.review_requests, "the review never reached the server"
        assert app._review_running

        await pilot.press(*"is IL-23 worth a look", "enter")
        await pilot.pause()
        backend.hold_reviews.set()

        await wait_for_app_workers(app)
        await pilot.pause()

        assert backend.requests, "the chat turn never reached the model"
        assert "Hello from the fake model." in "\n".join(
            str(m.source) for m in app.query(AssistantMessage)
        )
        assert "Literature review finished" in "\n".join(notices(app))


async def test_a_second_review_while_one_runs_is_refused(corpus_file, review_dir, models_yaml):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._review_running = True
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await pilot.pause()
        assert "already running" in notices(app)[-1]
    # Nothing started, so nothing to wait for.


async def test_review_stop_sets_the_cancel_event(corpus_file, review_dir, models_yaml):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/review stop")
        await pilot.pause()
        assert "No literature review is running" in notices(app)[-1]

        app._review_running = True
        assert not app._review_cancel.is_set()
        await run(app, "/review stop")
        await pilot.pause()
        assert app._review_cancel.is_set()
        assert "Stopping the literature review" in notices(app)[-1]
        app._review_running = False


async def test_quitting_stops_a_running_review_so_it_stays_resumable(review_dir, models_yaml):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        assert not app._review_cancel.is_set()
        await app.action_quit()
    assert app._review_cancel.is_set()


async def test_a_typo_never_starts_a_review(corpus_file, review_dir, models_yaml):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/review prevent psoriasis --corpus a.jsonl --search")
        await pilot.pause()
        assert app._review_running is False
        assert "not both" in notices(app)[-1]


# ---------------------------------------------------------------------------
# On a model
# ---------------------------------------------------------------------------


async def test_a_review_on_a_model_goes_out_at_temperature_zero_with_a_schema_and_a_seed(
    corpus_file, review_dir, models_yaml
):
    backend = ReviewBackend(models=[MODEL])
    app = make_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        assert backend.review_requests, "the review made no model calls"
        _, messages, options = backend.review_requests[0]
        assert [m["role"] for m in messages] == ["user"]
        assert options.temperature == 0.0
        assert options.seed == 0
        assert options.json_schema is not None

        text = "\n".join(notices(app))
        assert "Literature review finished" in text, text
        assert "Traceback" not in text


async def test_a_window_too_small_produces_the_preflight_notice_and_no_run(
    corpus_file, review_dir, models_yaml
):
    """The laptop profile's 8,192-token window cannot hold an extraction call."""
    backend = ReviewBackend(models=[MODEL])
    app = make_app(backend, profile=PROFILES["laptop"])
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        text = notices(app)[-1]
        assert "12,288" in text and "8,192" in text
        assert "OLLAMA_CONTEXT_LENGTH=12288" in text
    assert backend.review_requests == []


async def test_a_model_the_server_does_not_have_names_the_pull_command(
    corpus_file, review_dir, models_yaml
):
    backend = ReviewBackend(models=["something-else"])
    app = make_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert f"ollama pull {MODEL}" in notices(app)[-1]


# ---------------------------------------------------------------------------
# Agent-pack mode
# ---------------------------------------------------------------------------


async def test_agent_pack_mode_says_where_reviews_run():
    from bedrock.agentpack_app import AGENTPACK_REVIEW_NOTICE

    assert "default mode" in AGENTPACK_REVIEW_NOTICE
    assert "bedrock-litreview" in AGENTPACK_REVIEW_NOTICE


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_id(lines: list[str]) -> str:
    for line in reversed(lines):
        if "Literature review finished" in line:
            return line.split("run `")[1].split("`")[0]
    raise AssertionError(f"no finished review in: {lines}")
