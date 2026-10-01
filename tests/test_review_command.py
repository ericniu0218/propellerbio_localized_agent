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

from tests.conftest import FakeBackend, FakeHistoryStore, make_services
from tests.workers import wait_for_app_workers

from bedrock.app import BedrockApp
from bedrock.ui.services import ComplianceServices
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


def test_a_corpus_dropped_onto_the_terminal_loses_its_at_sign():
    """Dropping a file pastes `@/path`, as it does for chat attachments."""
    command = parse_review_command("prevent psoriasis --corpus @/tmp/papers.jsonl")
    assert command.request.corpus == Path("/tmp/papers.jsonl")


def test_a_patents_file_is_read_at_parse_time_not_inside_the_pipeline(tmp_path):
    """`--patents FILE` is read here, on the UI thread, before the request is
    built — the same reasoning `--corpus` gets, and the same reason a bad file
    is a usage error immediately rather than a failure minutes into a run."""
    from bedrock.litreview.models import PatentRecord
    from bedrock.litreview.sources.patent_corpus import write_patents

    path = tmp_path / "patents.jsonl"
    write_patents([PatentRecord(id="EP1A1", title="x")], path)

    command = parse_review_command(f"prevent psoriasis --patents {path}")
    assert command.request.patents == [PatentRecord(id="EP1A1", title="x")]


def test_a_patents_file_dropped_onto_the_terminal_loses_its_at_sign(tmp_path):
    from bedrock.litreview.models import PatentRecord
    from bedrock.litreview.sources.patent_corpus import write_patents

    path = tmp_path / "patents.jsonl"
    write_patents([PatentRecord(id="EP1A1", title="x")], path)

    command = parse_review_command(f"prevent psoriasis --patents @{path}")
    assert command.request.patents == [PatentRecord(id="EP1A1", title="x")]


def test_an_unreadable_patents_file_is_a_usage_error_not_a_crash():
    """A typo'd path must stop the command before a job starts, the same way
    every other bad option does — not raise from inside the worker thread."""
    with pytest.raises(CommandUsageError, match="patents"):
        parse_review_command("prevent psoriasis --patents /no/such/file.jsonl")


def test_no_patents_option_means_no_patents():
    assert parse_review_command("prevent psoriasis").request.patents is None


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


@pytest.fixture()
def patents_file(tmp_path):
    from bedrock.litreview.models import PatentRecord
    from bedrock.litreview.sources.patent_corpus import write_patents

    path = tmp_path / "patents.jsonl"
    write_patents(
        [
            PatentRecord(id="EP1A1", family_id="f1", title="Anti-IL-23 antibody",
                         assignee="Genentech, Inc.", kind_code="A1", priority_date="2018-01-01"),
            PatentRecord(id="US1A1", family_id="f1", title="Anti-IL-23 antibody",
                         assignee="Genentech, Inc.", kind_code="A1", priority_date="2018-01-01"),
        ],
        path,
    )
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


async def test_a_review_with_patents_shows_the_competitor_landscape(
    corpus_file, patents_file, review_dir, models_yaml
):
    """The one path this file had no coverage for: `/review --patents FILE`
    end to end, through the real app — `parse_review_command` reading the
    file, `ReviewRequest.patents`, `service.run_review`, and all the way down
    to `_competitors` actually writing the section. `--corpus` and `--patents`
    are wired through service.py identically, so this is the proof the app
    side was not left as an unverified guess."""
    backend = ReviewBackend()
    app = make_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file} --patents {patents_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        assert "Literature review finished" in "\n".join(notices(app))
        reports = list((review_dir / "reports").glob("*/*/report.md"))
        assert len(reports) == 1
        body = reports[0].read_text()
        assert "Competitor landscape" in body
        assert "Genentech, Inc." in body
        assert "1 patent famil" in body  # the two filings merged into one family, exactly as `--corpus` alone could not prove
        assert "2 filing" in body


async def test_the_app_hands_its_history_and_actor_to_the_review_job(corpus_file, review_dir, models_yaml):
    """`literature_review` must thread the app's own `ComplianceServices`
    journal and actor through to `run_review_job` - the same way `/digest`
    gets both (`ui/local_commands.py`) - so a review shows up in `/audit`."""
    backend = ReviewBackend()
    history = FakeHistoryStore()
    settings = Settings(model=MODEL, profile=PROFILES["workstation"], user_id="alice")
    services = make_services(settings, history=history)
    app = BedrockApp(settings=settings, backend=backend, services=services)

    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

    rows = [r for r in history.rows if r["action"] == "litreview.review"]
    assert [r["state"] for r in rows] == ["intent", "succeeded"]
    assert all(r["actor_id"] == "alice" for r in rows)


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
# /reviews check and /reviews forget
# ---------------------------------------------------------------------------


async def test_reviews_check_reports_the_failure_and_the_advice_beside_it(review_dir, models_yaml):
    """`/reviews check` is `bedrock-litreview doctor`'s implementation
    (`litreview/readiness.py`), asked through the app's own connection. What it
    adds over the error a review raises is that the advice that does not stop a
    review - here a timeout too short for an extraction call - is shown beside
    the one thing that does, instead of being found one failed `/review` later."""
    # The laptop profile's 8,192-token window is too small for a review (a
    # failure), and a 120-second timeout is too short (worth knowing).
    app = make_app(ReviewBackend(), profile=PROFILES["laptop"], request_timeout=120.0)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/reviews check")
        await wait_for_app_workers(app)
        await pilot.pause()

        report = notices(app)[-1]
        assert "**FAIL**" in report and "12,288" in report, report
        assert "request timeout" in report and "worth knowing" in report, report
        assert "cannot run yet" in report


async def test_reviews_check_passes_on_a_machine_that_is_set_up(review_dir, models_yaml):
    app = make_app(ReviewBackend(), profile=PROFILES["workstation"], request_timeout=600.0)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/reviews check")
        await wait_for_app_workers(app)
        await pilot.pause()

        report = notices(app)[-1]
        assert "A review can run." in report, report
        assert "**FAIL**" not in report
        assert "operational" in report, "it should have asked the server, and said what it found"


async def test_reviews_forget_deletes_the_run_and_says_what_went(
    corpus_file, review_dir, models_yaml
):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        run_id = _run_id(notices(app))
        assert list((review_dir / "reports").glob("*/*/report.html"))

        await run(app, f"/reviews forget {run_id}")
        await wait_for_app_workers(app)
        await pilot.pause()

        notice = notices(app)[-1]
        assert "Deleted review run" in notice and run_id in notice
        assert "stage artifact(s) removed" in notice
        assert "question and its papers were not touched" in notice
        assert not list((review_dir / "reports").glob("*/*/report.html"))

        # ...and it is gone from the list, not merely hidden.
        await run(app, "/reviews")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "No literature reviews on this machine yet" in notices(app)[-1]


async def test_review_warns_about_a_short_timeout_before_it_starts(
    corpus_file, review_dir, models_yaml
):
    """`/review` in the app uses the configured `request_timeout` — it shares the
    chat's connection, so unlike the shell command it cannot raise it for itself.
    INSTALL.md says to set 600; until now nothing said so when it mattered.

    Before the run, not after: the point is to go and change it.
    """
    app = make_app(ReviewBackend(), profile=PROFILES["workstation"], request_timeout=120.0)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        shown = notices(app)
        warning = [n for n in shown if "request_timeout" in n]
        assert warning, shown
        assert "request_timeout = 600" in warning[0]
        finished = [i for i, n in enumerate(shown) if "Literature review finished" in n]
        assert finished and shown.index(warning[0]) < finished[0], "the warning comes first"


async def test_no_timeout_warning_when_it_is_already_long_enough(
    corpus_file, review_dir, models_yaml
):
    app = make_app(ReviewBackend(), profile=PROFILES["workstation"], request_timeout=600.0)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert not [n for n in notices(app) if "request_timeout" in n]


async def test_reviews_forget_without_a_run_id_explains_rather_than_guessing(review_dir, models_yaml):
    """Deleting "the last one" because no id was given is how the wrong report
    disappears."""
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/reviews forget")
        await wait_for_app_workers(app)
        await pilot.pause()
        notice = notices(app)[-1]
        assert "name a run id" in notice
        assert "cannot be undone" in notice


async def test_reviews_forget_an_unknown_run_id_says_so(review_dir, models_yaml):
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/reviews forget 0123456789abcdef")
        await wait_for_app_workers(app)
        await pilot.pause()
        assert "no run" in notices(app)[-1]


# ---------------------------------------------------------------------------
# Chatting during a review on a machine that holds one model
# ---------------------------------------------------------------------------


async def _scheduler_after_chat(profile, chat_model: str | None):
    """A real Scheduler whose last chat turn (if any) was answered by `chat_model`."""
    from bedrock.inference import Priority, Scheduler

    scheduler = Scheduler(FakeBackend(models=[MODEL, "llama3.2:3b"]), profile)
    if chat_model:
        await scheduler.complete(chat_model, [{"role": "user", "content": "hi"}], priority=Priority.CHAT)
    return scheduler


def _review_settings():
    from bedrock.litreview.config import load_settings

    return load_settings(Settings(model=MODEL, profile=PROFILES["workstation"]), check_endpoint=False)


async def test_a_review_warns_when_the_chat_would_make_the_server_swap_models(models_yaml):
    from bedrock.litreview.service import ReviewRequest
    from bedrock.ui.review_command import model_swap_note

    laptop = await _scheduler_after_chat(PROFILES["laptop"], "llama3.2:3b")
    note = model_swap_note(ReviewRequest(question=QUESTION), _review_settings(), laptop)
    assert note is not None
    assert "llama3.2:3b" in note and MODEL in note and f"/model {MODEL}" in note


async def test_no_swap_warning_where_no_swap_would_happen(models_yaml):
    from bedrock.litreview.service import ReviewRequest
    from bedrock.ui.review_command import model_swap_note

    settings, request = _review_settings(), ReviewRequest(question=QUESTION)
    # The chat already uses the review's model.
    assert model_swap_note(request, settings, await _scheduler_after_chat(PROFILES["laptop"], MODEL)) is None
    # A shared server runs several calls at once and holds several models.
    assert model_swap_note(request, settings, await _scheduler_after_chat(PROFILES["server"], "llama3.2:3b")) is None
    # No chat has been answered yet, so nothing is loaded to swap out.
    assert model_swap_note(request, settings, await _scheduler_after_chat(PROFILES["laptop"], None)) is None
    # A review without a model never reaches the server.
    other = await _scheduler_after_chat(PROFILES["laptop"], "llama3.2:3b")
    assert model_swap_note(ReviewRequest(question=QUESTION, use_model=False), settings, other) is None


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


async def test_a_stale_install_says_to_reinstall_instead_of_showing_a_traceback(monkeypatch):
    """A virtual environment from before the review was added lacks pydantic,
    lxml, PyYAML and rapidfuzz. The pipeline is imported lazily, so that turns
    up on the first `/review` — as a full-screen traceback, until now.

    Patched at `parse_review_command` because that is the first thing to import
    the pipeline (`litreview.service`, for `ReviewRequest`), earlier than
    `start_review`'s own import of `litreview.config`.
    """
    def missing(_args):
        raise ImportError("No module named 'pydantic'")

    monkeypatch.setattr("bedrock.ui.review_mixin.parse_review_command", missing)
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, "/review does a stale venv crash the app?")
        await pilot.pause()
        notice = notices(app)[-1]
        assert "not installed in this environment" in notice
        assert 'pip install -e ".[dev]"' in notice
        assert "pydantic" in notice
        assert not app._review_running, "nothing should be marked as running"


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


async def test_an_unrecordable_intent_becomes_a_notice_not_a_crash(corpus_file, review_dir, models_yaml):
    """`run_review_job` must catch `ReviewNotRecorded` and turn it into a
    notice like every other failure it handles - it never raises, and the
    app's one-review-at-a-time bookkeeping still gets to run."""
    backend = ReviewBackend()
    history = FakeHistoryStore(fail_states={"intent"})
    settings = Settings(model=MODEL, profile=PROFILES["workstation"], user_id="alice")
    services = make_services(settings, history=history)
    app = BedrockApp(settings=settings, backend=backend, services=services)

    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        text = "\n".join(notices(app))
        assert "Traceback" not in text
        assert app._review_running is False

    assert backend.review_requests == [], "the pipeline must never have run"
async def test_a_journal_that_cannot_be_opened_stops_the_review_with_a_notice(
    corpus_file, review_dir, models_yaml
):
    """A local history that exists but cannot be opened stops the review
    before it starts, with a notice - never a crashed worker."""

    class BrokenServices(ComplianceServices):
        async def history(self):
            raise OSError("history.sqlite3 is read-only")

    backend = ReviewBackend()
    settings = Settings(model=MODEL, profile=PROFILES["workstation"], user_id="alice")
    app = BedrockApp(settings=settings, backend=backend, services=BrokenServices(settings))

    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        text = "\n".join(notices(app))
        assert "Local history is unavailable" in text
        assert "read-only" in text
        assert app._review_running is False

    assert backend.review_requests == []


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


# Agent-pack mode runs `/review` for real: tests/test_agentpack_review.py.


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _run_id(lines: list[str]) -> str:
    for line in reversed(lines):
        if "Literature review finished" in line:
            return line.split("run `")[1].split("`")[0]
    raise AssertionError(f"no finished review in: {lines}")


# ---------------------------------------------------------------------------
# The status bar during a run
# ---------------------------------------------------------------------------


async def test_the_status_bar_shows_the_position_while_a_review_runs(
    corpus_file, review_dir, models_yaml
):
    """The reported gap: rank/extract/write make one model call per item and
    said nothing between milestones, so a working run and a hung one looked
    the same. The count goes in the status bar rather than in notices, because
    one notice per graded paper would be 40 toasts that cover the screen and
    then fade."""
    from bedrock.ui.widgets import StatusBar

    backend = ReviewBackend()
    backend.hold_reviews = asyncio.Event()  # freeze mid-run and look at the bar
    app = make_app(backend)
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --corpus {corpus_file}")
        for _ in range(12):
            await pilot.pause()

        bar = app.query_one(StatusBar)
        assert bar.review_progress is not None, "nothing shown while a review is running"
        assert bar.review_progress.startswith("review: ")

        backend.hold_reviews.set()
        await wait_for_app_workers(app)
        await pilot.pause()

        assert bar.review_progress is None, "the bar must clear when the run ends"


async def test_the_status_bar_clears_even_when_a_review_fails(review_dir, models_yaml):
    """A position frozen on screen after a failed run is exactly the "is it
    still going?" confusion this feature removes."""
    from bedrock.ui.widgets import StatusBar

    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        # No corpus and no --search: the run stops with "no papers".
        await run(app, f"/review {QUESTION}")
        await wait_for_app_workers(app)
        await pilot.pause()

        assert app.query_one(StatusBar).review_progress is None
        assert not app._review_running


async def test_the_progress_counts_never_become_notices(corpus_file, review_dir, models_yaml):
    """If ticks were routed through `progress`, a 40-paper rank would post 40
    toasts. The milestone lines stay; the per-item counts do not."""
    app = make_app(ReviewBackend())
    async with app.run_test() as pilot:
        await pilot.pause()
        await run(app, f"/review {QUESTION} --no-model --corpus {corpus_file}")
        await wait_for_app_workers(app)
        await pilot.pause()

        assert not [n for n in notices(app) if n.startswith("review: ")]
