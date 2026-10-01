"""The `/review` and `/reviews` commands: a literature review beside the chat.

The pipeline itself is `bedrock.litreview`, and the one way into it is
`litreview.service.run_review`. This module is only the terminal's share of the
work: reading what was typed, running the job without ever letting it raise into
the app, and writing the notices. `bedrock-litreview` in a shell and the HTTP
API call the same service, so all three produce the same reports from the same
model calls; only what they say about it differs.

Two things here are load-bearing rather than tidy.

**The question is taken verbatim.** A review question is a sentence a scientist
typed - "What's worth targeting to prevent psoriasis?" - and putting that
through shell-style word splitting would turn the apostrophe into a parse error.
So everything before the first ` --` is the question exactly as written, and
only the options after it are parsed.

**A progress callback must never raise.** It is called from the pipeline's
worker thread; an exception there would end a run that was otherwise fine, tens
of model calls in. `progress_to_toasts` therefore swallows everything.

Notices are not saved into the conversation, for the same reason a fold's are
not: the run itself is recorded in litreview's own database, and `/reviews`
finds it. What is on screen is a receipt, not part of what the model is told.

The `bedrock.litreview` imports are inside the functions that use them, not at
the top. `app.py` imports this module on every launch, and importing the
pipeline pulls in pydantic, lxml and the whole of `sources/` and `stages/` -
about 115 ms of start-up for a feature most sessions never touch. By the time
someone has typed `/review` they are waiting for a job that takes minutes, so
that is the right moment to pay for it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from bedrock.ui.argparsing import CommandParser, CommandUsageError, split_arguments
from bedrock.ui.rendering import as_code, as_code_path, as_cell, markdown_text, safe_line

if TYPE_CHECKING:  # pragma: no cover - types only
    from bedrock.litreview.config import Settings as ReviewSettings
    from bedrock.litreview.service import ReviewRequest

Notify = Callable[[str, str], Awaitable[None]]

REVIEW_QUESTION_MISSING = (
    "Tell me what to review: `/review <question> [--corpus FILE | --search] [--deck]`. "
    "`/review stop` stops a running one, and `/reviews` lists what has run."
)

DECK_NEEDS_PPTX = (
    "The review is finished, but its PowerPoint could not be drawn: python-pptx is "
    'not installed. Install it with `pip install "python-pptx>=1.0,<2"` (or reinstall '
    "Bedrock, see INSTALL.md) and run the review again with `--deck`."
)

SEARCH_NOTICE = (
    "search terms and paper identifiers go to Europe PMC and NCBI; your "
    "question, papers and report stay on this machine"
)

REVIEW_STALE_INSTALL = (
    "**The literature review is not installed in this environment.** "
    "It needs four packages the rest of the app does not (`pydantic`, `lxml`, "
    "`PyYAML`, `rapidfuzz`), so a virtual environment set up before the review "
    "was added will not have them.\n\n"
    "Re-install and restart:\n\n"
    "```\n"
    'pip install -e ".[dev]"\n'
    "```\n\n"
    "If bedrock was installed with `uv tool install`, run `uv tool upgrade "
    "bedrock-private-ai` instead. Everything else in the app keeps working.\n\n"
    "The missing piece was: {detail}"
)


@dataclass(frozen=True)
class ReviewsVerb:
    """One `/reviews` verb, described once.

    Three help tables used to list these by hand - `REVIEWS_HELP` below,
    `ui/commands.py`'s default-mode `/help`, and `agentpack_app.py`'s - and
    they drifted: `settings` and `reinstate` shipped in only the first, so in
    agent-pack mode the retraction settings were undiscoverable. All three are
    now generated from this list, and `tests/test_reviews_help.py` fails if a
    dispatched verb is missing from any of them.
    """

    command: str
    """As typed, with its argument: "/reviews verify <run-id>"."""
    summary: str
    verb: str | None = None
    """The word `run_reviews_command` dispatches on; None for bare `/reviews`
    and for the `--all` variant, which is a flag on an existing verb."""
    writes: bool = False
    """Changes something on this machine. `settings` saves a preference and
    `reinstate` records a decision, so "reads local files only" was wrong of
    them - a claim worth keeping true, because it is what tells a scientist
    whether a command is safe to try."""
    needs_server: bool = False
    """Asks the model server something. Only `check`, and only which models it
    has - no prompt, no paper, no question."""
    compact: bool = False
    """Also shown in the short `/help` tables, which list commands across the
    whole app rather than just this one."""


REVIEWS_VERBS: tuple[ReviewsVerb, ...] = (
    ReviewsVerb("/reviews", "Literature reviews recorded on this machine", compact=True),
    ReviewsVerb("/reviews verify <run-id>", "Re-prove one report against the papers it pinned",
                verb="verify", compact=True),
    ReviewsVerb("/reviews verify --all", "Re-prove every finished report"),
    ReviewsVerb("/reviews check", "Can this machine run a review? What to change",
                verb="check", needs_server=True, compact=True),
    ReviewsVerb("/reviews settings", "Retraction checks: automatic or manual, and what each means",
                verb="settings", writes=True, compact=True),
    ReviewsVerb("/reviews reinstate <paper>",
                "Add back a paper that was retracted and later reinstated (your decision)",
                verb="reinstate", writes=True, compact=True),
    ReviewsVerb("/reviews forget <run-id>", "Delete one run's report and working notes",
                verb="forget", writes=True, compact=True),
)


def reviews_rows(*, compact: bool = False) -> str:
    """The `/reviews` rows of a Markdown help table, without a header."""
    verbs = [v for v in REVIEWS_VERBS if v.compact] if compact else list(REVIEWS_VERBS)
    return "\n".join(f"| `{v.command}` | {v.summary} |" for v in verbs)


REVIEWS_HELP = f"""\
**`/reviews` — literature reviews on this machine**

| Command | What it does |
|---|---|
{reviews_rows()}

Everything here works with the model server switched off, and nothing here
sends your question, your papers or a report to a model. Two of them do change
something on this machine: `settings` saves your choice, and `reinstate`
records your decision to add a paper back. `check` is the one that asks the
server anything, and only which models it has.
"""


@dataclass(frozen=True)
class ReviewCommand:
    """What `/review …` asked for. `stop` is the one verb; everything else
    describes a run."""

    stop: bool = False
    request: ReviewRequest | None = None


def parse_review_command(args: str) -> ReviewCommand:
    """Read `/review <question> [options]`, or `/review stop`.

    Raises `CommandUsageError` for anything that must not start a job, so a
    typo is an error notice rather than a review nobody wanted.
    """
    from bedrock.litreview.service import ReviewRequest

    text = args.strip()
    if not text:
        raise CommandUsageError(REVIEW_QUESTION_MISSING)
    if text.casefold() == "stop":
        return ReviewCommand(stop=True)

    question, option_text = _split_question(text)

    parser = CommandParser("/review")
    parser.add_argument("--corpus")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--no-model", action="store_true", dest="no_model")
    parser.add_argument("--models")
    parser.add_argument("--sources")
    parser.add_argument("--no-full-text", action="store_true", dest="no_full_text")
    parser.add_argument("--patents")
    parser.add_argument("--deck", action="store_true")
    parser.add_argument("--patent-search", action="store_true", dest="patent_search")
    options = parser.parse_args(split_arguments(option_text))

    if options.corpus and options.search:
        raise CommandUsageError(
            "/review: use `--corpus FILE` or `--search`, not both: papers come "
            "from one place so a run can be pinned to them."
        )
    if options.patents and options.patent_search:
        raise CommandUsageError(
            "/review: use `--patents FILE` or `--patent-search`, not both: the "
            "records in the file are the ones a run would use."
        )
    if not question and not options.resume:
        raise CommandUsageError(REVIEW_QUESTION_MISSING)

    return ReviewCommand(
        request=ReviewRequest(
            question=question,
            # A file dropped onto the terminal pastes as `@/path`, as it does
            # for chat attachments; the `@` is not part of the name.
            corpus=Path(options.corpus.removeprefix("@")).expanduser() if options.corpus else None,
            search=options.search,
            refresh=options.refresh,
            resume=options.resume,
            use_model=not options.no_model,
            model_set=options.models,
            sources=_sources(options.sources),
            full_text=not options.no_full_text,
            patents=_patents(options.patents),
            deck=options.deck,
            patent_search=options.patent_search,
        )
    )


def _split_question(text: str) -> tuple[str, str]:
    """Everything before the first ` --` is the question, exactly as typed.

    Word splitting the question would break on an apostrophe, which is common
    in the way people actually ask ("What's worth targeting?"), so it never
    goes near `shlex`.
    """
    marker = text.find(" --")
    if text.startswith("--"):
        return "", text
    if marker == -1:
        return text.strip(), ""
    return text[:marker].strip(), text[marker + 1 :]


def _sources(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise CommandUsageError("/review: --sources needs at least one source name")
    return names


def _patents(path: str | None):
    """Read `--patents FILE` (from `bedrock-litreview patents`) here, on the UI
    thread, before the request is built — not inside the pipeline. It is a
    handful of records at most, and reading it here means a bad or missing
    file is a usage error shown immediately, the same way an unreadable
    `--corpus` would be, rather than a failure discovered minutes into a run.

    `--patent-search` is the other way to the same section: the run searches EPO
    OPS itself, with the account configured on this machine. A file is the
    air-gapped way, and it is the one that needs reading here."""
    if not path:
        return None
    from bedrock.litreview.sources.patent_corpus import read_patents

    try:
        return read_patents(Path(path.removeprefix("@")).expanduser())
    except (OSError, ValueError) as exc:
        raise CommandUsageError(f"/review: --patents {path!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# Running one
# ---------------------------------------------------------------------------


def start_notice(request: ReviewRequest, settings: ReviewSettings) -> str:
    """What is said when a review begins: the question, the models, the papers."""
    if request.corpus is not None:
        papers = f"papers from {as_code_path(request.corpus)}"
    elif request.search:
        papers = f"a new literature search — {SEARCH_NOTICE}"
    elif request.resume:
        papers = f"resuming run {as_code(request.resume)} on the papers it pinned"
    else:
        papers = "the papers already collected for this question"
    model = "no model (rules and code only)" if not request.use_model else f"model set {as_code(settings.model_set)}"
    question = markdown_text(request.question) if request.question else "(resumed run)"
    if request.resume:
        checks = "as when the run started"
    elif getattr(settings, "integrity_checks", "auto") == "auto":
        checks = ("automatic — the papers' ID numbers (not your question) go to PubMed and Crossref; "
                  "`/reviews settings` to change")
    else:
        checks = "manual — nothing is sent; `/reviews settings` to change"
    if request.patents is not None:
        patents = "- Patents: the records given with `--patents` (competitor landscape only, never evidence)\n"
    elif request.patent_search:
        patents = ("- Patents: a new patent search — the question's exact terms go to the European Patent "
                   "Office (competitor landscape only, never evidence)\n")
    else:
        patents = ""
    return (
        f"**Literature review started** — {question}\n\n"
        f"- Papers: {papers}\n"
        f"- Models: {model}\n"
        f"{patents}"
        f"- Retraction check: {checks}\n\n"
        "It runs beside the chat: keep asking questions while it works. "
        "`/review stop` stops it, and the run can be resumed afterwards."
    )


def model_swap_note(request: ReviewRequest, settings: ReviewSettings, scheduler) -> str | None:
    """A warning when chatting during this review would make the server swap models.

    On a machine that serves one call at a time (the `laptop` and `workstation`
    profiles), Ollama is usually set to hold one model in memory. The review's
    models are pinned for determinism, so if the chat is on a different model,
    every chat turn sent during the review unloads one model and loads the
    other, twice - gigabytes each way. That undoes what the laptop profile is
    for. Nothing is changed automatically: which model the chat uses is the
    employee's choice, so this only says what it costs and how to avoid it.
    """
    if not request.use_model or scheduler is None or scheduler.profile.max_parallel > 1:
        return None
    chat_model = scheduler.loaded_model
    review_models = sorted({spec.name for spec in settings.models.values()})
    if chat_model is None or not review_models or chat_model in review_models:
        return None
    return (
        f"The chat is on {as_code(chat_model)} and this review uses "
        f"{', '.join(as_code(name) for name in review_models)}. On a machine that holds one "
        "model at a time, each chat message sent while the review runs makes the server "
        "unload one and load the other, which is slow. To avoid it, switch the chat with "
        f"`/model {review_models[0]}` until the review finishes."
    )


def finish_notice(result, settings: ReviewSettings) -> tuple[str, str]:
    """The receipt for a finished review, and whether it is an error notice.

    A failed integrity check is an error, not a footnote: the report exists and
    opens with a warning banner, and nobody should read it without knowing.
    """
    stats = result.report.stats
    integrity = result.report.integrity
    lines = [
        f"**Literature review finished** — run {as_code(result.run.id)}",
        "",
        f"- Report: {as_code(result.report_id)}",
        f"- Papers: {stats.get('papers_collected', 0)} collected, "
        f"{stats.get('gate_pass', 0)} about the exact subject, {stats.get('papers_read', 0)} read in detail",
        f"- Findings: {stats.get('evidence_units', 0)} verified, "
        f"from {stats.get('findings_proposed', 0)} proposed",
        f"- Sentences: {stats.get('sentences_kept', 0)} kept, "
        f"{stats.get('sentences_dropped', 0)} removed because the evidence did not carry them",
    ]
    if integrity:
        lines.append(f"- Integrity: **FAILED** — {len(integrity)} violation(s):")
        lines.extend(f"  - {safe_line(v, limit=200)}" for v in integrity[:5])
    else:
        lines.append("- Integrity: passed")
    lines.append("")
    lines.extend(f"- {name}: {as_code_path(path)}" for name, path in sorted(result.paths.items()))
    lines += ["", f"{as_code('/reviews verify ' + result.run.id)} re-checks it."]
    return "\n".join(lines), ("error" if integrity else "info")


def interrupted_notice(exc: Exception, run_id: str | None) -> str:
    """What to say when a run stopped partway. Always how to pick it up again."""
    from bedrock.litreview.llm import ModelServerUnavailable

    resume = f"`/review --resume {safe_line(run_id, limit=64)}`" if run_id else "`/reviews` to find its id, then `/review --resume <run-id>`"
    lines = [
        f"**Literature review stopped** — {safe_line(exc, limit=400)}",
        "",
        f"Every finished stage is saved. Resume it with {resume}.",
    ]
    if isinstance(exc, ModelServerUnavailable):
        lines.append(
            "If the server is running but slow, raise `request_timeout` in "
            "`~/.config/bedrock/config.toml` (600 or more on a laptop)."
        )
    return "\n".join(lines)


def progress_to_toasts(post: Callable[[str], None]) -> Callable[[str], None]:
    """Wrap a progress callback so nothing it does can end a run.

    Called from the pipeline's worker thread, tens of model calls in. A toast
    that fails is worth nothing; a run that fails because a toast failed would
    cost the whole review.
    """

    def progress(message: str) -> None:
        try:
            post(safe_line(message, limit=200))
        except Exception:  # noqa: BLE001 - see the docstring
            pass

    return progress


class _NoticeHandler(logging.Handler):
    """Forward the pipeline's warnings and errors to the same toasts.

    litreview logs through the standard library, which would otherwise write
    over the Textual screen. Nothing below WARNING is forwarded: the pipeline's
    INFO log is its progress, and that already arrives through `progress`.
    """

    def __init__(self, post: Callable[[str], None]) -> None:
        super().__init__(level=logging.WARNING)
        self._post = post

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._post(safe_line(record.getMessage(), limit=200))
        except Exception:  # noqa: BLE001 - a log line must not end a run
            pass


async def run_review_job(
    request: ReviewRequest,
    *,
    settings: ReviewSettings,
    scheduler,
    cancel: threading.Event,
    notify: Notify,
    progress: Callable[[str], None],
    history: object | None = None,
    actor_id: str = "",
    tick: Callable[[str, int, int, str], None] | None = None,
    timeout_floor_applied: bool = False,
) -> bool:
    """Run one review and report it. Never raises; True means a finished report.

    Every failure the pipeline can produce is turned into a notice here, so the
    app's own bookkeeping - the one-review-at-a-time flag - always gets to run.

    `history` and `actor_id` go straight to `service.run_review`, which records
    the run in local history when a journal is supplied. `history=None` runs
    unrecorded, which is what a build without `bedrock.provenance` gets.

    `tick` is the per-item signal from inside the stages, which the app puts in
    the status bar rather than in a notice; `progress` stays the milestone
    lines. See `litreview/progress.py`.
    """
    from bedrock.litreview.config import ConfigError
    from bedrock.litreview.journal import ReviewNotRecorded
    from bedrock.litreview.llm import ReviewInterrupted
    from bedrock.litreview.service import run_review
    from bedrock.litreview.sources.literature import SearchFailed
    from bedrock.litreview.stages.s0_question import QuestionError
    from bedrock.litreview.stages.s1_collect import NoPapers, NothingFound

    await notify(start_notice(request, settings), "info")
    note = model_swap_note(request, settings, scheduler)
    if note:
        await notify(note, "info")

    # Said before the run rather than through `run_review`'s `warn`, which would
    # only be read once the run was over. A short `request_timeout` and a nearly
    # full disk are both things to go and change, and the moment they are worth
    # knowing is now - a review takes hours, and `/review` in the app is the one
    # front end that really uses the configured timeout. A notice, not a toast,
    # for the same reason: a toast disappears.
    for advisory in await asyncio.to_thread(
            lambda: _setup_warnings(settings, timeout_floor_applied=timeout_floor_applied)):
        await notify(advisory, "info")

    handler = _NoticeHandler(progress)
    logger = logging.getLogger("bedrock.litreview")
    logger.addHandler(handler)
    try:
        result = await run_review(
            request,
            settings=settings,
            scheduler=scheduler,
            progress=progress_to_toasts(progress),
            cancel=cancel,
            tick=tick,
            history=history,
            actor_id=actor_id,
        )
    except ReviewNotRecorded as exc:
        # Nothing ran: local history could not record that this review was
        # starting - the same "did not start" meaning as a ConfigError.
        await notify(safe_line(exc, limit=600), "error")
        return False
    except ReviewInterrupted as exc:
        run_id = await asyncio.to_thread(_unfinished_run_id, settings)
        await notify(interrupted_notice(exc, run_id), "error")
        return False
    except NothingFound as exc:
        await notify(safe_line(exc, limit=600), "error")
        return False
    except NoPapers:
        await notify(
            "No papers for this question yet. Add `--corpus <file.jsonl>`, or "
            "`--search` to collect them now.",
            "error",
        )
        return False
    except (ConfigError, QuestionError, SearchFailed, OSError, ValueError) as exc:
        await notify(safe_line(exc, limit=600), "error")
        return False
    finally:
        logger.removeHandler(handler)

    text, kind = finish_notice(result, settings)
    await notify(text, kind)
    if request.deck:
        await _write_deck_notice(result, settings, request, notify, author=actor_id)
    return True


def _setup_warnings(settings: ReviewSettings, *, timeout_floor_applied: bool = False) -> list[str]:
    """What is worth knowing before a review starts, and does not stop it.

    Only configuration and the disk, which need no server and no event loop, so
    it is safe on a worker thread. What *stops* a review runs inside
    `run_review`'s `preflight`, where it belongs: this says what is worth
    knowing, not what is allowed. `timeout_floor_applied` is True when the review
    opens its own connection and so already has its timeout raised.
    """
    from bedrock.litreview.service import config_findings

    try:
        return [f.detail for f in config_findings(settings, timeout_floor_applied=timeout_floor_applied)
                if not f.ok]
    except Exception:  # noqa: BLE001 - an advisory must never stop a review
        return []


async def _write_deck_notice(result, settings: ReviewSettings, request: ReviewRequest, notify: Notify, *, author: str) -> None:
    """Draw the PowerPoint after a finished review and say where it is.

    Drawing reads the run's manifest and writes a file, so it runs in a worker
    thread and the screen stays alive. It never reaches the network (the
    figures were fetched during the run, and only with `--search`). Nothing
    here can undo the review: a missing python-pptx or a failed write is one
    notice, and the report stays finished.
    """
    from bedrock.litreview.service import write_deck

    try:
        path = await asyncio.to_thread(write_deck, result, settings, author=author)
    except ImportError:
        await notify(DECK_NEEDS_PPTX, "error")
        return
    except Exception as exc:  # noqa: BLE001 - a deck that fails to draw must not look like a failed review
        await notify(f"The review is finished, but its PowerPoint could not be written: {safe_line(exc, limit=400)}", "error")
        return
    lines = [f"**PowerPoint written** — {as_code_path(path)}"]
    if not request.search:
        lines += ["", "No paper figures are in it: they are fetched only with `--search`."]
    await notify("\n".join(lines), "info")


def _unfinished_run_id(settings: ReviewSettings) -> str | None:
    """The most recent run with no report, so a notice can name what to resume."""
    from bedrock.litreview.db import Database

    try:
        db = Database(settings.db_path)
        try:
            for run in db.recent_runs(20):
                if run.finished is None:
                    return run.id
        finally:
            db.conn.close()
    except Exception:  # noqa: BLE001 - a hint must not replace the real error
        pass
    return None


# ---------------------------------------------------------------------------
# /reviews
# ---------------------------------------------------------------------------


async def run_reviews_command(
    args: str,
    *,
    settings_for,
    notify: Notify,
    readiness_for=None,
) -> None:
    """`/reviews`, and every verb in `REVIEWS_VERBS`.

    `settings_for` is a zero-argument callable returning litreview's settings,
    so a broken `[litreview]` table is reported here rather than at startup.
    All the reading happens in a worker thread, with the database opened and
    closed there - SQLite connections belong to the thread that made them.

    `readiness_for` is an async zero-argument callable returning
    `bedrock.litreview.readiness.review_readiness`'s checks, which only `check`
    needs. Listing, verifying and forgetting want `db_path` and nothing about
    models, and a site whose `models.yaml` names only its own set must still be
    able to manage its past runs - so `check` loads its own settings, and a
    broken configuration is something it reports rather than something that
    stops the other verbs.
    """
    from bedrock.litreview.config import ConfigError

    text = args.strip()
    if text.casefold() in {"help", "--help", "-h"}:
        await notify(REVIEWS_HELP, "info")
        return

    if text.casefold() in {"check", "doctor"}:
        await _check(readiness_for, notify)
        return

    try:
        settings = await asyncio.to_thread(settings_for)
    except ConfigError as exc:
        await notify(safe_line(exc, limit=600), "error")
        return

    if text.casefold().startswith("verify"):
        await _verify(text[len("verify"):].strip(), settings, notify)
        return
    if text.casefold().startswith("settings"):
        await _settings(text[len("settings"):].strip(), settings, notify)
        return
    if text.casefold().startswith("reinstate"):
        await _reinstate(text[len("reinstate"):].strip(), settings, notify)
        return
    if text.casefold().startswith("forget"):
        await _forget(text[len("forget"):].strip(), settings, notify)
        return
    if text:
        await notify(f"/reviews: I do not know `{markdown_text(text)}`. Try `/reviews help`.", "error")
        return

    try:
        rows = await asyncio.to_thread(_recent_rows, settings)
    except OSError as exc:
        await notify(safe_line(exc, limit=400), "error")
        return
    if not rows:
        await notify(
            "No literature reviews on this machine yet. Start one with "
            "`/review <question> --corpus <file.jsonl>` (papers from a file), or "
            "`/review <question> --search` (a live search of Europe PMC and PubMed).",
            "info",
        )
        return

    lines = [
        "**Literature reviews on this machine**",
        "",
        "| Run | Started | Report | Question |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| `{as_cell(run_id, 20)}` | {as_cell(started, 20)} | {as_cell(report or 'unfinished', 24)} | {as_cell(question, 60)} |"
        for run_id, started, report, question in rows
    ]
    lines += ["", f"{as_code('/reviews verify <run-id>')} re-proves one report against the papers it pinned."]
    await notify("\n".join(lines), "info")


async def _settings(argument: str, settings: "ReviewSettings", notify: Notify) -> None:
    """`/reviews settings [auto|manual]`: show or change how reviews check
    their papers for retractions, explained for someone who is not technical."""
    import os

    from bedrock.litreview.preferences import (
        INTEGRITY_EXPLAINED,
        INTEGRITY_MODES,
        PreferenceError,
        integrity_mode,
        save_integrity_mode,
    )

    choice = argument.strip().casefold()
    if choice and choice not in INTEGRITY_MODES:
        await notify(f"/reviews settings: choose `auto` or `manual`, not `{markdown_text(argument)}`.", "error")
        return
    if choice:
        try:
            await asyncio.to_thread(save_integrity_mode, settings.data_dir, choice)
        except (OSError, PreferenceError) as exc:
            await notify(safe_line(exc, limit=400), "error")
            return
    try:
        current = integrity_mode(settings.data_dir)
    except PreferenceError as exc:
        await notify(safe_line(exc, limit=400), "error")
        return
    names = {"auto": "Automatic", "manual": "Manual"}
    lines = [f"**Retraction checks: {names[current]}**" + (" — saved." if choice else ""), ""]
    if os.environ.get("BEDROCK_LITREVIEW_INTEGRITY_CHECKS"):
        lines += ["This machine sets it with `BEDROCK_LITREVIEW_INTEGRITY_CHECKS`, which wins over the choice "
                  "saved here.", ""]
    lines += ["Before a review, the app can check whether any of its papers has been retracted, questioned "
              "by its journal (an *expression of concern*) or corrected. Your two choices:", ""]
    for mode in INTEGRITY_MODES:
        marker = " ← current" if mode == current else ""
        lines += [INTEGRITY_EXPLAINED[mode] + marker, ""]
    other = "manual" if current == "auto" else "auto"
    lines.append(f"To switch: {as_code(f'/reviews settings {other}')}. The choice applies to reviews started "
                 "from now on; reports already written do not change.")
    await notify("\n".join(lines), "info")


async def _reinstate(argument: str, settings: "ReviewSettings", notify: Notify) -> None:
    """`/reviews reinstate <paper> [--undo] [--note TEXT]`: a scientist adds
    back a paper that was retracted and later reinstated. Nothing is added
    back unless someone does this."""
    import getpass
    import os
    from datetime import datetime, timezone

    parser = CommandParser("/reviews reinstate")
    parser.add_argument("paper")
    parser.add_argument("--undo", action="store_true")
    parser.add_argument("--note")
    try:
        options = parser.parse_args(split_arguments(argument))
    except CommandUsageError as exc:
        await notify(str(exc), "error")
        return
    who = os.environ.get("BEDROCK_USER") or getpass.getuser()

    def decide() -> str:
        from bedrock.litreview.db import Database
        from bedrock.litreview.publication_integrity import decide_reinstated

        db = Database(settings.db_path)
        try:
            return decide_reinstated(db, options.paper, include=not options.undo, by=who, note=options.note,
                                     now=datetime.now(timezone.utc))
        finally:
            db.conn.close()

    try:
        message = await asyncio.to_thread(decide)
    except (ValueError, OSError) as exc:
        await notify(f"/reviews reinstate: {safe_line(exc, limit=400)}", "error")
        return
    await notify(message, "info")


def _recent_rows(settings: ReviewSettings) -> list[tuple[str, str, str | None, str]]:
    from bedrock.litreview.db import Database

    if not Path(settings.db_path).exists():
        return []
    db = Database(settings.db_path)
    try:
        rows = []
        for run in db.recent_runs(20):
            question = db.get_question(run.question_id)
            rows.append((run.id, run.started, db.report_id_for(run.id), question.text if question else ""))
        return rows
    finally:
        db.conn.close()


async def _verify(argument: str, settings: "ReviewSettings", notify: Notify) -> None:
    parser = CommandParser("/reviews verify")
    parser.add_argument("run_id", nargs="?")
    parser.add_argument("--all", action="store_true", dest="all_runs")
    try:
        options = parser.parse_args(split_arguments(argument))
    except CommandUsageError as exc:
        await notify(str(exc), "error")
        return
    if not (options.run_id or options.all_runs):
        await notify("/reviews verify: name a run id, or use `--all`.", "error")
        return

    try:
        results = await asyncio.to_thread(_verify_runs, settings, options.run_id, options.all_runs)
    except OSError as exc:
        await notify(safe_line(exc, limit=400), "error")
        return
    if not results:
        await notify("No finished reviews to check.", "info")
        return

    lines = ["**Verification**", ""]
    failed = False
    changed = False
    for run_id, violations, summary, differences, advisories in results:
        if violations:
            failed = True
            lines.append(f"- {as_code(run_id)}: **{len(violations)} violation(s)**")
            lines.extend(f"  - {safe_line(v, limit=200)}" for v in violations[:5])
        else:
            lines.append(f"- {as_code(run_id)}: OK")
        # The same three answers `bedrock-litreview verify-run` prints: the
        # environment is a disclosure beside the verdict, not a violation.
        if summary:
            lines.append(f"  - {safe_line(summary, limit=400)}")
            lines.extend(f"    - {safe_line(d, limit=200)}" for d in differences[:5])
            changed = changed or bool(differences)
        # Recorded retractions or concerns about cited papers: advice for a
        # person, from this machine's own records; nothing is fetched here.
        lines.extend(f"  - ! {safe_line(x, limit=300)}" for x in advisories[:5])
    if failed:
        lines += ["", "A violation is a bug in the pipeline, not a judgement about the papers."]
    if changed:
        lines += ["", "A changed environment means today's checks are not the ones the run was built "
                  "under; it does not mean the report is corrupt."]
    await notify("\n".join(lines), "error" if failed else "info")


def _verify_runs(
    settings: ReviewSettings, run_id: str | None, all_runs: bool
) -> list[tuple[str, list[str], str | None, list[str], list[str]]]:
    """(run id, violations, one-line summary, environment differences,
    publication advisories) per run."""
    from bedrock.litreview.db import Database
    from bedrock.litreview.report.invariants import assess_run

    if not Path(settings.db_path).exists():
        return []
    db = Database(settings.db_path)
    try:
        # Only runs that finished a report; an interrupted run has nothing to verify.
        run_ids = db.finished_run_ids() if all_runs else [run_id]
        out = []
        for one in run_ids:
            try:
                a = assess_run(db, one)
                changes = [d.line() for d in a.environment.differences
                           if d.kind in ("changed", "source_changed", "missing_now")]
                out.append((one, a.violations, a.summary(), changes, a.advisories))
            except Exception as exc:  # noqa: BLE001 - an unreadable run is a failure too
                out.append((one, [f"cannot be verified: {type(exc).__name__}: {exc}"], None, [], []))
        return out
    finally:
        db.conn.close()


# ---------------------------------------------------------------------------
# /reviews check
# ---------------------------------------------------------------------------


async def _check(readiness_for, notify: Notify) -> None:
    """`/reviews check`: whether a review could start, without starting one.

    The same implementation `bedrock-litreview doctor` prints
    (`litreview/readiness.py`), so the two cannot disagree. Worth having in the
    app as well as the shell, because the person who hits "a review needs a
    context window of at least 12,288 tokens" is in the app when they hit it,
    and the rest of what is wrong with the machine is worth knowing at the same
    time.
    """
    from bedrock.litreview.readiness import operational

    if readiness_for is None:
        await notify("/reviews check: not available here. `bedrock-litreview doctor` runs the same check "
                     "from a shell.", "error")
        return
    try:
        checks = await readiness_for()
    except Exception as exc:  # noqa: BLE001 - a check that crashes is a finding, not a crash of the app
        await notify(f"**Literature review setup**\n\n- **FAIL** — the check itself failed: "
                     f"{safe_line(exc, limit=400)}", "error")
        return

    marks = {True: "ok", False: "**FAIL**", None: "note"}
    lines = ["**Literature review setup**", ""]
    lines += [f"- {marks[c.ok]} — {markdown_text(c.name)}: {safe_line(c.detail, limit=500)}" for c in checks]
    ready = operational(checks)
    lines += ["", "A review can run." if ready else "A review cannot run yet; fix the FAIL lines above."]
    await notify("\n".join(lines), "info" if ready else "error")


# ---------------------------------------------------------------------------
# /reviews forget
# ---------------------------------------------------------------------------


async def _forget(argument: str, settings: ReviewSettings, notify: Notify) -> None:
    """`/reviews forget <run-id>`: delete one run's output.

    Deletes on the first ask, as `/digest forget` does, rather than putting a
    confirmation in front of it: naming a run id is already a deliberate act,
    and the app has no plain yes/no prompt. What it does instead is say exactly
    what went, so a mistake is visible immediately rather than discovered later.
    (`bedrock-litreview forget` does confirm, because a shell can ask.)
    """
    parser = CommandParser("/reviews forget")
    parser.add_argument("run_id", nargs="?")
    try:
        options = parser.parse_args(split_arguments(argument))
    except CommandUsageError as exc:
        await notify(str(exc), "error")
        return
    if not options.run_id:
        await notify(
            "/reviews forget: name a run id — `/reviews` lists them. This deletes that "
            "run's report and working notes and cannot be undone; the question and its "
            "papers stay, so it can be asked again.",
            "error",
        )
        return

    try:
        result = await asyncio.to_thread(_forget_run, settings, options.run_id)
    except OSError as exc:
        await notify(safe_line(exc, limit=400), "error")
        return
    if result is None:
        await notify(
            f"/reviews forget: no run {as_code(options.run_id)} on this machine. "
            f"{as_code('/reviews')} lists them.",
            "error",
        )
        return

    deleted, files = result
    lines = [
        f"**Deleted review run {as_code(deleted.run_id)}**",
        "",
        f"- {deleted.artifacts} stage artifact(s) removed",
    ]
    if deleted.report_deleted:
        lines.append(f"- Report {as_code(deleted.report_id)} removed, with {len(files)} file(s)")
        if deleted.ratings:
            lines.append(f"- {deleted.ratings} rating(s) of that report removed")
        if not files:
            lines.append("- No report files were found where they are normally written. "
                         "A report written to another directory has to be deleted by hand.")
    elif deleted.report_shared_with:
        lines.append(
            f"- Report {as_code(deleted.report_id)} was **kept**: "
            f"{', '.join(as_code(r) for r in deleted.report_shared_with)} produced the identical "
            "report and still points at it"
        )
    else:
        lines.append("- The run wrote no report, so there was none to remove")
    lines += ["", "The question and its papers were not touched, so the same question can be asked again."]
    await notify("\n".join(lines), "info")


def _forget_run(settings: ReviewSettings, run_id: str):
    from bedrock.litreview.api import forget_run
    from bedrock.litreview.db import Database

    if not Path(settings.db_path).exists():
        return None
    db = Database(settings.db_path)
    try:
        result = forget_run(run_id, db=db, settings=settings)
        return None if result is None else (result.deleted, result.files_removed)
    finally:
        db.conn.close()
