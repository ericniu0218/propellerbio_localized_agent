"""The one way into a review that uses a model.

`/review` in the terminal app, the `bedrock-litreview` command and the HTTP API
all come through here, so all three get the same pipeline, the same scheduler
and the same guarantees. Adding a fourth front end should mean calling
`run_review`, never reaching past it into `runner.answer_question`.

Two jobs, and they are both about the seam rather than the science:

`preflight` fails a run *before* it starts, with the fix written out. A review
is a long job - hundreds of calls - and the two ways it goes wrong on a fresh
machine are a context window too small to hold an extraction prompt and a model
that has not been pulled. Both are silent at call time: a small window makes the
server quietly drop the start of the prompt, so the report comes out thin rather
than failing. Neither is worth discovering forty minutes in.

`run_review` handles the thread arrangement. The pipeline is blocking code and
the scheduler is asyncio, so the pipeline runs in a worker thread and each model
call is handed back to the loop (see `llm/scheduled.py`). That also keeps the
terminal UI responsive while a review runs, which is the whole point of running
one beside the chat.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from bedrock.inference import InferenceError, Scheduler
from bedrock.litreview.config import ConfigError, Settings, private_dir
from bedrock.litreview.journal import ReviewNotRecorded, commit_intent, commit_outcome, new_operation_id
from bedrock.litreview.llm import ReviewInterrupted, make_client
from bedrock.litreview.models import Paper
from bedrock.litreview.runner import RunResult, answer_question
from bedrock.litreview.sources.corpus import CorpusSearcher
from bedrock.litreview.sources.literature import LiteratureSearcher, SearchFailed
from bedrock.litreview.stages.s0_question import QuestionError
from bedrock.litreview.stages.s1_collect import NoPapers, NothingFound


@dataclass(frozen=True)
class ReviewRequest:
    """One review, as any front end describes it."""

    question: str
    corpus: Path | None = None
    search: bool = False
    refresh: bool = False
    resume: str | None = None
    use_model: bool = True
    model_set: str | None = None
    sources: tuple[str, ...] | None = None
    full_text: bool = True
    out_dir: Path | None = None

    papers: list[Paper] | None = None
    """Papers posted with the question, over HTTP."""

    approved_terms: dict[str, list[str]] | None = None
    """Extra search terms a person approved by hand. HTTP only, as in the source."""


@dataclass(frozen=True)
class ReviewCapacity:
    """What preflight established about the models a review will use."""

    windows: dict[str, int]
    """Model -> the context window every call to it is sized against: the
    smaller of what the server reports and what was configured."""
    served_by: dict[str, str]
    """Model -> the weights behind it, when the server names models by alias."""
    notes: tuple[str, ...] = ()
    """Things worth telling the scientist that do not stop the run."""


def _where(settings: Settings) -> str:
    if settings.endpoint_source == "agentpack":
        return f"the agent server's inference API at {settings.base_url}"
    return f"the model server at {settings.base_url}"


def _window_fix(settings: Settings, model: str) -> str:
    need = settings.min_context_window
    if settings.endpoint_source == "agentpack":
        return (f"On the agent server, give the role behind {model!r} a context_length of at least {need} in its "
                f"models.yaml (and start its vLLM with --max-model-len {need} or more), or choose a model set "
                "whose models have one with `model_set` in the [litreview] table.")
    return (f"Start Ollama with OLLAMA_CONTEXT_LENGTH={need} (the Ollama app: Settings -> Context length) - that "
            f"fits on a 16 GB machine - or vLLM with --max-model-len {need}, and set `context_window = {need}` in "
            f"~/.config/bedrock/config.toml (or BEDROCK_CONTEXT_WINDOW={need}). Otherwise use the `workstation` "
            "profile, or the office server.")


async def preflight(settings: Settings, backend) -> ReviewCapacity:
    """Check the machine can run a review at all, and say what to change if not.

    Raises `ConfigError` with the fix spelled out. Every message names a number
    or a command, because the person reading it is usually a scientist rather
    than the person who set the server up.

    The window each model gets is checked against what the *server* says when
    it says anything: vLLM and bedrock's agent server report the window a model
    was started with, and Ollama reports the window of a model it has loaded.
    A configured number is a claim about the server; a reported one is the
    server's own account, and the smaller of the two is what a prompt really
    gets. A server that reports nothing is taken at the configured number, and
    one that reports nothing when nothing was configured is refused - a remote
    model's window is never assumed. Each call is checked again as it runs
    (`llm/scheduled.py` `PromptTruncated`), because Ollama loads a model with
    whatever window it was started with and only says which once it has.
    """
    need = settings.min_context_window
    configured = settings.context_window if settings.context_window_source != "unset" else None
    if configured is not None and configured < need:
        raise ConfigError(
            f"a literature review needs a context window of at least {need:,} tokens, and this machine is set to "
            f"{configured:,}. Extraction sends about 4,300 tokens of prompt and allows a 4,096-token reply, "
            f"which {configured:,} cannot hold.\n" + _window_fix(settings, "the review's models")
        )

    try:
        offered = {info.id: info for info in await backend.list_models()}
    except InferenceError as exc:
        hint = ("Check that the agent server is running and reachable, and that `agent_pack_api_base` points at it."
                if settings.endpoint_source == "agentpack" else "Start it, or point `api_base` at the office server.")
        raise ConfigError(f"{_where(settings)} is not answering: {exc}\n{hint}") from exc

    wanted = sorted({settings.model_for(job) for job in settings.jobs})
    missing = [name for name in wanted if name not in offered]
    if missing:
        have = ", ".join(sorted(offered)) or "none"
        if settings.endpoint_source == "agentpack":
            fix = ("Choose a model set whose names the server offers, with `model_set` in the [litreview] table or "
                   "BEDROCK_LITREVIEW_MODELS, or ask IT to expose those roles on the agent server.")
        else:
            pulls = "\n".join(f"  ollama pull {name}" for name in missing)
            fix = (f"On a laptop running Ollama:\n{pulls}\nOn the office server (vLLM), ask IT to serve it - a vLLM "
                   "server offers only the model it was started with.")
        raise ConfigError(
            f"{_where(settings)} does not have {', '.join(repr(m) for m in missing)}, which the "
            f"`{settings.model_set}` model set needs. It offers: {have}.\n{fix}"
        )

    windows: dict[str, int] = {}
    served_by: dict[str, str] = {}
    notes: list[str] = []
    for name in wanted:
        info = offered[name]
        if info.root:
            served_by[name] = info.root
        caps = info.capabilities
        if caps is not None and "json_schema" not in caps:
            raise ConfigError(
                f"{_where(settings)} says {name!r} cannot constrain its output to a JSON schema, and every call a "
                "review makes needs it: without it a reply can be malformed JSON, and findings are lost silently. "
                "Choose a model that supports structured output (vLLM guided decoding, Ollama structured outputs)."
            )
        if caps is not None and "seed" not in caps:
            notes.append(f"{name!r} ignores the random seed, so a fresh run may word things differently; "
                         "cached calls still replay exactly.")
        reported = info.context_length
        if reported and configured and reported < configured:
            notes.append(f"{name!r} is served with a {reported:,}-token window, smaller than the {configured:,} "
                         "configured here; calls are sized to the server's.")
        window = min(w for w in (reported, configured) if w) if (reported or configured) else None
        if window is None:
            raise ConfigError(
                f"{_where(settings)} does not say how large a context window {name!r} has, and none is "
                f"configured for the review. It needs at least {need:,} tokens. Set `context_window = <the "
                "model's window>` in the [litreview] table of ~/.config/bedrock/config.toml (or "
                "BEDROCK_LITREVIEW_CONTEXT_WINDOW) to the window the server really gives it."
            )
        if window < need:
            said = "reports" if reported == window else "is configured with"
            raise ConfigError(
                f"{_where(settings)} {said} a {window:,}-token context window for {name!r}, and a literature "
                f"review needs at least {need:,}: extraction sends about 4,300 tokens of prompt and allows a "
                f"4,096-token reply. A smaller window does not fail - the server drops the start of each prompt, "
                f"and the model answers without its instructions.\n" + _window_fix(settings, name)
            )
        windows[name] = window
    return ReviewCapacity(windows=windows, served_by=served_by, notes=tuple(notes))


async def run_review(
    request: ReviewRequest,
    *,
    settings: Settings,
    scheduler: Scheduler | None,
    progress: Callable[[str], None],
    cancel: threading.Event,
    history: Any | None = None,
    actor_id: str = "",
) -> RunResult:
    """Run one review and return its result.

    `scheduler` may be None only when `request.use_model` is False. A run
    without a model never reaches a server, so the shell commands that do that
    do not open one.

    `progress` is called from the worker thread, so a caller that touches a UI
    has to hand the message back to its own thread. It must never raise: a
    progress line that throws would take the run down with it.

    `ConfigError`, `QuestionError`, `NoPapers`, `SearchFailed`,
    `ReviewInterrupted`, `OSError`, `ValueError` and `ReviewNotRecorded` all
    reach the caller, which is the piece that knows how to show them.
    `ReviewNotRecorded` (`bedrock.litreview.journal`) means local history
    could not record that this review was starting, so nothing ran at all -
    the same "did not start" meaning as `ConfigError`, not a run that failed.

    `history` and `actor_id` are optional and duck-typed - see
    `bedrock.litreview.journal` for the shape and the rules. `history=None`
    (the default, and what every caller that predates this had) journals
    nothing and runs exactly as before. When a journal is supplied, this
    function commits a `litreview.review` intent - and, for a run that is
    about to contact a source outside this machine, one `litreview.search`
    intent per host - before any pipeline work; if any of those cannot be
    written, `ReviewNotRecorded` is raised and the pipeline never runs.

    After the pipeline finishes (or raises), one outcome is committed for the
    review *and one for every search intent already committed*, all with the
    same final state, because a search event says "this run was authorised
    and began to contact this host", and what became of that run is exactly
    what became of the authorisation: `succeeded` with the finished run's id
    (on the review's own outcome only - a search event's outcome is
    `succeeded` with nothing more than its own `source_kind`, never a copy of
    the run id or the question), `failed` with a bounded error code, or
    `cancelled` for a run `ReviewInterrupted` stopped (every stage's work up
    to that point is saved, and it can be resumed). A cancellation that
    reaches this function from *outside* the pipeline - a real
    `asyncio.CancelledError`, which `asyncio.to_thread` cannot use to stop a
    worker thread that is already running - commits `unknown` instead,
    because nothing here can prove whether that thread has actually stopped.
    If the pipeline itself never starts (the intents above failed to commit),
    no outcome is written for anything: an intent with no outcome is not a
    bug to paper over, it is this journal's honest way of saying the
    authorisation was recorded but nothing after it is confirmed. A failure
    to write any of these outcomes never re-runs the review to tidy the
    journal; it is reported through `progress` instead, exactly as
    `bedrock.ui.journal` does for `/digest`.
    """
    private_dir(settings.data_dir)

    client = None
    if request.use_model:
        if scheduler is None:
            raise ValueError("a review that uses a model needs bedrock's scheduler")
        capacity = await preflight(settings, scheduler.backend)
        for note in capacity.notes:
            progress(note)
        client = make_client(
            settings,
            scheduler=scheduler,
            loop=asyncio.get_running_loop(),
            cancel=cancel,
            context_windows=capacity.windows,
        )

    searcher = _searcher(request, settings)
    # A resumed run keeps what it knew when it started; only a new one checks.
    integrity_check = None if request.resume else auto_integrity_check(settings)

    review_operation_id = new_operation_id()
    # Search declarations first: if any of them - or the review's own intent
    # right after - cannot be written, nothing below has happened yet, and
    # raising here (rather than after) means no half-authorised run is ever
    # started. Each intent is kept with its source_kind so the matching
    # outcome, once the pipeline is done, can be written for exactly the
    # hosts this run actually declared. See the module docstring on why this
    # is metadata about authorisation, not a log of individual HTTP requests.
    search_operations: list[tuple[str, str]] = []
    for source_kind in _search_kinds(request, searcher, integrity_check):
        search_operation_id = new_operation_id()
        await commit_intent(
            history, search_operation_id, actor_id=actor_id, action="litreview.search",
            details={"source_kind": source_kind},
        )
        search_operations.append((search_operation_id, source_kind))
    await commit_intent(
        history, review_operation_id, actor_id=actor_id, action="litreview.review",
        subject_id=request.resume,
    )

    async def _finish_searches(state: str, *, error_code: str | None = None) -> None:
        """Every search intent committed above gets the run's own final
        state - see the docstring on why an authorisation's outcome mirrors
        what became of the run it authorised, not a separate verdict."""
        for search_operation_id, source_kind in search_operations:
            details = {"source_kind": source_kind}
            if error_code is not None:
                details["error_code"] = error_code
            await commit_outcome(
                history, search_operation_id, actor_id=actor_id, action="litreview.search",
                state=state, details=details, progress=progress,
            )

    try:
        result = await asyncio.to_thread(
            answer_question,
            request.question,
            settings=settings,
            client=client,
            use_model=request.use_model,
            searcher=searcher,
            papers=request.papers,
            refresh_papers=request.refresh,
            approved_terms=request.approved_terms,
            out_dir=request.out_dir,
            resume_run_id=request.resume,
            progress=progress,
            integrity_check=integrity_check,
        )
    except asyncio.CancelledError:
        # `asyncio.to_thread` cannot stop a thread that is already running the
        # pipeline, so whether it is still going is genuinely unknown - the
        # same reason `ui/digest_command.py` records an interrupted ingest
        # this way rather than claiming it stopped.
        await commit_outcome(
            history, review_operation_id, actor_id=actor_id, action="litreview.review",
            state="unknown", subject_id=request.resume,
            details={"error_code": "review_cancelled_outcome_unknown"}, progress=progress,
        )
        await _finish_searches("unknown", error_code="review_cancelled_outcome_unknown")
        raise
    except ReviewInterrupted:
        # The pipeline thread has already finished (by raising), and every
        # stage's work up to that point is saved - an unreachable server and
        # an explicit `/review stop` both land here. Resumable, not failed.
        await commit_outcome(
            history, review_operation_id, actor_id=actor_id, action="litreview.review",
            state="cancelled", subject_id=request.resume, progress=progress,
        )
        await _finish_searches("cancelled")
        raise
    except (ConfigError, QuestionError, NoPapers, SearchFailed, OSError, ValueError) as exc:
        code = _error_code(exc)
        await commit_outcome(
            history, review_operation_id, actor_id=actor_id, action="litreview.review",
            state="failed", subject_id=request.resume,
            details={"error_code": code}, progress=progress,
        )
        await _finish_searches("failed", error_code=code)
        raise
    else:
        await commit_outcome(
            history, review_operation_id, actor_id=actor_id, action="litreview.review",
            state="succeeded", subject_id=request.resume,
            details={"run_id": result.run.id}, progress=progress,
        )
        await _finish_searches("succeeded")
        return result


def _search_kinds(request: ReviewRequest, searcher, integrity_check) -> list[str]:
    """Which external hosts this run is about to be authorised to contact.

    Matches `stages.s1_collect.collect` exactly, rather than re-deriving the
    condition: a `LiteratureSearcher` is only actually called when no frozen
    `papers` were also handed in (`request.papers is None`), so recording a
    search here for a request that also posted papers directly would claim
    a contact that never happens. The retraction check is a run's own
    decision (`integrity_check is not None`), already computed by the caller.
    """
    kinds = []
    if isinstance(searcher, LiteratureSearcher) and request.papers is None:
        kinds.append("literature_search")
    if integrity_check is not None:
        kinds.append("retraction_check")
    return kinds


def _error_code(exc: Exception) -> str:
    """A bounded, code-chosen identifier for a failed review's outcome -
    never `str(exc)`, which could carry a file path or other detail this
    journal must not store. Order matters: `NothingFound` is a `NoPapers`."""
    if isinstance(exc, ConfigError):
        return "config_error"
    if isinstance(exc, QuestionError):
        return "question_error"
    if isinstance(exc, NothingFound):
        return "nothing_found"
    if isinstance(exc, NoPapers):
        return "no_papers"
    if isinstance(exc, SearchFailed):
        return "search_failed"
    if isinstance(exc, OSError):
        return "os_error"
    return "value_error"


def local_history(bedrock_settings) -> tuple[Any | None, str]:
    """(the local action journal, who to credit) for a review run outside the app.

    The CLI and the HTTP API record their runs in the same journal the app
    does (`action_history_dir`, from BEDROCK_ACTION_HISTORY_DIR, the config
    file, then the default). A build without `bedrock.provenance` runs
    unrecorded, as `/digest` does. A journal that exists but cannot be opened
    raises `ReviewNotRecorded`: the review must not start unrecorded.
    `user_id` is BEDROCK_USER - configuration, not authentication.
    """
    try:
        from bedrock.provenance import HistoryStore
    except ImportError:  # pragma: no cover - depends on the installed build
        return None, bedrock_settings.user_id
    try:
        return HistoryStore(Path(bedrock_settings.action_history_dir).expanduser()), bedrock_settings.user_id
    except Exception as exc:  # noqa: BLE001 - any failure here stops the run
        raise ReviewNotRecorded(f"local history could not be opened ({exc}); nothing was run.") from exc


def auto_integrity_check(settings: Settings):
    """The retraction check a new review runs before it reads its papers, or
    None when the setting is `manual` (`/reviews settings`, `preferences.py`).
    See `sources.integrity.automatic_check_for`."""
    from bedrock.litreview.sources.integrity import automatic_check_for

    return automatic_check_for(settings)


def _searcher(request: ReviewRequest, settings: Settings):
    """Where this run's papers come from.

    `None` means the question's frozen set, already in the database. The
    literature searcher is built only when someone asked for it: a review
    never *searches* by default. (The retraction check is separate and sends
    identifiers only; see `auto_integrity_check`.)
    """
    if request.corpus is not None:
        return CorpusSearcher(request.corpus)
    if request.search:
        overrides: dict[str, Any] = {}
        if request.sources:
            overrides["sources"] = tuple(request.sources)
        if not request.full_text:
            overrides["full_text_max"] = 0
        return LiteratureSearcher.from_settings(settings, **overrides)
    return None


def review_backend(settings: Settings, *, timeout_floor: float = 600.0):
    """A connection to the review's model server (`settings.base_url`), with
    the long timeout a review needs. Opening one starts nothing: it is an HTTP
    client, and the model runs wherever `base_url` says."""
    from bedrock.config import Settings as BedrockSettings
    from bedrock.inference import create_backend

    # Through bedrock's factory, the one place a backend class is chosen
    # (inference/factory.py) - and the seam tests replace.
    return create_backend(BedrockSettings(
        api_base=settings.base_url,
        api_key=settings.api_key,
        request_timeout=max(float(settings.request_timeout), timeout_floor),
    ))


@asynccontextmanager
async def standalone_scheduler(bedrock_settings, *, backend=None, timeout_floor: float = 600.0,
                               review_settings: Settings | None = None):
    """A scheduler for a process that is not the app: the CLI, the benchmark, the HTTP API.

    The app already has one, shared with the chat. Everywhere else there is no
    chat to share with, so this opens a backend, wraps it in a scheduler of its
    own, and closes it again afterwards.

    The timeout floor is not cosmetic. bedrock's default is 120 seconds, which
    is fine for a chat reply and not for a 4,096-token extraction on a laptop
    model; the review would fail as "server unavailable" halfway through.
    `backend` is for tests.
    """
    from dataclasses import replace

    from bedrock.inference import create_backend

    own = backend is None
    if own and review_settings is not None:
        # The review's own endpoint, which in agent-pack mode is the agent
        # server rather than bedrock's `api_base`.
        backend = review_backend(review_settings, timeout_floor=timeout_floor)
    elif own:
        timeout = max(float(bedrock_settings.request_timeout), timeout_floor)
        backend = create_backend(replace(bedrock_settings, request_timeout=timeout))
    try:
        yield Scheduler(backend, bedrock_settings.profile)
    finally:
        if own:
            await backend.close()
