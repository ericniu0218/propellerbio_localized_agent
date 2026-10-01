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
import json
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from bedrock.inference import InferenceError, Scheduler
from bedrock.litreview.config import ConfigError, Settings, config_dir, private_dir
from bedrock.litreview.db import Database
from bedrock.litreview.journal import ReviewNotRecorded, commit_intent, commit_outcome, new_operation_id
from bedrock.litreview.llm import ReviewInterrupted, make_client
from bedrock.litreview.models import Paper, PatentRecord
from bedrock.litreview.report.deck_model import FIGURES_ARTIFACT, FigureAsset
from bedrock.litreview.report.figures import choose_figures
from bedrock.litreview.runner import RunResult, answer_question
from bedrock.litreview.sources import figures as figure_files
from bedrock.litreview.sources.corpus import CorpusSearcher
from bedrock.litreview.sources.http import SourceHTTP
from bedrock.litreview.sources.literature import LiteratureSearcher, SearchFailed
from bedrock.litreview.stages.s0_question import QuestionError
from bedrock.litreview.stages.s1_collect import NoPapers, NothingFound

log = logging.getLogger(__name__)


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

    deck: bool = False
    """Also prepare what the target-proposal PowerPoint needs from outside the
    report: which paper figures it may show and, when `search` is also on,
    their image files and the article text needed to find them (see
    `_write_figure_manifest`). Off, a review does exactly what it did before
    the deck existed."""

    patents: list[PatentRecord] | None = None
    """Competitor-landscape input, from `bedrock-litreview patents` (EPO OPS).
    Never fetched here: unlike `search`, there is no `patents=True` flag that
    reaches the internet as part of answering a question - a patent search is
    always its own separate, explicit step. See `sources/patents_base.py`."""


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


REVIEW_TIMEOUT_FLOOR = 600.0
"""The shortest `request_timeout` a review should run with, in seconds.

bedrock's default is 120, which is right for a chat reply and not for a
4,096-token extraction on a laptop model: the review fails partway as "server
unavailable". `review_backend` and `standalone_scheduler` raise it to this for
anything that opens its own connection: the shell command, the HTTP API, the
benchmark, and agent-pack mode's review. `/review` in the default mode shares
the chat's connection, built at start-up, so there the number in the config
file is the number in use, and all a check can do is say so.
"""

FREE_DISK_WARNING_BYTES = 10 * 1024**3
"""Free space below which a review is worth warning about, on the disk holding
the review's data directory.

This is a judgement call, not a measurement, and it is not about this package's
own files: a run's reports and cache are megabytes. It is about the model
server. A 16 GB machine running an 8B model has no room to spare, and the
deploy audit's own laptop run (OPEN_ISSUES, "A 16 GB laptop is not a review
machine") had to be paused three times because macOS swapped until the disk
filled - the review's calls were what kept the model resident. 10 GB is roughly
one model's weights plus swap headroom. Worth replacing with a measured figure
once a review has been run to completion on the office server and on a 32 GB
laptop.
"""


@dataclass(frozen=True)
class Finding:
    """One thing worth saying about this machine before a review, that does not
    stop it. What *does* stop one is `preflight`, which raises `ConfigError`;
    these are the warnings that sit beside it, so a person fixing a machine sees
    both.

    `detail` is plain language and, when `ok` is False, says what to change and
    names a number or a command, because the reader is usually a scientist
    rather than whoever set the server up.
    """

    title: str
    ok: bool
    detail: str

    @property
    def mark(self) -> str:
        return "ok  " if self.ok else "warn"


def config_findings(settings: Settings, *, timeout_floor_applied: bool = False) -> list[Finding]:
    """Everything checkable from configuration and this disk. Never a network call.

    `timeout_floor_applied` says the review opens its own connection, which
    `review_backend` raises to `REVIEW_TIMEOUT_FLOOR`. Without it a front end
    would warn about a configured timeout it has already fixed for itself. The
    shell, the HTTP API, agent-pack mode and a default-mode review aimed at a
    different server all open their own; only a review that shares the chat's
    connection is exposed, which is the case worth warning about.
    """
    findings = [_timeout_finding(settings, timeout_floor_applied), _disk_finding(settings)]
    if (host_note := _allowed_hosts_finding(settings)) is not None:
        findings.append(host_note)
    return findings


def _timeout_finding(settings: Settings, floor_applied: bool) -> Finding:
    """`request_timeout`. A warning and never a refusal: a review that hits it
    stops between calls and resumes with `--resume`, so a short timeout costs
    time, not the run."""
    if floor_applied and settings.request_timeout < REVIEW_TIMEOUT_FLOOR:
        return Finding(
            title="request timeout",
            ok=True,
            detail=(
                f"raised to {REVIEW_TIMEOUT_FLOOR:.0f}s for this review, from the configured "
                f"{settings.request_timeout:.0f}s, because it opens its own connection. A review that "
                "shares the chat's connection cannot do that, so "
                f"`request_timeout = {REVIEW_TIMEOUT_FLOOR:.0f}` in ~/.config/bedrock/config.toml is still "
                "worth setting."
            ),
        )
    if settings.request_timeout >= REVIEW_TIMEOUT_FLOOR:
        return Finding(
            title="request timeout",
            ok=True,
            detail=f"{settings.request_timeout:.0f}s, enough for an extraction call",
        )
    return Finding(
        title="request timeout",
        ok=False,
        detail=(
            f"`request_timeout` is {settings.request_timeout:.0f}s. One extraction call writes up to 4,096 "
            f"tokens, which takes longer than that on a laptop model, and a review that shares the chat's "
            f"connection uses this number.\n"
            f"Set `request_timeout = {REVIEW_TIMEOUT_FLOOR:.0f}` in ~/.config/bedrock/config.toml and restart. "
            "A review that does time out stops between calls and resumes with `--resume`."
        ),
    )


def _disk_finding(settings: Settings) -> Finding:
    import shutil

    target = settings.data_dir
    # The directory may not exist before the first run; the disk it will live
    # on is what matters, so walk up to something that does.
    while not target.exists() and target != target.parent:
        target = target.parent
    try:
        free = shutil.disk_usage(target).free
    except OSError as exc:
        return Finding(title="free disk", ok=True, detail=f"could not be checked: {exc}")

    gb = free / 1024**3
    if free >= FREE_DISK_WARNING_BYTES:
        return Finding(title="free disk", ok=True, detail=f"{gb:.1f} GB free on {target}")
    return Finding(
        title="free disk",
        ok=False,
        detail=(
            f"only {gb:.1f} GB free on {target}. A review keeps the model loaded for hours, and on a 16 GB "
            f"machine that pushes the system into swap - a review on this hardware has filled a disk and had "
            f"to be paused. The reports themselves are small; the space is for the model server and swap. "
            f"Free up space, or run the review on the office server."
        ),
    )


def _allowed_hosts_finding(settings: Settings) -> Finding | None:
    """Why a company server name is refused until somebody says it is allowed.

    `assert_local_url` has already run by the time settings exist, so reaching
    here means the address passed. This is the other half: `allowed_hosts` is
    the one piece of review configuration nobody discovers by accident, so a
    name that passed *because* it is listed is worth showing, and a name that
    passed some other way is worth saying how to list it if a review ever
    refuses it.
    """
    from urllib.parse import urlparse

    host = (urlparse(settings.base_url).hostname or "").lower()
    if not host or host == "localhost" or host.startswith("127.") or host == "::1":
        return None
    if host in settings.allowed_hosts:
        return Finding(
            title="model server address",
            ok=True,
            detail=f"{host} is allowed by `allowed_hosts` in the [litreview] table",
        )
    return Finding(
        title="model server address",
        ok=True,
        detail=(
            f"{host} was accepted as a local address. If a review ever refuses it "
            f'("not a local address"), add it to the [litreview] table of '
            f'~/.config/bedrock/config.toml as `allowed_hosts = ["{host}"]`.'
        ),
    )


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
                   "server offers only the model it was started with, under whatever name it was started with "
                   "(`--served-model-name`): often a full path such as 'Qwen/Qwen3-30B-A3B-Instruct-2507', not a "
                   "short name.\nTo use the names this server really offers, without a new release, write them "
                   f"into {config_dir() / 'models.yaml'} - a whole replacement for the packaged "
                   "defaults/models.yaml, which the run then records as the file it read.")
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
    warn: Callable[[str], None] | None = None,
    timeout_floor_applied: bool = False,
    tick=None,
) -> RunResult:
    """Run one review and return its result.

    `warn` receives the non-blocking setup findings - a short `request_timeout`,
    a nearly full disk - before the run starts. Optional: skipping it loses a
    warning and nothing else. `timeout_floor_applied` is True when this review
    opens its own connection (see `config_findings`). `tick` is the per-item
    signal from inside the stages; see `litreview/progress.py`.

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
    about to contact a source outside this machine (the literature search, the
    retraction check, the deck's figure download), one `litreview.search`
    intent per kind - before any pipeline work; if any of those cannot be
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
        if warn is not None:
            for advisory in config_findings(settings, timeout_floor_applied=timeout_floor_applied):
                if not advisory.ok:
                    warn(advisory.detail)
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
            patents=request.patents,
            refresh_papers=request.refresh,
            approved_terms=request.approved_terms,
            out_dir=request.out_dir,
            resume_run_id=request.resume,
            progress=progress,
            tick=tick,
            integrity_check=integrity_check,
        )
        if request.deck:
            # The report is finished and saved; this only adds the deck's
            # figures beside it. It cannot fail the review (see the helper).
            await _prepare_deck_figures(request, settings, result, progress, cancel)
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
    if request.deck and request.search:
        # Figure images, and the article text re-read to find them, come from
        # Europe PMC (`www.ebi.ac.uk`) and only when the review was asked to
        # search: the deck toggle alone never reaches outside this machine.
        # See `_write_figure_manifest`.
        kinds.append("figure_download")
    return kinds


async def _prepare_deck_figures(
    request: ReviewRequest, settings: Settings, result: RunResult,
    progress: Callable[[str], None], cancel: threading.Event,
) -> None:
    """Choose, and when allowed fetch, the figures the run's deck may show.

    Runs once, right after the report is finished, for a fresh run and a
    resumed one alike - and never again at download, report viewing or
    `verify-run`, so nothing reaches the network later. The blocking work (the
    database, one download per paper) goes to a worker thread, like the
    pipeline itself.

    A failure here is logged and ignored: a deck without pictures is a normal
    deck, and the review's report is already saved and checked.
    """
    try:
        await asyncio.to_thread(_write_figure_manifest, request, settings, result, progress, cancel)
    except Exception:  # noqa: BLE001 - a missing picture never fails a review
        log.warning("the deck's figures could not be prepared for run %s", result.run.id, exc_info=True)
        progress("The PowerPoint's figures could not be prepared; the review itself is unaffected.")


def _write_figure_manifest(
    request: ReviewRequest, settings: Settings, result: RunResult,
    progress: Callable[[str], None], cancel: threading.Event,
) -> None:
    """Save the run's figure manifest (`FIGURES_ARTIFACT`).

    Picking a figure needs each cited paper's JATS, and ingestion discards the
    raw text once it has split a paper into sections, so a stored run's papers
    carry none. When the review was asked to search, `ArticleTexts` re-reads
    the articles the report cites from Europe PMC (cached on disk, a bounded
    number per deck) and the images are then downloaded through the same
    client. That text is used only to find the figure the paper cites beside a
    quote the review already verified: if the article has changed upstream the
    quote is not found and no figure is shown. Without a search nothing is
    requested, and `choose_figures` sees the papers as stored (normally no
    figure at all).

    The manifest is written even when nothing was downloaded - every figure
    then has `path=None`, which the deck reads as "not shown" - so a run made
    with the deck on always says what it chose. It lives in a run artifact,
    never in the report, so a stored report's bytes do not change.
    """
    papers = {paper.id: paper for paper in result.papers}
    http = None
    try:
        if request.search:
            progress("Fetching figures for the PowerPoint...")
            http = SourceHTTP(min_interval=0.34, timeout=60.0)  # the retraction check's pace: polite to Europe PMC
            figures = choose_figures(result.report, figure_files.ArticleTexts(
                papers, figures_dir(settings), http=http, cancel=cancel))
            if figures:
                figures = figure_files.download_figures(figures, figures_dir(settings), http=http, cancel=cancel)
        else:
            figures = choose_figures(result.report, papers)
    finally:
        if http is not None:
            http.close()
    db = Database(settings.db_path)
    try:
        db.put_artifact(result.run.id, FIGURES_ARTIFACT, json.dumps([f.to_dict() for f in figures], sort_keys=True))
    finally:
        db.close()


def figures_dir(settings: Settings) -> Path:
    """Where downloaded figure images are kept, one folder per paper. Private
    like the rest of the data directory: which papers a scientist looked at is
    not for other accounts on the machine."""
    return private_dir(Path(settings.data_dir) / "figures")


def deck_figures(settings: Settings, run_id: str, *, db: Database | None = None) -> list[FigureAsset]:
    """The figures a run's deck may show, read back from its manifest.

    Empty for a run made without the deck, one that predates it, or a manifest
    that cannot be read. A figure whose image file has since disappeared (the
    cache was cleared) comes back with `path=None`, so the renderer never
    meets a path that is not there.
    """
    own = db is None
    db = db or Database(settings.db_path)
    try:
        body = db.get_artifact(run_id, FIGURES_ARTIFACT)
    finally:
        if own:
            db.close()
    if not body:
        return []
    try:
        figures = [FigureAsset.from_dict(item) for item in json.loads(body)]
    except (ValueError, TypeError, KeyError):
        log.warning("the figure manifest of run %s could not be read", run_id)
        return []
    return [f if f.path is None or Path(f.path).is_file() else replace(f, path=None) for f in figures]


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
