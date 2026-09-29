"""The one way into a review that uses a model.

`/review` in the terminal app, the `bedrock-litreview` command and the HTTP API
all come through here, so all three get the same pipeline, the same scheduler
and the same guarantees. Adding a fourth front end should mean calling
`run_review`, never reaching past it into `runner.answer_question`.

Two jobs, and they are both about the seam rather than the science:

**Checking the machine can do this at all**, before a run starts, with the fix
written out. A review is a long job - hundreds of calls - and the ways it goes
wrong on a fresh machine are silent at call time: a context window too small to
hold an extraction prompt makes the server quietly drop the start of the prompt,
so the report comes out thin rather than failing. None of it is worth
discovering forty minutes in.

There is one set of checks, and two ways to ask for it, because a run and a
person want different things from the same knowledge:

* `preflight` is what a run calls. It raises `ConfigError` on the first thing
  that stops a review, and returns the advisories, which are for the front end
  to show.
* `findings` is what `bedrock-litreview doctor` and `/reviews check` call. It
  returns everything - what is fine, what is merely worth knowing, and what
  blocks - so a person can fix all of it in one pass instead of one error per
  attempt.

Both are the same code, so a check cannot exist in one and not the other.

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
from bedrock.litreview.config import ConfigError, Settings, config_dir, private_dir
from bedrock.litreview.llm import make_client
from bedrock.litreview.models import Paper, PatentRecord
from bedrock.litreview.runner import RunResult, answer_question
from bedrock.litreview.sources.corpus import CorpusSearcher
from bedrock.litreview.sources.literature import LiteratureSearcher


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

    patents: list[PatentRecord] | None = None
    """Competitor-landscape input, from `bedrock-litreview patents` (EPO OPS).
    Never fetched here: unlike `search`, there is no `patents=True` flag that
    reaches the internet as part of answering a question — a patent search is
    always its own separate, explicit step. See `sources/patents_base.py`."""


REVIEW_TIMEOUT_FLOOR = 600.0
"""The shortest `request_timeout` a review should run with, in seconds.

bedrock's default is 120, which is right for a chat reply and not for a
4,096-token extraction on a laptop model: the review fails partway as "server
unavailable". `standalone_scheduler` raises it to this for the shell command,
the HTTP API and the benchmark, which build their own backend. `/review` in the
app cannot - it shares the chat's backend, built at start-up - so there the
number in the config file is the number in use, and all the check can do is say
so.
"""

FREE_DISK_WARNING_BYTES = 10 * 1024**3
"""Free space below which a review is worth warning about, on the disk holding
the review's data directory.

This is a judgement call, not a measurement, and it is not about this package's
own files: a run's reports and cache are megabytes. It is about the model
server. A 16 GB machine running an 8B model has no room to spare, and the
deploy audit's own laptop run (OPEN_ISSUES item 21) had to be paused three times
because macOS swapped until the disk filled - the review's calls were what kept
the model resident. 10 GB is roughly one model's weights plus swap headroom.
Worth replacing with a measured figure once a review has been run to completion
on the office server and on a 32 GB laptop.
"""


@dataclass(frozen=True)
class Finding:
    """One thing the machine's setup says about running a review.

    `blocking` is the distinction that matters: it is the difference between
    "this review will not work" and "this review will work and you should know
    something". Only a blocking finding may stop a run.
    """

    title: str
    """A few words, for a list: "context window", "models", "free disk"."""
    ok: bool
    detail: str
    """Plain language. When `ok` is False this says what to change, and names a
    number or a command, because the person reading it is usually a scientist
    rather than whoever set the server up."""
    blocking: bool = False

    @property
    def mark(self) -> str:
        return "ok  " if self.ok else ("FAIL" if self.blocking else "warn")


def config_findings(settings: Settings, *, timeout_floor_applied: bool = False) -> list[Finding]:
    """Everything checkable from configuration and this disk. Never a network call.

    Separate from `server_findings` because these hold whether or not a model
    server exists yet, which is the normal state of a machine on the day
    somebody installs this.

    `timeout_floor_applied` says the caller built its own backend through
    `standalone_scheduler`, which raises the timeout to `REVIEW_TIMEOUT_FLOOR`
    for that process. The shell command, the HTTP API and the benchmark all do;
    `/review` in the app cannot, because it shares the chat's backend. Without
    this, the shell command would warn about a timeout it had already fixed for
    itself - and `doctor` deliberately leaves it False, because what it reports
    is how the machine is configured for every front end, not what this one
    process happens to have done.
    """
    findings: list[Finding] = []

    if settings.context_window < settings.min_context_window:
        findings.append(Finding(
            title="context window",
            ok=False,
            blocking=True,
            detail=(
                f"a literature review needs a context window of at least "
                f"{settings.min_context_window:,} tokens, and this machine is set to "
                f"{settings.context_window:,}. Extraction sends about 4,300 tokens of prompt "
                f"and allows a 4,096-token reply, which {settings.context_window:,} cannot hold.\n"
                f"Set `context_window = {settings.min_context_window}` in ~/.config/bedrock/config.toml "
                f"(or BEDROCK_CONTEXT_WINDOW={settings.min_context_window}), and start Ollama with "
                f"OLLAMA_CONTEXT_LENGTH={settings.min_context_window} — that fits on a 16 GB machine. "
                f"Otherwise use the `workstation` profile, or the office server."
            ),
        ))
    else:
        findings.append(Finding(
            title="context window",
            ok=True,
            detail=f"{settings.context_window:,} tokens ({settings.profile_name} profile); "
                   f"a review needs {settings.min_context_window:,}",
        ))

    findings.append(_timeout_finding(settings, timeout_floor_applied))
    findings.append(_disk_finding(settings))
    if (host_note := _allowed_hosts_finding(settings)) is not None:
        findings.append(host_note)
    return findings


def _timeout_finding(settings: Settings, floor_applied: bool) -> Finding:
    """`request_timeout`, which only `/review` in the app is really exposed to.

    A warning rather than a refusal, and never blocking: a review that hits it
    stops between calls and resumes with `--resume`, so a short timeout costs
    time, not the run.
    """
    if floor_applied and settings.request_timeout < REVIEW_TIMEOUT_FLOOR:
        return Finding(
            title="request timeout",
            ok=True,
            detail=(
                f"raised to {REVIEW_TIMEOUT_FLOOR:.0f}s for this process, from the configured "
                f"{settings.request_timeout:.0f}s. `/review` in the app cannot do that - it shares "
                f"the chat's connection - so `request_timeout = {REVIEW_TIMEOUT_FLOOR:.0f}` in "
                "~/.config/bedrock/config.toml is still worth setting for it."
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
            f"`request_timeout` is {settings.request_timeout:.0f}s. One extraction call writes up "
            f"to 4,096 tokens, which takes longer than that on a laptop model, and `/review` in "
            f"the app uses this number — it shares the chat's connection.\n"
            f"Set `request_timeout = {REVIEW_TIMEOUT_FLOOR:.0f}` in ~/.config/bedrock/config.toml "
            f"and restart. (`bedrock-litreview` and the HTTP API already raise it to "
            f"{REVIEW_TIMEOUT_FLOOR:.0f} for themselves, so this only affects `/review`.) "
            "A run that does time out stops between calls and resumes with `--resume`."
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
            f"only {gb:.1f} GB free on {target}. A review keeps the model loaded for hours, and "
            f"on a 16 GB machine that pushes the system into swap — a review on this hardware has "
            f"filled a disk and had to be paused. The reports themselves are small; the space is "
            f"for the model server and swap. Free up space, or run the review on the office server."
        ),
    )


def _allowed_hosts_finding(settings: Settings) -> Finding | None:
    """Why a company server name is refused until somebody says it is allowed.

    `assert_local_url` has already run by the time settings exist, so reaching
    here means the address passed. This is the other half: a name that passed
    *because* it is in `allowed_hosts` is worth showing, because that list is
    the one piece of review configuration nobody discovers by accident, and the
    first-run check has been known to suggest exactly such a host.
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


async def server_findings(settings: Settings, backend) -> list[Finding]:
    """Reachability and model names. Needs a backend, so it is the half a
    machine with the server switched off cannot answer."""
    try:
        available = {info.id for info in await backend.list_models()}
    except InferenceError as exc:
        return [Finding(
            title="model server",
            ok=False,
            blocking=True,
            detail=(
                f"the model server at {settings.base_url} is not answering: {exc}\n"
                "Start it, or point `api_base` at the office server."
            ),
        )]

    findings = [Finding(
        title="model server",
        ok=True,
        detail=f"{settings.base_url} answered, offering {len(available)} model(s)",
    )]

    wanted = sorted({settings.model_for(job) for job in settings.jobs})
    missing = [name for name in wanted if name not in available]
    if missing:
        pulls = "\n".join(f"  ollama pull {name}" for name in missing)
        findings.append(Finding(
            title="models",
            ok=False,
            blocking=True,
            detail=(
                f"the model server at {settings.base_url} does not have "
                f"{', '.join(repr(m) for m in missing)}, which the `{settings.model_set}` model set needs.\n"
                f"On a laptop running Ollama:\n{pulls}\n"
                "On the office server (vLLM), ask IT to serve it — a vLLM server offers "
                "only the model it was started with. A vLLM server's model name is whatever it "
                "was started with (`--served-model-name`), which is often the full path such as "
                "'Qwen/Qwen3-30B-A3B-Instruct-2507' and not a short name.\n"
                f"To use the names this server really offers, without a new release, write them "
                f"into {config_dir()/'models.yaml'} — a whole replacement for the packaged "
                f"defaults/models.yaml, which the run then records as the file it read. "
                f"This server offers: {', '.join(sorted(available)) or '(none)'}"
            ),
        ))
    else:
        findings.append(Finding(
            title="models",
            ok=True,
            detail=f"the `{settings.model_set}` set needs {', '.join(wanted)}; all present",
        ))
    return findings


async def findings(settings: Settings, backend=None, *, timeout_floor_applied: bool = False) -> list[Finding]:
    """Everything that can be said about this machine's readiness, in order.

    `backend` may be None, for a check on a machine with no server running: the
    configuration half is still worth reporting, and is where most first-run
    problems are.
    """
    out = config_findings(settings, timeout_floor_applied=timeout_floor_applied)
    if backend is not None:
        out.extend(await server_findings(settings, backend))
    return out


async def preflight(settings: Settings, backend, *, timeout_floor_applied: bool = False) -> list[Finding]:
    """Refuse a run the machine cannot do, and return what is merely worth saying.

    Raises `ConfigError` on the first blocking finding, with the fix spelled
    out. The non-blocking findings come back for the front end to show; a
    caller that ignores them loses a warning, never a guarantee.
    """
    found = await findings(settings, backend, timeout_floor_applied=timeout_floor_applied)
    for finding in found:
        if finding.blocking:
            raise ConfigError(finding.detail)
    return [finding for finding in found if not finding.ok]


async def run_review(
    request: ReviewRequest,
    *,
    settings: Settings,
    scheduler: Scheduler | None,
    progress: Callable[[str], None],
    cancel: threading.Event,
    warn: Callable[[str], None] | None = None,
    timeout_floor_applied: bool = False,
) -> RunResult:
    """Run one review and return its result.

    `scheduler` may be None only when `request.use_model` is False. A run
    without a model never reaches a server, so the shell commands that do that
    do not open one.

    `warn` receives preflight's non-blocking findings - a short `request_timeout`,
    a nearly full disk - before the run starts. It is called from this thread,
    not the worker's, and is optional: skipping it loses a warning and nothing
    else. A front end that has somewhere better to put them can call
    `config_findings` itself instead, which is what the app does so the notice
    appears before the run rather than after it.

    `timeout_floor_applied` is True for a caller that opened its own backend
    through `standalone_scheduler`, so it is not warned about a timeout that
    context manager has already raised for it. See `config_findings`.

    `progress` is called from the worker thread, so a caller that touches a UI
    has to hand the message back to its own thread. It must never raise: a
    progress line that throws would take the run down with it.

    `ConfigError`, `QuestionError`, `NoPapers`, `SearchFailed`,
    `ReviewInterrupted`, `OSError` and `ValueError` all reach the caller, which
    is the piece that knows how to show them.
    """
    private_dir(settings.data_dir)

    client = None
    if request.use_model:
        if scheduler is None:
            raise ValueError("a review that uses a model needs bedrock's scheduler")
        advisories = await preflight(
            settings, scheduler.backend, timeout_floor_applied=timeout_floor_applied
        )
        for advisory in advisories:
            if warn is not None:
                warn(advisory.detail)
        client = make_client(
            settings,
            scheduler=scheduler,
            loop=asyncio.get_running_loop(),
            cancel=cancel,
        )

    searcher = _searcher(request, settings)

    return await asyncio.to_thread(
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
    )


def _searcher(request: ReviewRequest, settings: Settings):
    """Where this run's papers come from.

    `None` means the question's frozen set, already in the database. The
    literature searcher is built only when someone asked for it: a review must
    never reach the internet by default.
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


@asynccontextmanager
async def standalone_scheduler(bedrock_settings, *, backend=None, timeout_floor: float = 600.0):
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
    if own:
        timeout = max(float(bedrock_settings.request_timeout), timeout_floor)
        backend = create_backend(replace(bedrock_settings, request_timeout=timeout))
    try:
        yield Scheduler(backend, bedrock_settings.profile)
    finally:
        if own:
            await backend.close()
