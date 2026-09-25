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
from bedrock.litreview.llm import make_client
from bedrock.litreview.models import Paper
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


async def preflight(settings: Settings, backend) -> None:
    """Check the machine can run a review at all, and say what to change if not.

    Raises `ConfigError` with the fix spelled out. Every message names a number
    or a command, because the person reading it is usually a scientist rather
    than the person who set the server up.
    """
    if settings.context_window < settings.min_context_window:
        raise ConfigError(
            f"a literature review needs a context window of at least "
            f"{settings.min_context_window:,} tokens, and this machine is set to "
            f"{settings.context_window:,}. Extraction sends about 4,300 tokens of prompt "
            f"and allows a 4,096-token reply, which {settings.context_window:,} cannot hold.\n"
            f"Set `context_window = {settings.min_context_window}` in ~/.config/bedrock/config.toml "
            f"(or BEDROCK_CONTEXT_WINDOW={settings.min_context_window}), and start Ollama with "
            f"OLLAMA_CONTEXT_LENGTH={settings.min_context_window} — that fits on a 16 GB machine. "
            f"Otherwise use the `workstation` profile, or the office server."
        )

    try:
        available = {info.id for info in await backend.list_models()}
    except InferenceError as exc:
        raise ConfigError(
            f"the model server at {settings.base_url} is not answering: {exc}\n"
            "Start it, or point `api_base` at the office server."
        ) from exc

    wanted = sorted({settings.model_for(job) for job in settings.jobs})
    missing = [name for name in wanted if name not in available]
    if missing:
        pulls = "\n".join(f"  ollama pull {name}" for name in missing)
        raise ConfigError(
            f"the model server at {settings.base_url} does not have "
            f"{', '.join(repr(m) for m in missing)}, which the `{settings.model_set}` model set needs.\n"
            f"On a laptop running Ollama:\n{pulls}\n"
            "On the office server (vLLM), ask IT to serve it — a vLLM server offers "
            "only the model it was started with."
        )


async def run_review(
    request: ReviewRequest,
    *,
    settings: Settings,
    scheduler: Scheduler,
    progress: Callable[[str], None],
    cancel: threading.Event,
) -> RunResult:
    """Run one review and return its result.

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
        await preflight(settings, scheduler.backend)
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
