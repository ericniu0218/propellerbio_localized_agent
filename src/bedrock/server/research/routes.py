"""The API the workspace's Research tab calls.

The tab is a front end to the literature review, ``bedrock.litreview`` - the
same pipeline as ``/review`` in the terminal app and ``bedrock-litreview ask``
in a shell. A scientist types a question; the review searches Europe PMC,
preprints and PubMed, keeps the papers about the exact subject, reads them,
extracts findings whose quotes are checked against the paper, and writes a
report in which every sentence cites them.

Everything goes through ``service.run_review``, as ``service.py`` asks of every
front end, so the browser gets the same pipeline, the same checks and the same
reports as the terminal. Reports land in the review's own data directory
(``[litreview] data_dir``), so on the same machine ``/reviews`` in the terminal
lists them too.

    GET    /workspace/api/research/status            can this server run a review?
    POST   /workspace/api/research/runs              start one
    GET    /workspace/api/research/runs              this person's reviews
    GET    /workspace/api/research/runs/{id}         one review, with its progress
    POST   /workspace/api/research/runs/{id}/cancel  stop it (finished stages are kept)
    POST   /workspace/api/research/runs/{id}/resume  carry on from the last finished stage
    GET    /workspace/api/research/runs/{id}/report  ?format=html|md|json
    POST   /workspace/api/research/runs/{id}/save    report -> a workspace file (project / chat context)
    DELETE /workspace/api/research/runs/{id}         remove it from this person's list

Who owns a review is recorded in the workspace store (kind ``research``), so
each person sees their own. The review's database keeps every run regardless:
it is the audit trail, and removing a review from the list does not touch it.

A review is long - hundreds of model calls - and runs in the background. At
most ``MAX_RUNNING`` run at once on a server, one per person: they share one
model server, and a queue of reviews would starve everyone's chat.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field

from bedrock.server.config import get_settings
from bedrock.server.workspace.routes import identity, require_object, store
from bedrock.server.workspace.store import uid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workspace/api/research", tags=["research"])

KIND = "research"
MAX_RUNNING = 2
MAX_EVENTS = 200
SOURCE_NAMES = {"europepmc": "Europe PMC", "biorxiv": "Preprints (bioRxiv / medRxiv)", "pubmed": "PubMed"}
STAT_KEYS = ("papers_collected", "gate_pass", "papers_read", "evidence_units", "findings_proposed",
             "sentences_kept", "sentences_dropped")


# --------------------------------------------------------------------------- the review's side
# Small seams, so the tests can hand in their own settings and model server.

def review_settings():
    """The literature review's settings: bedrock's config.toml and its [litreview] table."""
    from bedrock.litreview.config import load_settings

    return load_settings()


_stack: AsyncExitStack | None = None
_scheduler = None


async def review_scheduler():
    """One scheduler for every review this server runs, opened on first use.

    Reviews are not agents, so they do not use the model hub's mock backend in
    a demo: they always go to the model server bedrock is configured for, with
    the long timeout ``standalone_scheduler`` sets for them. One scheduler means
    one queue, so two reviews take turns rather than both hitting the GPU.
    """
    global _stack, _scheduler
    if _scheduler is None:
        from bedrock.config import load_settings as load_bedrock_settings
        from bedrock.litreview.service import standalone_scheduler

        stack = AsyncExitStack()
        scheduler = await stack.enter_async_context(standalone_scheduler(load_bedrock_settings()))
        if _scheduler is None:
            _stack, _scheduler = stack, scheduler
        else:  # another request opened one while this one waited
            await stack.aclose()
    return _scheduler


async def aclose() -> None:
    """Stop running reviews and close the model connection (server shutdown)."""
    global _stack, _scheduler
    for job in list(JOBS.values()):
        job.cancel.set()
        if job.task is not None:
            job.task.cancel()
    if _stack is not None:
        stack, _stack, _scheduler = _stack, None, None
        await stack.aclose()


def _airgapped() -> bool:
    s = get_settings()
    return s.mode == "real" and s.airgap_check_enabled


# --------------------------------------------------------------------------- live jobs

@dataclass
class Job:
    id: str
    owner: str
    cancel: threading.Event = field(default_factory=threading.Event)
    events: list[dict] = field(default_factory=list)
    task: asyncio.Task | None = None

    def say(self, message: str) -> None:
        # Called from the review's worker thread; must never raise.
        try:
            self.events.append({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                "message": str(message)[:300]})
            del self.events[:-MAX_EVENTS]
        except Exception:  # noqa: BLE001
            pass


JOBS: dict[str, Job] = {}


class _WarningsToJob(logging.Handler):
    """The pipeline's warnings (a source that failed, a paper skipped) belong in
    the progress log a scientist is watching, not only the server log."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.WARNING)
        self.job = job

    def emit(self, record: logging.LogRecord) -> None:
        self.job.say("warning: " + record.getMessage())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _save(owner: str, review: dict) -> dict:
    return store().put(owner, KIND, review)


def _get(owner: str, review_id: str) -> dict:
    return require_object(store(), owner, review_id, KIND)


def _public(review: dict, *, events: bool = False) -> dict:
    out = dict(review)
    job = JOBS.get(review["id"])
    if review.get("status") in ("queued", "running") and job is None:
        # The server restarted mid-review. Every finished stage is saved.
        out["status"] = review["status"] = "stopped"
        review["error"] = "The server restarted while this review was running."
        review["finishedAt"] = review.get("finishedAt") or _now()
        _save(review["ownerId"], review)
        out = dict(review)
    if job is not None:
        out["lastEvent"] = job.events[-1]["message"] if job.events else "Starting…"
        if events:
            out["events"] = job.events[-120:]
    elif events:
        out["events"] = []
    out["canResume"] = bool(out.get("status") in ("stopped", "failed", "cancelled") and out.get("reviewRunId"))
    return out


async def _run(job: Job, review: dict, request) -> None:
    """Run one review in the background and record how it ended."""
    from bedrock.litreview.config import ConfigError
    from bedrock.litreview.llm import ReviewInterrupted
    from bedrock.litreview.service import run_review
    from bedrock.litreview.sources.literature import SearchFailed
    from bedrock.litreview.stages.s0_question import QuestionError
    from bedrock.litreview.stages.s1_collect import NoPapers, NothingFound

    owner = review["ownerId"]
    handler = _WarningsToJob(job)
    log = logging.getLogger("bedrock.litreview")
    log.addHandler(handler)
    settings = None
    started_before: set[str] = set()
    try:
        settings = await asyncio.to_thread(review_settings)
        scheduler = await review_scheduler() if request.use_model else None
        review["status"] = "running"
        _save(owner, review)
        started_before = await asyncio.to_thread(_run_ids, settings)
        job.say("Reading the question…")
        result = await run_review(request, settings=settings, scheduler=scheduler, progress=job.say, cancel=job.cancel)
        report = result.report
        review.update(
            status="completed",
            reviewRunId=result.run.id,
            reportId=result.report_id,
            questionId=result.spec.id,
            stats={k: int(report.stats.get(k, 0)) for k in STAT_KEYS},
            references=len(report.references),
            integrity=list(report.integrity[:5]),
            integrityPassed=not report.integrity,
            error=None,
        )
        job.say("Report written.")
    except asyncio.CancelledError:
        review.update(status="cancelled", error="Stopped.")
        raise
    except ReviewInterrupted as exc:
        review.update(status="cancelled" if job.cancel.is_set() else "stopped",
                      error="Stopped." if job.cancel.is_set() else f"The review stopped partway: {exc}")
    except NothingFound as exc:
        review.update(status="failed", error=str(exc)[:600])
    except NoPapers:
        review.update(status="failed", error="No papers were found for this question. Turn on "
                                             "“Search the literature” to collect them.")
    except (ConfigError, QuestionError, SearchFailed, OSError, ValueError) as exc:
        review.update(status="failed", error=str(exc)[:600])
    except Exception as exc:  # noqa: BLE001 - a background job must record how it ended
        logger.exception("Research review %s failed", review["id"])
        review.update(status="failed", error=f"The review could not finish ({type(exc).__name__}).")
    finally:
        log.removeHandler(handler)
        if review["status"] != "completed" and settings is not None and not review.get("reviewRunId"):
            # So "Resume" can pick up from the last finished stage.
            try:
                review["reviewRunId"] = await asyncio.to_thread(_new_unfinished_run, settings, started_before)
            except Exception:  # noqa: BLE001
                pass
        review["finishedAt"] = _now()
        _save(owner, review)
        store().audit(owner, "research " + review["status"], review["id"])
        JOBS.pop(job.id, None)


def _run_ids(settings) -> set[str]:
    from bedrock.litreview.db import Database

    if not settings.db_path.exists():
        return set()
    db = Database(settings.db_path)
    try:
        return {r.id for r in db.recent_runs(10_000)}
    finally:
        db.close()


def _new_unfinished_run(settings, before: set[str]) -> str | None:
    from bedrock.litreview.db import Database

    if not settings.db_path.exists():
        return None
    db = Database(settings.db_path)
    try:
        for run in db.recent_runs(50):
            if run.id not in before and run.finished is None:
                return run.id
    finally:
        db.close()
    return None


def _start(owner: str, review: dict, request) -> dict:
    running = [j for j in JOBS.values()]
    if any(j.owner == owner for j in running):
        raise HTTPException(429, "You already have a review running. Wait for it to finish, or stop it.")
    if len(running) >= MAX_RUNNING:
        raise HTTPException(429, f"{MAX_RUNNING} reviews are already running on this server. Try again when one finishes.")
    review = _save(owner, review)
    job = Job(id=review["id"], owner=owner)
    JOBS[job.id] = job
    job.task = asyncio.get_running_loop().create_task(_run(job, review, request))
    return _public(review, events=True)


# --------------------------------------------------------------------------- routes

class StartReview(BaseModel):
    question: str = Field(min_length=5, max_length=600)
    search: bool = True
    useModel: bool = True
    fullText: bool = True
    sources: list[str] | None = None
    projectId: str | None = None


class SaveReport(BaseModel):
    projectId: str | None = None
    chatId: str | None = None


_status_cache: dict = {"at": 0.0, "body": None}


@router.get("/status")
async def status(user=Depends(identity)):
    """What this server can do right now, so the page can say so before a run."""
    if _status_cache["body"] is not None and time.monotonic() - _status_cache["at"] < 20:
        return _status_cache["body"]
    from bedrock.litreview.config import ConfigError
    from bedrock.litreview.service import preflight
    from bedrock.litreview.sources.literature import SOURCES

    body = {"available": True, "reason": "", "searchAvailable": not _airgapped(),
            "searchReason": "This server is air-gapped, so it cannot search the literature. Reviews can "
                            "only re-read questions whose papers are already here." if _airgapped() else "",
            "sources": [{"name": s, "label": SOURCE_NAMES.get(s, s)} for s in SOURCES],
            "model": {"ready": False, "reason": "", "baseUrl": "", "modelSet": ""}}
    try:
        settings = await asyncio.to_thread(review_settings)
    except Exception as exc:  # noqa: BLE001 - a broken config is the answer here
        body.update(available=False, reason=f"The literature review is not set up on this server: {exc}")
        return body
    body["model"].update(baseUrl=settings.base_url, modelSet=settings.model_set)
    body["integrityChecks"] = settings.integrity_checks
    try:
        body["model"]["models"] = sorted({settings.model_for(j) for j in settings.jobs})
        scheduler = await review_scheduler()
        await asyncio.wait_for(preflight(settings, scheduler.backend), timeout=8)
        body["model"]["ready"] = True
    except (ConfigError, asyncio.TimeoutError) as exc:
        body["model"]["reason"] = str(exc) or "The model server did not answer in time."
    except Exception as exc:  # noqa: BLE001
        body["model"]["reason"] = f"The model server could not be checked: {exc}"
    _status_cache.update(at=time.monotonic(), body=body)
    return body


@router.post("/runs")
async def start(body: StartReview, user=Depends(identity)):
    from bedrock.litreview.service import ReviewRequest
    from bedrock.litreview.sources.literature import SOURCES

    question = " ".join(body.question.split())
    if body.search and _airgapped():
        raise HTTPException(409, "This server is air-gapped and cannot search the literature. Turn off "
                                 "“Search the literature” to re-read papers already collected for this question.")
    sources = tuple(s for s in (body.sources or SOURCES) if s in SOURCES)
    if body.search and not sources:
        raise HTTPException(422, "Choose at least one place to search.")
    if body.projectId:
        require_object(store(), user["id"], body.projectId, "projects")
    request = ReviewRequest(question=question, search=body.search, use_model=body.useModel,
                            full_text=body.fullText, sources=sources if body.search else None)
    review = {"id": uid(), "question": question, "status": "queued", "search": body.search,
              "useModel": body.useModel, "fullText": body.fullText, "sources": list(sources),
              "projectId": body.projectId, "startedAt": _now(), "ownerName": user["name"]}
    return _start(user["id"], review, request)


@router.get("/runs")
def list_reviews(user=Depends(identity)):
    reviews = [_public(r) for r in store().list(user["id"], KIND)]
    return sorted(reviews, key=lambda r: r.get("startedAt") or "", reverse=True)


@router.get("/runs/{review_id}")
def get_review(review_id: str, user=Depends(identity)):
    return _public(_get(user["id"], review_id), events=True)


@router.post("/runs/{review_id}/cancel")
def cancel(review_id: str, user=Depends(identity)):
    _get(user["id"], review_id)
    job = JOBS.get(review_id)
    if job is None:
        raise HTTPException(404, "This review is not running.")
    job.cancel.set()
    job.say("Stopping after the current step…")
    return {"ok": True}


@router.post("/runs/{review_id}/resume")
async def resume(review_id: str, user=Depends(identity)):
    from bedrock.litreview.service import ReviewRequest

    review = _get(user["id"], review_id)
    if review_id in JOBS:
        raise HTTPException(409, "This review is already running.")
    if not review.get("reviewRunId") or review.get("status") == "completed":
        raise HTTPException(409, "There is nothing to resume for this review; start it again instead.")
    request = ReviewRequest(question=review["question"], resume=review["reviewRunId"],
                            use_model=review.get("useModel", True))
    review.update(status="queued", error=None, finishedAt=None)
    return _start(user["id"], review, request)


def _report(review: dict):
    from bedrock.litreview.db import Database
    from bedrock.litreview.models import Report

    if review.get("status") != "completed" or not review.get("reportId"):
        raise HTTPException(404, "This review has no report yet.")
    settings = review_settings()
    db = Database(settings.db_path)
    try:
        body = db.get_report(review["reportId"])
    finally:
        db.close()
    if body is None:
        raise HTTPException(404, "The report is no longer in the review database on this server.")
    return Report.model_validate_json(body)


@router.get("/runs/{review_id}/report")
async def report(review_id: str, format: str = "html", download: bool = False, user=Depends(identity)):
    from bedrock.litreview.stages.s8_render import to_html, to_json, to_markdown

    review = _get(user["id"], review_id)
    rep = await asyncio.to_thread(_report, review)
    name = f"review-{review['reportId']}"
    headers = {"Content-Disposition": f'attachment; filename="{name}.{format}"'} if download else {}
    if format == "html":
        html = to_html(rep)
        if not download:
            # Shown in a frame on the Research page: links open in a new tab.
            html = html.replace("<head>", '<head><base target="_blank">', 1)
        return HTMLResponse(html, headers=headers)
    if format == "md":
        return PlainTextResponse(to_markdown(rep), media_type="text/markdown", headers=headers)
    if format == "json":
        return Response(to_json(rep), media_type="application/json", headers=headers)
    raise HTTPException(400, "format is html, md or json.")


@router.post("/runs/{review_id}/save")
async def save_report(review_id: str, body: SaveReport, user=Depends(identity)):
    """Put the report into the workspace as a file, so chats can use it.

    With a project it is a project file: every chat in that project gets it as
    context, the same as an uploaded file. With a chat it is attached to that
    chat. With neither it goes to the person's own library (Memory), which
    every chat of theirs can read.
    """
    from bedrock.litreview.stages.s8_render import to_markdown

    s = store()
    review = _get(user["id"], review_id)
    chat = require_object(s, user["id"], body.chatId, "chats") if body.chatId else None
    project_id = body.projectId or (chat or {}).get("projectId")
    if project_id:
        require_object(s, user["id"], project_id, "projects")
    rep = await asyncio.to_thread(_report, review)
    text = to_markdown(rep)
    raw = text.encode("utf-8")
    artifact_id = uid()
    s.blobs.joinpath(artifact_id).write_bytes(raw)
    title = " ".join(review["question"].split()[:10])
    stats = review.get("stats") or {}
    artifact = s.put(user["id"], "artifacts", {
        "id": artifact_id, "name": f"Literature review - {title}.md"[:200], "size": len(raw), "type": ".md",
        "text": text[:300000], "truncated": len(text) > 300000,
        "profile": f"Literature review · {stats.get('evidence_units', 0)} verified findings from "
                   f"{review.get('references', 0)} papers",
        "parseStatus": "parsed", "scanStatus": "no obvious matches",
        "projectId": project_id, "chatId": chat["id"] if chat else None,
        "scope": "Project" if project_id else "Personal", "researchId": review["id"]})
    if chat:
        chat.setdefault("artifactIds", []).append(artifact_id)
        s.put(user["id"], "chats", chat)
    review.setdefault("savedArtifactIds", []).append(artifact_id)
    _save(user["id"], review)
    s.audit(user["id"], "saved research report", artifact_id)
    return {k: v for k, v in artifact.items() if k != "text"}


@router.delete("/runs/{review_id}")
def delete_review(review_id: str, user=Depends(identity)):
    _get(user["id"], review_id)
    if review_id in JOBS:
        raise HTTPException(409, "Stop this review before removing it.")
    store().delete(user["id"], review_id)
    store().audit(user["id"], "removed research from list", review_id)
    return {"ok": True}
