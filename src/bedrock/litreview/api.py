"""The front door for the front end.

Two ways in:

- Python: `ask(question, papers=...)` returns the run, with the report as a
  model and rendered files on disk.
- HTTP (optional, `pip install -e ".[api]"`): `create_app()` returns a small
  FastAPI app — POST /ask, GET /reports/{id}, POST /reports/{id}/ratings.
  Serve it on the office machine only:
  `uvicorn bedrock.litreview.api:create_app --factory --host 127.0.0.1`.

  This is kept on purpose. bedrock is terminal-first today and is going to
  become an app, and this is the front door that app will knock on. The "not a
  website" rule in the root CLAUDE.md governs the *employee interface*, which is
  the terminal; this API is not that interface, and it is optional. What matters
  is that it is not a second pipeline: a POST /ask goes through the same
  `service.run_review`, the same scheduler and the same guarantees as `/review`
  in the app and `bedrock-litreview ask` in a shell.

  It has no authentication yet. Keep it on localhost or the office network.

Papers can be posted with the question, or `"search": true` on POST /ask runs
the literature searcher (`sources/literature.py`) for that question — the only
way the HTTP app goes to the internet, and only when asked. `set_searcher`
registers a default searcher for the Python `ask()` entry point; the HTTP app
builds its own through `service.run_review`, so that one request's `search`
flag is the whole of what decides.

Ratings are stored now so that, once enough exist, they can be compared with
the automatic benchmark (`litreview.bench`) — the benchmark is the check that
works before any scientist has rated anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bedrock.litreview.config import Settings, load_settings
from bedrock.litreview.db import Database, DeletedRun
from bedrock.litreview.models import Paper, Rating
from bedrock.litreview.runner import RunResult, answer_question
from bedrock.litreview.sources.base import PaperSearcher

_searcher: PaperSearcher | None = None


# Request bodies live at module level: with postponed annotations, FastAPI
# resolves a parameter's type from the module, and a class defined inside
# create_app() is invisible to it — the body would be read as a query string.
class AskBody(BaseModel):
    question: str
    papers: list[Paper] | None = None
    search: bool = False
    """Search the literature for this question (needs network)."""
    refresh_papers: bool = False
    approved_terms: dict[str, list[str]] | None = None


class RateBody(BaseModel):
    score: int
    rater: str
    comments: str | None = None
    flagged_sentences: list[str] = []


def set_searcher(searcher: PaperSearcher) -> None:
    """Register the paper searcher the Python `ask()` uses by default."""
    global _searcher
    _searcher = searcher


def ask(
    question: str,
    *,
    papers: list[Paper] | None = None,
    searcher: PaperSearcher | None = None,
    settings: Settings | None = None,
    refresh_papers: bool = False,
    approved_terms: dict[str, list[str]] | None = None,
    **kwargs: Any,
) -> RunResult:
    """Run the pipeline in this thread, on a client the caller supplies.

    A model-using review needs `client=` (there is no longer a client this can
    build for itself: the only route to a model is bedrock's scheduler, which
    lives on an event loop). Use `service.run_review` instead unless you are
    passing a fake, or running with `use_model=False`.
    """
    return answer_question(
        question,
        papers=papers,
        searcher=searcher or _searcher,
        settings=settings,
        refresh_papers=refresh_papers,
        approved_terms=approved_terms,
        **kwargs,
    )


def rate_report(
    report_id: str,
    score: int,
    *,
    rater: str,
    comments: str | None = None,
    flagged_sentences: list[str] | None = None,
    db: Database | None = None,
    settings: Settings | None = None,
) -> Rating:
    # Recording a rating needs the database, not a model set.
    db = db or Database((settings or load_settings(check_endpoint=False, require_models=False)).db_path)
    if db.get_report(report_id) is None:
        raise KeyError(f"no report {report_id}")
    rating = Rating(
        report_id=report_id,
        rater=rater,
        score=score,
        comments=comments,
        flagged_sentences=flagged_sentences or [],
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    db.add_rating(rating)
    return rating


REPORT_FILES = ("report.json", "report.md", "report.html", "run.json")
"""What `s8_render.render` and the runner write into a report's directory."""


@dataclass(frozen=True)
class Forgotten:
    """What deleting one run removed, from the database and from disk."""

    deleted: DeletedRun
    files_removed: tuple[Path, ...]
    report_dir: Path | None
    """Where the rendered files were looked for, or None if the run wrote none."""


def forget_run(
    run_id: str,
    *,
    db: Database | None = None,
    settings: Settings | None = None,
) -> Forgotten | None:
    """Delete one run's own output. Returns None when there is no such run.

    This is the review's half of `/forget`, and it is deliberately narrower: it
    removes the run, the working notes of its stages and its rendered report,
    and it leaves the question and the papers alone. The papers are shared —
    `paper_versions` is append-only, and it is what lets every *other* run be
    re-proved against the exact text it read — so a tidy-up after one run must
    not touch them.

    Not reversible. There is no trash: on an air-gapped machine a second copy of
    a confidential report is a liability rather than a safety net.

    One honest limit: a run's rendered files are found at the default reports
    location. A report written with `--out` elsewhere is not recorded anywhere
    (`Run` has no such field), so the database entry goes and that copy of the
    files has to be deleted by hand. The result says which files were removed,
    so a caller can tell the difference rather than promising too much.
    """
    settings = settings or load_settings(check_endpoint=False, require_models=False)
    db = db or Database(settings.db_path)
    deleted = db.delete_run(run_id)
    if deleted is None:
        return None

    report_dir: Path | None = None
    removed: list[Path] = []
    # Only when this run's report row really went. Two identical runs share one
    # report and one directory, so deleting the files with the first would take
    # away the second run's report while leaving that run looking finished.
    if deleted.report_deleted and deleted.report_id:
        report_dir = Path(settings.out_dir) / deleted.question_id / deleted.report_id
        removed = _remove_report_files(report_dir)
    return Forgotten(deleted=deleted, files_removed=tuple(removed), report_dir=report_dir)


def _remove_report_files(directory: Path) -> list[Path]:
    """Delete the files a run rendered, then the directory if it is left empty.

    Named files rather than `shutil.rmtree`: this path is built from a question
    id and a report id that came out of the database, and a recursive delete
    driven by stored values is not worth writing at all. Anything a person put in
    that directory by hand therefore survives, and the directory survives with
    it.
    """
    if not directory.is_dir():
        return []
    removed = []
    for name in REPORT_FILES:
        path = directory / name
        if path.is_file():
            path.unlink()
            removed.append(path)
    for empty in (directory, directory.parent):
        try:
            empty.rmdir()  # refuses a directory that still holds anything
        except OSError:
            break
    return removed


def create_app(settings: Settings | None = None, *, bedrock_settings=None, backend=None):
    """The FastAPI app, with one backend and one `Scheduler` per server process.

    They are opened by the app's lifespan and closed on shutdown, so every
    request shares one queue to the model server rather than each opening its
    own - which is the same reason the terminal app has exactly one.
    `backend` is for tests.
    """
    import threading
    from contextlib import asynccontextmanager

    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('the HTTP API needs: pip install -e ".[api]"') from exc

    from bedrock.config import load_settings as load_bedrock_settings
    from bedrock.litreview.config import ConfigError
    from bedrock.litreview.llm import ReviewInterrupted
    from bedrock.litreview.models import Report
    from bedrock.litreview.service import ReviewRequest, run_review, standalone_scheduler
    from bedrock.litreview.sources.literature import SearchFailed
    from bedrock.litreview.stages.s0_question import QuestionError
    from bedrock.litreview.stages.s1_collect import NoPapers
    from bedrock.litreview.stages.s8_render import to_html

    settings = settings or load_settings()
    bedrock_settings = bedrock_settings or load_bedrock_settings()

    @asynccontextmanager
    async def lifespan(app):
        async with standalone_scheduler(bedrock_settings, backend=backend) as scheduler:
            app.state.scheduler = scheduler
            yield

    app = FastAPI(title="litreview", version="0.2", lifespan=lifespan)

    @app.post("/ask")
    async def post_ask(body: AskBody) -> dict:
        request = ReviewRequest(
            question=body.question,
            papers=body.papers,
            search=body.search,
            refresh=body.refresh_papers,
            approved_terms=body.approved_terms,
        )
        try:
            result = await run_review(
                request,
                settings=settings,
                scheduler=app.state.scheduler,
                progress=lambda msg: None,
                cancel=threading.Event(),
                # This process's scheduler came from `standalone_scheduler`,
                # which already raised the timeout for it.
                timeout_floor_applied=True,
            )
        except QuestionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NoPapers as exc:
            raise HTTPException(
                status_code=422, detail="no papers for this question: post papers, or set search=true",
            ) from exc
        except SearchFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except (ConfigError, ReviewInterrupted) as exc:
            # The machine cannot run a review, or the run stopped partway. Both
            # are "try again once something changes", not "your question is bad".
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"report_id": result.report_id, "report": result.report.model_dump(mode="json")}

    @app.get("/reports/{report_id}")
    def get_report(report_id: str, format: str = "json"):
        body = Database(settings.db_path).get_report(report_id)
        if body is None:
            raise HTTPException(status_code=404, detail="no such report")
        if format == "html":
            return HTMLResponse(to_html(Report.model_validate_json(body)))
        return JSONResponse(content=Report.model_validate_json(body).model_dump(mode="json"))

    @app.post("/reports/{report_id}/ratings")
    def post_rating(report_id: str, body: RateBody) -> dict:
        try:
            rating = rate_report(
                report_id, body.score, rater=body.rater, comments=body.comments,
                flagged_sentences=body.flagged_sentences, settings=settings,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return rating.model_dump()

    return app
