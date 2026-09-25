"""The front door for the front end.

Two ways in:

- Python: `ask(question, papers=...)` returns the run, with the report as a
  model and rendered files on disk.
- HTTP (optional, `pip install -e ".[api]"`): `create_app()` returns a small
  FastAPI app — POST /ask, GET /reports/{id}, POST /reports/{id}/ratings.
  Serve it on the office machine only: `uvicorn litreview.api:app`.

The searcher plugs in through `set_searcher`, or its papers can be posted with
the question, or `"search": true` on POST /ask runs the literature searcher
(`sources/literature.py`) for that question — the only way the HTTP app goes
to the internet, and only when asked.

Ratings are stored now so that, once enough exist, they can be compared with
the automatic benchmark (`litreview.bench`) — the benchmark is the check that
works before any scientist has rated anything.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from bedrock.litreview.config import Settings, load_settings
from bedrock.litreview.db import Database
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
    """Register the paper searcher the HTTP app and `ask` use by default."""
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
    db = db or Database((settings or load_settings(check_endpoint=False)).db_path)
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


def create_app(settings: Settings | None = None):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse, JSONResponse
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError('the HTTP API needs: pip install -e ".[api]"') from exc

    from bedrock.litreview.models import Report
    from bedrock.litreview.sources.literature import LiteratureSearcher, SearchFailed
    from bedrock.litreview.stages.s0_question import QuestionError
    from bedrock.litreview.stages.s1_collect import NoPapers
    from bedrock.litreview.stages.s8_render import to_html

    settings = settings or load_settings()
    app = FastAPI(title="litreview", version="0.2")

    @app.post("/ask")
    def post_ask(body: AskBody) -> dict:
        searcher = LiteratureSearcher.from_settings(settings) if body.search else None
        try:
            result = ask(
                body.question, papers=body.papers, searcher=searcher, settings=settings,
                refresh_papers=body.refresh_papers, approved_terms=body.approved_terms,
            )
        except QuestionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except NoPapers as exc:
            raise HTTPException(
                status_code=422, detail="no papers for this question: post papers, or set search=true",
            ) from exc
        except SearchFailed as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
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
