"""Fetch the actual content behind a result_ref.

Agents pass references (result_ref) to each other internally rather than
full content — see docs/ARCHITECTURE.md section 6. A client needs a way to
resolve a ref from a `task.completed` event into the text to actually show
the user; this is that one small, additive endpoint.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.api import ResultResponse
from app.task_store.store import get_task_store

router = APIRouter(tags=["results"])


@router.get("/results/{result_ref}", response_model=ResultResponse)
async def get_result(result_ref: str) -> ResultResponse:
    content = get_task_store().get_result(result_ref)
    if content is None:
        raise HTTPException(status_code=404, detail="result not found")
    return ResultResponse(result_ref=result_ref, content=content)
