from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Body, Depends, Request

from app.api.detections import require_api_key
from app.services.eval_service import (
    complete_teacher_run,
    get_eval_summary,
    list_eval_disagreements,
    list_teacher_pending,
    list_teacher_runs,
    receive_teacher_batch,
    run_backfill,
    start_teacher_run,
)

router = APIRouter()


@router.get("/eval/summary", dependencies=[Depends(require_api_key)])
async def read_eval_summary(request: Request) -> dict[str, Any]:
    return await get_eval_summary(request.app.state.db)


@router.get("/eval/disagreements", dependencies=[Depends(require_api_key)])
async def read_eval_disagreements(
    request: Request,
    limit: int = 50,
) -> dict[str, Any]:
    return await list_eval_disagreements(request.app.state.db, limit=limit)


# New uploads are scored at ingest; the backfill covers history and any
# uploads whose ingest-time scoring failed. rescore=true re-walks every
# detection to apply changed matching rules or thresholds.
@router.post("/eval/backfill", dependencies=[Depends(require_api_key)])
async def backfill_eval(
    request: Request,
    rescore: bool = False,
    max_images: int = 5000,
) -> dict[str, Any]:
    return await run_backfill(
        request.app.state.db,
        rescore=rescore,
        max_images=max_images,
    )


# --- Offline teacher runner API -------------------------------------------


@router.get("/eval/teacher/pending", dependencies=[Depends(require_api_key)])
async def read_teacher_pending(
    request: Request,
    teacher_source: str,
    limit: int = 200,
) -> dict[str, Any]:
    return await list_teacher_pending(
        request.app.state.db,
        teacher_source=teacher_source,
        limit=limit,
    )


@router.post("/eval/teacher/batch", dependencies=[Depends(require_api_key)])
async def receive_teacher_annotations(
    request: Request,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    return await receive_teacher_batch(request.app.state.db, payload)


@router.get("/eval/teacher/runs", dependencies=[Depends(require_api_key)])
async def read_teacher_runs(
    request: Request,
    limit: int = 5,
) -> dict[str, Any]:
    return await list_teacher_runs(request.app.state.db, limit=limit)


@router.post("/eval/teacher/runs", dependencies=[Depends(require_api_key)])
async def create_teacher_run(
    request: Request,
    payload: dict[str, Any] = Body(default={}),
) -> dict[str, Any]:
    return await start_teacher_run(request.app.state.db, payload)


@router.patch(
    "/eval/teacher/runs/{run_id}", dependencies=[Depends(require_api_key)]
)
async def finish_teacher_run_route(
    request: Request,
    run_id: UUID,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    return await complete_teacher_run(request.app.state.db, run_id, payload)
