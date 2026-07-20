from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request

from app.api.detections import require_api_key
from app.services.eval_service import (
    get_eval_summary,
    list_eval_disagreements,
    run_backfill,
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
