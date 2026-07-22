from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Request

from app.api.detections import require_api_key
from app.services.receipt_service import list_receipts, receive_receipt_batch

router = APIRouter()


@router.post("/receipts/batch", dependencies=[Depends(require_api_key)])
async def receive_receipts(
    request: Request,
    payload: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    return await receive_receipt_batch(request.app.state.db, payload)


@router.get("/receipts", dependencies=[Depends(require_api_key)])
async def read_receipts(
    request: Request,
    device_id: str | None = None,
    event: str | None = None,
    limit: int = 100,
) -> dict[str, Any]:
    return await list_receipts(
        request.app.state.db,
        device_id=device_id,
        event=event,
        limit=limit,
    )
