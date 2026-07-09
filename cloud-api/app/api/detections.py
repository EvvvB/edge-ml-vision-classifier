from __future__ import annotations

from typing import Any
from uuid import UUID

from fastapi import APIRouter, File, Form, Request, UploadFile
from starlette.status import HTTP_201_CREATED

from app.services.detection_service import (
    get_detection,
    list_detections,
    receive_detection_upload,
)
from app.config import settings
from app.storage.postgres import check_db
from app.storage.s3 import check_s3


router = APIRouter()


@router.post("/detections", status_code=HTTP_201_CREATED)
async def receive_detection(
    request: Request,
    image: UploadFile = File(...),
    metadata: str = Form(...),
) -> dict[str, Any]:
    return await receive_detection_upload(
        db=request.app.state.db,
        s3_client=request.app.state.s3_client,
        image=image,
        raw_metadata=metadata,
    )


@router.get("/detections/{image_id}")
async def read_detection(request: Request, image_id: UUID) -> dict[str, Any]:
    return await get_detection(request.app.state.db, image_id=image_id)


@router.get("/detections")
async def read_detections(
    request: Request,
    device_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    return await list_detections(
        request.app.state.db,
        device_id=device_id,
        limit=limit,
        offset=offset,
    )


@router.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}


@router.get("/ready")
async def ready(request: Request) -> dict[str, bool]:
    await check_db(request.app.state.db)
    await check_s3(request.app.state.s3_client, settings.s3_bucket)
    return {"ok": True}
