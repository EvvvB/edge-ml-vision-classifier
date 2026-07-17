from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from starlette.status import HTTP_202_ACCEPTED

from app.services.detection_service import receive_detection_upload


router = APIRouter()


@router.post("/detections", status_code=HTTP_202_ACCEPTED)
async def receive_detection(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    metadata: str = Form(...),
) -> dict[str, Any]:
    return await receive_detection_upload(
        image=image,
        raw_metadata=metadata,
        background_tasks=background_tasks,
        client_host=request.client.host if request.client else None,
    )


@router.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
