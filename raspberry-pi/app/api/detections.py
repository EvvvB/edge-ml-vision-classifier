from __future__ import annotations

from typing import Any

from fastapi import APIRouter, File, Form, UploadFile

from app.services.detection_service import receive_detection_upload


router = APIRouter()


@router.post("/detections")
async def receive_detection(
    image: UploadFile = File(...),
    metadata: str = Form(...),
) -> dict[str, Any]:
    return await receive_detection_upload(image=image, raw_metadata=metadata)


@router.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
