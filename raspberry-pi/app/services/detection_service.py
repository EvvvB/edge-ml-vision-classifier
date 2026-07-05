from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, HTTPException, UploadFile

from app.config import settings
from app.inference.model import predict_image
from app.storage.filesystem import save_upload, update_metadata


async def receive_detection_upload(
    image: UploadFile,
    raw_metadata: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    if image.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported image type: {image.content_type}",
        )

    parsed_metadata = parse_metadata(raw_metadata)
    suffix = safe_image_suffix(image.filename)
    image_id = uuid4().hex
    image_path, metadata_path = await save_upload(
        image_id=image_id,
        image=image,
        suffix=suffix,
        metadata=parsed_metadata,
    )
    update_metadata(metadata_path, {"inference_status": "queued"})
    background_tasks.add_task(run_inference_job, image_path, metadata_path)

    return {
        "ok": True,
        "status": "accepted",
        "image_id": image_id,
        "filename": image.filename,
        "content_type": image.content_type,
        "saved_to": str(image_path),
        "metadata_saved_to": str(metadata_path),
        "inference_status": "queued",
    }


def run_inference_job(image_path: Path, metadata_path: Path) -> None:
    try:
        detections = predict_image(image_path)
    except Exception as exc:
        update_metadata(
            metadata_path,
            {
                "inference_status": "failed",
                "inference_error": str(exc),
            },
        )
        raise

    update_metadata(
        metadata_path,
        {
            "inference_status": "complete",
            "detections": detections,
        },
    )


def parse_metadata(raw_metadata: str) -> dict[str, Any]:
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="metadata must be valid JSON",
        ) from exc

    if not isinstance(metadata, dict):
        raise HTTPException(
            status_code=400,
            detail="metadata must be a JSON object",
        )

    return metadata


def safe_image_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_image_suffixes:
        return settings.default_image_suffix
    return suffix
