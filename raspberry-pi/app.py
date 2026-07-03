from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile


app = FastAPI(title="Raspberry Pi Animal Detector Receiver")

UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
METADATA_DIR = Path(__file__).resolve().parent / "metadata"
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


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


@app.post("/detections")
async def receive_detection(
    image: UploadFile = File(...),
    metadata: str = Form(...),
) -> dict[str, Any]:
    if image.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported image type: {image.content_type}",
        )

    parsed_metadata = parse_metadata(metadata)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"

    image_id = uuid4().hex
    saved_path = UPLOAD_DIR / f"{image_id}{suffix}"
    metadata_path = METADATA_DIR / f"{image_id}.json"

    with saved_path.open("wb") as output_file:
        while chunk := await image.read(1024 * 1024):
            output_file.write(chunk)

    saved_metadata = {
        "image_id": image_id,
        "filename": image.filename,
        "content_type": image.content_type,
        "image_path": str(saved_path),
        "metadata": parsed_metadata,
    }
    metadata_path.write_text(
        json.dumps(saved_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return {
        "ok": True,
        "image_id": image_id,
        "filename": image.filename,
        "content_type": image.content_type,
        "saved_to": str(saved_path),
        "metadata_saved_to": str(metadata_path),
        "metadata": parsed_metadata,
    }


@app.get("/health")
async def health() -> dict[str, bool]:
    return {"ok": True}
