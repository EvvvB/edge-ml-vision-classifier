from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from app.config import settings


async def save_upload(
    image_id: str,
    image: UploadFile,
    suffix: str,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.metadata_dir.mkdir(parents=True, exist_ok=True)

    image_path = settings.upload_dir / f"{image_id}{suffix}"
    metadata_path = settings.metadata_dir / f"{image_id}.json"

    with image_path.open("wb") as output_file:
        while chunk := await image.read(1024 * 1024):
            output_file.write(chunk)

    saved_metadata = {
        "image_id": image_id,
        "filename": image.filename,
        "content_type": image.content_type,
        "image_path": str(image_path),
        "metadata": metadata,
    }
    metadata_path.write_text(
        json.dumps(saved_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    return image_path, metadata_path
