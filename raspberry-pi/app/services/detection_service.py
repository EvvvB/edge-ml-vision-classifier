from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy
from fastapi import BackgroundTasks, HTTPException, UploadFile
from PIL import Image

from app.config import settings
from app.inference.model import predict_image
from app.services.coco import normalize_fomo_detections, normalize_yolo_detections
from app.services.tile_dedupe import deduplicate_tile_detections
from app.storage.filesystem import save_upload, save_upload_bytes, update_metadata


async def receive_detection_upload(
    image: UploadFile,
    raw_metadata: str,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    parsed_metadata = deduplicate_tile_detections(parse_metadata(raw_metadata))
    device_metadata, fomo_detections = split_device_detections(parsed_metadata)
    image_id = uuid4().hex

    if image.content_type in settings.allowed_raw_image_types:
        image_path, metadata_path = await save_raw_upload(
            image_id=image_id,
            image=image,
            metadata=device_metadata,
        )
        saved_content_type = "image/jpeg"
    elif image.content_type in settings.allowed_image_types:
        suffix = safe_image_suffix(image.filename)
        image_path, metadata_path = await save_upload(
            image_id=image_id,
            image=image,
            suffix=suffix,
            metadata=device_metadata,
        )
        saved_content_type = image.content_type
    else:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported image type: {image.content_type}",
        )

    update_metadata(
        metadata_path,
        {
            "fomo_detections": fomo_detections,
            "yolo_detections": [],
            "inference_status": "queued",
        },
    )
    background_tasks.add_task(run_inference_job, image_path, metadata_path)

    return {
        "ok": True,
        "status": "accepted",
        "image_id": image_id,
        "filename": image.filename,
        "content_type": saved_content_type,
        "source_content_type": image.content_type,
        "saved_to": str(image_path),
        "metadata_saved_to": str(metadata_path),
        "inference_status": "queued",
    }


async def save_raw_upload(
    image_id: str,
    image: UploadFile,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    raw_bytes = await image.read()
    converted_bytes = convert_raw_image_to_jpeg(raw_bytes, metadata)

    metadata = {
        **metadata,
        "source_content_type": image.content_type,
        "stored_image_encoding": "jpeg",
        "stored_image_content_type": "image/jpeg",
        "stored_image_suffix": ".jpg",
    }

    return save_upload_bytes(
        image_id=image_id,
        filename=converted_filename(image.filename),
        content_type="image/jpeg",
        suffix=".jpg",
        image_bytes=converted_bytes,
        metadata=metadata,
    )


def convert_raw_image_to_jpeg(
    raw_bytes: bytes,
    metadata: dict[str, Any],
) -> bytes:
    frame_width = positive_metadata_int(metadata, "frame_width")
    frame_height = positive_metadata_int(metadata, "frame_height")
    image_encoding = metadata.get("image_encoding")
    expected_byte_count = expected_raw_byte_count(
        frame_width,
        frame_height,
        image_encoding,
    )

    if len(raw_bytes) != expected_byte_count:
        raise HTTPException(
            status_code=400,
            detail=(
                "raw image byte count mismatch: "
                f"expected {expected_byte_count}, got {len(raw_bytes)}"
            ),
        )

    if image_encoding == "grayscale":
        pil_image = Image.frombytes(
            "L",
            (frame_width, frame_height),
            raw_bytes,
        )
    elif image_encoding == "rgb565":
        pil_image = Image.frombytes(
            "RGB",
            (frame_width, frame_height),
            rgb565_to_rgb888(raw_bytes),
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported raw image encoding: {image_encoding}",
        )

    output = BytesIO()
    pil_image.save(output, format="JPEG", quality=95)
    return output.getvalue()


def positive_metadata_int(metadata: dict[str, Any], key: str) -> int:
    try:
        value = int(metadata[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail=f"metadata field must be a positive integer: {key}",
        ) from exc

    if value <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"metadata field must be a positive integer: {key}",
        )

    return value


def expected_raw_byte_count(
    frame_width: int,
    frame_height: int,
    image_encoding: Any,
) -> int:
    pixel_count = frame_width * frame_height

    if image_encoding == "grayscale":
        return pixel_count

    if image_encoding == "rgb565":
        return pixel_count * 2

    raise HTTPException(
        status_code=400,
        detail=f"unsupported raw image encoding: {image_encoding}",
    )


def rgb565_to_rgb888(raw_bytes: bytes) -> bytes:
    # OpenMV exposes RGB565 framebuffer bytes little-endian, so each pixel
    # reads as one little-endian uint16. Unpacking the whole frame at once
    # keeps a per-pixel Python loop off the event loop, which matters on the
    # Pi's much slower cores.
    packed = numpy.frombuffer(raw_bytes, dtype="<u2")

    # 255 // 31 and 255 // 63 rescale each channel to a full byte. The
    # intermediate products peak at 16065, so uint16 does not overflow.
    red = (packed >> 11) & 0x1F
    green = (packed >> 5) & 0x3F
    blue = packed & 0x1F

    rgb = numpy.empty((packed.size, 3), dtype=numpy.uint8)
    rgb[:, 0] = (red * 255) // 31
    rgb[:, 1] = (green * 255) // 63
    rgb[:, 2] = (blue * 255) // 31

    return rgb.tobytes()


def converted_filename(filename: str | None) -> str:
    stem = Path(filename or "nicla-frame").stem or "nicla-frame"
    return f"{stem}.jpg"


def split_device_detections(
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Separate the device's own detections from the rest of its metadata.

    Detections are stored top level alongside the YOLO results so both models
    share one schema and one nesting depth. What remains under "metadata" is
    then purely what the device reported about itself and the frame.
    """
    device_metadata = {
        key: value for key, value in metadata.items() if key != "detections"
    }
    return device_metadata, normalize_fomo_detections(metadata.get("detections"))


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
            "yolo_detections": normalize_yolo_detections(detections),
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
