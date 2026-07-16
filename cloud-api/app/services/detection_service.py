from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import asyncpg
from botocore.exceptions import ClientError
from fastapi import HTTPException, Response, UploadFile

from app.config import settings
from app.storage.postgres import (
    fetch_detection,
    fetch_detections,
    insert_detection_upload,
    mark_detection_failed,
    mark_detection_stored,
)
from app.storage.s3 import download_image, upload_image

# Images are keyed by UUID and never rewritten, so clients may cache them
# forever without revalidating.
IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"


async def receive_detection_upload(
    db: asyncpg.Pool,
    s3_client: Any,
    image: UploadFile,
    raw_metadata: str,
) -> dict[str, Any]:
    if image.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported image type: {image.content_type}",
        )

    if not settings.s3_bucket:
        raise HTTPException(
            status_code=500,
            detail="S3 bucket is not configured. Set CLOUD_S3_BUCKET or S3_BUCKET.",
        )

    parsed_metadata = parse_metadata(raw_metadata)
    image_bytes = await read_image_bytes(image)
    image_id = uuid4()
    suffix = safe_image_suffix(image.filename)
    s3_key = build_s3_key(image_id=image_id, suffix=suffix)
    s3_url = build_s3_url(bucket=settings.s3_bucket, key=s3_key)
    device_id = optional_string(parsed_metadata.get("device_id"))
    captured_at = parse_optional_datetime(parsed_metadata.get("captured_at"))

    await insert_detection_upload(
        db,
        image_id=image_id,
        device_id=device_id,
        filename=image.filename,
        content_type=image.content_type or "application/octet-stream",
        file_size_bytes=len(image_bytes),
        metadata=parsed_metadata,
        s3_bucket=settings.s3_bucket,
        s3_key=s3_key,
        s3_url=s3_url,
        captured_at=captured_at,
    )

    try:
        upload_result = await upload_image(
            s3_client=s3_client,
            bucket=settings.s3_bucket,
            key=s3_key,
            body=image_bytes,
            content_type=image.content_type or "application/octet-stream",
            image_id=image_id,
            metadata=parsed_metadata,
        )
    except Exception as exc:
        await mark_detection_failed(db, image_id=image_id, error=str(exc))
        raise HTTPException(
            status_code=502,
            detail="image metadata was recorded, but upload to S3 failed",
        ) from exc

    await mark_detection_stored(
        db,
        image_id=image_id,
        s3_etag=upload_result.get("ETag"),
    )

    return {
        "ok": True,
        "status": "stored",
        "image_id": str(image_id),
        "filename": image.filename,
        "content_type": image.content_type,
        "file_size_bytes": len(image_bytes),
        "s3_bucket": settings.s3_bucket,
        "s3_key": s3_key,
        "s3_url": s3_url,
    }


async def get_detection(db: asyncpg.Pool, image_id: UUID) -> dict[str, Any]:
    detection = await fetch_detection(db, image_id=image_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")
    return {"ok": True, "detection": detection}


async def get_detection_image(
    db: asyncpg.Pool,
    s3_client: Any,
    image_id: UUID,
    if_none_match: str | None = None,
) -> Response:
    detection = await fetch_detection(db, image_id=image_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="detection not found")
    if detection["upload_status"] != "stored":
        raise HTTPException(
            status_code=404,
            detail=f"detection image is not stored (status: {detection['upload_status']})",
        )

    stored_etag = detection.get("s3_etag")
    if if_none_match and stored_etag and if_none_match == stored_etag:
        return Response(
            status_code=304,
            headers={
                "Cache-Control": IMMUTABLE_CACHE_CONTROL,
                "ETag": stored_etag,
            },
        )

    try:
        image = await download_image(
            s3_client=s3_client,
            bucket=detection["s3_bucket"],
            key=detection["s3_key"],
        )
    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code")
        if error_code in {"NoSuchKey", "404"}:
            raise HTTPException(
                status_code=404,
                detail="detection image is missing from storage",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail="failed to fetch detection image from storage",
        ) from exc

    headers = {"Cache-Control": IMMUTABLE_CACHE_CONTROL}
    etag = image.get("etag") or stored_etag
    if etag:
        headers["ETag"] = etag
    return Response(
        content=image["body"],
        media_type=image.get("content_type") or detection["content_type"],
        headers=headers,
    )


async def list_detections(
    db: asyncpg.Pool,
    device_id: str | None,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="offset must be greater than or equal to 0",
        )

    detections = await fetch_detections(
        db,
        device_id=device_id,
        limit=limit,
        offset=offset,
    )
    return {"ok": True, "detections": detections}


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


async def read_image_bytes(image: UploadFile) -> bytes:
    data = await image.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image is larger than {settings.max_image_bytes} bytes",
        )
    if not data:
        raise HTTPException(status_code=400, detail="image cannot be empty")
    return data


def safe_image_suffix(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in settings.allowed_image_suffixes:
        return settings.default_image_suffix
    return suffix


def build_s3_key(image_id: UUID, suffix: str) -> str:
    now = datetime.now(UTC)
    prefix = settings.normalized_s3_prefix
    date_path = f"{now:%Y/%m/%d}"
    filename = f"{image_id}{suffix}"
    if not prefix:
        return f"{date_path}/{filename}"
    return f"{prefix}/{date_path}/{filename}"


def build_s3_url(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def parse_optional_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="captured_at must be a string")

    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="captured_at must be an ISO 8601 datetime",
        ) from exc
