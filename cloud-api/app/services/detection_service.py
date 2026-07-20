from __future__ import annotations

import io
import json
import re
import zipfile
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
    fetch_detection_facets,
    fetch_detections,
    insert_detection_upload,
    mark_detection_failed,
    mark_detection_stored,
    touch_device_upload,
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

    if device_id:
        # Registry bookkeeping must never fail an upload; the detection is
        # already stored either way.
        manifest = parsed_metadata.get("model_manifest")
        try:
            await touch_device_upload(
                db,
                device_id=device_id,
                model_hash=optional_string(parsed_metadata.get("model_hash")),
                model_manifest=manifest if isinstance(manifest, dict) else None,
            )
        except Exception:
            pass

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


VALID_DETECTION_FILTERS = frozenset({"any", "some", "none"})
VALID_SOURCE_FILTERS = frozenset({"any", "fomo", "yolo"})
MAX_LABEL_FILTERS = 25
MAX_MODEL_FILTERS = 25

# Devices stamp truncated SHA-256 hex (12 chars today); accept any plausible
# hex length so the filter survives a future change to the truncation.
MODEL_HASH_PATTERN = re.compile(r"^[0-9a-f]{4,64}$")


def parse_label_filters(labels: str | None) -> list[str]:
    if not labels:
        return []
    parsed = sorted({label.strip().lower() for label in labels.split(",") if label.strip()})
    if len(parsed) > MAX_LABEL_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_LABEL_FILTERS} labels may be filtered at once",
        )
    return parsed


def parse_model_filters(models: str | None) -> list[str]:
    if not models:
        return []
    parsed = sorted({model.strip().lower() for model in models.split(",") if model.strip()})
    if len(parsed) > MAX_MODEL_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_MODEL_FILTERS} models may be filtered at once",
        )
    for model_hash in parsed:
        if not MODEL_HASH_PATTERN.fullmatch(model_hash):
            raise HTTPException(
                status_code=400,
                detail="models must be hex model hashes",
            )
    return parsed


def validate_filter_value(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be one of: {', '.join(sorted(allowed))}",
        )
    return value


async def list_detections(
    db: asyncpg.Pool,
    device_id: str | None,
    labels: str | None,
    models: str | None,
    detections: str,
    source: str,
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
    validate_filter_value("detections", detections, VALID_DETECTION_FILTERS)
    validate_filter_value("source", source, VALID_SOURCE_FILTERS)

    rows, total = await fetch_detections(
        db,
        device_id=device_id,
        labels=parse_label_filters(labels),
        models=parse_model_filters(models),
        detections=detections,
        source=source,
        limit=limit,
        offset=offset,
    )
    return {"ok": True, "detections": rows, "total": total}


async def get_detection_facets(
    db: asyncpg.Pool,
    device_id: str | None,
    source: str,
) -> dict[str, Any]:
    validate_filter_value("source", source, VALID_SOURCE_FILTERS)
    facets = await fetch_detection_facets(db, device_id=device_id, source=source)
    return {"ok": True, **facets}


EXPORT_MAX_IMAGES = 1000


def build_coco_dataset(
    rows: list[dict[str, Any]],
    detections_key: str,
    description: str,
) -> dict[str, Any]:
    """Assemble a COCO object-detection dataset from stored detection rows.

    Images without annotations stay in images[] — that is how COCO
    represents negative samples.
    """
    categories: dict[str, int] = {}
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    annotation_id = 1

    for image_number, row in enumerate(rows, start=1):
        metadata = row.get("metadata") or {}
        images.append(
            {
                "id": image_number,
                "file_name": Path(row["s3_key"]).name,
                "width": metadata.get("frame_width") or 0,
                "height": metadata.get("frame_height") or 0,
            }
        )

        entries = metadata.get(detections_key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            bbox = entry.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            label = str(entry.get("label") or "object")
            category_id = categories.setdefault(label, len(categories) + 1)
            annotation: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image_number,
                "category_id": category_id,
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0,
            }
            if isinstance(entry.get("confidence"), (int, float)):
                annotation["score"] = entry["confidence"]
            annotations.append(annotation)
            annotation_id += 1

    return {
        "info": {"description": description},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": category_id, "name": name}
            for name, category_id in categories.items()
        ],
    }


async def export_detections(
    db: asyncpg.Pool,
    s3_client: Any,
    device_id: str | None,
    labels: str | None,
    models: str | None,
    detections: str,
    source: str,
) -> Response:
    validate_filter_value("detections", detections, VALID_DETECTION_FILTERS)
    validate_filter_value("source", source, VALID_SOURCE_FILTERS)

    rows, total = await fetch_detections(
        db,
        device_id=device_id,
        labels=parse_label_filters(labels),
        models=parse_model_filters(models),
        detections=detections,
        source=source,
        limit=EXPORT_MAX_IMAGES + 1,
        offset=0,
    )
    if total > EXPORT_MAX_IMAGES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"export matches {total} detections; narrow the filters to at "
                f"most {EXPORT_MAX_IMAGES}"
            ),
        )

    stored = [row for row in rows if row["upload_status"] == "stored"]
    exported: list[dict[str, Any]] = []
    missing_images = 0

    buffer = io.BytesIO()
    # JPEGs don't recompress; ZIP_STORED keeps CPU cost near zero.
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for row in stored:
            try:
                image = await download_image(
                    s3_client=s3_client,
                    bucket=row["s3_bucket"],
                    key=row["s3_key"],
                )
            except ClientError:
                missing_images += 1
                continue
            archive.writestr(f"images/{Path(row['s3_key']).name}", image["body"])
            exported.append(row)

        generated_at = datetime.now(UTC).isoformat()
        for source_name in ("fomo", "yolo"):
            dataset = build_coco_dataset(
                exported,
                f"{source_name}_detections",
                f"{source_name.upper()} detections export ({generated_at})",
            )
            archive.writestr(
                f"{source_name}.coco.json",
                json.dumps(dataset, indent=2),
            )

        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "generated_at": generated_at,
                    "query": {
                        "device_id": device_id,
                        "labels": parse_label_filters(labels),
                        "models": parse_model_filters(models),
                        "detections": detections,
                        "source": source,
                    },
                    "matched": total,
                    "exported_images": len(exported),
                    "skipped_not_stored": len(rows) - len(stored),
                    "skipped_missing_from_s3": missing_images,
                },
                indent=2,
            ),
        )

    filename = f"detections-export-{datetime.now(UTC):%Y-%m-%d}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
