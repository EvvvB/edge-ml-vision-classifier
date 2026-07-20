from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.storage.filesystem import update_metadata


logger = logging.getLogger(__name__)


class CloudForwardError(Exception):
    pass


def forward_detection(image_path: Path, metadata_path: Path) -> None:
    """Send a stored image and its metadata to the cloud API.

    Runs inside the inference background task, so it never raises: the
    outcome is recorded in the local metadata file either way, which is what
    a later resync would scan for.
    """
    if not settings.cloud_api_url:
        return

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    try:
        response_body = post_detection(image_path, saved_metadata)
    except Exception as exc:
        logger.warning("cloud forward failed for %s: %s", image_path.name, exc)
        update_metadata(
            metadata_path,
            {
                "cloud_sync_status": "failed",
                "cloud_sync_error": str(exc),
            },
        )
        return

    update_metadata(
        metadata_path,
        {
            "cloud_sync_status": "synced",
            "cloud_sync_error": None,
            "cloud_image_id": response_body.get("image_id"),
        },
    )


def post_detection(
    image_path: Path,
    saved_metadata: dict[str, Any],
) -> dict[str, Any]:
    url = settings.cloud_api_url.rstrip("/") + "/detections"
    headers = {}
    if settings.cloud_api_key:
        headers["X-API-Key"] = settings.cloud_api_key

    files = {
        "image": (
            saved_metadata.get("filename") or image_path.name,
            image_path.read_bytes(),
            saved_metadata.get("content_type") or "image/jpeg",
        ),
    }
    data = {"metadata": json.dumps(build_cloud_metadata(saved_metadata))}

    delay = settings.cloud_forward_retry_seconds
    attempts = max(1, settings.cloud_forward_attempts)

    for attempt in range(1, attempts + 1):
        error: Exception
        try:
            response = httpx.post(
                url,
                files=files,
                data=data,
                headers=headers,
                timeout=settings.cloud_forward_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            error = exc
        else:
            if response.status_code < 400:
                return response.json()
            if response.status_code < 500:
                # The cloud API rejected the payload; retrying cannot help.
                raise CloudForwardError(
                    f"cloud API rejected upload: {response.status_code} {response.text}"
                )
            error = CloudForwardError(
                f"cloud API server error: {response.status_code}"
            )

        if attempt == attempts:
            raise error
        time.sleep(delay)
        delay *= 2

    raise CloudForwardError("unreachable")


def build_cloud_metadata(saved_metadata: dict[str, Any]) -> dict[str, Any]:
    """Flatten the Pi's stored metadata into the cloud API's shape.

    The cloud API reads device_id and captured_at from the top level of the
    metadata object, so the device's own fields are hoisted up and the Pi's
    results sit alongside them.
    """
    device_metadata = saved_metadata.get("metadata") or {}
    cloud_metadata = {
        **device_metadata,
        "pi_image_id": saved_metadata.get("image_id"),
        "inference_status": saved_metadata.get("inference_status"),
        "fomo_detections": saved_metadata.get("fomo_detections") or [],
        "yolo_detections": saved_metadata.get("yolo_detections") or [],
    }

    # The Nicla stamps its own model identity (model_hash/model_manifest)
    # inside device_metadata; these are the Pi model's counterparts. Absent
    # on records whose inference failed, so those stay unstamped.
    for key in ("yolo_model_hash", "yolo_model_manifest"):
        if saved_metadata.get(key) is not None:
            cloud_metadata[key] = saved_metadata[key]

    received_at = saved_metadata.get("received_at")
    if received_at and not cloud_metadata.get("captured_at"):
        # The Nicla has no clock, so the Pi's receive time is the closest
        # thing to a capture timestamp.
        cloud_metadata["captured_at"] = received_at

    return cloud_metadata
