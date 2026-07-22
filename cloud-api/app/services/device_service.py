from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime
from typing import Any

import asyncpg
from fastapi import HTTPException, Response

from app.config import settings
from app.storage.postgres import (
    delete_device,
    expire_stale_positioning,
    fetch_devices,
    record_reported_config,
    record_reported_mode,
    set_desired_config,
    set_desired_mode,
    touch_device_seen,
    upsert_device_hello,
)

VALID_MODES = frozenset({"automated", "positioning"})

# Remote camera config: bounds mirror the firmware's own clamps, so a value
# accepted here is a value the device will actually apply.
CONFIG_CROP_SIZES = frozenset({96, 192})
CONFIG_MIN_SWEEP_INTERVAL_MS = 5_000
CONFIG_MAX_SWEEP_INTERVAL_MS = 24 * 60 * 60 * 1000
CONFIG_MIN_DIFF_THRESHOLD = 5
CONFIG_MAX_DIFF_THRESHOLD = 128


def validate_mode(mode: Any) -> str:
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"mode must be one of: {', '.join(sorted(VALID_MODES))}",
        )
    return mode


def validate_config(payload: Any) -> dict[str, Any]:
    """Check a desired-config payload; only known keys, all in range.

    Returns the validated subset. Partial payloads are fine — the storage
    layer merges keys into the existing desired config.
    """
    if not isinstance(payload, dict) or not payload:
        raise HTTPException(
            status_code=400,
            detail="config must be a non-empty JSON object",
        )

    config: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "full_sweep_interval_ms":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not (
                    CONFIG_MIN_SWEEP_INTERVAL_MS
                    <= value
                    <= CONFIG_MAX_SWEEP_INTERVAL_MS
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "full_sweep_interval_ms must be an integer between "
                        f"{CONFIG_MIN_SWEEP_INTERVAL_MS} and "
                        f"{CONFIG_MAX_SWEEP_INTERVAL_MS}"
                    ),
                )
            config[key] = value
        elif key == "crop_size":
            if value not in CONFIG_CROP_SIZES:
                raise HTTPException(
                    status_code=400,
                    detail="crop_size must be one of: 96, 192",
                )
            config[key] = value
        elif key == "motion_diff_threshold":
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not (
                    CONFIG_MIN_DIFF_THRESHOLD
                    <= value
                    <= CONFIG_MAX_DIFF_THRESHOLD
                )
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "motion_diff_threshold must be an integer between "
                        f"{CONFIG_MIN_DIFF_THRESHOLD} and "
                        f"{CONFIG_MAX_DIFF_THRESHOLD}"
                    ),
                )
            config[key] = value
        elif key == "min_confidence":
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not (0.0 <= value <= 1.0)
            ):
                raise HTTPException(
                    status_code=400,
                    detail="min_confidence must be a number between 0 and 1",
                )
            config[key] = float(value)
        elif key in ("model_enabled", "silent_mode"):
            if not isinstance(value, bool):
                raise HTTPException(
                    status_code=400,
                    detail=f"{key} must be a boolean",
                )
            config[key] = value
        else:
            raise HTTPException(
                status_code=400,
                detail=f"unknown config key: {key}",
            )
    return config


def optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    return str(value)


async def handle_device_hello(
    db: asyncpg.Pool,
    device_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    manifest = payload.get("model_manifest")
    if manifest is not None and not isinstance(manifest, dict):
        raise HTTPException(
            status_code=400,
            detail="model_manifest must be a JSON object",
        )

    await expire_stale_positioning(
        db, settings.positioning_ttl_seconds, device_id
    )
    device = await upsert_device_hello(
        db,
        device_id=device_id,
        hardware_id=optional_str(payload, "hardware_id"),
        firmware_build=optional_str(payload, "firmware_build"),
        model_hash=optional_str(payload, "model_hash"),
        model_manifest=manifest,
        pi_id=optional_str(payload, "pi_id"),
    )

    # Cameras self-report their running config at boot, so the dashboard
    # shows real device values before any desired config ever existed.
    # Stored as reported truth, unvalidated; the newest-seq guard keeps a
    # boot-default report (seq 0) from clobbering a later ack.
    reported = payload.get("config")
    if isinstance(reported, dict):
        try:
            reported_seq = int(payload.get("config_seq") or 0)
        except (TypeError, ValueError):
            reported_seq = 0
        await record_reported_config(
            db, device_id=device_id, config=reported, seq=reported_seq
        )
    answer = {
        "ok": True,
        "mode": device["desired_mode"],
        "seq": device["desired_mode_seq"],
        "server_time": datetime.now(UTC).isoformat(),
    }
    if device.get("desired_config") is not None:
        answer["config"] = device["desired_config"]
        answer["config_seq"] = device["desired_config_seq"]
    return answer


async def list_devices(
    db: asyncpg.Pool,
    gateway_connected: Any,
) -> dict[str, Any]:
    """List registry rows, annotated with per-device gateway liveness.

    gateway_connected is a callable(device_id) -> bool backed by the SSE
    broadcaster: a subscribed stream means the Pi fronting that device is
    online, which lets the dashboard distinguish a dead camera from a dead
    gateway.
    """
    await expire_stale_positioning(db, settings.positioning_ttl_seconds)
    devices = await fetch_devices(db)
    for device in devices:
        device["gateway_connected"] = bool(gateway_connected(device["device_id"]))
    return {"ok": True, "devices": devices}


async def set_device_mode(
    db: asyncpg.Pool,
    broadcaster: Any,
    device_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    mode = validate_mode(payload.get("mode"))
    device = await set_desired_mode(db, device_id=device_id, mode=mode)
    # Wake the device's SSE stream so the Pi learns about the change now
    # rather than at the next heartbeat.
    await broadcaster.notify(device_id)
    return {
        "ok": True,
        "device_id": device_id,
        "desired_mode": device["desired_mode"],
        "desired_mode_seq": device["desired_mode_seq"],
        "reported_mode": device["reported_mode"],
        "reported_mode_seq": device["reported_mode_seq"],
    }


async def set_device_config(
    db: asyncpg.Pool,
    broadcaster: Any,
    device_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    config = validate_config(payload.get("config"))
    device = await set_desired_config(db, device_id=device_id, config=config)
    # Wake the device's SSE stream so the Pi learns about the change now
    # rather than at the next heartbeat.
    await broadcaster.notify(device_id)
    return {
        "ok": True,
        "device_id": device_id,
        "desired_config": device["desired_config"],
        "desired_config_seq": device["desired_config_seq"],
        "reported_config": device["reported_config"],
        "reported_config_seq": device["reported_config_seq"],
    }


async def record_device_config_ack(
    db: asyncpg.Pool,
    device_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise HTTPException(
            status_code=400,
            detail="config must be a JSON object",
        )
    try:
        seq = int(payload.get("seq"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="seq must be an integer",
        ) from exc
    # The reported config is what the camera says it runs — stored as-is,
    # not validated against desired bounds, so drift is visible.
    await record_reported_config(db, device_id=device_id, config=config, seq=seq)
    return {"ok": True}


async def record_device_mode_ack(
    db: asyncpg.Pool,
    device_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    mode = validate_mode(payload.get("mode"))
    try:
        seq = int(payload.get("seq"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="seq must be an integer",
        ) from exc
    await record_reported_mode(db, device_id=device_id, mode=mode, seq=seq)
    return {"ok": True}


async def record_device_seen(
    db: asyncpg.Pool,
    device_id: str,
) -> dict[str, Any]:
    await touch_device_seen(db, device_id)
    return {"ok": True}


async def remove_device(
    db: asyncpg.Pool,
    store: PreviewStore,
    device_id: str,
) -> dict[str, Any]:
    deleted = await delete_device(db, device_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="device not found")
    store.discard(device_id)
    return {"ok": True, "device_id": device_id}


class PreviewStore:
    """Latest positioning-preview frame per device, in memory only.

    Frames are ephemeral by design: they never touch S3 or Postgres, and a
    slot reads as empty once the camera has stopped sending for
    expiry_seconds. The clock is injectable for tests.
    """

    def __init__(
        self,
        expiry_seconds: float | None = None,
        clock=time.monotonic,
    ) -> None:
        self.expiry_seconds = (
            settings.preview_expiry_seconds
            if expiry_seconds is None
            else expiry_seconds
        )
        self._clock = clock
        self._frames: dict[str, dict[str, Any]] = {}

    def put(self, device_id: str, body: bytes, content_type: str) -> str:
        etag = f'"{hashlib.sha256(body).hexdigest()[:16]}"'
        self._frames[device_id] = {
            "body": body,
            "content_type": content_type,
            "etag": etag,
            "stored_at": self._clock(),
        }
        return etag

    def get(self, device_id: str) -> dict[str, Any] | None:
        frame = self._frames.get(device_id)
        if frame is None:
            return None
        if self._clock() - frame["stored_at"] > self.expiry_seconds:
            del self._frames[device_id]
            return None
        return frame

    def discard(self, device_id: str) -> None:
        self._frames.pop(device_id, None)


async def receive_preview_frame(
    store: PreviewStore,
    device_id: str,
    body: bytes,
    content_type: str | None,
) -> dict[str, Any]:
    if not body:
        raise HTTPException(status_code=400, detail="preview frame is empty")
    if len(body) > settings.preview_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"preview frame exceeds {settings.preview_max_bytes} bytes",
        )
    if content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"unsupported preview content type: {content_type}",
        )
    store.put(device_id, body, content_type)
    return {"ok": True}


def get_preview_frame(
    store: PreviewStore,
    device_id: str,
    if_none_match: str | None,
) -> Response:
    frame = store.get(device_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="no recent preview frame")
    headers = {"Cache-Control": "no-store", "ETag": frame["etag"]}
    if if_none_match and if_none_match == frame["etag"]:
        return Response(status_code=304, headers=headers)
    return Response(
        content=frame["body"],
        media_type=frame["content_type"],
        headers=headers,
    )
