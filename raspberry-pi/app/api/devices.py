from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.config import settings
from app.services.device_gateway import (
    handle_config_ack,
    handle_hello,
    handle_mode_ack,
    handle_preview,
    handle_tick,
)


router = APIRouter()


async def json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="body must be JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="body must be a JSON object")
    return payload


def required_device_id(payload: dict[str, Any]) -> str:
    device_id = payload.get("device_id")
    if not device_id:
        raise HTTPException(status_code=400, detail="device_id is required")
    return str(device_id)


@router.post("/hello")
async def device_hello(request: Request) -> dict[str, Any]:
    payload = await json_body(request)
    return await handle_hello(
        required_device_id(payload),
        payload,
        request.client.host if request.client else None,
    )


@router.post("/mode-ack")
async def device_mode_ack(request: Request) -> dict[str, Any]:
    payload = await json_body(request)
    try:
        seq = int(payload.get("seq"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="seq must be an integer") from exc
    mode = payload.get("mode")
    if mode not in ("automated", "positioning"):
        raise HTTPException(
            status_code=400,
            detail="mode must be automated or positioning",
        )
    return await handle_mode_ack(
        required_device_id(payload),
        mode,
        seq,
        request.client.host if request.client else None,
    )


@router.post("/config-ack")
async def device_config_ack(request: Request) -> dict[str, Any]:
    payload = await json_body(request)
    try:
        seq = int(payload.get("seq"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="seq must be an integer") from exc
    config = payload.get("config")
    if not isinstance(config, dict):
        raise HTTPException(status_code=400, detail="config must be an object")
    return await handle_config_ack(
        required_device_id(payload),
        config,
        seq,
        request.client.host if request.client else None,
    )


@router.post("/tick")
async def device_tick(request: Request) -> dict[str, Any]:
    payload = await json_body(request)
    return await handle_tick(
        required_device_id(payload),
        request.client.host if request.client else None,
    )


@router.post("/preview")
async def device_preview(request: Request, device_id: str) -> dict[str, Any]:
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="preview frame is empty")
    if len(body) > settings.preview_max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"preview frame exceeds {settings.preview_max_bytes} bytes",
        )
    return await handle_preview(
        device_id,
        body,
        request.headers.get("content-type"),
        request.client.host if request.client else None,
    )
