from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request, Response

from app.api.detections import require_api_key, require_api_key_or_query
from app.config import settings
from app.services.device_service import (
    get_preview_frame,
    handle_device_hello,
    list_devices,
    receive_preview_frame,
    record_device_config_ack,
    record_device_mode_ack,
    record_device_seen,
    remove_device,
    set_device_config,
    set_device_mode,
)


router = APIRouter()


@router.get("/devices", dependencies=[Depends(require_api_key)])
async def read_devices(request: Request) -> dict[str, Any]:
    return await list_devices(
        request.app.state.db,
        gateway_connected=request.app.state.capture_broadcaster.has_subscriber,
    )


@router.delete(
    "/devices/{device_id}",
    dependencies=[Depends(require_api_key)],
)
async def prune_device(request: Request, device_id: str) -> dict[str, Any]:
    return await remove_device(
        request.app.state.db,
        request.app.state.preview_store,
        device_id,
    )


@router.post(
    "/devices/{device_id}/hello",
    dependencies=[Depends(require_api_key)],
)
async def device_hello(request: Request, device_id: str) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    return await handle_device_hello(request.app.state.db, device_id, payload)


@router.post(
    "/devices/{device_id}/mode",
    dependencies=[Depends(require_api_key)],
)
async def set_mode(request: Request, device_id: str) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    return await set_device_mode(
        request.app.state.db,
        request.app.state.capture_broadcaster,
        device_id,
        payload,
    )


@router.post(
    "/devices/{device_id}/mode-ack",
    dependencies=[Depends(require_api_key)],
)
async def mode_ack(request: Request, device_id: str) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    return await record_device_mode_ack(
        request.app.state.db, device_id, payload
    )


@router.post(
    "/devices/{device_id}/config",
    dependencies=[Depends(require_api_key)],
)
async def set_config(request: Request, device_id: str) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    return await set_device_config(
        request.app.state.db,
        request.app.state.capture_broadcaster,
        device_id,
        payload,
    )


@router.post(
    "/devices/{device_id}/config-ack",
    dependencies=[Depends(require_api_key)],
)
async def config_ack(request: Request, device_id: str) -> dict[str, Any]:
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    return await record_device_config_ack(
        request.app.state.db, device_id, payload
    )


@router.post(
    "/devices/{device_id}/seen",
    dependencies=[Depends(require_api_key)],
)
async def device_seen(request: Request, device_id: str) -> dict[str, Any]:
    return await record_device_seen(request.app.state.db, device_id)


@router.post(
    "/devices/{device_id}/preview",
    dependencies=[Depends(require_api_key)],
)
async def upload_preview(request: Request, device_id: str) -> dict[str, Any]:
    body = await request.body()
    return await receive_preview_frame(
        request.app.state.preview_store,
        device_id,
        body,
        request.headers.get("content-type"),
    )


# <img> tags cannot send headers, so like the detection images the preview
# also accepts the key as a query parameter.
@router.get(
    "/devices/{device_id}/preview",
    dependencies=[Depends(require_api_key_or_query)],
)
async def read_preview(
    request: Request,
    device_id: str,
    if_none_match: str | None = Header(default=None),
) -> Response:
    return get_preview_frame(
        request.app.state.preview_store,
        device_id,
        if_none_match,
    )
