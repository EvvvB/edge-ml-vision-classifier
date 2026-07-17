from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.detections import require_api_key, require_api_key_or_query
from app.services.capture_service import capture_event_stream, request_capture


router = APIRouter()


@router.post(
    "/devices/{device_id}/capture",
    dependencies=[Depends(require_api_key)],
)
async def request_device_capture(
    request: Request,
    device_id: str,
) -> dict[str, Any]:
    return await request_capture(
        request.app.state.db,
        request.app.state.capture_broadcaster,
        device_id,
    )


# EventSource cannot send headers, so like the image endpoints the stream
# also accepts the key as a ?key= query parameter.
@router.get(
    "/devices/{device_id}/capture/stream",
    dependencies=[Depends(require_api_key_or_query)],
)
async def stream_device_captures(
    request: Request,
    device_id: str,
) -> StreamingResponse:
    return StreamingResponse(
        capture_event_stream(
            request.app.state.db,
            request.app.state.capture_broadcaster,
            device_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Caddy disables buffering for text/event-stream on its own;
            # this covers any other proxy in front.
            "X-Accel-Buffering": "no",
        },
    )
