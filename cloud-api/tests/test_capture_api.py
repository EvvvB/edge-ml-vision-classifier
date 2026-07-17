import asyncio
import json
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import app.main as main
import app.services.capture_service as capture_service


def patch_lifespan(monkeypatch) -> None:
    monkeypatch.setattr(main, "create_db_pool", AsyncMock(return_value=Mock()))
    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "close_db_pool", AsyncMock())
    monkeypatch.setattr(main, "create_s3_client", Mock(return_value=Mock()))


@pytest.mark.asyncio
async def test_capture_request_increments_counter(monkeypatch) -> None:
    patch_lifespan(monkeypatch)
    monkeypatch.setattr(
        capture_service,
        "increment_capture_counter",
        AsyncMock(return_value=7),
    )

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post("/devices/nicla-vision-01/capture")

    assert response.status_code == 200
    assert response.json() == {"device_id": "nicla-vision-01", "counter": 7}


# httpx's ASGITransport buffers whole response bodies, so an endless SSE
# response cannot be consumed through it; the generator is tested directly.
@pytest.mark.asyncio
async def test_capture_stream_sends_snapshot_then_updates(monkeypatch) -> None:
    counters = iter([3, 4])
    monkeypatch.setattr(
        capture_service,
        "fetch_capture_counter",
        AsyncMock(side_effect=lambda pool, device_id: next(counters)),
    )
    monkeypatch.setattr(
        capture_service,
        "increment_capture_counter",
        AsyncMock(return_value=4),
    )

    broadcaster = capture_service.CaptureBroadcaster()
    stream = capture_service.capture_event_stream(
        Mock(), broadcaster, "nicla-vision-01"
    )
    events = []

    async def read_two_events() -> None:
        async for chunk in stream:
            if chunk.startswith("data:"):
                events.append(json.loads(chunk[len("data:") :]))
            if len(events) == 2:
                return

    reader = asyncio.create_task(read_two_events())
    # Let the stream send its snapshot event, then trigger a capture.
    await asyncio.sleep(0.05)
    await capture_service.request_capture(Mock(), broadcaster, "nicla-vision-01")
    await asyncio.wait_for(reader, timeout=5)

    assert events == [
        {"device_id": "nicla-vision-01", "counter": 3},
        {"device_id": "nicla-vision-01", "counter": 4},
    ]


@pytest.mark.asyncio
async def test_capture_stream_heartbeats_without_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        capture_service,
        "fetch_capture_counter",
        AsyncMock(return_value=5),
    )
    monkeypatch.setattr(capture_service, "HEARTBEAT_SECONDS", 0.05)

    broadcaster = capture_service.CaptureBroadcaster()
    stream = capture_service.capture_event_stream(
        Mock(), broadcaster, "nicla-vision-01"
    )

    first = await anext(stream)
    second = await asyncio.wait_for(anext(stream), timeout=5)

    assert json.loads(first[len("data:") :]) == {
        "device_id": "nicla-vision-01",
        "counter": 5,
    }
    assert second.startswith(": keepalive")
