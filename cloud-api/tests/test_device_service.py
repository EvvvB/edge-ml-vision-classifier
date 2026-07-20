from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import HTTPException

import app.main as main
from app.services.capture_service import CaptureBroadcaster, sse_event
from app.services.device_service import PreviewStore, validate_mode


def test_validate_mode_accepts_known_modes() -> None:
    assert validate_mode("automated") == "automated"
    assert validate_mode("positioning") == "positioning"


@pytest.mark.parametrize("mode", ["bogus", None, "", "POSITIONING"])
def test_validate_mode_rejects_unknown_values(mode) -> None:
    with pytest.raises(HTTPException) as error:
        validate_mode(mode)
    assert error.value.status_code == 400


def test_sse_event_carries_counter_and_mode() -> None:
    event = sse_event("nicla-vision-01", 7, "positioning", 3)
    assert event.startswith("data: ")
    assert '"counter": 7' in event
    assert '"mode": "positioning"' in event
    assert '"mode_seq": 3' in event
    assert event.endswith("\n\n")


def test_broadcaster_tracks_subscribers() -> None:
    broadcaster = CaptureBroadcaster()
    assert not broadcaster.has_subscriber("nicla-vision-01")

    broadcaster.subscribe("nicla-vision-01")
    broadcaster.subscribe("nicla-vision-01")
    broadcaster.unsubscribe("nicla-vision-01")
    assert broadcaster.has_subscriber("nicla-vision-01")

    broadcaster.unsubscribe("nicla-vision-01")
    assert not broadcaster.has_subscriber("nicla-vision-01")


def test_preview_store_discard_clears_slot() -> None:
    store = PreviewStore(expiry_seconds=10, clock=lambda: 0.0)
    store.put("nicla-vision-01", b"frame", "image/jpeg")
    store.discard("nicla-vision-01")
    assert store.get("nicla-vision-01") is None
    # Discarding an unknown device is a no-op.
    store.discard("never-registered")


@pytest.mark.asyncio
async def test_delete_device_endpoint(monkeypatch) -> None:
    import app.services.device_service as device_service

    pool = Mock()
    monkeypatch.setattr(main, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "close_db_pool", AsyncMock())
    monkeypatch.setattr(main, "create_s3_client", Mock(return_value=Mock()))
    deleted = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr(device_service, "delete_device", deleted)

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            removed = await client.delete("/devices/nicla-vision-01")
            missing = await client.delete("/devices/never-registered")

    assert removed.status_code == 200
    assert removed.json() == {"ok": True, "device_id": "nicla-vision-01"}
    assert missing.status_code == 404


def test_preview_store_serves_latest_until_expiry() -> None:
    clock = {"now": 100.0}
    store = PreviewStore(expiry_seconds=10, clock=lambda: clock["now"])

    assert store.get("nicla-vision-01") is None

    first_etag = store.put("nicla-vision-01", b"frame-1", "image/jpeg")
    second_etag = store.put("nicla-vision-01", b"frame-2", "image/jpeg")
    assert first_etag != second_etag

    frame = store.get("nicla-vision-01")
    assert frame is not None
    assert frame["body"] == b"frame-2"
    assert frame["etag"] == second_etag

    clock["now"] = 111.0
    assert store.get("nicla-vision-01") is None


@pytest.mark.asyncio
async def test_device_endpoints_validate_and_authenticate(monkeypatch) -> None:
    import dataclasses

    import app.api.detections as detections

    pool = Mock()
    monkeypatch.setattr(main, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "close_db_pool", AsyncMock())
    monkeypatch.setattr(main, "create_s3_client", Mock(return_value=Mock()))
    monkeypatch.setattr(
        detections,
        "settings",
        dataclasses.replace(detections.settings, api_key="test-key"),
    )

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            headers = {"X-API-Key": "test-key"}
            unauthenticated = await client.get("/devices")
            bad_mode = await client.post(
                "/devices/nicla-vision-01/mode",
                json={"mode": "bogus"},
                headers=headers,
            )
            bad_ack_seq = await client.post(
                "/devices/nicla-vision-01/mode-ack",
                json={"mode": "automated", "seq": "not-a-number"},
                headers=headers,
            )
            empty_preview = await client.post(
                "/devices/nicla-vision-01/preview",
                content=b"",
                headers={**headers, "Content-Type": "image/jpeg"},
            )
            missing_preview = await client.get(
                "/devices/nicla-vision-01/preview",
                headers=headers,
            )

    assert unauthenticated.status_code == 401
    assert bad_mode.status_code == 400
    assert "mode must be one of" in bad_mode.json()["detail"]
    assert bad_ack_seq.status_code == 400
    assert empty_preview.status_code == 400
    assert missing_preview.status_code == 404


@pytest.mark.asyncio
async def test_preview_roundtrip_with_etag(monkeypatch) -> None:
    pool = Mock()
    monkeypatch.setattr(main, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "close_db_pool", AsyncMock())
    monkeypatch.setattr(main, "create_s3_client", Mock(return_value=Mock()))

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            upload = await client.post(
                "/devices/nicla-vision-01/preview",
                content=b"jpeg-bytes",
                headers={"Content-Type": "image/jpeg"},
            )
            fetch = await client.get("/devices/nicla-vision-01/preview")
            not_modified = await client.get(
                "/devices/nicla-vision-01/preview",
                headers={"If-None-Match": fetch.headers["etag"]},
            )

    assert upload.status_code == 200
    assert fetch.status_code == 200
    assert fetch.content == b"jpeg-bytes"
    assert fetch.headers["cache-control"] == "no-store"
    assert not_modified.status_code == 304


@pytest.mark.asyncio
async def test_hello_returns_mode_and_server_time(monkeypatch) -> None:
    import app.services.device_service as device_service

    pool = Mock()
    monkeypatch.setattr(main, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "close_db_pool", AsyncMock())
    monkeypatch.setattr(main, "create_s3_client", Mock(return_value=Mock()))
    monkeypatch.setattr(
        device_service, "expire_stale_positioning", AsyncMock(return_value=0)
    )
    upsert = AsyncMock(
        return_value={
            "device_id": "nicla-vision-01",
            "desired_mode": "positioning",
            "desired_mode_seq": 4,
        }
    )
    monkeypatch.setattr(device_service, "upsert_device_hello", upsert)

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post(
                "/devices/nicla-vision-01/hello",
                json={
                    "hardware_id": "a1b2c3",
                    "firmware_build": "2026-07-19.1",
                    "model_hash": "ffff00ffff00",
                    "model_manifest": {"model_version": "v1"},
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "positioning"
    assert body["seq"] == 4
    assert "server_time" in body
    kwargs = upsert.await_args.kwargs
    assert kwargs["device_id"] == "nicla-vision-01"
    assert kwargs["hardware_id"] == "a1b2c3"
    assert kwargs["model_manifest"] == {"model_version": "v1"}
