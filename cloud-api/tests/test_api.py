from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import app.main as main


@pytest.mark.asyncio
async def test_health_endpoint(monkeypatch) -> None:
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
            response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_content_type(monkeypatch) -> None:
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
            response = await client.post(
                "/detections",
                files={"image": ("frame.txt", b"not an image", "text/plain")},
                data={"metadata": "{}"},
            )

    assert response.status_code == 400
    assert "unsupported image type" in response.json()["detail"]
