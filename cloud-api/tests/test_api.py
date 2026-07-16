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


@pytest.mark.asyncio
async def test_detection_image_serves_with_cache_headers(monkeypatch) -> None:
    import dataclasses

    import app.api.detections as detections
    import app.services.detection_service as detection_service

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

    image_id = "3f0c2f4e-9c48-4d55-8a70-6f6f0f9dd3a1"
    detection = {
        "image_id": image_id,
        "upload_status": "stored",
        "content_type": "image/jpeg",
        "s3_bucket": "bucket",
        "s3_key": "detections/2026/07/16/frame.jpg",
        "s3_etag": '"abc123"',
    }
    monkeypatch.setattr(
        detection_service,
        "fetch_detection",
        AsyncMock(return_value=detection),
    )
    download = AsyncMock(
        return_value={
            "body": b"jpeg-bytes",
            "content_type": "image/jpeg",
            "etag": '"abc123"',
        }
    )
    monkeypatch.setattr(detection_service, "download_image", download)

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            no_key = await client.get(f"/detections/{image_id}/image")
            with_query_key = await client.get(
                f"/detections/{image_id}/image",
                params={"key": "test-key"},
            )
            not_modified = await client.get(
                f"/detections/{image_id}/image",
                params={"key": "test-key"},
                headers={"If-None-Match": '"abc123"'},
            )

    assert no_key.status_code == 401

    assert with_query_key.status_code == 200
    assert with_query_key.content == b"jpeg-bytes"
    assert with_query_key.headers["content-type"] == "image/jpeg"
    assert (
        with_query_key.headers["cache-control"]
        == "public, max-age=31536000, immutable"
    )
    assert with_query_key.headers["etag"] == '"abc123"'

    assert not_modified.status_code == 304
    # The 304 must be answered from the stored ETag without re-fetching S3.
    assert download.await_count == 1


@pytest.mark.asyncio
async def test_detection_image_404_when_not_stored(monkeypatch) -> None:
    import app.services.detection_service as detection_service

    pool = Mock()
    monkeypatch.setattr(main, "create_db_pool", AsyncMock(return_value=pool))
    monkeypatch.setattr(main, "init_db", AsyncMock())
    monkeypatch.setattr(main, "close_db_pool", AsyncMock())
    monkeypatch.setattr(main, "create_s3_client", Mock(return_value=Mock()))

    image_id = "3f0c2f4e-9c48-4d55-8a70-6f6f0f9dd3a1"
    monkeypatch.setattr(
        detection_service,
        "fetch_detection",
        AsyncMock(return_value={"image_id": image_id, "upload_status": "failed"}),
    )

    transport = httpx.ASGITransport(app=main.app)
    async with main.lifespan(main.app):
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get(f"/detections/{image_id}/image")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_detections_reject_invalid_filter_values(monkeypatch) -> None:
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
            bad_detections = await client.get(
                "/detections", params={"detections": "bogus"}
            )
            bad_source = await client.get("/detections", params={"source": "ssd"})
            bad_facet_source = await client.get(
                "/detections/facets", params={"source": "ssd"}
            )

    assert bad_detections.status_code == 400
    assert "detections must be one of" in bad_detections.json()["detail"]
    assert bad_source.status_code == 400
    assert bad_facet_source.status_code == 400


def test_parse_label_filters_normalizes() -> None:
    from app.services.detection_service import parse_label_filters

    assert parse_label_filters(None) == []
    assert parse_label_filters(" Dog ,person,,DOG ") == ["dog", "person"]


def test_build_coco_dataset() -> None:
    from app.services.detection_service import build_coco_dataset

    rows = [
        {
            "s3_key": "detections/2026/07/16/aaa.jpg",
            "metadata": {
                "frame_width": 320,
                "frame_height": 240,
                "fomo_detections": [
                    {"label": "dog", "confidence": 0.45, "bbox": [141, 51, 8, 8]},
                    {"label": "dog", "bbox": "not-a-box"},
                ],
            },
        },
        {
            # Negative sample: listed in images[], no annotations.
            "s3_key": "detections/2026/07/16/bbb.jpg",
            "metadata": {"frame_width": 320, "frame_height": 240},
        },
    ]

    dataset = build_coco_dataset(rows, "fomo_detections", "test export")

    assert [img["file_name"] for img in dataset["images"]] == ["aaa.jpg", "bbb.jpg"]
    assert dataset["images"][0] == {
        "id": 1,
        "file_name": "aaa.jpg",
        "width": 320,
        "height": 240,
    }
    assert len(dataset["annotations"]) == 1
    annotation = dataset["annotations"][0]
    assert annotation["bbox"] == [141, 51, 8, 8]
    assert annotation["area"] == 64
    assert annotation["image_id"] == 1
    assert annotation["score"] == 0.45
    assert dataset["categories"] == [{"id": 1, "name": "dog"}]


@pytest.mark.asyncio
async def test_detections_require_api_key_when_configured(monkeypatch) -> None:
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
            missing = await client.get("/detections")
            wrong = await client.get("/detections", headers={"X-API-Key": "nope"})
            health = await client.get("/health")

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert health.status_code == 200
