from __future__ import annotations

import asyncio
import dataclasses
import json
from io import BytesIO

import pytest
from fastapi import BackgroundTasks, HTTPException
from starlette.datastructures import Headers, UploadFile

import app.services.detection_service as detection_service
import app.storage.filesystem as filesystem


@pytest.fixture
def receipts(monkeypatch):
    recorded: list[dict] = []
    monkeypatch.setattr(
        detection_service,
        "record_receipt",
        lambda event, **fields: recorded.append({"event": event, **fields}),
    )
    return recorded


@pytest.fixture
def tmp_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        filesystem,
        "settings",
        dataclasses.replace(
            filesystem.settings,
            upload_dir=tmp_path / "uploads",
            metadata_dir=tmp_path / "metadata",
        ),
    )


def make_upload(
    content: bytes = b"jpeg-bytes",
    filename: str = "frame.jpg",
    content_type: str = "image/jpeg",
) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def receive(image: UploadFile, metadata: str) -> dict:
    return asyncio.run(
        detection_service.receive_detection_upload(
            image=image,
            raw_metadata=metadata,
            background_tasks=BackgroundTasks(),
            client_host=None,
        )
    )


def test_accepted_upload_records_receipt(receipts, tmp_storage) -> None:
    metadata = json.dumps({"device_id": "nicla-vision-01", "detections": []})

    result = receive(make_upload(), metadata)

    assert result["status"] == "accepted"
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["event"] == "accepted"
    assert receipt["device_id"] == "nicla-vision-01"
    assert receipt["image_id"] == result["image_id"]
    assert receipt["content_type"] == "image/jpeg"
    assert receipt["fomo_count"] == 0


def test_unparseable_metadata_records_rejection(receipts) -> None:
    with pytest.raises(HTTPException):
        receive(make_upload(), "not json")

    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["event"] == "rejected"
    assert receipt["reason"] == "metadata must be valid JSON"
    assert receipt["device_id"] is None


def test_rejected_content_type_still_carries_device_id(receipts) -> None:
    metadata = json.dumps({"device_id": "nicla-vision-01"})

    with pytest.raises(HTTPException):
        receive(make_upload(content_type="text/plain"), metadata)

    receipt = receipts[0]
    assert receipt["event"] == "rejected"
    assert receipt["device_id"] == "nicla-vision-01"
    assert "unsupported image type" in receipt["reason"]
