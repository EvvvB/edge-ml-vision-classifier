from __future__ import annotations

from app.services.cloud_forwarder import build_cloud_metadata


def test_build_cloud_metadata_forwards_model_stamps() -> None:
    saved = {
        "image_id": "abc123",
        "inference_status": "complete",
        "metadata": {
            "device_id": "nicla-vision-01",
            "model_hash": "ffff00ffff00",
            "model_manifest": {"model_version": "v1"},
        },
        "fomo_detections": [],
        "yolo_detections": [],
        "yolo_model_hash": "a1b2c3d4e5f6",
        "yolo_model_manifest": {"model_version": "yolo26m-stock"},
        "received_at": "2026-07-19T00:00:00+00:00",
    }

    cloud = build_cloud_metadata(saved)

    # The Nicla's own stamp rides along inside device metadata untouched.
    assert cloud["model_hash"] == "ffff00ffff00"
    assert cloud["model_manifest"] == {"model_version": "v1"}
    assert cloud["yolo_model_hash"] == "a1b2c3d4e5f6"
    assert cloud["yolo_model_manifest"] == {"model_version": "yolo26m-stock"}


def test_build_cloud_metadata_leaves_failed_inference_unstamped() -> None:
    saved = {
        "image_id": "abc123",
        "inference_status": "failed",
        "metadata": {"device_id": "nicla-vision-01"},
    }

    cloud = build_cloud_metadata(saved)

    assert "yolo_model_hash" not in cloud
    assert "yolo_model_manifest" not in cloud
