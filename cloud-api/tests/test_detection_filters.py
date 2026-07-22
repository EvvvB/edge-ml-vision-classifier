from datetime import UTC, datetime
from typing import Any

import pytest

from app.storage.postgres import build_detection_filters, delete_detections


def build(**overrides: Any) -> tuple[str, list[Any]]:
    params: list[Any] = []
    kwargs: dict[str, Any] = {
        "device_id": None,
        "labels": [],
        "models": [],
        "detections": "any",
        "source": "any",
        "since": None,
        "until": None,
        "params": params,
    }
    kwargs.update(overrides)
    return build_detection_filters(**kwargs), params


def test_no_filters_builds_no_where_clause() -> None:
    where, params = build()
    assert where == ""
    assert params == []


def test_models_filter_matches_either_sources_hash() -> None:
    where, params = build(models=["a1b2c3d4e5f6"])
    assert "metadata->>'model_hash' = ANY($1::text[])" in where
    assert "metadata->>'yolo_model_hash' = ANY($1::text[])" in where
    assert params == [["a1b2c3d4e5f6"]]


def test_models_filter_respects_source() -> None:
    where, _ = build(models=["a1b2c3d4e5f6"], source="yolo")
    assert "metadata->>'yolo_model_hash' = ANY($1::text[])" in where
    # The FOMO field must not appear; note "yolo_model_hash" contains
    # "model_hash", so match on the full field expression.
    assert "metadata->>'model_hash'" not in where


def test_models_filter_composes_with_device_and_labels() -> None:
    where, params = build(
        device_id="nicla-vision-01",
        labels=["dog"],
        models=["ffff00"],
        source="fomo",
    )
    assert where.startswith(" WHERE ")
    assert "device_id = $1" in where
    assert "= ANY($2::text[])" in where
    assert "metadata->>'model_hash' = ANY($3::text[])" in where
    assert params == ["nicla-vision-01", ["dog"], ["ffff00"]]


def test_time_range_filters_display_timestamp() -> None:
    since = datetime(2026, 7, 20, tzinfo=UTC)
    until = datetime(2026, 7, 21, tzinfo=UTC)
    where, params = build(since=since, until=until)
    assert "coalesce(captured_at, created_at) >= $1" in where
    assert "coalesce(captured_at, created_at) <= $2" in where
    assert params == [since, until]


def test_time_range_composes_with_device() -> None:
    since = datetime(2026, 7, 20, tzinfo=UTC)
    where, params = build(device_id="nicla-vision-01", since=since)
    assert "device_id = $1" in where
    assert "coalesce(captured_at, created_at) >= $2" in where
    assert params == ["nicla-vision-01", since]


@pytest.mark.asyncio
async def test_delete_detections_refuses_empty_filters() -> None:
    # The guard fires before the pool is touched, so None stands in for it.
    with pytest.raises(ValueError):
        await delete_detections(
            None,
            device_id=None,
            labels=[],
            models=[],
            detections="any",
            source="any",
            since=None,
            until=None,
        )
