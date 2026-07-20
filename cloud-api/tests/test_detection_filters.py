from typing import Any

from app.storage.postgres import build_detection_filters


def build(**overrides: Any) -> tuple[str, list[Any]]:
    params: list[Any] = []
    kwargs: dict[str, Any] = {
        "device_id": None,
        "labels": [],
        "models": [],
        "detections": "any",
        "source": "any",
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
