from datetime import timezone
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.services.detection_service import (
    build_s3_key,
    parse_metadata,
    parse_model_filters,
    parse_optional_datetime,
    parse_time_range,
    remove_detections,
    safe_image_suffix,
)


def test_parse_metadata_accepts_object() -> None:
    assert parse_metadata('{"device_id": "nicla-01"}') == {
        "device_id": "nicla-01"
    }


@pytest.mark.parametrize("raw_metadata", ["not-json", "[]", '"text"'])
def test_parse_metadata_rejects_invalid_values(raw_metadata: str) -> None:
    with pytest.raises(HTTPException) as error:
        parse_metadata(raw_metadata)
    assert error.value.status_code == 400


def test_parse_optional_datetime_accepts_utc_z_suffix() -> None:
    result = parse_optional_datetime("2026-07-08T12:30:00Z")
    assert result is not None
    assert result.tzinfo == timezone.utc


def test_parse_time_range_parses_both_bounds() -> None:
    since, until = parse_time_range("2026-07-20T00:00:00Z", "2026-07-21T00:00:00Z")
    assert since is not None and until is not None
    assert since < until
    assert parse_time_range(None, None) == (None, None)


def test_parse_time_range_rejects_inverted_bounds() -> None:
    with pytest.raises(HTTPException) as error:
        parse_time_range("2026-07-21T00:00:00Z", "2026-07-20T00:00:00Z")
    assert error.value.status_code == 400


def test_parse_time_range_names_the_bad_field() -> None:
    with pytest.raises(HTTPException) as error:
        parse_time_range(None, "not-a-date")
    assert error.value.status_code == 400
    assert "until" in error.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["any", "fomo"])
async def test_remove_detections_refuses_unrestrictive_filters(source: str) -> None:
    # A source filter alone matches everything, so it must not authorize a
    # bulk delete. The guard fires before db/s3 are touched.
    with pytest.raises(HTTPException) as error:
        await remove_detections(
            None,
            s3_client=None,
            device_id=None,
            labels=None,
            models=None,
            detections="any",
            source=source,
            since=None,
            until=None,
        )
    assert error.value.status_code == 400


def test_parse_model_filters_normalizes() -> None:
    assert parse_model_filters(None) == []
    assert parse_model_filters(" A1B2C3D4E5F6 ,ffff00,,a1b2c3d4e5f6") == [
        "a1b2c3d4e5f6",
        "ffff00",
    ]


@pytest.mark.parametrize("models", ["not-a-hash", "abc123,xyz", "12g4"])
def test_parse_model_filters_rejects_non_hex(models: str) -> None:
    with pytest.raises(HTTPException) as error:
        parse_model_filters(models)
    assert error.value.status_code == 400


def test_safe_image_suffix_falls_back_for_unknown_extension() -> None:
    assert safe_image_suffix("frame.exe") == ".jpg"


def test_build_s3_key_contains_prefix_date_and_image_id() -> None:
    image_id = UUID("12345678-1234-5678-1234-567812345678")
    key = build_s3_key(image_id, ".jpg")
    assert key.startswith("detections/")
    assert key.endswith(f"/{image_id}.jpg")
