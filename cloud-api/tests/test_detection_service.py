from datetime import timezone
from uuid import UUID

import pytest
from fastapi import HTTPException

from app.services.detection_service import (
    build_s3_key,
    parse_metadata,
    parse_optional_datetime,
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


def test_safe_image_suffix_falls_back_for_unknown_extension() -> None:
    assert safe_image_suffix("frame.exe") == ".jpg"


def test_build_s3_key_contains_prefix_date_and_image_id() -> None:
    image_id = UUID("12345678-1234-5678-1234-567812345678")
    key = build_s3_key(image_id, ".jpg")
    assert key.startswith("detections/")
    assert key.endswith(f"/{image_id}.jpg")
