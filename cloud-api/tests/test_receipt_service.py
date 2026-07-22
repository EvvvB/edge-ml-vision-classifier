from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import HTTPException

import app.services.receipt_service as receipt_service
from app.services.receipt_service import parse_receipt


def receipt_line(**overrides):
    line = {
        "receipt_id": "a" * 32,
        "logged_at": "2026-07-21T10:00:00+00:00",
        "pi_id": "pi-01",
        "event": "accepted",
        "device_id": "nicla-vision-01",
        "image_id": "b" * 32,
        "filename": "frame.jpg",
        "content_type": "image/jpeg",
        "fomo_count": 2,
        "client_host": "192.168.50.60",
    }
    line.update(overrides)
    return line


def test_parse_receipt_coerces_fields() -> None:
    row = parse_receipt(receipt_line(), None)
    assert row["receipt_id"] == UUID("a" * 32)
    assert row["event"] == "accepted"
    assert row["fomo_count"] == 2
    assert row["logged_at"].isoformat() == "2026-07-21T10:00:00+00:00"


def test_parse_receipt_falls_back_to_batch_pi_id() -> None:
    row = parse_receipt(receipt_line(pi_id=None), "pi-02")
    assert row["pi_id"] == "pi-02"


@pytest.mark.parametrize(
    "entry",
    [
        "not-a-dict",
        receipt_line(receipt_id="not-a-uuid"),
        receipt_line(receipt_id=None),
        receipt_line(event="exploded"),
        receipt_line(event=None),
    ],
)
def test_parse_receipt_rejects_bad_entries(entry) -> None:
    assert parse_receipt(entry, None) is None


def test_parse_receipt_nulls_unusable_optional_fields() -> None:
    row = parse_receipt(
        receipt_line(fomo_count=True, logged_at="whenever"), None
    )
    assert row is not None
    assert row["fomo_count"] is None
    assert row["logged_at"] is None


@pytest.mark.asyncio
async def test_receive_receipt_batch_counts_outcomes(monkeypatch) -> None:
    insert = AsyncMock(return_value=1)
    monkeypatch.setattr(receipt_service, "insert_pi_receipts", insert)

    payload = {
        "pi_id": "pi-01",
        "receipts": [
            receipt_line(),
            receipt_line(receipt_id="c" * 32),
            {"event": "accepted"},  # no receipt_id: invalid, dropped
        ],
    }
    result = await receipt_service.receive_receipt_batch(None, payload)

    assert result == {
        "ok": True,
        "received": 3,
        "inserted": 1,
        "duplicates": 1,
        "invalid": 1,
    }
    rows = insert.await_args.args[1]
    assert [row["receipt_id"] for row in rows] == [
        UUID("a" * 32),
        UUID("c" * 32),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("receipts", [None, [], "not-a-list"])
async def test_receive_receipt_batch_rejects_bad_payload(receipts) -> None:
    with pytest.raises(HTTPException) as error:
        await receipt_service.receive_receipt_batch(
            None, {"receipts": receipts}
        )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_receive_receipt_batch_rejects_oversized_batch() -> None:
    receipts = [receipt_line()] * (receipt_service.MAX_BATCH_RECEIPTS + 1)
    with pytest.raises(HTTPException) as error:
        await receipt_service.receive_receipt_batch(
            None, {"receipts": receipts}
        )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_list_receipts_validates_inputs(monkeypatch) -> None:
    monkeypatch.setattr(
        receipt_service, "fetch_pi_receipts", AsyncMock(return_value=[])
    )

    result = await receipt_service.list_receipts(
        None, device_id=None, event="rejected", limit=10
    )
    assert result == {"receipts": []}

    with pytest.raises(HTTPException):
        await receipt_service.list_receipts(
            None, device_id=None, event="meh", limit=10
        )
    with pytest.raises(HTTPException):
        await receipt_service.list_receipts(
            None, device_id=None, event=None, limit=0
        )
