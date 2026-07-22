from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import HTTPException

from app.storage.postgres import fetch_pi_receipts, insert_pi_receipts


RECEIPT_EVENTS = frozenset({"accepted", "rejected"})

MAX_BATCH_RECEIPTS = 1000
MAX_LIST_LIMIT = 500


async def receive_receipt_batch(
    pool: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    receipts = payload.get("receipts")
    if not isinstance(receipts, list) or not receipts:
        raise HTTPException(
            status_code=400, detail="receipts must be a non-empty list"
        )
    if len(receipts) > MAX_BATCH_RECEIPTS:
        raise HTTPException(
            status_code=400,
            detail=f"batch too large: max {MAX_BATCH_RECEIPTS} receipts",
        )

    batch_pi_id = payload.get("pi_id")
    rows: list[dict[str, Any]] = []
    invalid = 0
    for entry in receipts:
        row = parse_receipt(entry, batch_pi_id)
        # An invalid line is dropped, not fatal: it would come back verbatim
        # on every retry, so failing the batch would wedge the Pi's sync.
        if row is None:
            invalid += 1
        else:
            rows.append(row)

    inserted = await insert_pi_receipts(pool, rows) if rows else 0
    return {
        "ok": True,
        "received": len(receipts),
        "inserted": inserted,
        "duplicates": len(rows) - inserted,
        "invalid": invalid,
    }


def parse_receipt(entry: Any, batch_pi_id: Any) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    try:
        receipt_id = UUID(str(entry.get("receipt_id")))
    except ValueError:
        return None
    event = entry.get("event")
    if event not in RECEIPT_EVENTS:
        return None

    logged_at: datetime | None = None
    if isinstance(entry.get("logged_at"), str):
        try:
            logged_at = datetime.fromisoformat(entry["logged_at"])
        except ValueError:
            logged_at = None

    fomo_count = entry.get("fomo_count")
    if not isinstance(fomo_count, int) or isinstance(fomo_count, bool):
        fomo_count = None

    def text(key: str) -> str | None:
        value = entry.get(key)
        return str(value) if value is not None else None

    return {
        "receipt_id": receipt_id,
        "pi_id": text("pi_id")
        or (str(batch_pi_id) if batch_pi_id is not None else None),
        "device_id": text("device_id"),
        "event": event,
        "image_id": text("image_id"),
        "filename": text("filename"),
        "content_type": text("content_type"),
        "fomo_count": fomo_count,
        "reason": text("reason"),
        "client_host": text("client_host"),
        "logged_at": logged_at,
    }


async def list_receipts(
    pool: Any,
    *,
    device_id: str | None,
    event: str | None,
    limit: int,
) -> dict[str, Any]:
    if event is not None and event not in RECEIPT_EVENTS:
        raise HTTPException(
            status_code=400,
            detail=f"event must be one of {sorted(RECEIPT_EVENTS)}",
        )
    if not 1 <= limit <= MAX_LIST_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be between 1 and {MAX_LIST_LIMIT}",
        )
    receipts = await fetch_pi_receipts(
        pool, device_id=device_id, event=event, limit=limit
    )
    return {"receipts": receipts}
