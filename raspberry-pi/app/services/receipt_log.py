from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import settings


logger = logging.getLogger(__name__)


def receipt_log_path(now: datetime) -> Path:
    return settings.receipt_log_dir / f"receipts-{now:%Y-%m-%d}.jsonl"


def record_receipt(event: str, **fields: Any) -> None:
    """Append one line to today's receipt log.

    The log is observability, not correctness: any filesystem problem is
    logged and swallowed so recording a receipt can never fail the upload
    it describes. receipt_id is the idempotency key the cloud dedupes on,
    so the sync job may resend lines freely.
    """
    now = datetime.now(UTC)
    line = {
        "receipt_id": uuid4().hex,
        "logged_at": now.isoformat(),
        "pi_id": settings.pi_id,
        "event": event,
        **fields,
    }
    try:
        path = receipt_log_path(now)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(line) + "\n")
    except OSError as exc:
        logger.warning("could not record %s receipt: %s", event, exc)
