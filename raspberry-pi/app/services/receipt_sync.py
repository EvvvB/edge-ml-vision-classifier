"""Ship the Pi's receipt log to the cloud in batches.

Run by receipt-sync.timer as `python -m app.services.receipt_sync`. A state
file next to the logs maps each log file to the byte offset already shipped;
lines past the offset are POSTed in batches and the offset advances only
after the cloud acknowledges the whole file. Resends after a crash or a
mid-file failure are safe: receipt_id is the primary key cloud-side and
duplicates are dropped there.

Fully-synced files older than the retention window are deleted, so the log
directory stays bounded without ever discarding unshipped lines.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from app.config import settings


log = logging.getLogger("receipt-sync")


def sync_state_path() -> Path:
    return settings.receipt_log_dir / "sync_state.json"


def load_sync_state() -> dict[str, int]:
    try:
        stored = json.loads(sync_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(stored, dict):
        return {}
    return {
        str(name): int(offset)
        for name, offset in stored.items()
        if isinstance(offset, int)
    }


def save_sync_state(state: dict[str, int]) -> None:
    sync_state_path().write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def pending_receipts(
    path: Path, offset: int
) -> tuple[list[dict[str, Any]], int]:
    """Complete, parseable lines past `offset`, and the new offset.

    Only newline-terminated lines count: the receiver may be mid-append on
    the last line, so an unterminated tail is left for the next run. A line
    that does not parse (a crash mid-write, later appended over) is skipped
    with a warning rather than blocking the file forever.
    """
    data = path.read_bytes()[offset:]
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset

    complete = data[: end + 1]
    receipts: list[dict[str, Any]] = []
    for raw_line in complete.splitlines():
        if not raw_line.strip():
            continue
        try:
            line = json.loads(raw_line)
        except ValueError:
            log.warning("skipping corrupt receipt line in %s", path.name)
            continue
        if isinstance(line, dict):
            receipts.append(line)
    return receipts, offset + len(complete)


def post_receipts(client: httpx.Client, receipts: list[dict[str, Any]]) -> None:
    response = client.post(
        "/receipts/batch",
        json={"pi_id": settings.pi_id, "receipts": receipts},
    )
    response.raise_for_status()


def sync_file(
    client: httpx.Client, path: Path, offset: int
) -> tuple[int, int]:
    """Ship a file's pending lines; return (new offset, lines shipped)."""
    receipts, new_offset = pending_receipts(path, offset)
    batch_size = max(1, settings.receipt_sync_batch_size)
    for start in range(0, len(receipts), batch_size):
        post_receipts(client, receipts[start : start + batch_size])
    return new_offset, len(receipts)


def receipt_file_date(path: Path) -> datetime | None:
    try:
        return datetime.strptime(path.stem, "receipts-%Y-%m-%d").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None


def prune_synced_files(state: dict[str, int]) -> None:
    """Drop fully-synced files past retention, and state for missing files."""
    cutoff = datetime.now(UTC) - timedelta(days=settings.receipt_retention_days)
    for path in sorted(settings.receipt_log_dir.glob("receipts-*.jsonl")):
        file_date = receipt_file_date(path)
        if file_date is None or file_date >= cutoff:
            continue
        if state.get(path.name, 0) < path.stat().st_size:
            continue
        path.unlink()
        state.pop(path.name, None)
        log.info("pruned %s", path.name)

    remaining = {path.name for path in settings.receipt_log_dir.glob("*.jsonl")}
    for name in list(state):
        if name not in remaining:
            del state[name]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not settings.cloud_api_url:
        raise SystemExit(
            "CLOUD_API_URL is not set; the receipt sync needs the cloud API"
        )
    if not settings.receipt_log_dir.is_dir():
        log.info("no receipt log directory yet; nothing to sync")
        return

    headers = {}
    if settings.cloud_api_key:
        headers["X-API-Key"] = settings.cloud_api_key
    client = httpx.Client(
        base_url=settings.cloud_api_url.rstrip("/"),
        headers=headers,
        timeout=settings.cloud_forward_timeout_seconds,
    )

    state = load_sync_state()
    shipped = 0
    try:
        for path in sorted(settings.receipt_log_dir.glob("receipts-*.jsonl")):
            old_offset = state.get(path.name, 0)
            new_offset, count = sync_file(client, path, old_offset)
            if new_offset != old_offset:
                # Saved per file so a failure mid-run keeps earlier progress.
                state[path.name] = new_offset
                save_sync_state(state)
                shipped += count
    finally:
        client.close()

    prune_synced_files(state)
    save_sync_state(state)
    log.info("shipped %d receipts", shipped)


if __name__ == "__main__":
    main()
