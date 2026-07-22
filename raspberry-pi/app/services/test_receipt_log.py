from __future__ import annotations

import dataclasses
import json
from datetime import datetime

import app.services.receipt_log as receipt_log


def use_tmp_log_dir(monkeypatch, log_dir) -> None:
    monkeypatch.setattr(
        receipt_log,
        "settings",
        dataclasses.replace(receipt_log.settings, receipt_log_dir=log_dir),
    )


def test_record_receipt_appends_to_daily_file(monkeypatch, tmp_path) -> None:
    use_tmp_log_dir(monkeypatch, tmp_path / "receipts")

    receipt_log.record_receipt(
        "accepted", device_id="nicla-vision-01", fomo_count=2
    )
    receipt_log.record_receipt("rejected", reason="metadata must be valid JSON")

    files = sorted((tmp_path / "receipts").glob("receipts-*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    accepted = json.loads(lines[0])
    assert accepted["event"] == "accepted"
    assert accepted["device_id"] == "nicla-vision-01"
    assert accepted["fomo_count"] == 2
    assert accepted["pi_id"] == receipt_log.settings.pi_id
    assert len(accepted["receipt_id"]) == 32
    # The filename date and the line timestamp come from the same clock.
    logged_at = datetime.fromisoformat(accepted["logged_at"])
    assert files[0].name == f"receipts-{logged_at:%Y-%m-%d}.jsonl"

    assert json.loads(lines[1])["event"] == "rejected"


def test_record_receipt_swallows_filesystem_errors(
    monkeypatch, tmp_path
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    use_tmp_log_dir(monkeypatch, blocker / "receipts")

    receipt_log.record_receipt("accepted", device_id="nicla-vision-01")
