from __future__ import annotations

import dataclasses
import json

import app.services.receipt_sync as receipt_sync


def configure(monkeypatch, tmp_path, **overrides) -> None:
    monkeypatch.setattr(
        receipt_sync,
        "settings",
        dataclasses.replace(
            receipt_sync.settings,
            receipt_log_dir=tmp_path,
            cloud_api_url="http://cloud.test",
            **overrides,
        ),
    )


def capture_posts(monkeypatch) -> list[list[dict]]:
    batches: list[list[dict]] = []
    monkeypatch.setattr(
        receipt_sync,
        "post_receipts",
        lambda client, receipts: batches.append(receipts),
    )
    return batches


def receipt_line(number: int) -> bytes:
    return (
        json.dumps({"receipt_id": f"{number:032x}", "event": "accepted"})
        + "\n"
    ).encode()


def test_pending_receipts_leaves_unterminated_tail(tmp_path) -> None:
    path = tmp_path / "receipts-2026-07-20.jsonl"
    path.write_bytes(receipt_line(1) + receipt_line(2) + b'{"partial": 1')

    receipts, offset = receipt_sync.pending_receipts(path, 0)

    assert [r["receipt_id"] for r in receipts] == [f"{1:032x}", f"{2:032x}"]
    assert offset == len(receipt_line(1) + receipt_line(2))

    # Once the tail is completed, the next run picks it up from the offset.
    path.write_bytes(path.read_bytes() + b"}\n")
    receipts, offset = receipt_sync.pending_receipts(path, offset)
    assert receipts == [{"partial": 1}]
    assert offset == path.stat().st_size


def test_pending_receipts_skips_corrupt_lines(tmp_path) -> None:
    path = tmp_path / "receipts-2026-07-20.jsonl"
    path.write_bytes(b"garbage\n" + receipt_line(7))

    receipts, offset = receipt_sync.pending_receipts(path, 0)

    assert [r["receipt_id"] for r in receipts] == [f"{7:032x}"]
    assert offset == path.stat().st_size


def test_main_ships_batches_and_saves_offsets(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path, receipt_sync_batch_size=2)
    batches = capture_posts(monkeypatch)
    path = tmp_path / "receipts-2026-07-20.jsonl"
    path.write_bytes(receipt_line(1) + receipt_line(2) + receipt_line(3))

    receipt_sync.main()

    assert [len(batch) for batch in batches] == [2, 1]
    state = json.loads((tmp_path / "sync_state.json").read_text())
    assert state[path.name] == path.stat().st_size


def test_main_resumes_from_saved_offset(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path)
    batches = capture_posts(monkeypatch)
    path = tmp_path / "receipts-2026-07-20.jsonl"
    path.write_bytes(receipt_line(1))
    (tmp_path / "sync_state.json").write_text(
        json.dumps({path.name: path.stat().st_size})
    )

    receipt_sync.main()

    assert batches == []


def test_main_prunes_old_synced_files(monkeypatch, tmp_path) -> None:
    configure(monkeypatch, tmp_path, receipt_retention_days=14)
    capture_posts(monkeypatch)
    old = tmp_path / "receipts-2000-01-01.jsonl"
    old.write_bytes(receipt_line(1))
    fresh = tmp_path / "receipts-2026-07-20.jsonl"
    fresh.write_bytes(receipt_line(2))

    receipt_sync.main()

    assert not old.exists()
    assert fresh.exists()
    state = json.loads((tmp_path / "sync_state.json").read_text())
    assert old.name not in state
    assert state[fresh.name] == fresh.stat().st_size
