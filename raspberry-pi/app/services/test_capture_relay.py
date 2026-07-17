from __future__ import annotations

import asyncio

import pytest

from app.services import capture_relay


async def stream_of(lines: list[str]):
    for line in lines:
        yield line


def collect_counters(lines: list[str]) -> list[int]:
    async def run() -> list[int]:
        return [
            counter
            async for counter in capture_relay.iter_sse_counters(stream_of(lines))
        ]

    return asyncio.run(run())


def test_parses_counter_events_and_ignores_heartbeats() -> None:
    lines = [
        'data: {"device_id": "nicla-vision-01", "counter": 3}',
        "",
        ": keepalive",
        "",
        'data: {"device_id": "nicla-vision-01", "counter": 4}',
        "",
    ]
    assert collect_counters(lines) == [3, 4]


def test_ignores_malformed_event_data() -> None:
    lines = [
        "data: not json",
        "",
        'data: {"device_id": "nicla-vision-01"}',
        "",
        'data: {"counter": 9}',
        "",
    ]
    assert collect_counters(lines) == [9]


def test_device_addresses_persist_and_reload(tmp_path, monkeypatch) -> None:
    path = tmp_path / "device_addresses.json"
    monkeypatch.setattr(capture_relay, "device_addresses_path", lambda: path)
    monkeypatch.setattr(capture_relay, "_device_addresses", {})
    monkeypatch.setattr(capture_relay, "_addresses_loaded", False)

    capture_relay.remember_device_address("nicla-vision-01", "192.168.50.60")
    assert capture_relay.known_device_address("nicla-vision-01") == "192.168.50.60"

    # A fresh process should read the persisted map back from disk.
    monkeypatch.setattr(capture_relay, "_device_addresses", {})
    monkeypatch.setattr(capture_relay, "_addresses_loaded", False)
    assert capture_relay.known_device_address("nicla-vision-01") == "192.168.50.60"


@pytest.mark.parametrize(
    ("data_lines", "expected"),
    [
        ([], None),
        (['{"counter": "12"}'], 12),
        (['{"counter": null}'], None),
    ],
)
def test_parse_capture_event_edge_cases(data_lines, expected) -> None:
    assert capture_relay.parse_capture_event(data_lines) == expected
