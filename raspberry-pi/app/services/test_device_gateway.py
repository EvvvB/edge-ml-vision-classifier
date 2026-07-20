from __future__ import annotations

from app.services import device_gateway


def reset_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        device_gateway,
        "device_state_path",
        lambda: tmp_path / "device_state.json",
    )
    monkeypatch.setattr(device_gateway, "_desired_states", {})
    monkeypatch.setattr(device_gateway, "_states_loaded", False)


def test_desired_state_defaults_to_automated(tmp_path, monkeypatch) -> None:
    reset_state(tmp_path, monkeypatch)
    assert device_gateway.cached_desired_state("nicla-vision-01") == {
        "mode": "automated",
        "seq": 0,
    }


def test_desired_state_persists_and_reloads(tmp_path, monkeypatch) -> None:
    reset_state(tmp_path, monkeypatch)
    device_gateway.remember_desired_state("nicla-vision-01", "positioning", 3)

    # A fresh process should read the persisted state back from disk.
    monkeypatch.setattr(device_gateway, "_desired_states", {})
    monkeypatch.setattr(device_gateway, "_states_loaded", False)
    assert device_gateway.cached_desired_state("nicla-vision-01") == {
        "mode": "positioning",
        "seq": 3,
    }


def test_desired_state_ignores_stale_seq(tmp_path, monkeypatch) -> None:
    reset_state(tmp_path, monkeypatch)
    device_gateway.remember_desired_state("nicla-vision-01", "positioning", 5)
    device_gateway.remember_desired_state("nicla-vision-01", "automated", 4)
    assert device_gateway.cached_desired_state("nicla-vision-01") == {
        "mode": "positioning",
        "seq": 5,
    }


def test_seen_relay_due_rate_limits_per_device() -> None:
    last_relayed: dict[str, float] = {}
    assert device_gateway.seen_relay_due("a", 100.0, 300.0, last_relayed)

    last_relayed["a"] = 100.0
    assert not device_gateway.seen_relay_due("a", 250.0, 300.0, last_relayed)
    assert device_gateway.seen_relay_due("a", 400.0, 300.0, last_relayed)
    # Other devices keep their own clocks.
    assert device_gateway.seen_relay_due("b", 250.0, 300.0, last_relayed)
