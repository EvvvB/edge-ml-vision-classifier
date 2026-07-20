from __future__ import annotations

import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import AsyncIterator

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Device address registry
# ---------------------------------------------------------
#
# The Nicla gets its address from DHCP, so instead of configuring it the Pi
# remembers the source address of each device's most recent upload. The map
# is persisted so a Pi restart does not lose the target before the next
# upload arrives.

_device_addresses: dict[str, str] = {}
_addresses_loaded = False


def device_addresses_path() -> Path:
    return settings.metadata_dir / "device_addresses.json"


def _load_addresses() -> None:
    global _addresses_loaded
    if _addresses_loaded:
        return
    _addresses_loaded = True
    try:
        stored = json.loads(device_addresses_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(stored, dict):
        _device_addresses.update(
            (str(key), str(value)) for key, value in stored.items()
        )


def remember_device_address(device_id: str, host: str) -> None:
    _load_addresses()
    if _device_addresses.get(device_id) == host:
        return
    _device_addresses[device_id] = host
    try:
        path = device_addresses_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_device_addresses, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("could not persist device addresses: %s", exc)


def known_device_address(device_id: str) -> str | None:
    _load_addresses()
    return _device_addresses.get(device_id)


# ---------------------------------------------------------
# UDP trigger
# ---------------------------------------------------------


def resolve_trigger_host() -> str | None:
    if settings.nicla_udp_host:
        return settings.nicla_udp_host
    return known_device_address(settings.capture_device_id)


async def send_capture_trigger(counter: int) -> bool:
    """Fire the capture datagram at the Nicla a few times.

    The payload is the cloud's monotonic press counter; the firmware keeps a
    high-water mark, so repeats and stale packets are harmless.
    """
    host = resolve_trigger_host()
    if not host:
        logger.warning(
            "capture %s requested but no address is known for %s yet",
            counter,
            settings.capture_device_id,
        )
        return False

    payload = f"snap:{counter}".encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for repeat in range(max(1, settings.capture_udp_repeats)):
            if repeat > 0:
                # A short gap decorrelates the repeats from a single burst
                # of interference.
                await asyncio.sleep(0.05)
            sock.sendto(payload, (host, settings.nicla_udp_port))
    except OSError as exc:
        logger.warning("capture trigger send failed: %s", exc)
        return False
    finally:
        sock.close()

    logger.info(
        "capture trigger %s sent to %s:%s", counter, host, settings.nicla_udp_port
    )
    return True


# ---------------------------------------------------------
# Cloud SSE subscription
# ---------------------------------------------------------


def parse_capture_event(data_lines: list[str]) -> dict | None:
    """Parse one SSE event's data into {counter, mode, mode_seq}.

    counter is required; mode fields are optional so the parser keeps
    working against a cloud API that predates modes.
    """
    if not data_lines:
        return None
    try:
        payload = json.loads("\n".join(data_lines))
        event = {"counter": int(payload["counter"])}
    except (ValueError, TypeError, KeyError):
        return None

    mode = payload.get("mode")
    mode_seq = payload.get("mode_seq")
    if mode in ("automated", "positioning") and mode_seq is not None:
        try:
            event["mode"] = mode
            event["mode_seq"] = int(mode_seq)
        except (ValueError, TypeError):
            event.pop("mode", None)
    return event


async def iter_sse_events(lines: AsyncIterator[str]) -> AsyncIterator[dict]:
    """Yield parsed events from a raw SSE line stream.

    Heartbeat comments (leading ':') and unknown fields are ignored; an
    event's data lines are parsed when the blank separator line arrives.
    """
    data_lines: list[str] = []
    async for line in lines:
        if line == "":
            event = parse_capture_event(data_lines)
            data_lines = []
            if event is not None:
                yield event
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())


async def capture_stream_worker() -> None:
    """Hold the capture stream open forever, reconnecting with backoff.

    The first counter seen after process start only sets the baseline;
    afterwards any increase is relayed to the Nicla. Reconnect snapshots
    replay the current counter, so presses made during a brief disconnect
    are still relayed once the stream returns.

    Desired mode rides the same events. Unlike the counter, the snapshot is
    NOT baselined away: desired state must converge on every (re)connect,
    and the push layer's seq high-water makes re-applying it harmless.
    """
    # Imported here because device_gateway imports this module for the
    # address registry.
    from app.services.device_gateway import apply_desired_mode
    stream_url = (
        settings.cloud_api_url.rstrip("/")
        + f"/devices/{settings.capture_device_id}/capture/stream"
    )
    headers = {}
    if settings.cloud_api_key:
        headers["X-API-Key"] = settings.cloud_api_key

    timeout = httpx.Timeout(
        connect=10.0,
        read=settings.capture_stream_read_timeout_seconds,
        write=10.0,
        pool=10.0,
    )

    last_relayed: int | None = None
    backoff_seconds = 1.0

    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("GET", stream_url, headers=headers) as response:
                    response.raise_for_status()
                    logger.info("capture stream connected: %s", stream_url)
                    backoff_seconds = 1.0
                    async for event in iter_sse_events(response.aiter_lines()):
                        if "mode" in event:
                            apply_desired_mode(
                                settings.capture_device_id,
                                event["mode"],
                                event["mode_seq"],
                            )
                        counter = event["counter"]
                        if last_relayed is None:
                            last_relayed = counter
                            continue
                        if counter > last_relayed:
                            last_relayed = counter
                            await send_capture_trigger(counter)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("capture stream disconnected: %s", exc)

        await asyncio.sleep(backoff_seconds)
        backoff_seconds = min(backoff_seconds * 2, 60.0)
