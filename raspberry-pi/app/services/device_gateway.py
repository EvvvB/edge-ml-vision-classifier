from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.services.capture_relay import (
    known_device_address,
    remember_device_address,
)


logger = logging.getLogger(__name__)

DEFAULT_DESIRED_STATE = {"mode": "automated", "seq": 0}


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def cloud_url(path: str) -> str:
    return settings.cloud_api_url.rstrip("/") + path


def cloud_headers() -> dict[str, str]:
    if settings.cloud_api_key:
        return {"X-API-Key": settings.cloud_api_key}
    return {}


# ---------------------------------------------------------
# Desired-state cache
# ---------------------------------------------------------
#
# The cloud owns desired mode; this cache is what lets the Pi answer a
# camera's boot hello while the WAN is down. Persisted like the device
# address map so a Pi restart does not forget it.

_desired_states: dict[str, dict[str, Any]] = {}
_states_loaded = False


def device_state_path() -> Path:
    return settings.metadata_dir / "device_state.json"


def _load_states() -> None:
    global _states_loaded
    if _states_loaded:
        return
    _states_loaded = True
    try:
        stored = json.loads(device_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(stored, dict):
        for device_id, state in stored.items():
            if isinstance(state, dict):
                _desired_states[str(device_id)] = state


def cached_desired_state(device_id: str) -> dict[str, Any]:
    _load_states()
    state = _desired_states.get(device_id) or {}
    cached = {
        "mode": state.get("mode") or DEFAULT_DESIRED_STATE["mode"],
        "seq": int(state.get("seq") or 0),
    }
    config = state.get("config")
    if isinstance(config, dict):
        cached["config"] = config
        cached["config_seq"] = int(state.get("config_seq") or 0)
    return cached


def _persist_states() -> None:
    try:
        path = device_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_desired_states, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("could not persist device states: %s", exc)


def remember_desired_state(device_id: str, mode: str, seq: int) -> None:
    _load_states()
    current = _desired_states.setdefault(device_id, {})
    if int(current.get("seq") or 0) >= seq:
        return
    current["mode"] = mode
    current["seq"] = seq
    _persist_states()


def remember_desired_config(
    device_id: str, config: dict[str, Any], seq: int
) -> None:
    # Config versions independently of mode: each has its own seq lane in
    # the same per-device cache entry.
    _load_states()
    current = _desired_states.setdefault(device_id, {})
    if int(current.get("config_seq") or 0) >= seq:
        return
    current["config"] = config
    current["config_seq"] = seq
    _persist_states()


# ---------------------------------------------------------
# Hello relay
# ---------------------------------------------------------

_pending_hello_tasks: dict[str, asyncio.Task] = {}


async def handle_hello(
    device_id: str,
    payload: dict[str, Any],
    client_host: str | None,
) -> dict[str, Any]:
    """Answer a camera's boot hello, relaying to the cloud when it can.

    The camera's boot sequence is blocked on this response, so the cloud
    call runs under a short timeout and the cached desired state is the
    fallback answer. A failed relay is retried in the background until the
    registry has the row.
    """
    if client_host:
        remember_device_address(device_id, client_host)

    enriched = {**payload, "pi_id": settings.pi_id}

    if settings.cloud_api_url:
        try:
            async with httpx.AsyncClient(
                timeout=settings.hello_relay_timeout_seconds
            ) as client:
                response = await client.post(
                    cloud_url(f"/devices/{device_id}/hello"),
                    json=enriched,
                    headers=cloud_headers(),
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            logger.warning("hello relay failed for %s: %s", device_id, exc)
            _schedule_hello_retry(device_id, enriched)
        else:
            mode = body.get("mode") or DEFAULT_DESIRED_STATE["mode"]
            seq = int(body.get("seq") or 0)
            remember_desired_state(device_id, mode, seq)
            answer = {
                "ok": True,
                "mode": mode,
                "seq": seq,
                "server_time": body.get("server_time") or now_iso(),
            }
            config = body.get("config")
            if isinstance(config, dict):
                config_seq = int(body.get("config_seq") or 0)
                remember_desired_config(device_id, config, config_seq)
                answer["config"] = config
                answer["config_seq"] = config_seq
            return answer

    state = cached_desired_state(device_id)
    # The Pi keeps NTP time, so its clock is a fine stand-in for the
    # cloud's when answering from cache.
    answer = {
        "ok": True,
        "mode": state["mode"],
        "seq": state["seq"],
        "server_time": now_iso(),
    }
    if "config" in state:
        answer["config"] = state["config"]
        answer["config_seq"] = state["config_seq"]
    return answer


def _schedule_hello_retry(device_id: str, payload: dict[str, Any]) -> None:
    existing = _pending_hello_tasks.get(device_id)
    if existing is not None and not existing.done():
        existing.cancel()
    _pending_hello_tasks[device_id] = asyncio.create_task(
        _retry_hello(device_id, payload)
    )


async def _retry_hello(device_id: str, payload: dict[str, Any]) -> None:
    delay = 5.0
    while True:
        await asyncio.sleep(delay)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    cloud_url(f"/devices/{device_id}/hello"),
                    json=payload,
                    headers=cloud_headers(),
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:
            logger.warning(
                "hello retry failed for %s: %s", device_id, exc
            )
            delay = min(delay * 2, 300.0)
            continue
        remember_desired_state(
            device_id,
            body.get("mode") or DEFAULT_DESIRED_STATE["mode"],
            int(body.get("seq") or 0),
        )
        if isinstance(body.get("config"), dict):
            remember_desired_config(
                device_id,
                body["config"],
                int(body.get("config_seq") or 0),
            )
        logger.info("hello for %s reached the cloud", device_id)
        return


# ---------------------------------------------------------
# Desired-mode push (UDP, retried until acked)
# ---------------------------------------------------------

_acked_seqs: dict[str, int] = {}
_mode_push_tasks: dict[str, asyncio.Task] = {}


def resolve_device_host(device_id: str) -> str | None:
    if settings.nicla_udp_host:
        return settings.nicla_udp_host
    return known_device_address(device_id)


def send_mode_datagram(device_id: str, mode: str, seq: int) -> bool:
    host = resolve_device_host(device_id)
    if not host:
        logger.warning(
            "mode %s:%s pending but no address is known for %s yet",
            seq,
            mode,
            device_id,
        )
        return False

    payload = f"mode:{seq}:{mode}".encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, (host, settings.nicla_udp_port))
    except OSError as exc:
        logger.warning("mode datagram send failed: %s", exc)
        return False
    finally:
        sock.close()
    return True


def apply_desired_mode(device_id: str, mode: str, seq: int) -> None:
    """Adopt a desired state seen on the SSE stream and push until acked.

    Traffic is proportional to state changes: once the device acks the
    seq (or a later one), the push loop exits and the LAN goes quiet.
    """
    remember_desired_state(device_id, mode, seq)
    if _acked_seqs.get(device_id, 0) >= seq:
        return

    existing = _mode_push_tasks.get(device_id)
    if existing is not None and not existing.done():
        existing.cancel()
    _mode_push_tasks[device_id] = asyncio.create_task(
        _push_mode_until_acked(device_id, mode, seq)
    )


async def _push_mode_until_acked(device_id: str, mode: str, seq: int) -> None:
    delay = settings.mode_push_retry_base_seconds
    while _acked_seqs.get(device_id, 0) < seq:
        send_mode_datagram(device_id, mode, seq)
        await asyncio.sleep(delay)
        delay = min(delay * 2, settings.mode_push_retry_max_seconds)
    logger.info("mode %s:%s acked by %s", seq, mode, device_id)


async def handle_mode_ack(
    device_id: str,
    mode: str,
    seq: int,
    client_host: str | None,
) -> dict[str, Any]:
    if client_host:
        remember_device_address(device_id, client_host)
    _acked_seqs[device_id] = max(_acked_seqs.get(device_id, 0), seq)
    asyncio.create_task(_relay_mode_ack(device_id, mode, seq))
    return {"ok": True}


async def _relay_mode_ack(device_id: str, mode: str, seq: int) -> None:
    if not settings.cloud_api_url:
        return
    delay = 2.0
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    cloud_url(f"/devices/{device_id}/mode-ack"),
                    json={"mode": mode, "seq": seq},
                    headers=cloud_headers(),
                )
                response.raise_for_status()
            return
        except Exception as exc:
            logger.warning("mode-ack relay failed for %s: %s", device_id, exc)
            await asyncio.sleep(delay)
            delay *= 2


# ---------------------------------------------------------
# Desired-config push (UDP, retried until acked)
# ---------------------------------------------------------
#
# Same seq/ack machinery as modes, in its own lane: a config datagram
# carries the full desired-config JSON, so re-sends and reordering are
# harmless and the newest seq always wins on the device.

_config_acked_seqs: dict[str, int] = {}
_config_push_tasks: dict[str, asyncio.Task] = {}


def send_config_datagram(
    device_id: str, config: dict[str, Any], seq: int
) -> bool:
    host = resolve_device_host(device_id)
    if not host:
        logger.warning(
            "config %s pending but no address is known for %s yet",
            seq,
            device_id,
        )
        return False

    payload = f"config:{seq}:{json.dumps(config)}".encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(payload, (host, settings.nicla_udp_port))
    except OSError as exc:
        logger.warning("config datagram send failed: %s", exc)
        return False
    finally:
        sock.close()
    return True


def apply_desired_config(
    device_id: str, config: dict[str, Any], seq: int
) -> None:
    """Adopt a desired config seen on the SSE stream and push until acked."""
    remember_desired_config(device_id, config, seq)
    if _config_acked_seqs.get(device_id, 0) >= seq:
        return

    existing = _config_push_tasks.get(device_id)
    if existing is not None and not existing.done():
        existing.cancel()
    _config_push_tasks[device_id] = asyncio.create_task(
        _push_config_until_acked(device_id, config, seq)
    )


async def _push_config_until_acked(
    device_id: str, config: dict[str, Any], seq: int
) -> None:
    delay = settings.mode_push_retry_base_seconds
    while _config_acked_seqs.get(device_id, 0) < seq:
        send_config_datagram(device_id, config, seq)
        await asyncio.sleep(delay)
        delay = min(delay * 2, settings.mode_push_retry_max_seconds)
    logger.info("config %s acked by %s", seq, device_id)


async def handle_config_ack(
    device_id: str,
    config: dict[str, Any],
    seq: int,
    client_host: str | None,
) -> dict[str, Any]:
    if client_host:
        remember_device_address(device_id, client_host)
    _config_acked_seqs[device_id] = max(
        _config_acked_seqs.get(device_id, 0), seq
    )
    asyncio.create_task(_relay_config_ack(device_id, config, seq))
    return {"ok": True}


async def _relay_config_ack(
    device_id: str, config: dict[str, Any], seq: int
) -> None:
    if not settings.cloud_api_url:
        return
    delay = 2.0
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    cloud_url(f"/devices/{device_id}/config-ack"),
                    json={"config": config, "seq": seq},
                    headers=cloud_headers(),
                )
                response.raise_for_status()
            return
        except Exception as exc:
            logger.warning(
                "config-ack relay failed for %s: %s", device_id, exc
            )
            await asyncio.sleep(delay)
            delay *= 2


# ---------------------------------------------------------
# Presence ticks
# ---------------------------------------------------------

_last_seen_relay: dict[str, float] = {}


def seen_relay_due(
    device_id: str,
    now: float,
    interval: float,
    last_relayed: dict[str, float],
) -> bool:
    last = last_relayed.get(device_id)
    return last is None or now - last >= interval


async def handle_tick(
    device_id: str,
    client_host: str | None,
) -> dict[str, Any]:
    if client_host:
        remember_device_address(device_id, client_host)
    now = time.monotonic()
    if settings.cloud_api_url and seen_relay_due(
        device_id, now, settings.seen_relay_interval_seconds, _last_seen_relay
    ):
        _last_seen_relay[device_id] = now
        asyncio.create_task(_relay_seen(device_id))
    return {"ok": True}


async def _relay_seen(device_id: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                cloud_url(f"/devices/{device_id}/seen"),
                headers=cloud_headers(),
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning("seen relay failed for %s: %s", device_id, exc)


# ---------------------------------------------------------
# Positioning preview
# ---------------------------------------------------------

_preview_frames: dict[str, dict[str, Any]] = {}
_last_preview_forward: dict[str, float] = {}


def latest_preview(device_id: str) -> dict[str, Any] | None:
    return _preview_frames.get(device_id)


async def handle_preview(
    device_id: str,
    body: bytes,
    content_type: str | None,
    client_host: str | None,
) -> dict[str, Any]:
    if client_host:
        remember_device_address(device_id, client_host)
    _preview_frames[device_id] = {
        "body": body,
        "content_type": content_type or "image/jpeg",
        "received_at": time.monotonic(),
    }
    now = time.monotonic()
    last = _last_preview_forward.get(device_id)
    if settings.cloud_api_url and (
        last is None
        or now - last >= settings.preview_forward_min_interval_seconds
    ):
        _last_preview_forward[device_id] = now
        asyncio.create_task(
            _forward_preview(device_id, body, content_type or "image/jpeg")
        )
    return {"ok": True}


async def _forward_preview(
    device_id: str,
    body: bytes,
    content_type: str,
) -> None:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                cloud_url(f"/devices/{device_id}/preview"),
                content=body,
                headers={**cloud_headers(), "Content-Type": content_type},
            )
            response.raise_for_status()
    except Exception as exc:
        logger.warning("preview forward failed for %s: %s", device_id, exc)
