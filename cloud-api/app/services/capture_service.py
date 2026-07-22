from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import asyncpg

from app.config import settings
from app.storage.postgres import (
    expire_stale_positioning,
    fetch_capture_counter,
    fetch_desired_config,
    fetch_desired_mode,
    increment_capture_counter,
)


# Comment lines keep NAT tables and proxies from culling an idle stream and
# let the client detect a dead connection with a read timeout.
HEARTBEAT_SECONDS = 20.0


class CaptureBroadcaster:
    """In-process fan-out from capture requests to open SSE streams.

    The API runs as a single uvicorn worker, so an asyncio.Condition per
    device is enough. Notifications are only a wake-up hint: stream handlers
    re-read the counter from Postgres after every wake, so a notification
    lost between yields is recovered on the next heartbeat.

    Subscriber counts double as gateway liveness: the Pi is the only client
    that holds a device's stream open, so a subscribed stream means that
    device's gateway is up.
    """

    def __init__(self) -> None:
        self._conditions: dict[str, asyncio.Condition] = {}
        self._subscribers: dict[str, int] = {}

    def _condition(self, device_id: str) -> asyncio.Condition:
        if device_id not in self._conditions:
            self._conditions[device_id] = asyncio.Condition()
        return self._conditions[device_id]

    async def notify(self, device_id: str) -> None:
        condition = self._condition(device_id)
        async with condition:
            condition.notify_all()

    async def wait_for_change(self, device_id: str, timeout: float) -> None:
        condition = self._condition(device_id)
        async with condition:
            try:
                await asyncio.wait_for(condition.wait(), timeout)
            except TimeoutError:
                pass

    def subscribe(self, device_id: str) -> None:
        self._subscribers[device_id] = self._subscribers.get(device_id, 0) + 1

    def unsubscribe(self, device_id: str) -> None:
        remaining = self._subscribers.get(device_id, 0) - 1
        if remaining > 0:
            self._subscribers[device_id] = remaining
        else:
            self._subscribers.pop(device_id, None)

    def has_subscriber(self, device_id: str) -> bool:
        return self._subscribers.get(device_id, 0) > 0


async def request_capture(
    pool: asyncpg.Pool,
    broadcaster: CaptureBroadcaster,
    device_id: str,
) -> dict[str, int | str]:
    counter = await increment_capture_counter(pool, device_id)
    await broadcaster.notify(device_id)
    return {"device_id": device_id, "counter": counter}


def sse_event(
    device_id: str,
    counter: int,
    mode: str,
    mode_seq: int,
    config: dict | None,
    config_seq: int,
) -> str:
    payload = {
        "device_id": device_id,
        "counter": counter,
        "mode": mode,
        "mode_seq": mode_seq,
    }
    if config is not None:
        payload["config"] = config
        payload["config_seq"] = config_seq
    return f"data: {json.dumps(payload)}\n\n"


async def read_stream_state(
    pool: asyncpg.Pool,
    device_id: str,
) -> tuple[int, str, int, dict | None, int]:
    # TTL expiry has no HTTP trigger of its own, so this read path (hit at
    # least every heartbeat) is what notices a lapsed positioning mode and
    # turns it into a pushed change.
    await expire_stale_positioning(
        pool, settings.positioning_ttl_seconds, device_id
    )
    counter = await fetch_capture_counter(pool, device_id)
    mode, mode_seq = await fetch_desired_mode(pool, device_id)
    config, config_seq = await fetch_desired_config(pool, device_id)
    return counter, mode, mode_seq, config, config_seq


async def capture_event_stream(
    pool: asyncpg.Pool,
    broadcaster: CaptureBroadcaster,
    device_id: str,
) -> AsyncIterator[str]:
    """SSE stream of the device's capture counter, desired mode, and
    desired config.

    The first event is a snapshot so a reconnecting client can re-baseline
    the counter and converge on the current desired state; afterwards an
    event is sent whenever any of the three advances.
    """
    broadcaster.subscribe(device_id)
    try:
        state = await read_stream_state(pool, device_id)
        last_counter, last_mode_seq, last_config_seq = (
            state[0],
            state[2],
            state[4],
        )
        yield sse_event(device_id, *state)

        while True:
            await broadcaster.wait_for_change(
                device_id, timeout=HEARTBEAT_SECONDS
            )
            state = await read_stream_state(pool, device_id)
            counter, mode_seq, config_seq = state[0], state[2], state[4]
            if (
                counter > last_counter
                or mode_seq > last_mode_seq
                or config_seq > last_config_seq
            ):
                last_counter = counter
                last_mode_seq = mode_seq
                last_config_seq = config_seq
                yield sse_event(device_id, *state)
            else:
                yield ": keepalive\n\n"
    finally:
        broadcaster.unsubscribe(device_id)
