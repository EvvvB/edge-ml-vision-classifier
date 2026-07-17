from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import asyncpg

from app.storage.postgres import fetch_capture_counter, increment_capture_counter


# Comment lines keep NAT tables and proxies from culling an idle stream and
# let the client detect a dead connection with a read timeout.
HEARTBEAT_SECONDS = 20.0


class CaptureBroadcaster:
    """In-process fan-out from capture requests to open SSE streams.

    The API runs as a single uvicorn worker, so an asyncio.Condition per
    device is enough. Notifications are only a wake-up hint: stream handlers
    re-read the counter from Postgres after every wake, so a notification
    lost between yields is recovered on the next heartbeat.
    """

    def __init__(self) -> None:
        self._conditions: dict[str, asyncio.Condition] = {}

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


async def request_capture(
    pool: asyncpg.Pool,
    broadcaster: CaptureBroadcaster,
    device_id: str,
) -> dict[str, int | str]:
    counter = await increment_capture_counter(pool, device_id)
    await broadcaster.notify(device_id)
    return {"device_id": device_id, "counter": counter}


def sse_event(device_id: str, counter: int) -> str:
    payload = json.dumps({"device_id": device_id, "counter": counter})
    return f"data: {payload}\n\n"


async def capture_event_stream(
    pool: asyncpg.Pool,
    broadcaster: CaptureBroadcaster,
    device_id: str,
) -> AsyncIterator[str]:
    """SSE stream of the device's capture counter.

    The first event is the current counter so a reconnecting client can
    re-baseline; afterwards an event is sent whenever the counter grows.
    """
    last_sent = await fetch_capture_counter(pool, device_id)
    yield sse_event(device_id, last_sent)

    while True:
        await broadcaster.wait_for_change(device_id, timeout=HEARTBEAT_SECONDS)
        counter = await fetch_capture_counter(pool, device_id)
        if counter > last_sent:
            last_sent = counter
            yield sse_event(device_id, counter)
        else:
            yield ": keepalive\n\n"
