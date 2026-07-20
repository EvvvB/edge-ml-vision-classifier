from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.detections import router as detections_router
from app.api.devices import router as devices_router
from app.config import settings
from app.services.capture_relay import capture_stream_worker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Without a cloud API there is nothing to subscribe to; the Pi keeps
    # working standalone, matching the forwarder's behavior.
    capture_task = (
        asyncio.create_task(capture_stream_worker())
        if settings.cloud_api_url
        else None
    )
    try:
        yield
    finally:
        if capture_task is not None:
            capture_task.cancel()
            with suppress(asyncio.CancelledError):
                await capture_task


app = FastAPI(title="Edge ML Vision Classifier", lifespan=lifespan)
app.include_router(detections_router)
app.include_router(devices_router)
