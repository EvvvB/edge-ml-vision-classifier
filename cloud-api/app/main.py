from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.captures import router as captures_router
from app.api.detections import router as detections_router
from app.config import settings
from app.services.capture_service import CaptureBroadcaster
from app.storage.postgres import close_db_pool, create_db_pool, init_db
from app.storage.s3 import create_s3_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.db = await create_db_pool()
    try:
        await init_db(app.state.db)
        app.state.s3_client = create_s3_client()
        app.state.capture_broadcaster = CaptureBroadcaster()
        yield
    finally:
        await close_db_pool(app.state.db)


app = FastAPI(title="Edge ML Vision Cloud API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(detections_router)
app.include_router(captures_router)

# The built frontend (frontend/ -> npm run build) lands in static/; serve it
# from the same origin as the API. API routes above take precedence over the
# mount. In development the directory is absent and vite serves the app.
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
