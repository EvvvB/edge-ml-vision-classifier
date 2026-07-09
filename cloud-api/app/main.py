from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.detections import router as detections_router
from app.config import settings
from app.storage.postgres import close_db_pool, create_db_pool, init_db
from app.storage.s3 import create_s3_client


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.db = await create_db_pool()
    try:
        await init_db(app.state.db)
        app.state.s3_client = create_s3_client()
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
