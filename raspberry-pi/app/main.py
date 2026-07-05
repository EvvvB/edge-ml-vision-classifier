from __future__ import annotations

from fastapi import FastAPI

from app.api.detections import router as detections_router


app = FastAPI(title="Edge ML Vision Classifier")
app.include_router(detections_router)
