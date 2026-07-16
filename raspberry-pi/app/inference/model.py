from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


INFERENCE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = INFERENCE_DIR / "models" / "yolo26m.pt"
RUNS_DIR = INFERENCE_DIR / "runs"

# The Nicla sends HVGA frames, so 480 is the long side of what actually
# arrives. Ultralytics scales the long side up to imgsz, so a 640 default
# would upscale every frame past its own resolution before inference.
DEFAULT_IMAGE_SIZE = 480

os.environ.setdefault("YOLO_CONFIG_DIR", str(INFERENCE_DIR / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(INFERENCE_DIR / ".matplotlib"))


@lru_cache(maxsize=1)
def load_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    from ultralytics import YOLO

    return YOLO(str(model_path))


def predict_image(
    image_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    imgsz: int = DEFAULT_IMAGE_SIZE,
    device: str = "cpu",
    save: bool = False,
) -> list[dict[str, Any]]:
    model = load_model(model_path)
    results = model(
        str(image_path),
        imgsz=imgsz,
        device=device,
        save=save,
        project=str(RUNS_DIR),
    )
    return summarize_result(results[0])


def summarize_result(result: Any) -> list[dict[str, Any]]:
    return result.summary()
