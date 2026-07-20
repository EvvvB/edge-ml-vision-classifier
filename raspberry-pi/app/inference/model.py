from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


INFERENCE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = INFERENCE_DIR / "models" / "yolo26m.pt"
RUNS_DIR = INFERENCE_DIR / "runs"

# Same convention as the Nicla firmware: a truncated SHA-256 of the deployed
# weights uniquely identifies the handful of models this project will ever
# run, and keeps upload metadata small.
MODEL_HASH_HEX_CHARS = 12

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


def manifest_path_for(model_path: str | Path) -> Path:
    return Path(model_path).with_suffix(".manifest.json")


@lru_cache(maxsize=4)
def model_identity(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Identity of the deployed model, stamped onto every detection.

    The hash is the ground-truth identity: two records share a hash only when
    byte-identical weights produced them, even if the manifest was forgotten
    or wrong. The manifest carries what the hash cannot: version label,
    training date, notes. It lives next to the weights file
    (models/yolo26m.manifest.json) so the two cannot drift apart.
    """
    return {
        "hash": compute_model_hash(model_path),
        "manifest": load_model_manifest(manifest_path_for(model_path)),
    }


def compute_model_hash(model_path: str | Path) -> str | None:
    try:
        sha = hashlib.sha256()
        with open(model_path, "rb") as model_file:
            while chunk := model_file.read(1 << 20):
                sha.update(chunk)
        return sha.hexdigest()[:MODEL_HASH_HEX_CHARS]
    except OSError:
        return None


def load_model_manifest(manifest_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


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
