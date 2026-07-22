from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.inference.model import MODELS_DIR, model_identity

# Teacher inference runs offline with no latency budget, so every teacher
# is configured for recall: a low confidence floor (the cloud stores raw
# scores and applies its own threshold at scoring time, so this can stay
# permissive) and test-time augmentation where the architecture supports
# it. 640 covers the current uploads; raise imgsz when captures move to
# the sensor's full 1600x1200.


@dataclass(frozen=True)
class TeacherConfig:
    source: str            # teacher_source slug the cloud groups eval rows by
    weights_name: str      # file under inference/models/, auto-downloaded
    loader: str            # 'yolo' | 'rtdetr'
    imgsz: int = 640
    conf: float = 0.1
    augment: bool = False

    @property
    def weights_path(self) -> Path:
        return MODELS_DIR / self.weights_name

    def identity(self) -> dict[str, Any]:
        return model_identity(self.weights_path)

    def load(self):
        ensure_weights(self)
        if self.loader == "rtdetr":
            from ultralytics import RTDETR

            return RTDETR(str(self.weights_path))
        from ultralytics import YOLO

        return YOLO(str(self.weights_path))


# Neither teacher gets test-time augmentation: YOLO26's end-to-end head
# and RT-DETR both lack TTA support in Ultralytics (augment=True just
# logs a warning per image and reverts).
TEACHERS: tuple[TeacherConfig, ...] = (
    TeacherConfig(
        source="yolo26x",
        weights_name="yolo26x.pt",
        loader="yolo",
    ),
    # A DETR-family second opinion: its misses are uncorrelated with the
    # YOLO models', which is what makes teacher disagreement informative.
    TeacherConfig(
        source="rtdetr-x",
        weights_name="rtdetr-x.pt",
        loader="rtdetr",
    ),
)


def teachers_by_source(sources: list[str] | None) -> list[TeacherConfig]:
    if not sources:
        return list(TEACHERS)
    by_source = {teacher.source: teacher for teacher in TEACHERS}
    unknown = [source for source in sources if source not in by_source]
    if unknown:
        known = ", ".join(sorted(by_source))
        raise SystemExit(f"unknown teacher(s): {', '.join(unknown)} (known: {known})")
    return [by_source[source] for source in sources]


def ensure_weights(teacher: TeacherConfig) -> None:
    """Download official weights on first use, into inference/models/.

    Ultralytics resolves a bare weights filename by downloading the
    official release asset into the current working directory, so chdir
    into the models dir for the duration.
    """
    if teacher.weights_path.exists():
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with contextlib.chdir(MODELS_DIR):
        teacher_class = None
        if teacher.loader == "rtdetr":
            from ultralytics import RTDETR as teacher_class
        else:
            from ultralytics import YOLO as teacher_class
        teacher_class(teacher.weights_name)
    if not teacher.weights_path.exists():
        raise FileNotFoundError(
            f"weights for {teacher.source} did not appear at {teacher.weights_path}"
        )
    # A downloaded weights file must never drift apart from its committed
    # manifest, so recompute identity now rather than trusting the cache.
    model_identity.cache_clear()
