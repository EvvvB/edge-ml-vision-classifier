from __future__ import annotations

from typing import Any

FOMO_SOURCE = "fomo"
YOLO_SOURCE = "yolo"

# COCO stores a box as [x, y, width, height] anchored at the top-left corner.
# The Nicla already reports boxes that way, so FOMO boxes pass through
# unconverted. Ultralytics reports {x1, y1, x2, y2}, which does need converting.

# COCO category names are lowercase. The Edge Impulse model reports "Dog" while
# YOLO reports "dog", so labels are folded to lowercase and the two models share
# one vocabulary.


def normalize_fomo_detections(detections: Any) -> list[dict[str, Any]]:
    """Nicla FOMO detections to the shared COCO-style detection schema."""
    if not isinstance(detections, list):
        return []

    normalized = (normalize_fomo_detection(entry) for entry in detections)
    return [entry for entry in normalized if entry is not None]


def normalize_yolo_detections(detections: Any) -> list[dict[str, Any]]:
    """Ultralytics summary() output to the shared COCO-style detection schema."""
    if not isinstance(detections, list):
        return []

    normalized = (normalize_yolo_detection(entry) for entry in detections)
    return [entry for entry in normalized if entry is not None]


def normalize_fomo_detection(detection: Any) -> dict[str, Any] | None:
    if not isinstance(detection, dict):
        return None

    bbox = coco_bbox_from_xywh(detection.get("box"))
    confidence = optional_float(detection.get("score"))
    if bbox is None or confidence is None:
        return None

    normalized: dict[str, Any] = {
        "source": FOMO_SOURCE,
        "label": normalize_label(detection.get("label")),
        "confidence": confidence,
        "bbox": bbox,
    }

    # Tile and center are FOMO-specific and worth keeping: the tile explains
    # which inference produced the detection, and FOMO's centroid is more
    # meaningful than its blob extents.
    if isinstance(detection.get("tile"), int):
        normalized["tile"] = detection["tile"]

    center = detection.get("center")
    if isinstance(center, (list, tuple)) and len(center) == 2:
        if all(isinstance(value, (int, float)) for value in center):
            normalized["center"] = list(center)

    return normalized


def normalize_yolo_detection(detection: Any) -> dict[str, Any] | None:
    if not isinstance(detection, dict):
        return None

    bbox = coco_bbox_from_xyxy(detection.get("box"))
    confidence = optional_float(detection.get("confidence"))
    if bbox is None or confidence is None:
        return None

    normalized: dict[str, Any] = {
        "source": YOLO_SOURCE,
        "label": normalize_label(detection.get("name")),
        "confidence": confidence,
        "bbox": bbox,
    }

    # The Ultralytics class index is already a COCO category id.
    if isinstance(detection.get("class"), int):
        normalized["category_id"] = detection["class"]

    return normalized


def coco_bbox_from_xywh(box: Any) -> list[float] | None:
    """Pass through a [x, y, width, height] box, which is COCO's convention."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not all(isinstance(value, (int, float)) for value in box):
        return None
    return list(box)


def coco_bbox_from_xyxy(box: Any) -> list[float] | None:
    """Convert a {x1, y1, x2, y2} corner box to COCO [x, y, width, height]."""
    if not isinstance(box, dict):
        return None

    corners = [optional_float(box.get(key)) for key in ("x1", "y1", "x2", "y2")]
    if any(corner is None for corner in corners):
        return None

    x1, y1, x2, y2 = corners
    return [x1, y1, x2 - x1, y2 - y1]


def normalize_label(label: Any) -> str:
    return str(label if label is not None else "").strip().lower()


def optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
