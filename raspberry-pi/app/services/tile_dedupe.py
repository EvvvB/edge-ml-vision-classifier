from __future__ import annotations

import math
from typing import Any, NamedTuple

from app.config import settings


class TileGeometry(NamedTuple):
    frame_width: int
    frame_height: int
    grid_columns: int
    grid_rows: int


def deduplicate_tile_detections(metadata: dict[str, Any]) -> dict[str, Any]:
    """Drop cross-tile duplicates from Nicla FOMO detections.

    The firmware runs non-max suppression per tile, so an object sitting in the
    overlap band between two tiles is reported once by each. FOMO emits centroid
    blobs rather than true bounding boxes, and the two halves of a split object
    often produce boxes that do not overlap at all, so IoU-based suppression
    cannot merge them. Centroid distance can.

    Candidates are limited to same-label detections in adjacent tiles whose
    centers both sit near their shared boundary, which is the only place the
    per-tile suppression can leave a duplicate behind.
    """
    detections = metadata.get("detections")
    if not isinstance(detections, list) or len(detections) < 2:
        return metadata

    geometry = tile_geometry(metadata)
    if geometry is None:
        return metadata

    kept = surviving_detections(detections, geometry)
    removed_count = len(detections) - len(kept)
    if removed_count == 0:
        return metadata

    return {
        **metadata,
        "detections": kept,
        "detection_count": len(kept),
        "duplicates_removed": removed_count,
    }


def tile_geometry(metadata: dict[str, Any]) -> TileGeometry | None:
    try:
        geometry = TileGeometry(
            frame_width=int(metadata["frame_width"]),
            frame_height=int(metadata["frame_height"]),
            grid_columns=int(metadata["grid_columns"]),
            grid_rows=int(metadata["grid_rows"]),
        )
    except (KeyError, TypeError, ValueError):
        return None

    if min(geometry) <= 0:
        return None

    return geometry


def surviving_detections(
    detections: list[Any],
    geometry: TileGeometry,
) -> list[Any]:
    ranked = sorted(
        enumerate(detections),
        key=lambda pair: detection_score(pair[1]),
        reverse=True,
    )

    kept_indexes: list[int] = []
    for index, detection in ranked:
        if not is_tiled_detection(detection):
            kept_indexes.append(index)
            continue

        duplicate = any(
            is_duplicate(detection, detections[kept_index], geometry)
            for kept_index in kept_indexes
            if is_tiled_detection(detections[kept_index])
        )
        if not duplicate:
            kept_indexes.append(index)

    return [detections[index] for index in sorted(kept_indexes)]


def is_duplicate(
    detection: dict[str, Any],
    kept: dict[str, Any],
    geometry: TileGeometry,
) -> bool:
    if detection.get("label") != kept.get("label"):
        return False

    if detection["tile"] == kept["tile"]:
        # Same-tile duplicates are already suppressed on the device.
        return False

    if not shares_overlap_band(detection, kept, geometry):
        return False

    distance = math.dist(detection["center"], kept["center"])
    return distance <= settings.tile_duplicate_distance_pixels


def shares_overlap_band(
    detection: dict[str, Any],
    kept: dict[str, Any],
    geometry: TileGeometry,
) -> bool:
    """True when both centers sit in the overlap band of adjacent tiles."""
    column, row = tile_position(detection["tile"], geometry)
    kept_column, kept_row = tile_position(kept["tile"], geometry)

    if abs(column - kept_column) > 1 or abs(row - kept_row) > 1:
        # Non-adjacent tiles never observe the same object.
        return False

    band = settings.tile_boundary_band_pixels

    if column != kept_column:
        boundary = (
            max(column, kept_column) * geometry.frame_width
        ) // geometry.grid_columns
        if not near_boundary(detection, kept, axis=0, boundary=boundary, band=band):
            return False

    if row != kept_row:
        boundary = (
            max(row, kept_row) * geometry.frame_height
        ) // geometry.grid_rows
        if not near_boundary(detection, kept, axis=1, boundary=boundary, band=band):
            return False

    return True


def near_boundary(
    detection: dict[str, Any],
    kept: dict[str, Any],
    axis: int,
    boundary: int,
    band: int,
) -> bool:
    return (
        abs(detection["center"][axis] - boundary) <= band
        and abs(kept["center"][axis] - boundary) <= band
    )


def tile_position(tile: int, geometry: TileGeometry) -> tuple[int, int]:
    return tile % geometry.grid_columns, tile // geometry.grid_columns


def detection_score(detection: Any) -> float:
    if not isinstance(detection, dict):
        return 0.0
    score = detection.get("score")
    return float(score) if isinstance(score, (int, float)) else 0.0


def is_tiled_detection(detection: Any) -> bool:
    """Guard against foreign or legacy metadata that lacks tile geometry."""
    if not isinstance(detection, dict):
        return False

    center = detection.get("center")
    return (
        isinstance(detection.get("tile"), int)
        and isinstance(center, (list, tuple))
        and len(center) == 2
        and all(isinstance(value, (int, float)) for value in center)
        and isinstance(detection.get("score"), (int, float))
    )
