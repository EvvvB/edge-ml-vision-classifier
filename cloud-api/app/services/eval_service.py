from __future__ import annotations

import math
from typing import Any
from uuid import UUID

import asyncpg

from fastapi import HTTPException

from app.config import settings
from app.storage.postgres import (
    fetch_detections_for_rescore,
    fetch_eval_disagreements,
    fetch_eval_summary,
    fetch_unscored_detections,
    upsert_eval_result,
)

# FOMO is the student and Pi YOLO the teacher. The teacher is silver, not
# gold: everything here measures agreement with YOLO, never accuracy. A
# larger cloud-side teacher can slot in later by writing rows with a
# different teacher_source.
STUDENT_SOURCE = "fomo"
TEACHER_SOURCE = "yolo"

# FOMO predicts centroid blobs, not object extents, so IoU against a real
# bounding box is meaningless (tile_dedupe.py on the Pi makes the same
# call). A student detection agrees with a teacher box when its centroid
# lands inside the box and the labels match.


def score_detections(
    student_detections: list[dict[str, Any]],
    teacher_detections: list[dict[str, Any]],
    *,
    labels: frozenset[str] | None = None,
    teacher_min_confidence: float | None = None,
) -> dict[str, Any]:
    """Match student centroids against teacher boxes.

    Both inputs use the shared COCO-style detection schema. Detections
    outside the eval label set are excluded from both sides: the student
    model cannot know labels it was never trained on, so an off-label
    teacher box must not count as a student miss.
    """
    if labels is None:
        labels = settings.eval_labels
    if teacher_min_confidence is None:
        teacher_min_confidence = settings.eval_teacher_min_confidence

    students, student_excluded = eligible_detections(student_detections, labels)
    teachers, teacher_excluded = eligible_detections(
        teacher_detections, labels, min_confidence=teacher_min_confidence
    )

    # Each student centroid matches at most one teacher box: the containing
    # box whose center is nearest. A teacher box may absorb several student
    # centroids (FOMO often fires multiple blobs on one large object) but
    # counts as one agreed object.
    absorbed: dict[int, list[dict[str, Any]]] = {}
    student_only: list[dict[str, Any]] = []
    for student in students:
        center = detection_center(student)
        best_index: int | None = None
        best_distance = math.inf
        for index, teacher in enumerate(teachers):
            if teacher["label"] != student["label"]:
                continue
            if not contains(teacher["bbox"], center):
                continue
            distance = math.dist(center, bbox_center(teacher["bbox"]))
            if distance < best_distance:
                best_index = index
                best_distance = distance
        entry = {
            "label": student["label"],
            "center": [round(value, 1) for value in center],
            "confidence": student["confidence"],
        }
        if best_index is None:
            student_only.append(entry)
        else:
            absorbed.setdefault(best_index, []).append(entry)

    matched = [
        {
            "label": teachers[index]["label"],
            "teacher_bbox": teachers[index]["bbox"],
            "teacher_confidence": teachers[index]["confidence"],
            "students": absorbed[index],
        }
        for index in sorted(absorbed)
    ]
    teacher_only = [
        {
            "label": teacher["label"],
            "bbox": teacher["bbox"],
            "confidence": teacher["confidence"],
        }
        for index, teacher in enumerate(teachers)
        if index not in absorbed
    ]

    return {
        "student_total": len(students),
        "teacher_total": len(teachers),
        "matched_count": len(matched),
        "student_matched": len(students) - len(student_only),
        "detail": {
            "matched": matched,
            "student_only": student_only,
            "teacher_only": teacher_only,
            "student_excluded": student_excluded,
            "teacher_excluded": teacher_excluded,
            "labels": sorted(labels),
            "teacher_min_confidence": teacher_min_confidence,
        },
    }


def eligible_detections(
    detections: list[dict[str, Any]],
    labels: frozenset[str],
    *,
    min_confidence: float = 0.0,
) -> tuple[list[dict[str, Any]], int]:
    """Filter to well-formed detections in the eval label set.

    Returns the survivors and the count excluded (off-label, below the
    confidence floor, or malformed) so the stored detail stays honest about
    what the score ignored.
    """
    eligible: list[dict[str, Any]] = []
    excluded = 0
    for detection in detections:
        if not isinstance(detection, dict):
            excluded += 1
            continue
        label = detection.get("label")
        confidence = detection.get("confidence")
        bbox = detection.get("bbox")
        if (
            not isinstance(label, str)
            or label not in labels
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or confidence < min_confidence
            or not valid_bbox(bbox)
        ):
            excluded += 1
            continue
        eligible.append(detection)
    return eligible, excluded


def valid_bbox(bbox: Any) -> bool:
    return (
        isinstance(bbox, (list, tuple))
        and len(bbox) == 4
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in bbox
        )
    )


def detection_center(detection: dict[str, Any]) -> tuple[float, float]:
    """FOMO's reported centroid when present, else the bbox center."""
    center = detection.get("center")
    if (
        isinstance(center, (list, tuple))
        and len(center) == 2
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in center
        )
    ):
        return float(center[0]), float(center[1])
    return bbox_center(detection["bbox"])


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    x, y, width, height = bbox
    return x + width / 2, y + height / 2


def contains(bbox: list[float], point: tuple[float, float]) -> bool:
    x, y, width, height = bbox
    px, py = point
    return x <= px <= x + width and y <= py <= y + height


def build_eval_row(metadata: dict[str, Any]) -> dict[str, Any]:
    """Score one upload's flat cloud metadata into an eval_results row.

    Rows where YOLO never ran are recorded as skipped rather than scored:
    an empty yolo_detections after a failed inference says nothing about
    what the teacher saw, and treating it as "teacher saw nothing" would
    manufacture student false positives.
    """
    student_detections = metadata.get("fomo_detections")
    teacher_detections = metadata.get("yolo_detections")
    inference_status = metadata.get("inference_status")

    skip_reason: str | None = None
    if not isinstance(teacher_detections, list):
        skip_reason = "no yolo_detections in metadata"
    elif inference_status not in (None, "complete"):
        skip_reason = f"inference_status is {inference_status!r}"

    row: dict[str, Any] = {
        "student_source": STUDENT_SOURCE,
        "teacher_source": TEACHER_SOURCE,
        "student_hash": optional_string(metadata.get("model_hash")),
        "student_version": manifest_version(metadata.get("model_manifest")),
        "teacher_hash": optional_string(metadata.get("yolo_model_hash")),
        "teacher_version": manifest_version(metadata.get("yolo_model_manifest")),
    }

    if skip_reason is not None:
        row.update(
            status="skipped",
            skip_reason=skip_reason,
            student_total=0,
            teacher_total=0,
            matched_count=0,
            student_matched=0,
            detail={},
        )
        return row

    if not isinstance(student_detections, list):
        student_detections = []
    row.update(status="scored", skip_reason=None)
    row.update(score_detections(student_detections, teacher_detections))
    return row


def manifest_version(manifest: Any) -> str | None:
    if not isinstance(manifest, dict):
        return None
    return optional_string(manifest.get("model_version"))


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


async def record_image_eval(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
    metadata: dict[str, Any],
    captured_at: Any,
) -> None:
    await upsert_eval_result(
        pool,
        image_id=image_id,
        captured_at=captured_at,
        **build_eval_row(metadata),
    )


async def get_eval_summary(pool: asyncpg.Pool) -> dict[str, Any]:
    summary = await fetch_eval_summary(pool)
    pairs = [summarize_pair(pair) for pair in summary["pairs"]]
    return {
        "ok": True,
        "labels": sorted(settings.eval_labels),
        "teacher_min_confidence": settings.eval_teacher_min_confidence,
        "scored_images": summary["scored_images"],
        "skipped_images": summary["skipped_images"],
        "unscored_images": summary["unscored_images"],
        "pairs": pairs,
    }


def summarize_pair(pair: dict[str, Any]) -> dict[str, Any]:
    """Attach agreement rates to a raw per-model-pair aggregate row.

    Precision: of the student's detections, the share that landed inside a
    teacher box. Recall: of the teacher's boxes, the share the student hit.
    Both deliberately named "agreement" — the teacher is a reference, not
    ground truth.
    """
    result = dict(pair)
    result["agreement_precision"] = safe_ratio(
        pair["student_matched"], pair["student_total"]
    )
    result["agreement_recall"] = safe_ratio(
        pair["matched_count"], pair["teacher_total"]
    )
    return result


def safe_ratio(numerator: int | None, denominator: int | None) -> float | None:
    if not denominator:
        return None
    return round((numerator or 0) / denominator, 4)


async def list_eval_disagreements(
    pool: asyncpg.Pool,
    *,
    limit: int,
) -> dict[str, Any]:
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    rows = await fetch_eval_disagreements(pool, limit=limit)
    for row in rows:
        detail = row.pop("detail", None) or {}
        row["student_only"] = detail.get("student_only", [])
        row["teacher_only"] = detail.get("teacher_only", [])
    return {"ok": True, "disagreements": rows}


async def run_backfill(
    pool: asyncpg.Pool,
    *,
    rescore: bool = False,
    max_images: int = 5000,
    batch_size: int = 500,
) -> dict[str, Any]:
    """Score stored detections that have no eval row yet.

    rescore walks every detection instead, overwriting existing rows — for
    re-running history after the matching rules or thresholds change.
    """
    if max_images < 1 or max_images > 50000:
        raise HTTPException(
            status_code=400,
            detail="max_images must be between 1 and 50000",
        )

    scored = 0
    skipped = 0
    processed = 0
    offset = 0
    exhausted = False

    while processed < max_images:
        batch = min(batch_size, max_images - processed)
        if rescore:
            rows = await fetch_detections_for_rescore(
                pool, limit=batch, offset=offset
            )
            offset += len(rows)
        else:
            # Unscored rows disappear from the fetch as their eval rows are
            # written, so this loop needs no offset to make progress.
            rows = await fetch_unscored_detections(pool, limit=batch)
        if not rows:
            exhausted = True
            break

        for row in rows:
            eval_row = build_eval_row(row["metadata"])
            await upsert_eval_result(
                pool,
                image_id=row["image_id"],
                captured_at=row["captured_at"],
                **eval_row,
            )
            processed += 1
            if eval_row["status"] == "scored":
                scored += 1
            else:
                skipped += 1

    return {
        "ok": True,
        "scored": scored,
        "skipped": skipped,
        "complete": exhausted,
    }
