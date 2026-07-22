from __future__ import annotations

import math
from typing import Any
from uuid import UUID, uuid4

import asyncpg

import re

from fastapi import HTTPException

from app.config import settings
from app.storage.postgres import (
    fetch_detections_by_ids,
    fetch_detections_for_rescore,
    fetch_eval_disagreements,
    fetch_eval_summary,
    fetch_pending_teacher_images,
    fetch_teacher_runs,
    fetch_unscored_detections,
    finish_teacher_run,
    insert_teacher_run,
    upsert_eval_result,
    upsert_teacher_annotation,
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


# ---------------------------------------------------------------------------
# Teacher batch ingest (offline runner uploads)
# ---------------------------------------------------------------------------

TEACHER_SOURCE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")
TEACHER_HASH_PATTERN = re.compile(r"^[0-9a-f]{4,64}$")
MAX_BATCH_ANNOTATIONS = 200
# The live pipeline's own sources can never be teacher families; allowing
# them would let a teacher row silently collide with phase-0 semantics.
RESERVED_TEACHER_SOURCES = frozenset({STUDENT_SOURCE, TEACHER_SOURCE})


def validate_teacher_identity(payload: dict[str, Any]) -> tuple[str, str]:
    teacher_source = payload.get("teacher_source")
    teacher_hash = payload.get("teacher_hash")
    if (
        not isinstance(teacher_source, str)
        or not TEACHER_SOURCE_PATTERN.fullmatch(teacher_source)
        or teacher_source in RESERVED_TEACHER_SOURCES
    ):
        raise HTTPException(
            status_code=400,
            detail="teacher_source must be a short lowercase slug"
            f" and not one of: {', '.join(sorted(RESERVED_TEACHER_SOURCES))}",
        )
    if not isinstance(teacher_hash, str) or not TEACHER_HASH_PATTERN.fullmatch(
        teacher_hash
    ):
        raise HTTPException(
            status_code=400, detail="teacher_hash must be a hex model hash"
        )
    return teacher_source, teacher_hash


def build_teacher_eval_rows(
    metadata: dict[str, Any],
    *,
    teacher_source: str,
    teacher_hash: str,
    teacher_version: str | None,
    teacher_detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Score both edge models against one teacher's annotations.

    FOMO is always scoreable (its detections rode in with the upload). The
    Pi YOLO is scored too — that pairing measures what the bigger teacher
    catches that the live model misses — but only when its inference
    actually completed.
    """
    rows = []
    students = [
        (
            STUDENT_SOURCE,
            metadata.get("fomo_detections"),
            optional_string(metadata.get("model_hash")),
            manifest_version(metadata.get("model_manifest")),
            True,
        ),
        (
            TEACHER_SOURCE,
            metadata.get("yolo_detections"),
            optional_string(metadata.get("yolo_model_hash")),
            manifest_version(metadata.get("yolo_model_manifest")),
            metadata.get("inference_status") in (None, "complete"),
        ),
    ]
    for source, detections, student_hash, student_version, scoreable in students:
        if not scoreable:
            continue
        if not isinstance(detections, list):
            detections = []
        row = {
            "student_source": source,
            "teacher_source": teacher_source,
            "student_hash": student_hash,
            "student_version": student_version,
            "teacher_hash": teacher_hash,
            "teacher_version": teacher_version,
            "status": "scored",
            "skip_reason": None,
        }
        row.update(score_detections(detections, teacher_detections))
        rows.append(row)
    return rows


async def receive_teacher_batch(
    pool: asyncpg.Pool,
    payload: dict[str, Any],
) -> dict[str, Any]:
    teacher_source, teacher_hash = validate_teacher_identity(payload)
    manifest = payload.get("teacher_manifest")
    if manifest is not None and not isinstance(manifest, dict):
        raise HTTPException(
            status_code=400, detail="teacher_manifest must be an object"
        )
    annotations = payload.get("annotations")
    if not isinstance(annotations, list) or not annotations:
        raise HTTPException(
            status_code=400, detail="annotations must be a non-empty list"
        )
    if len(annotations) > MAX_BATCH_ANNOTATIONS:
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_BATCH_ANNOTATIONS} annotations per batch",
        )

    parsed: list[tuple[UUID, dict[str, Any]]] = []
    for entry in annotations:
        if not isinstance(entry, dict):
            raise HTTPException(
                status_code=400, detail="each annotation must be an object"
            )
        try:
            image_id = UUID(str(entry.get("image_id")))
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="annotation image_id must be a UUID"
            ) from exc
        detections = entry.get("detections")
        error = entry.get("error")
        if error is not None and not isinstance(error, str):
            raise HTTPException(
                status_code=400, detail="annotation error must be a string"
            )
        if error is None and not isinstance(detections, list):
            raise HTTPException(
                status_code=400,
                detail="annotation detections must be a list unless error is set",
            )
        parsed.append((image_id, entry))

    detections_by_id = await fetch_detections_by_ids(
        pool, [image_id for image_id, _ in parsed]
    )
    teacher_version = manifest_version(manifest)

    annotated = 0
    scored_rows = 0
    unknown_images = 0
    for image_id, entry in parsed:
        detection_row = detections_by_id.get(image_id)
        if detection_row is None:
            # Never seen this image; nothing to attach the annotation to.
            unknown_images += 1
            continue
        error = entry.get("error")
        teacher_detections = entry.get("detections") if error is None else []
        if not isinstance(teacher_detections, list):
            teacher_detections = []
        await upsert_teacher_annotation(
            pool,
            image_id=image_id,
            teacher_source=teacher_source,
            teacher_hash=teacher_hash,
            teacher_manifest=manifest,
            detections=teacher_detections,
            error=error,
            inference_ms=entry.get("inference_ms")
            if isinstance(entry.get("inference_ms"), int)
            else None,
            imgsz=entry.get("imgsz") if isinstance(entry.get("imgsz"), int) else None,
        )
        annotated += 1
        if error is not None:
            continue
        for row in build_teacher_eval_rows(
            detection_row["metadata"],
            teacher_source=teacher_source,
            teacher_hash=teacher_hash,
            teacher_version=teacher_version,
            teacher_detections=teacher_detections,
        ):
            await upsert_eval_result(
                pool,
                image_id=image_id,
                captured_at=detection_row["captured_at"],
                **row,
            )
            scored_rows += 1

    return {
        "ok": True,
        "annotated": annotated,
        "scored_rows": scored_rows,
        "unknown_images": unknown_images,
    }


async def list_teacher_pending(
    pool: asyncpg.Pool,
    *,
    teacher_source: str,
    limit: int,
) -> dict[str, Any]:
    if not TEACHER_SOURCE_PATTERN.fullmatch(teacher_source or ""):
        raise HTTPException(
            status_code=400, detail="teacher_source must be a short lowercase slug"
        )
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400, detail="limit must be between 1 and 1000"
        )
    image_ids = await fetch_pending_teacher_images(
        pool, teacher_source=teacher_source, limit=limit
    )
    return {"ok": True, "image_ids": image_ids}


async def start_teacher_run(
    pool: asyncpg.Pool,
    payload: dict[str, Any],
) -> dict[str, Any]:
    run_id = uuid4()
    runner = payload.get("runner")
    await insert_teacher_run(
        pool,
        run_id=run_id,
        runner=str(runner) if runner is not None else None,
    )
    return {"ok": True, "run_id": str(run_id)}


VALID_RUN_STATUSES = frozenset({"complete", "failed"})


async def complete_teacher_run(
    pool: asyncpg.Pool,
    run_id: UUID,
    payload: dict[str, Any],
) -> dict[str, Any]:
    status = payload.get("status")
    if status not in VALID_RUN_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"status must be one of: {', '.join(sorted(VALID_RUN_STATUSES))}",
        )
    detail = payload.get("detail")
    if detail is not None and not isinstance(detail, dict):
        raise HTTPException(status_code=400, detail="detail must be an object")
    updated = await finish_teacher_run(
        pool, run_id=run_id, status=status, detail=detail
    )
    if not updated:
        raise HTTPException(status_code=404, detail="teacher run not found")
    return {"ok": True}


async def list_teacher_runs(
    pool: asyncpg.Pool,
    *,
    limit: int,
) -> dict[str, Any]:
    if limit < 1 or limit > 50:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 50")
    runs = await fetch_teacher_runs(pool, limit=limit)
    return {"ok": True, "runs": runs}
