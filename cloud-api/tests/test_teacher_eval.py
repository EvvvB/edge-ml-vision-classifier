import dataclasses
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import HTTPException

import app.services.eval_service as eval_service
from app.services.eval_service import (
    build_teacher_eval_rows,
    validate_teacher_identity,
)

DOG_LABELS = frozenset({"dog"})

IMAGE_A = UUID("11111111-1111-1111-1111-111111111111")
IMAGE_B = UUID("22222222-2222-2222-2222-222222222222")


@pytest.fixture
def dog_settings(monkeypatch):
    monkeypatch.setattr(
        eval_service,
        "settings",
        dataclasses.replace(
            eval_service.settings,
            eval_labels=DOG_LABELS,
            eval_teacher_min_confidence=0.5,
        ),
    )


def teacher_detection(confidence=0.9, bbox=None):
    return {
        "source": "yolo26x",
        "label": "dog",
        "confidence": confidence,
        "bbox": bbox or [80, 80, 120, 100],
    }


def upload_metadata(**overrides):
    metadata = {
        "model_hash": "aaaa11112222",
        "model_manifest": {"model_version": "v1"},
        "yolo_model_hash": "bbbb33334444",
        "yolo_model_manifest": {"model_version": "yolo26m-stock"},
        "inference_status": "complete",
        "fomo_detections": [
            {
                "source": "fomo",
                "label": "dog",
                "confidence": 0.8,
                "bbox": [128, 118, 24, 24],
                "center": [140, 130],
            }
        ],
        "yolo_detections": [
            {
                "source": "yolo",
                "label": "dog",
                "confidence": 0.9,
                "bbox": [85, 85, 110, 95],
            }
        ],
    }
    metadata.update(overrides)
    return metadata


def batch_payload(**overrides):
    payload = {
        "teacher_source": "yolo26x",
        "teacher_hash": "cccc55556666",
        "teacher_manifest": {"model_version": "yolo26x-stock"},
        "annotations": [
            {
                "image_id": str(IMAGE_A),
                "detections": [teacher_detection()],
                "inference_ms": 6400,
                "imgsz": 640,
            }
        ],
    }
    payload.update(overrides)
    return payload


def test_validate_teacher_identity_accepts_slug_and_hash() -> None:
    assert validate_teacher_identity(batch_payload()) == (
        "yolo26x",
        "cccc55556666",
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"teacher_source": "YOLO26X"},
        {"teacher_source": "fomo"},
        {"teacher_source": "yolo"},
        {"teacher_source": None},
        {"teacher_hash": "not-hex"},
        {"teacher_hash": None},
    ],
)
def test_validate_teacher_identity_rejects_bad_identity(overrides) -> None:
    with pytest.raises(HTTPException) as error:
        validate_teacher_identity(batch_payload(**overrides))
    assert error.value.status_code == 400


def test_teacher_eval_rows_score_both_students(dog_settings) -> None:
    rows = build_teacher_eval_rows(
        upload_metadata(),
        teacher_source="yolo26x",
        teacher_hash="cccc55556666",
        teacher_version="yolo26x-stock",
        teacher_detections=[teacher_detection()],
    )
    assert [row["student_source"] for row in rows] == ["fomo", "yolo"]
    fomo_row, yolo_row = rows
    assert fomo_row["teacher_source"] == "yolo26x"
    assert fomo_row["student_hash"] == "aaaa11112222"
    assert fomo_row["matched_count"] == 1  # centroid (140,130) in teacher box
    assert yolo_row["student_hash"] == "bbbb33334444"
    assert yolo_row["student_total"] == 1
    assert yolo_row["status"] == "scored"


def test_teacher_eval_rows_skip_yolo_student_when_inference_failed(
    dog_settings,
) -> None:
    rows = build_teacher_eval_rows(
        upload_metadata(inference_status="failed", yolo_detections=[]),
        teacher_source="yolo26x",
        teacher_hash="cccc55556666",
        teacher_version=None,
        teacher_detections=[teacher_detection()],
    )
    assert [row["student_source"] for row in rows] == ["fomo"]


@pytest.mark.asyncio
async def test_receive_teacher_batch_upserts_and_scores(
    dog_settings, monkeypatch
) -> None:
    upsert_annotation = AsyncMock()
    upsert_eval = AsyncMock()
    monkeypatch.setattr(
        eval_service, "upsert_teacher_annotation", upsert_annotation
    )
    monkeypatch.setattr(eval_service, "upsert_eval_result", upsert_eval)
    monkeypatch.setattr(
        eval_service,
        "fetch_detections_by_ids",
        AsyncMock(
            return_value={
                IMAGE_A: {
                    "image_id": IMAGE_A,
                    "metadata": upload_metadata(),
                    "captured_at": None,
                }
            }
        ),
    )

    result = await eval_service.receive_teacher_batch(None, batch_payload())

    assert result == {
        "ok": True,
        "annotated": 1,
        "scored_rows": 2,
        "unknown_images": 0,
    }
    assert upsert_annotation.await_count == 1
    assert upsert_annotation.await_args.kwargs["teacher_source"] == "yolo26x"
    student_sources = [
        call.kwargs["student_source"] for call in upsert_eval.await_args_list
    ]
    assert student_sources == ["fomo", "yolo"]
    teacher_sources = {
        call.kwargs["teacher_source"] for call in upsert_eval.await_args_list
    }
    assert teacher_sources == {"yolo26x"}


@pytest.mark.asyncio
async def test_receive_teacher_batch_errored_annotation_skips_scoring(
    dog_settings, monkeypatch
) -> None:
    upsert_annotation = AsyncMock()
    upsert_eval = AsyncMock()
    monkeypatch.setattr(
        eval_service, "upsert_teacher_annotation", upsert_annotation
    )
    monkeypatch.setattr(eval_service, "upsert_eval_result", upsert_eval)
    monkeypatch.setattr(
        eval_service,
        "fetch_detections_by_ids",
        AsyncMock(
            return_value={
                IMAGE_A: {
                    "image_id": IMAGE_A,
                    "metadata": upload_metadata(),
                    "captured_at": None,
                }
            }
        ),
    )

    payload = batch_payload(
        annotations=[{"image_id": str(IMAGE_A), "error": "image download failed"}]
    )
    result = await eval_service.receive_teacher_batch(None, payload)

    assert result["annotated"] == 1
    assert result["scored_rows"] == 0
    assert upsert_annotation.await_args.kwargs["error"] == "image download failed"
    assert upsert_annotation.await_args.kwargs["detections"] == []
    upsert_eval.assert_not_awaited()


@pytest.mark.asyncio
async def test_receive_teacher_batch_counts_unknown_images(
    dog_settings, monkeypatch
) -> None:
    monkeypatch.setattr(
        eval_service, "upsert_teacher_annotation", AsyncMock()
    )
    monkeypatch.setattr(eval_service, "upsert_eval_result", AsyncMock())
    monkeypatch.setattr(
        eval_service, "fetch_detections_by_ids", AsyncMock(return_value={})
    )

    result = await eval_service.receive_teacher_batch(None, batch_payload())

    assert result["annotated"] == 0
    assert result["unknown_images"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "annotations",
    [
        [],
        "not-a-list",
        [{"image_id": "not-a-uuid", "detections": []}],
        [{"image_id": str(IMAGE_A)}],  # no detections and no error
        [{"image_id": str(IMAGE_A), "detections": [], "error": 42}],
    ],
)
async def test_receive_teacher_batch_rejects_bad_annotations(
    annotations,
) -> None:
    with pytest.raises(HTTPException) as error:
        await eval_service.receive_teacher_batch(
            None, batch_payload(annotations=annotations)
        )
    assert error.value.status_code == 400


@pytest.mark.asyncio
async def test_teacher_pending_validates_inputs() -> None:
    with pytest.raises(HTTPException):
        await eval_service.list_teacher_pending(
            None, teacher_source="BAD SOURCE", limit=10
        )
    with pytest.raises(HTTPException):
        await eval_service.list_teacher_pending(
            None, teacher_source="yolo26x", limit=0
        )


@pytest.mark.asyncio
async def test_complete_teacher_run_validates_status(monkeypatch) -> None:
    monkeypatch.setattr(
        eval_service, "finish_teacher_run", AsyncMock(return_value=True)
    )
    result = await eval_service.complete_teacher_run(
        None, IMAGE_A, {"status": "complete", "detail": {"images": 3}}
    )
    assert result == {"ok": True}

    with pytest.raises(HTTPException) as error:
        await eval_service.complete_teacher_run(None, IMAGE_A, {"status": "meh"})
    assert error.value.status_code == 400
