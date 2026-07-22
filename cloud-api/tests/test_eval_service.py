import dataclasses
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from fastapi import HTTPException

import app.services.eval_service as eval_service
from app.services.eval_service import (
    build_eval_row,
    safe_ratio,
    score_detections,
    summarize_pair,
)

DOG_LABELS = frozenset({"dog"})


def fomo(label="dog", confidence=0.8, bbox=None, center=None):
    detection = {
        "source": "fomo",
        "label": label,
        "confidence": confidence,
        "bbox": bbox or [100, 100, 24, 24],
    }
    if center is not None:
        detection["center"] = center
    return detection


def yolo(label="dog", confidence=0.9, bbox=None):
    return {
        "source": "yolo",
        "label": label,
        "confidence": confidence,
        "bbox": bbox or [80, 80, 120, 100],
    }


def score(students, teachers, **overrides):
    kwargs = {"labels": DOG_LABELS, "teacher_min_confidence": 0.5}
    kwargs.update(overrides)
    return score_detections(students, teachers, **kwargs)


def test_centroid_inside_teacher_box_matches() -> None:
    result = score([fomo(center=[140, 130])], [yolo(bbox=[80, 80, 120, 100])])
    assert result["matched_count"] == 1
    assert result["student_matched"] == 1
    assert result["detail"]["student_only"] == []
    assert result["detail"]["teacher_only"] == []


def test_centroid_outside_teacher_box_is_student_only() -> None:
    result = score([fomo(center=[400, 300])], [yolo(bbox=[80, 80, 120, 100])])
    assert result["matched_count"] == 0
    assert result["student_matched"] == 0
    assert len(result["detail"]["student_only"]) == 1
    assert len(result["detail"]["teacher_only"]) == 1


def test_label_mismatch_never_matches() -> None:
    result = score(
        [fomo(center=[140, 130])],
        [yolo(label="cat", bbox=[80, 80, 120, 100])],
        labels=frozenset({"dog", "cat"}),
    )
    assert result["matched_count"] == 0
    assert len(result["detail"]["student_only"]) == 1
    assert len(result["detail"]["teacher_only"]) == 1


def test_bbox_center_used_when_centroid_missing() -> None:
    # bbox [100, 100, 24, 24] centers at (112, 112), inside the teacher box.
    result = score([fomo(center=None)], [yolo(bbox=[80, 80, 120, 100])])
    assert result["matched_count"] == 1


def test_multiple_centroids_absorbed_by_one_box_count_once() -> None:
    students = [fomo(center=[100, 120]), fomo(center=[150, 130])]
    result = score(students, [yolo(bbox=[80, 80, 120, 100])])
    assert result["matched_count"] == 1
    assert result["student_matched"] == 2
    assert len(result["detail"]["matched"]) == 1
    assert len(result["detail"]["matched"][0]["students"]) == 2


def test_overlapping_boxes_nearest_center_wins() -> None:
    left = yolo(bbox=[0, 0, 100, 100])
    right = yolo(bbox=[50, 0, 100, 100])
    # Centroid at (90, 50): inside both; right box centers at (100, 50),
    # left at (50, 50), so the right box is nearer.
    result = score([fomo(center=[90, 50])], [left, right])
    assert result["matched_count"] == 1
    assert result["detail"]["matched"][0]["teacher_bbox"] == [50, 0, 100, 100]
    assert len(result["detail"]["teacher_only"]) == 1


def test_low_confidence_teacher_is_excluded_not_a_miss() -> None:
    result = score([], [yolo(confidence=0.3)])
    assert result["teacher_total"] == 0
    assert result["detail"]["teacher_only"] == []
    assert result["detail"]["teacher_excluded"] == 1


def test_off_label_detections_excluded_from_both_sides() -> None:
    result = score(
        [fomo(label="person", center=[140, 130])],
        [yolo(label="person")],
    )
    assert result["student_total"] == 0
    assert result["teacher_total"] == 0
    assert result["detail"]["student_excluded"] == 1
    assert result["detail"]["teacher_excluded"] == 1


def test_malformed_detections_are_excluded() -> None:
    result = score(
        ["not-a-dict", {"label": "dog"}, fomo(bbox=[1, 2, 3])],
        [yolo(confidence=True)],
    )
    assert result["student_total"] == 0
    assert result["teacher_total"] == 0
    assert result["detail"]["student_excluded"] == 3
    assert result["detail"]["teacher_excluded"] == 1


def test_empty_inputs_score_as_trivial_agreement() -> None:
    result = score([], [])
    assert result["student_total"] == 0
    assert result["teacher_total"] == 0
    assert result["matched_count"] == 0
    assert result["student_matched"] == 0


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


def upload_metadata(**overrides):
    metadata = {
        "device_id": "nicla-vision-01",
        "model_hash": "aaaa11112222",
        "model_manifest": {"model_version": "v1"},
        "yolo_model_hash": "bbbb33334444",
        "yolo_model_manifest": {"model_version": "yolo26m-stock"},
        "inference_status": "complete",
        "fomo_detections": [fomo(center=[140, 130])],
        "yolo_detections": [yolo(bbox=[80, 80, 120, 100])],
    }
    metadata.update(overrides)
    return metadata


def test_build_eval_row_scores_complete_upload(dog_settings) -> None:
    row = build_eval_row(upload_metadata())
    assert row["status"] == "scored"
    assert row["skip_reason"] is None
    assert row["student_hash"] == "aaaa11112222"
    assert row["student_version"] == "v1"
    assert row["teacher_hash"] == "bbbb33334444"
    assert row["teacher_version"] == "yolo26m-stock"
    assert row["matched_count"] == 1


def test_build_eval_row_skips_when_yolo_never_ran(dog_settings) -> None:
    row = build_eval_row(upload_metadata(yolo_detections=None))
    assert row["status"] == "skipped"
    assert "yolo_detections" in row["skip_reason"]


def test_build_eval_row_skips_motion_only_captures(dog_settings) -> None:
    # model_enabled False means FOMO never ran; an empty fomo_detections
    # there is absence of inference, not absence of dogs.
    row = build_eval_row(
        upload_metadata(model_enabled=False, inference_mode="motion_only")
    )
    assert row["status"] == "skipped"
    assert "fomo did not run" in row["skip_reason"]

    # The stamp alone is enough even if model_enabled is missing (older
    # metadata shape).
    row = build_eval_row(upload_metadata(inference_mode="reference_frame"))
    assert row["status"] == "skipped"


def test_build_eval_row_skips_failed_inference(dog_settings) -> None:
    row = build_eval_row(
        upload_metadata(inference_status="failed", yolo_detections=[])
    )
    assert row["status"] == "skipped"
    assert "failed" in row["skip_reason"]


def test_build_eval_row_treats_missing_fomo_as_empty(dog_settings) -> None:
    row = build_eval_row(upload_metadata(fomo_detections=None))
    assert row["status"] == "scored"
    assert row["student_total"] == 0
    assert row["teacher_total"] == 1


def test_build_eval_row_tolerates_missing_model_stamps(dog_settings) -> None:
    row = build_eval_row(
        upload_metadata(
            model_hash=None,
            model_manifest=None,
            yolo_model_hash=None,
            yolo_model_manifest=None,
        )
    )
    assert row["status"] == "scored"
    assert row["student_hash"] is None
    assert row["teacher_version"] is None


def test_summarize_pair_computes_agreement_rates() -> None:
    pair = summarize_pair(
        {
            "student_total": 200,
            "student_matched": 190,
            "teacher_total": 250,
            "matched_count": 200,
        }
    )
    assert pair["agreement_precision"] == 0.95
    assert pair["agreement_recall"] == 0.8


def test_safe_ratio_handles_zero_denominator() -> None:
    assert safe_ratio(5, 0) is None
    assert safe_ratio(5, None) is None
    assert safe_ratio(None, 10) == 0.0


@pytest.mark.asyncio
async def test_backfill_scores_until_unscored_runs_dry(
    dog_settings, monkeypatch
) -> None:
    rows = [
        {
            "image_id": UUID("11111111-1111-1111-1111-111111111111"),
            "metadata": upload_metadata(),
            "captured_at": None,
        },
        {
            "image_id": UUID("22222222-2222-2222-2222-222222222222"),
            "metadata": upload_metadata(yolo_detections=None),
            "captured_at": None,
        },
    ]
    fetch = AsyncMock(side_effect=[rows, []])
    upsert = AsyncMock()
    monkeypatch.setattr(eval_service, "fetch_unscored_detections", fetch)
    monkeypatch.setattr(eval_service, "upsert_eval_result", upsert)

    result = await eval_service.run_backfill(pool=None)

    assert result == {"ok": True, "scored": 1, "skipped": 1, "complete": True}
    assert upsert.await_count == 2
    statuses = [call.kwargs["status"] for call in upsert.await_args_list]
    assert statuses == ["scored", "skipped"]


@pytest.mark.asyncio
async def test_backfill_rescore_walks_all_detections_with_offset(
    dog_settings, monkeypatch
) -> None:
    row = {
        "image_id": UUID("11111111-1111-1111-1111-111111111111"),
        "metadata": upload_metadata(),
        "captured_at": None,
    }
    fetch = AsyncMock(side_effect=[[row], []])
    upsert = AsyncMock()
    monkeypatch.setattr(eval_service, "fetch_detections_for_rescore", fetch)
    monkeypatch.setattr(eval_service, "upsert_eval_result", upsert)

    result = await eval_service.run_backfill(pool=None, rescore=True)

    assert result["scored"] == 1
    offsets = [call.kwargs["offset"] for call in fetch.await_args_list]
    assert offsets == [0, 1]


@pytest.mark.asyncio
async def test_backfill_stops_at_max_images(dog_settings, monkeypatch) -> None:
    def make_row(index: int):
        return {
            "image_id": UUID(f"00000000-0000-0000-0000-{index:012d}"),
            "metadata": upload_metadata(),
            "captured_at": None,
        }

    fetch = AsyncMock(side_effect=[[make_row(1)], [make_row(2)], [make_row(3)]])
    upsert = AsyncMock()
    monkeypatch.setattr(eval_service, "fetch_unscored_detections", fetch)
    monkeypatch.setattr(eval_service, "upsert_eval_result", upsert)

    result = await eval_service.run_backfill(pool=None, max_images=2)

    assert result["scored"] == 2
    assert result["complete"] is False


@pytest.mark.asyncio
async def test_backfill_rejects_bad_max_images() -> None:
    with pytest.raises(HTTPException) as error:
        await eval_service.run_backfill(pool=None, max_images=0)
    assert error.value.status_code == 400
