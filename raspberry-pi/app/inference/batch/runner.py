"""Offline teacher-model annotation runner.

Pulls images the configured teachers have not yet annotated from the cloud
API, runs each teacher over them, and posts the annotations back in
batches. The cloud scores FOMO and the Pi's live YOLO against every
teacher batch as it lands, so by the time this exits the eval dashboard is
current.

Designed to run unattended (systemd timer on the Pi) or by hand anywhere
with the API key:

    python -m app.inference.batch.runner --max-images 500
    python -m app.inference.batch.runner --teacher yolo26x --device mps
"""

from __future__ import annotations

import argparse
import gc
import logging
import socket
import time
from typing import Any

import numpy as np

from app.config import settings
from app.inference.batch.cloud import CloudEvalClient
from app.inference.batch.teachers import TeacherConfig, teachers_by_source
from app.services.coco import normalize_yolo_detections

log = logging.getLogger("teacher-runner")

BATCH_SIZE = 25


def annotate_with_teacher(
    client: CloudEvalClient,
    teacher: TeacherConfig,
    *,
    max_images: int,
    device: str,
) -> dict[str, Any]:
    identity = teacher.identity()
    teacher_hash = identity["hash"]
    if teacher_hash is None:
        # identity() reads the weights file; hash can only be missing
        # before the first download.
        teacher.load()
        identity = teacher.identity()
        teacher_hash = identity["hash"]
    if teacher_hash is None:
        raise RuntimeError(f"cannot hash weights for {teacher.source}")

    model = teacher.load()
    log.info(
        "%s: loaded %s (hash %s)", teacher.source, teacher.weights_name, teacher_hash
    )

    stats = {"images": 0, "errors": 0, "total_ms": 0}
    try:
        while stats["images"] + stats["errors"] < max_images:
            remaining = max_images - stats["images"] - stats["errors"]
            image_ids = client.pending_image_ids(
                teacher.source, min(BATCH_SIZE, remaining)
            )
            if not image_ids:
                break
            annotations = [
                annotate_image(client, model, teacher, image_id, device)
                for image_id in image_ids
            ]
            client.post_batch(
                teacher_source=teacher.source,
                teacher_hash=teacher_hash,
                teacher_manifest=identity["manifest"],
                annotations=annotations,
            )
            for annotation in annotations:
                if "error" in annotation:
                    stats["errors"] += 1
                else:
                    stats["images"] += 1
                    stats["total_ms"] += annotation["inference_ms"]
            log.info(
                "%s: %d annotated, %d errors, avg %.1fs/image",
                teacher.source,
                stats["images"],
                stats["errors"],
                stats["total_ms"] / stats["images"] / 1000 if stats["images"] else 0,
            )
    finally:
        del model
        gc.collect()

    return stats


def annotate_image(
    client: CloudEvalClient,
    model: Any,
    teacher: TeacherConfig,
    image_id: str,
    device: str,
) -> dict[str, Any]:
    """One image through one teacher; failures become error annotations.

    An error annotation still creates a row cloud-side, which is what keeps
    a broken image from staying pending forever.
    """
    import cv2

    try:
        image_bytes = client.download_image(image_id)
        frame = cv2.imdecode(
            np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if frame is None:
            raise ValueError("image bytes did not decode")
        started = time.perf_counter()
        results = model.predict(
            frame,
            imgsz=teacher.imgsz,
            conf=teacher.conf,
            augment=teacher.augment,
            device=device,
            verbose=False,
        )
        inference_ms = int((time.perf_counter() - started) * 1000)
        detections = normalize_yolo_detections(results[0].summary())
        for detection in detections:
            detection["source"] = teacher.source
        return {
            "image_id": image_id,
            "detections": detections,
            "inference_ms": inference_ms,
            "imgsz": teacher.imgsz,
        }
    except Exception as exc:  # noqa: BLE001 — one bad image must not kill the run
        log.warning("%s: %s failed: %s", teacher.source, image_id, exc)
        return {"image_id": image_id, "error": str(exc)[:500]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        action="append",
        help="teacher source to run (repeatable; default: all)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="per-teacher cap per run; backlog drains across runs (default 500)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="inference device: cpu (Pi) or mps (Mac) (default cpu)",
    )
    parser.add_argument("--base-url", default=settings.cloud_api_url)
    parser.add_argument("--api-key", default=settings.cloud_api_key)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    teachers = teachers_by_source(args.teacher)
    client = CloudEvalClient(args.base_url, args.api_key)
    run_id = client.start_run(socket.gethostname())

    detail: dict[str, Any] = {}
    status = "complete"
    started = time.monotonic()
    try:
        for teacher in teachers:
            detail[teacher.source] = annotate_with_teacher(
                client,
                teacher,
                max_images=args.max_images,
                device=args.device,
            )
    except Exception as exc:  # noqa: BLE001 — record the failure, then re-raise
        status = "failed"
        detail["error"] = str(exc)[:500]
        raise
    finally:
        detail["duration_s"] = int(time.monotonic() - started)
        client.finish_run(run_id, status, detail)
        client.close()
        log.info("run %s: %s %s", run_id, status, detail)


if __name__ == "__main__":
    main()
