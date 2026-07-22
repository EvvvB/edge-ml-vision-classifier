from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from app.config import settings


async def create_db_pool() -> asyncpg.Pool:
    return await asyncpg.create_pool(dsn=settings.database_url, min_size=1, max_size=10)


async def close_db_pool(pool: asyncpg.Pool) -> None:
    await pool.close()


async def check_db(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute("SELECT 1")


async def init_db(pool: asyncpg.Pool) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                image_id UUID PRIMARY KEY,
                device_id TEXT,
                filename TEXT,
                content_type TEXT NOT NULL,
                file_size_bytes BIGINT NOT NULL,
                metadata JSONB NOT NULL,
                s3_bucket TEXT NOT NULL,
                s3_key TEXT NOT NULL,
                s3_url TEXT NOT NULL,
                s3_etag TEXT,
                upload_status TEXT NOT NULL,
                upload_error TEXT,
                captured_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_detections_created_at
            ON detections (created_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_detections_device_id
            ON detections (device_id)
            """
        )
        # Monotonic per-device press counter for manual capture requests. The
        # counter value itself is the command: devices compare it against a
        # high-water mark, so retries and replays are naturally idempotent.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS capture_requests (
                device_id TEXT PRIMARY KEY,
                counter BIGINT NOT NULL DEFAULT 0,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # One row per (image, student, teacher) pair: how well the student
        # model's detections agreed with the teacher's on that image. Rows
        # are pure derivations of detections.metadata — safe to truncate and
        # rebuild via the eval backfill whenever scoring rules change.
        # Skipped rows (teacher inference never ran) are recorded too, so
        # "unscored" always means "backfill has not visited this image yet".
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS eval_results (
                image_id UUID NOT NULL
                    REFERENCES detections (image_id) ON DELETE CASCADE,
                student_source TEXT NOT NULL,
                teacher_source TEXT NOT NULL,
                student_hash TEXT,
                student_version TEXT,
                teacher_hash TEXT,
                teacher_version TEXT,
                status TEXT NOT NULL,
                skip_reason TEXT,
                student_total INT NOT NULL,
                teacher_total INT NOT NULL,
                matched_count INT NOT NULL,
                student_matched INT NOT NULL,
                detail JSONB NOT NULL,
                captured_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (image_id, student_source, teacher_source)
            )
            """
        )
        # Offline teacher-model annotations, one row per (image, teacher
        # family). teacher_hash records which weights produced the row;
        # upgrading a teacher's weights annotates new images under the new
        # hash without re-annotating history (re-runs can do that
        # explicitly). Rows with error set mark images the teacher could not
        # process, so they stop showing up as pending.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS teacher_annotations (
                image_id UUID NOT NULL
                    REFERENCES detections (image_id) ON DELETE CASCADE,
                teacher_source TEXT NOT NULL,
                teacher_hash TEXT NOT NULL,
                teacher_manifest JSONB,
                detections JSONB NOT NULL,
                error TEXT,
                inference_ms INT,
                imgsz INT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (image_id, teacher_source)
            )
            """
        )
        # One row per teacher-runner invocation, so the dashboard can show
        # whether the nightly batch actually ran and what it did.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS teacher_runs (
                run_id UUID PRIMARY KEY,
                runner TEXT,
                status TEXT NOT NULL,
                detail JSONB,
                started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                finished_at TIMESTAMPTZ
            )
            """
        )
        # Arrival log shipped from each Pi's receipt files: one row per
        # upload a Pi received from a camera, rejected uploads included —
        # the only record of those, since rejected frames are never stored.
        # receipt_id comes from the Pi and dedupes resent sync batches.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS pi_receipts (
                receipt_id UUID PRIMARY KEY,
                pi_id TEXT,
                device_id TEXT,
                event TEXT NOT NULL,
                image_id TEXT,
                filename TEXT,
                content_type TEXT,
                fomo_count INT,
                reason TEXT,
                client_host TEXT,
                logged_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pi_receipts_logged_at
            ON pi_receipts (logged_at DESC)
            """
        )
        await conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pi_receipts_device_id
            ON pi_receipts (device_id)
            """
        )
        # Device registry, upserted from hellos and uploads. desired_mode is
        # the cloud's command (versioned by desired_mode_seq, the same
        # high-water-mark pattern as capture counters); reported_mode is what
        # the device last acked, so the dashboard can show desired vs actual.
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS devices (
                device_id TEXT PRIMARY KEY,
                hardware_id TEXT,
                display_name TEXT,
                firmware_build TEXT,
                model_hash TEXT,
                model_manifest JSONB,
                pi_id TEXT,
                desired_mode TEXT NOT NULL DEFAULT 'automated',
                desired_mode_seq BIGINT NOT NULL DEFAULT 0,
                desired_mode_at TIMESTAMPTZ,
                reported_mode TEXT,
                reported_mode_seq BIGINT,
                reported_mode_at TIMESTAMPTZ,
                desired_config JSONB,
                desired_config_seq BIGINT NOT NULL DEFAULT 0,
                desired_config_at TIMESTAMPTZ,
                reported_config JSONB,
                reported_config_seq BIGINT,
                reported_config_at TIMESTAMPTZ,
                first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_hello_at TIMESTAMPTZ,
                last_upload_at TIMESTAMPTZ,
                last_seen_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Remote-config columns postdate deployed registries; these alters
        # bring an existing table up to the schema above and no-op on a
        # fresh one.
        await conn.execute(
            """
            ALTER TABLE devices
                ADD COLUMN IF NOT EXISTS desired_config JSONB,
                ADD COLUMN IF NOT EXISTS desired_config_seq BIGINT
                    NOT NULL DEFAULT 0,
                ADD COLUMN IF NOT EXISTS desired_config_at TIMESTAMPTZ,
                ADD COLUMN IF NOT EXISTS reported_config JSONB,
                ADD COLUMN IF NOT EXISTS reported_config_seq BIGINT,
                ADD COLUMN IF NOT EXISTS reported_config_at TIMESTAMPTZ
            """
        )


async def insert_detection_upload(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
    device_id: str | None,
    filename: str | None,
    content_type: str,
    file_size_bytes: int,
    metadata: dict[str, Any],
    s3_bucket: str,
    s3_key: str,
    s3_url: str,
    captured_at: datetime | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO detections (
                image_id,
                device_id,
                filename,
                content_type,
                file_size_bytes,
                metadata,
                s3_bucket,
                s3_key,
                s3_url,
                upload_status,
                captured_at
            )
            VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, 'uploading', $10)
            """,
            image_id,
            device_id,
            filename,
            content_type,
            file_size_bytes,
            json.dumps(metadata),
            s3_bucket,
            s3_key,
            s3_url,
            captured_at,
        )


async def mark_detection_stored(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
    s3_etag: str | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE detections
            SET upload_status = 'stored',
                upload_error = NULL,
                s3_etag = $2,
                updated_at = now()
            WHERE image_id = $1
            """,
            image_id,
            s3_etag,
        )


async def mark_detection_failed(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
    error: str,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE detections
            SET upload_status = 'failed',
                upload_error = $2,
                updated_at = now()
            WHERE image_id = $1
            """,
            image_id,
            error,
        )


async def fetch_detection(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT *
            FROM detections
            WHERE image_id = $1
            """,
            image_id,
        )
    return serialize_row(row) if row else None


async def increment_capture_counter(pool: asyncpg.Pool, device_id: str) -> int:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            """
            INSERT INTO capture_requests (device_id, counter)
            VALUES ($1, 1)
            ON CONFLICT (device_id)
            DO UPDATE SET counter = capture_requests.counter + 1,
                          updated_at = now()
            RETURNING counter
            """,
            device_id,
        )


async def fetch_capture_counter(pool: asyncpg.Pool, device_id: str) -> int:
    async with pool.acquire() as conn:
        counter = await conn.fetchval(
            "SELECT counter FROM capture_requests WHERE device_id = $1",
            device_id,
        )
    return counter or 0


DETECTION_SOURCES = ("fomo", "yolo")

# Where each source's model identity lives in upload metadata. The Nicla
# predates the two-model schema, so its FOMO fields are unprefixed.
MODEL_HASH_FIELDS = {
    "fomo": "metadata->>'model_hash'",
    "yolo": "metadata->>'yolo_model_hash'",
}
MODEL_VERSION_FIELDS = {
    "fomo": "metadata->'model_manifest'->>'model_version'",
    "yolo": "metadata->'yolo_model_manifest'->>'model_version'",
}


def detection_array(source: str) -> str:
    return f"coalesce(metadata->'{source}_detections', '[]'::jsonb)"


# Time filters and ordering use the timestamp the dashboard displays:
# capture time when the device reported one, arrival time otherwise.
DISPLAY_TIMESTAMP = "coalesce(captured_at, created_at)"


def build_detection_filters(
    *,
    device_id: str | None,
    labels: list[str],
    models: list[str],
    detections: str,
    source: str,
    since: datetime | None,
    until: datetime | None,
    params: list[Any],
) -> str:
    """Append filter params to `params` and return a WHERE clause (or '')."""
    sources = DETECTION_SOURCES if source == "any" else (source,)
    clauses: list[str] = []

    if device_id is not None:
        params.append(device_id)
        clauses.append(f"device_id = ${len(params)}")

    if since is not None:
        params.append(since)
        clauses.append(f"{DISPLAY_TIMESTAMP} >= ${len(params)}")

    if until is not None:
        params.append(until)
        clauses.append(f"{DISPLAY_TIMESTAMP} <= ${len(params)}")

    if detections == "some":
        clauses.append(
            "("
            + " OR ".join(
                f"jsonb_array_length({detection_array(s)}) > 0" for s in sources
            )
            + ")"
        )
    elif detections == "none":
        clauses.append(
            "("
            + " AND ".join(
                f"jsonb_array_length({detection_array(s)}) = 0" for s in sources
            )
            + ")"
        )

    if labels:
        params.append(labels)
        position = len(params)
        clauses.append(
            "("
            + " OR ".join(
                f"EXISTS (SELECT 1 FROM jsonb_array_elements({detection_array(s)}) AS d"
                f" WHERE lower(d->>'label') = ANY(${position}::text[]))"
                for s in sources
            )
            + ")"
        )

    if models:
        params.append(models)
        position = len(params)
        clauses.append(
            "("
            + " OR ".join(
                f"{MODEL_HASH_FIELDS[s]} = ANY(${position}::text[])"
                for s in sources
            )
            + ")"
        )

    if not clauses:
        return ""
    return " WHERE " + " AND ".join(clauses)


async def fetch_detections(
    pool: asyncpg.Pool,
    *,
    device_id: str | None,
    labels: list[str],
    models: list[str],
    detections: str,
    source: str,
    since: datetime | None,
    until: datetime | None,
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    params: list[Any] = []
    where = build_detection_filters(
        device_id=device_id,
        labels=labels,
        models=models,
        detections=detections,
        source=source,
        since=since,
        until=until,
        params=params,
    )

    async with pool.acquire() as conn:
        total = await conn.fetchval(
            f"SELECT count(*) FROM detections{where}",
            *params,
        )
        rows = await conn.fetch(
            f"""
            SELECT *
            FROM detections{where}
            ORDER BY created_at DESC
            LIMIT ${len(params) + 1} OFFSET ${len(params) + 2}
            """,
            *params,
            limit,
            offset,
        )
    return [serialize_row(row) for row in rows], total


async def delete_detection(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
) -> dict[str, Any] | None:
    """Delete one detection row; returns its S3 location, or None if absent.

    eval_results and teacher_annotations rows go with it via ON DELETE
    CASCADE; the S3 object is the caller's to clean up.
    """
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            DELETE FROM detections
            WHERE image_id = $1
            RETURNING s3_bucket, s3_key
            """,
            image_id,
        )
    return dict(row) if row else None


async def delete_detections(
    pool: asyncpg.Pool,
    *,
    device_id: str | None,
    labels: list[str],
    models: list[str],
    detections: str,
    source: str,
    since: datetime | None,
    until: datetime | None,
) -> list[dict[str, Any]]:
    """Delete every detection matching the filters; returns S3 locations."""
    params: list[Any] = []
    where = build_detection_filters(
        device_id=device_id,
        labels=labels,
        models=models,
        detections=detections,
        source=source,
        since=since,
        until=until,
        params=params,
    )
    # Backstop against wiping the table: the service validates that the
    # filters are restrictive, but this function must never run unfiltered.
    if not where:
        raise ValueError("refusing to delete detections without filters")

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"DELETE FROM detections{where} RETURNING s3_bucket, s3_key",
            *params,
        )
    return [dict(row) for row in rows]


async def fetch_detection_facets(
    pool: asyncpg.Pool,
    *,
    device_id: str | None,
    source: str,
) -> dict[str, Any]:
    sources = DETECTION_SOURCES if source == "any" else (source,)

    device_params: list[Any] = []
    device_where = ""
    if device_id is not None:
        device_params.append(device_id)
        device_where = " WHERE device_id = $1"

    # One (image_id, label) pair per record and label; UNION dedupes a label
    # reported by both models, so counts mean "records containing this label".
    label_arms = " UNION ".join(
        f"SELECT image_id, lower(d->>'label') AS label"
        f" FROM detections, jsonb_array_elements({detection_array(s)}) AS d"
        f"{device_where}"
        for s in sources
    )
    none_clause = " AND ".join(
        f"jsonb_array_length({detection_array(s)}) = 0" for s in sources
    )
    # One row per (source, model hash). UNION ALL because each arm carries a
    # distinct source constant. max(model_version) picks a display label if
    # manifests ever disagreed for the same hash; the hash stays the truth.
    model_arms = " UNION ALL ".join(
        f"SELECT '{s}' AS source, {MODEL_HASH_FIELDS[s]} AS model_hash,"
        f" {MODEL_VERSION_FIELDS[s]} AS model_version"
        f" FROM detections{device_where}"
        for s in sources
    )

    async with pool.acquire() as conn:
        label_rows = await conn.fetch(
            f"""
            SELECT label, count(*) AS record_count
            FROM ({label_arms}) AS labeled
            WHERE label IS NOT NULL AND label <> ''
            GROUP BY label
            ORDER BY record_count DESC, label
            """,
            *device_params,
        )
        total = await conn.fetchval(
            f"SELECT count(*) FROM detections{device_where}",
            *device_params,
        )
        none_where = device_where + (" AND " if device_where else " WHERE ")
        none_count = await conn.fetchval(
            f"SELECT count(*) FROM detections{none_where}{none_clause}",
            *device_params,
        )
        model_rows = await conn.fetch(
            f"""
            SELECT source, model_hash, max(model_version) AS model_version,
                   count(*) AS record_count
            FROM ({model_arms}) AS stamped
            WHERE model_hash IS NOT NULL AND model_hash <> ''
            GROUP BY source, model_hash
            ORDER BY source, record_count DESC, model_hash
            """,
            *device_params,
        )
        device_rows = await conn.fetch(
            """
            SELECT device_id, count(*) AS record_count
            FROM detections
            GROUP BY device_id
            ORDER BY record_count DESC
            """
        )
        timeline_device_clause = (
            " AND device_id = $1" if device_id is not None else ""
        )
        timeline_rows = await conn.fetch(
            f"""
            SELECT hours.bucket AS bucket, coalesce(counted.record_count, 0) AS record_count
            FROM generate_series(
                date_trunc('hour', now()) - interval '23 hours',
                date_trunc('hour', now()),
                interval '1 hour'
            ) AS hours(bucket)
            LEFT JOIN (
                SELECT date_trunc('hour', created_at) AS bucket, count(*) AS record_count
                FROM detections
                WHERE created_at > now() - interval '24 hours'{timeline_device_clause}
                GROUP BY 1
            ) AS counted USING (bucket)
            ORDER BY hours.bucket
            """,
            *device_params,
        )

    return {
        "total": total,
        "none": none_count,
        "timeline": [
            {"hour": row["bucket"].isoformat(), "count": row["record_count"]}
            for row in timeline_rows
        ],
        "labels": [
            {"label": row["label"], "count": row["record_count"]}
            for row in label_rows
        ],
        "models": [
            {
                "source": row["source"],
                "hash": row["model_hash"],
                "version": row["model_version"],
                "count": row["record_count"],
            }
            for row in model_rows
        ],
        "devices": [
            {"device_id": row["device_id"], "count": row["record_count"]}
            for row in device_rows
        ],
    }


def serialize_row(row: asyncpg.Record) -> dict[str, Any]:
    data = dict(row)
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    for key in ("image_id", "captured_at", "created_at", "updated_at"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data


# ---------------------------------------------------------------------------
# Eval results (student model scored against teacher model)
# ---------------------------------------------------------------------------


async def upsert_eval_result(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
    student_source: str,
    teacher_source: str,
    student_hash: str | None,
    student_version: str | None,
    teacher_hash: str | None,
    teacher_version: str | None,
    status: str,
    skip_reason: str | None,
    student_total: int,
    teacher_total: int,
    matched_count: int,
    student_matched: int,
    detail: dict[str, Any],
    captured_at: datetime | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO eval_results (
                image_id, student_source, teacher_source,
                student_hash, student_version, teacher_hash, teacher_version,
                status, skip_reason,
                student_total, teacher_total, matched_count, student_matched,
                detail, captured_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                    $14::jsonb, $15)
            ON CONFLICT (image_id, student_source, teacher_source)
            DO UPDATE SET
                student_hash = $4,
                student_version = $5,
                teacher_hash = $6,
                teacher_version = $7,
                status = $8,
                skip_reason = $9,
                student_total = $10,
                teacher_total = $11,
                matched_count = $12,
                student_matched = $13,
                detail = $14::jsonb,
                captured_at = $15,
                updated_at = now()
            """,
            image_id,
            student_source,
            teacher_source,
            student_hash,
            student_version,
            teacher_hash,
            teacher_version,
            status,
            skip_reason,
            student_total,
            teacher_total,
            matched_count,
            student_matched,
            json.dumps(detail),
            captured_at,
        )


async def fetch_unscored_detections(
    pool: asyncpg.Pool,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Detections the eval backfill has not visited yet, oldest first.

    Skipped rows count as visited, so this shrinks monotonically as the
    backfill writes rows — the caller needs no offset to make progress.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.image_id, d.metadata, d.captured_at
            FROM detections d
            LEFT JOIN eval_results e ON e.image_id = d.image_id
            WHERE e.image_id IS NULL
            ORDER BY d.created_at, d.image_id
            LIMIT $1
            """,
            limit,
        )
    return [eval_input_row(row) for row in rows]


async def fetch_detections_for_rescore(
    pool: asyncpg.Pool,
    *,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT image_id, metadata, captured_at
            FROM detections
            ORDER BY created_at, image_id
            LIMIT $1 OFFSET $2
            """,
            limit,
            offset,
        )
    return [eval_input_row(row) for row in rows]


def eval_input_row(row: asyncpg.Record) -> dict[str, Any]:
    """Keep image_id/captured_at native so they round-trip into upserts."""
    metadata = row["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return {
        "image_id": row["image_id"],
        "metadata": metadata if isinstance(metadata, dict) else {},
        "captured_at": row["captured_at"],
    }


async def fetch_eval_summary(pool: asyncpg.Pool) -> dict[str, Any]:
    async with pool.acquire() as conn:
        pair_rows = await conn.fetch(
            """
            SELECT student_source, teacher_source,
                   student_hash, max(student_version) AS student_version,
                   teacher_hash, max(teacher_version) AS teacher_version,
                   count(*) AS images,
                   count(*) FILTER (
                       WHERE student_total = 0 AND teacher_total = 0
                   ) AS empty_images,
                   count(*) FILTER (
                       WHERE student_total > student_matched
                          OR teacher_total > matched_count
                   ) AS disagreement_images,
                   sum(student_total) AS student_total,
                   sum(teacher_total) AS teacher_total,
                   sum(matched_count) AS matched_count,
                   sum(student_matched) AS student_matched
            FROM eval_results
            WHERE status = 'scored'
            GROUP BY student_source, teacher_source, student_hash, teacher_hash
            ORDER BY images DESC, student_hash
            """
        )
        status_rows = await conn.fetch(
            """
            SELECT status, count(*) AS record_count
            FROM eval_results
            GROUP BY status
            """
        )
        unscored = await conn.fetchval(
            """
            SELECT count(*)
            FROM detections d
            LEFT JOIN eval_results e ON e.image_id = d.image_id
            WHERE e.image_id IS NULL
            """
        )

    counts = {row["status"]: row["record_count"] for row in status_rows}
    return {
        "scored_images": counts.get("scored", 0),
        "skipped_images": counts.get("skipped", 0),
        "unscored_images": unscored,
        "pairs": [dict(row) for row in pair_rows],
    }


async def fetch_eval_disagreements(
    pool: asyncpg.Pool,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT e.image_id, d.device_id,
                   e.student_hash, e.student_version,
                   e.teacher_hash, e.teacher_version,
                   e.student_total, e.teacher_total,
                   e.matched_count, e.student_matched,
                   e.detail, e.captured_at, e.created_at
            FROM eval_results e
            JOIN detections d ON d.image_id = e.image_id
            WHERE e.status = 'scored'
              AND (e.student_total > e.student_matched
                   OR e.teacher_total > e.matched_count)
            ORDER BY coalesce(e.captured_at, e.created_at) DESC
            LIMIT $1
            """,
            limit,
        )
    return [serialize_eval_row(row) for row in rows]


def serialize_eval_row(row: asyncpg.Record) -> dict[str, Any]:
    data = dict(row)
    if isinstance(data.get("detail"), str):
        data["detail"] = json.loads(data["detail"])
    for key in ("image_id", "captured_at", "created_at"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data


# ---------------------------------------------------------------------------
# Teacher annotations (offline batch runner)
# ---------------------------------------------------------------------------


async def fetch_pending_teacher_images(
    pool: asyncpg.Pool,
    *,
    teacher_source: str,
    limit: int,
) -> list[str]:
    """Stored images this teacher has not annotated yet, newest first.

    Newest first so last night's captures get teacher labels on the next
    run even while a large historical backlog drains over many nights.
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT d.image_id
            FROM detections d
            LEFT JOIN teacher_annotations t
                ON t.image_id = d.image_id AND t.teacher_source = $1
            WHERE t.image_id IS NULL AND d.upload_status = 'stored'
            ORDER BY d.created_at DESC, d.image_id
            LIMIT $2
            """,
            teacher_source,
            limit,
        )
    return [str(row["image_id"]) for row in rows]


async def upsert_teacher_annotation(
    pool: asyncpg.Pool,
    *,
    image_id: UUID,
    teacher_source: str,
    teacher_hash: str,
    teacher_manifest: dict[str, Any] | None,
    detections: list[dict[str, Any]],
    error: str | None,
    inference_ms: int | None,
    imgsz: int | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO teacher_annotations (
                image_id, teacher_source, teacher_hash, teacher_manifest,
                detections, error, inference_ms, imgsz
            )
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6, $7, $8)
            ON CONFLICT (image_id, teacher_source) DO UPDATE SET
                teacher_hash = $3,
                teacher_manifest = $4::jsonb,
                detections = $5::jsonb,
                error = $6,
                inference_ms = $7,
                imgsz = $8,
                updated_at = now()
            """,
            image_id,
            teacher_source,
            teacher_hash,
            json.dumps(teacher_manifest) if teacher_manifest is not None else None,
            json.dumps(detections),
            error,
            inference_ms,
            imgsz,
        )


async def fetch_detections_by_ids(
    pool: asyncpg.Pool,
    image_ids: list[UUID],
) -> dict[UUID, dict[str, Any]]:
    """Metadata and captured_at for a batch of images, keyed by native id."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT image_id, metadata, captured_at
            FROM detections
            WHERE image_id = ANY($1::uuid[])
            """,
            image_ids,
        )
    result: dict[UUID, dict[str, Any]] = {}
    for row in rows:
        parsed = eval_input_row(row)
        result[row["image_id"]] = parsed
    return result


async def insert_teacher_run(
    pool: asyncpg.Pool,
    *,
    run_id: UUID,
    runner: str | None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO teacher_runs (run_id, runner, status)
            VALUES ($1, $2, 'running')
            """,
            run_id,
            runner,
        )


async def finish_teacher_run(
    pool: asyncpg.Pool,
    *,
    run_id: UUID,
    status: str,
    detail: dict[str, Any] | None,
) -> bool:
    async with pool.acquire() as conn:
        result = await conn.execute(
            """
            UPDATE teacher_runs
            SET status = $2,
                detail = $3::jsonb,
                finished_at = now()
            WHERE run_id = $1
            """,
            run_id,
            status,
            json.dumps(detail) if detail is not None else None,
        )
    return result.split()[-1] != "0"


async def fetch_teacher_runs(
    pool: asyncpg.Pool,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT run_id, runner, status, detail, started_at, finished_at
            FROM teacher_runs
            ORDER BY started_at DESC
            LIMIT $1
            """,
            limit,
        )
    runs = []
    for row in rows:
        data = dict(row)
        if isinstance(data.get("detail"), str):
            data["detail"] = json.loads(data["detail"])
        for key in ("run_id", "started_at", "finished_at"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        runs.append(data)
    return runs


# ---------------------------------------------------------------------------
# Pi receipt log
# ---------------------------------------------------------------------------


async def insert_pi_receipts(
    pool: asyncpg.Pool,
    receipts: list[dict[str, Any]],
) -> int:
    """Insert receipt rows, dropping ones already shipped; returns inserts."""
    inserted = 0
    async with pool.acquire() as conn:
        async with conn.transaction():
            for receipt in receipts:
                result = await conn.execute(
                    """
                    INSERT INTO pi_receipts (
                        receipt_id, pi_id, device_id, event, image_id,
                        filename, content_type, fomo_count, reason,
                        client_host, logged_at
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (receipt_id) DO NOTHING
                    """,
                    receipt["receipt_id"],
                    receipt.get("pi_id"),
                    receipt.get("device_id"),
                    receipt["event"],
                    receipt.get("image_id"),
                    receipt.get("filename"),
                    receipt.get("content_type"),
                    receipt.get("fomo_count"),
                    receipt.get("reason"),
                    receipt.get("client_host"),
                    receipt.get("logged_at"),
                )
                if result.split()[-1] != "0":
                    inserted += 1
    return inserted


async def fetch_pi_receipts(
    pool: asyncpg.Pool,
    *,
    device_id: str | None,
    event: str | None,
    limit: int,
) -> list[dict[str, Any]]:
    params: list[Any] = []
    clauses: list[str] = []
    if device_id is not None:
        params.append(device_id)
        clauses.append(f"device_id = ${len(params)}")
    if event is not None:
        params.append(event)
        clauses.append(f"event = ${len(params)}")
    where = " WHERE " + " AND ".join(clauses) if clauses else ""

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT *
            FROM pi_receipts{where}
            ORDER BY coalesce(logged_at, created_at) DESC
            LIMIT ${len(params) + 1}
            """,
            *params,
            limit,
        )

    receipts = []
    for row in rows:
        data = dict(row)
        for key in ("receipt_id", "logged_at", "created_at"):
            if data.get(key) is not None:
                data[key] = str(data[key])
        receipts.append(data)
    return receipts


# ---------------------------------------------------------------------------
# Device registry
# ---------------------------------------------------------------------------

DEVICE_TIMESTAMP_KEYS = (
    "desired_mode_at",
    "reported_mode_at",
    "desired_config_at",
    "reported_config_at",
    "first_seen_at",
    "last_hello_at",
    "last_upload_at",
    "last_seen_at",
    "updated_at",
)

DEVICE_JSON_KEYS = (
    "model_manifest",
    "desired_config",
    "reported_config",
)


def serialize_device_row(row: asyncpg.Record) -> dict[str, Any]:
    data = dict(row)
    for key in DEVICE_JSON_KEYS:
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key])
    for key in DEVICE_TIMESTAMP_KEYS:
        if data.get(key) is not None:
            data[key] = data[key].isoformat()
    return data


async def upsert_device_hello(
    pool: asyncpg.Pool,
    *,
    device_id: str,
    hardware_id: str | None,
    firmware_build: str | None,
    model_hash: str | None,
    model_manifest: dict[str, Any] | None,
    pi_id: str | None,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO devices (
                device_id, hardware_id, firmware_build, model_hash,
                model_manifest, pi_id, last_hello_at, last_seen_at
            )
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, now(), now())
            ON CONFLICT (device_id) DO UPDATE SET
                hardware_id = coalesce($2, devices.hardware_id),
                firmware_build = coalesce($3, devices.firmware_build),
                model_hash = coalesce($4, devices.model_hash),
                model_manifest = coalesce($5::jsonb, devices.model_manifest),
                pi_id = coalesce($6, devices.pi_id),
                last_hello_at = now(),
                last_seen_at = now(),
                updated_at = now()
            RETURNING *
            """,
            device_id,
            hardware_id,
            firmware_build,
            model_hash,
            json.dumps(model_manifest) if model_manifest is not None else None,
            pi_id,
        )
    return serialize_device_row(row)


async def touch_device_upload(
    pool: asyncpg.Pool,
    *,
    device_id: str,
    model_hash: str | None = None,
    model_manifest: dict[str, Any] | None = None,
) -> None:
    # Uploads register devices too, so cameras running pre-hello firmware
    # still appear in the registry (with model identity when stamped).
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO devices (
                device_id, model_hash, model_manifest,
                last_upload_at, last_seen_at
            )
            VALUES ($1, $2, $3::jsonb, now(), now())
            ON CONFLICT (device_id) DO UPDATE SET
                model_hash = coalesce($2, devices.model_hash),
                model_manifest = coalesce($3::jsonb, devices.model_manifest),
                last_upload_at = now(),
                last_seen_at = now(),
                updated_at = now()
            """,
            device_id,
            model_hash,
            json.dumps(model_manifest) if model_manifest is not None else None,
        )


async def touch_device_seen(pool: asyncpg.Pool, device_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO devices (device_id, last_seen_at)
            VALUES ($1, now())
            ON CONFLICT (device_id) DO UPDATE SET
                last_seen_at = now(),
                updated_at = now()
            """,
            device_id,
        )


async def expire_stale_positioning(
    pool: asyncpg.Pool,
    ttl_seconds: float,
    device_id: str | None = None,
) -> int:
    """Flip desired positioning back to automated once the TTL lapses.

    Called lazily from every read path (hello, SSE stream, device listing),
    so a forgotten positioning mode cannot outlive the TTL by more than one
    read. Bumping the seq makes the reversion push out to the device like
    any other change.
    """
    params: list[Any] = [ttl_seconds]
    device_clause = ""
    if device_id is not None:
        params.append(device_id)
        device_clause = " AND device_id = $2"
    async with pool.acquire() as conn:
        result = await conn.execute(
            f"""
            UPDATE devices
            SET desired_mode = 'automated',
                desired_mode_seq = desired_mode_seq + 1,
                desired_mode_at = now(),
                updated_at = now()
            WHERE desired_mode = 'positioning'
              AND desired_mode_at < now() - $1 * interval '1 second'
              {device_clause}
            """,
            *params,
        )
    return int(result.split()[-1])


async def fetch_devices(pool: asyncpg.Pool) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT * FROM devices ORDER BY device_id"
        )
    return [serialize_device_row(row) for row in rows]


async def fetch_device(
    pool: asyncpg.Pool,
    device_id: str,
) -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM devices WHERE device_id = $1",
            device_id,
        )
    return serialize_device_row(row) if row else None


async def delete_device(pool: asyncpg.Pool, device_id: str) -> bool:
    """Prune a registry row and its capture counter.

    Detections are untouched: they are the historical record and stay
    queryable by device filter. A pruned device that hellos again simply
    re-registers with a fresh row.
    """
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM devices WHERE device_id = $1",
            device_id,
        )
        await conn.execute(
            "DELETE FROM capture_requests WHERE device_id = $1",
            device_id,
        )
    return result.split()[-1] != "0"


async def set_desired_mode(
    pool: asyncpg.Pool,
    *,
    device_id: str,
    mode: str,
) -> dict[str, Any]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO devices (device_id, desired_mode, desired_mode_seq,
                                 desired_mode_at)
            VALUES ($1, $2, 1, now())
            ON CONFLICT (device_id) DO UPDATE SET
                desired_mode = $2,
                desired_mode_seq = devices.desired_mode_seq + 1,
                desired_mode_at = now(),
                updated_at = now()
            RETURNING *
            """,
            device_id,
            mode,
        )
    return serialize_device_row(row)


async def fetch_desired_mode(
    pool: asyncpg.Pool,
    device_id: str,
) -> tuple[str, int]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT desired_mode, desired_mode_seq FROM devices WHERE device_id = $1",
            device_id,
        )
    if row is None:
        return "automated", 0
    return row["desired_mode"], row["desired_mode_seq"]


async def set_desired_config(
    pool: asyncpg.Pool,
    *,
    device_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    # The jsonb merge lets one request adjust a single knob without
    # clobbering the others already set on the device.
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO devices (device_id, desired_config,
                                 desired_config_seq, desired_config_at)
            VALUES ($1, $2::jsonb, 1, now())
            ON CONFLICT (device_id) DO UPDATE SET
                desired_config =
                    coalesce(devices.desired_config, '{}'::jsonb)
                    || $2::jsonb,
                desired_config_seq = devices.desired_config_seq + 1,
                desired_config_at = now(),
                updated_at = now()
            RETURNING *
            """,
            device_id,
            json.dumps(config),
        )
    return serialize_device_row(row)


async def fetch_desired_config(
    pool: asyncpg.Pool,
    device_id: str,
) -> tuple[dict[str, Any] | None, int]:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT desired_config, desired_config_seq
            FROM devices WHERE device_id = $1
            """,
            device_id,
        )
    if row is None:
        return None, 0
    config = row["desired_config"]
    if config is None:
        return None, row["desired_config_seq"]
    if isinstance(config, str):
        config = json.loads(config)
    return config, row["desired_config_seq"]


async def record_reported_config(
    pool: asyncpg.Pool,
    *,
    device_id: str,
    config: dict[str, Any],
    seq: int,
) -> None:
    # Same stale-ack guard as reported modes.
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO devices (device_id, reported_config,
                                 reported_config_seq, reported_config_at,
                                 last_seen_at)
            VALUES ($1, $2::jsonb, $3, now(), now())
            ON CONFLICT (device_id) DO UPDATE SET
                reported_config = $2::jsonb,
                reported_config_seq = $3,
                reported_config_at = now(),
                last_seen_at = now(),
                updated_at = now()
            WHERE coalesce(devices.reported_config_seq, -1) <= $3
            """,
            device_id,
            json.dumps(config),
            seq,
        )


async def record_reported_mode(
    pool: asyncpg.Pool,
    *,
    device_id: str,
    mode: str,
    seq: int,
) -> None:
    # Acks can arrive out of order after retries; a stale seq must not
    # overwrite a newer report.
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO devices (device_id, reported_mode, reported_mode_seq,
                                 reported_mode_at, last_seen_at)
            VALUES ($1, $2, $3, now(), now())
            ON CONFLICT (device_id) DO UPDATE SET
                reported_mode = $2,
                reported_mode_seq = $3,
                reported_mode_at = now(),
                last_seen_at = now(),
                updated_at = now()
            WHERE coalesce(devices.reported_mode_seq, -1) <= $3
            """,
            device_id,
            mode,
            seq,
        )
