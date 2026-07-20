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


def build_detection_filters(
    *,
    device_id: str | None,
    labels: list[str],
    models: list[str],
    detections: str,
    source: str,
    params: list[Any],
) -> str:
    """Append filter params to `params` and return a WHERE clause (or '')."""
    sources = DETECTION_SOURCES if source == "any" else (source,)
    clauses: list[str] = []

    if device_id is not None:
        params.append(device_id)
        clauses.append(f"device_id = ${len(params)}")

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
