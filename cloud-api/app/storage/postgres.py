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


async def fetch_detections(
    pool: asyncpg.Pool,
    *,
    device_id: str | None,
    limit: int,
    offset: int,
) -> list[dict[str, Any]]:
    async with pool.acquire() as conn:
        if device_id is None:
            rows = await conn.fetch(
                """
                SELECT *
                FROM detections
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT *
                FROM detections
                WHERE device_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                device_id,
                limit,
                offset,
            )
    return [serialize_row(row) for row in rows]


def serialize_row(row: asyncpg.Record) -> dict[str, Any]:
    data = dict(row)
    if isinstance(data.get("metadata"), str):
        data["metadata"] = json.loads(data["metadata"])
    for key in ("image_id", "captured_at", "created_at", "updated_at"):
        if data.get(key) is not None:
            data[key] = str(data[key])
    return data
