from __future__ import annotations

from typing import Any
from uuid import UUID

import boto3
from botocore.config import Config
from starlette.concurrency import run_in_threadpool

from app.config import settings


def create_s3_client() -> Any:
    client_options: dict[str, Any] = {
        "region_name": settings.aws_region,
    }
    if settings.s3_endpoint_url:
        client_options["endpoint_url"] = settings.s3_endpoint_url
    if settings.s3_force_path_style:
        client_options["config"] = Config(s3={"addressing_style": "path"})
    return boto3.client("s3", **client_options)


async def check_s3(s3_client: Any, bucket: str) -> None:
    await run_in_threadpool(s3_client.head_bucket, Bucket=bucket)


async def download_image(
    *,
    s3_client: Any,
    bucket: str,
    key: str,
) -> dict[str, Any]:
    response = await run_in_threadpool(s3_client.get_object, Bucket=bucket, Key=key)
    body = await run_in_threadpool(response["Body"].read)
    return {
        "body": body,
        "content_type": response.get("ContentType"),
        "etag": response.get("ETag"),
    }


# DeleteObjects accepts at most 1000 keys per request.
DELETE_BATCH_SIZE = 1000


async def delete_images(
    *,
    s3_client: Any,
    bucket: str,
    keys: list[str],
) -> None:
    for start in range(0, len(keys), DELETE_BATCH_SIZE):
        batch = keys[start : start + DELETE_BATCH_SIZE]
        await run_in_threadpool(
            s3_client.delete_objects,
            Bucket=bucket,
            Delete={
                "Objects": [{"Key": key} for key in batch],
                "Quiet": True,
            },
        )


async def upload_image(
    *,
    s3_client: Any,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str,
    image_id: UUID,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    object_metadata = {
        "image_id": str(image_id),
    }
    device_id = metadata.get("device_id")
    if device_id is not None:
        object_metadata["device_id"] = str(device_id)

    return await run_in_threadpool(
        s3_client.put_object,
        Bucket=bucket,
        Key=key,
        Body=body,
        ContentType=content_type,
        Metadata=object_metadata,
    )
