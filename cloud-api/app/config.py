from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "CLOUD_DATABASE_URL",
        os.getenv(
            "DATABASE_URL",
            "postgresql://vision:vision@localhost:5432/vision_classifier",
        ),
    )
    s3_bucket: str = os.getenv("CLOUD_S3_BUCKET", os.getenv("S3_BUCKET", ""))
    s3_prefix: str = os.getenv("CLOUD_S3_PREFIX", "detections")
    s3_endpoint_url: str | None = os.getenv("CLOUD_S3_ENDPOINT_URL") or None
    s3_force_path_style: bool = env_bool("CLOUD_S3_FORCE_PATH_STYLE")
    aws_region: str = os.getenv("AWS_REGION", "us-west-2")
    allowed_image_types: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp"}
    )
    allowed_image_suffixes: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp"}
    )
    default_image_suffix: str = ".jpg"
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CLOUD_CORS_ORIGINS",
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )

    @property
    def normalized_s3_prefix(self) -> str:
        return self.s3_prefix.strip("/")


settings = Settings()
