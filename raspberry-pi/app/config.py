from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    upload_dir: Path = BASE_DIR / "uploads"
    metadata_dir: Path = BASE_DIR / "metadata"
    allowed_image_types: frozenset[str] = frozenset(
        {"image/jpeg", "image/png", "image/webp"}
    )
    allowed_raw_image_types: frozenset[str] = frozenset(
        {"application/octet-stream"}
    )
    allowed_image_suffixes: frozenset[str] = frozenset(
        {".jpg", ".jpeg", ".png", ".webp"}
    )
    default_image_suffix: str = ".jpg"


settings = Settings()
