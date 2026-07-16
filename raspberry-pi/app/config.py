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

    # Cross-tile duplicate suppression for Nicla FOMO detections. Two centers
    # closer than this, in adjacent tiles, are treated as one object.
    tile_duplicate_distance_pixels: float = 48.0
    # How far from a shared tile boundary a center may sit and still be a
    # duplicate candidate. Roughly the firmware's tile overlap plus slack for
    # the centroid drift of a split object.
    tile_boundary_band_pixels: int = 40


settings = Settings()
