from __future__ import annotations

import os
import socket
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

    # Receipt log: one JSONL line per upload the Pi receives from a camera,
    # accepted or rejected. Shipped to the cloud by receipt-sync.timer and
    # pruned locally once synced and older than the retention window.
    receipt_log_dir: Path = BASE_DIR / "receipts"
    receipt_retention_days: int = int(os.environ.get("RECEIPT_RETENTION_DAYS", "14"))
    receipt_sync_batch_size: int = int(
        os.environ.get("RECEIPT_SYNC_BATCH_SIZE", "500")
    )

    # Cloud forwarding. Leaving CLOUD_API_URL unset disables forwarding, so
    # the Pi keeps working standalone.
    cloud_api_url: str = os.environ.get("CLOUD_API_URL", "")
    cloud_api_key: str = os.environ.get("CLOUD_API_KEY", "")
    cloud_forward_timeout_seconds: float = float(
        os.environ.get("CLOUD_FORWARD_TIMEOUT_SECONDS", "30")
    )
    cloud_forward_attempts: int = int(os.environ.get("CLOUD_FORWARD_ATTEMPTS", "3"))
    cloud_forward_retry_seconds: float = float(
        os.environ.get("CLOUD_FORWARD_RETRY_SECONDS", "2")
    )

    # Manual capture relay: the Pi subscribes to the cloud API's capture
    # stream (SSE) and forwards each press to the Nicla as a UDP datagram on
    # the LAN. Requires CLOUD_API_URL; the trigger target defaults to the
    # address the device last uploaded from.
    capture_device_id: str = os.environ.get("CAPTURE_DEVICE_ID", "nicla-vision-01")
    nicla_udp_host: str = os.environ.get("NICLA_UDP_HOST", "")
    nicla_udp_port: int = int(os.environ.get("NICLA_UDP_PORT", "5005"))
    # Datagrams are cheap and duplicates are idempotent on the Nicla, so each
    # press is sent a few times to ride out packet loss.
    capture_udp_repeats: int = int(os.environ.get("CAPTURE_UDP_REPEATS", "3"))
    # Must comfortably exceed the cloud stream's 20s heartbeat interval so a
    # healthy but quiet stream is not treated as dead.
    capture_stream_read_timeout_seconds: float = float(
        os.environ.get("CAPTURE_STREAM_READ_TIMEOUT_SECONDS", "45")
    )

    # Device gateway: the Pi answers camera hellos, relays them to the
    # cloud, pushes desired-mode changes over UDP until acked, and damps
    # heartbeat/preview chatter before it reaches the WAN.
    pi_id: str = os.environ.get("PI_ID", socket.gethostname())
    # The camera's boot sequence is held open during the relay, so this
    # stays short; on timeout the Pi answers from its cached state.
    hello_relay_timeout_seconds: float = float(
        os.environ.get("HELLO_RELAY_TIMEOUT_SECONDS", "2")
    )
    mode_push_retry_base_seconds: float = float(
        os.environ.get("MODE_PUSH_RETRY_BASE_SECONDS", "1")
    )
    mode_push_retry_max_seconds: float = float(
        os.environ.get("MODE_PUSH_RETRY_MAX_SECONDS", "30")
    )
    seen_relay_interval_seconds: float = float(
        os.environ.get("SEEN_RELAY_INTERVAL_SECONDS", "300")
    )
    preview_forward_min_interval_seconds: float = float(
        os.environ.get("PREVIEW_FORWARD_MIN_INTERVAL_SECONDS", "1")
    )
    preview_max_bytes: int = int(
        os.environ.get("PREVIEW_MAX_BYTES", str(1024 * 1024))
    )


settings = Settings()
