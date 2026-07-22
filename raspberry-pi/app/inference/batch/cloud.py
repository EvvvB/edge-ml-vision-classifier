from __future__ import annotations

from typing import Any

import httpx


class CloudEvalClient:
    """HTTP client for the cloud eval API.

    The runner is deliberately API-only — no database credentials — so the
    same code runs on the Pi, a laptop, or anywhere else with the API key.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 60.0):
        if not base_url:
            raise SystemExit(
                "CLOUD_API_URL is not set; the teacher runner needs the cloud API"
            )
        headers = {"X-API-Key": api_key} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def pending_image_ids(self, teacher_source: str, limit: int) -> list[str]:
        response = self._client.get(
            "/eval/teacher/pending",
            params={"teacher_source": teacher_source, "limit": limit},
        )
        response.raise_for_status()
        return response.json().get("image_ids", [])

    def download_image(self, image_id: str) -> bytes:
        response = self._client.get(f"/detections/{image_id}/image")
        response.raise_for_status()
        return response.content

    def post_batch(
        self,
        *,
        teacher_source: str,
        teacher_hash: str,
        teacher_manifest: dict[str, Any] | None,
        annotations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        response = self._client.post(
            "/eval/teacher/batch",
            json={
                "teacher_source": teacher_source,
                "teacher_hash": teacher_hash,
                "teacher_manifest": teacher_manifest,
                "annotations": annotations,
            },
        )
        response.raise_for_status()
        return response.json()

    # Run bookkeeping is observability, not correctness: a failure here
    # must never stop the annotation work, so both calls swallow errors.

    def start_run(self, runner: str) -> str | None:
        try:
            response = self._client.post(
                "/eval/teacher/runs", json={"runner": runner}
            )
            response.raise_for_status()
            return response.json().get("run_id")
        except httpx.HTTPError:
            return None

    def finish_run(
        self, run_id: str | None, status: str, detail: dict[str, Any]
    ) -> None:
        if run_id is None:
            return
        try:
            self._client.patch(
                f"/eval/teacher/runs/{run_id}",
                json={"status": status, "detail": detail},
            )
        except httpx.HTTPError:
            pass
