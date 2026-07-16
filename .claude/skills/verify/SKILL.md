---
name: verify
description: Build, launch, and drive this project's cloud-api + frontend to verify changes end-to-end.
---

# Verifying edge-ml-vision-classifier changes

## cloud-api (FastAPI, port 8000)

```bash
cd cloud-api
bin/compose up -d --wait                      # postgres :5432 + minio :9000 (bucket auto-created)
CLOUD_API_KEY=verify-key-123 .venv/bin/uvicorn app.main:app --port 8000   # background this
curl -s localhost:8000/ready                  # {"ok": true} when DB+S3 reachable
```

`.env` already points at the compose stack. Setting `CLOUD_API_KEY` in the
environment overrides `.env` (load_dotenv does not override existing vars) and
turns on auth, which you want when verifying the frontend key flow.

Seed a detection (must be a real decodable image — the API only checks the
content type, and the browser will show a broken image otherwise):

```bash
curl -s -X POST localhost:8000/detections -H "X-API-Key: verify-key-123" \
  -F "image=@frame.png;type=image/png" \
  -F 'metadata={"device_id": "nicla-vision-01", "captured_at": "2026-07-16T08:00:00Z"}'
```

## frontend (Vite + React, port 5173)

```bash
cd frontend && npm run dev    # background this; proxies /detections + /health to :8000
```

Drive with Playwright (not a project dep — install `playwright` +
`npx playwright install chromium` in the scratchpad). Key selectors:
`.keygate-card` (key entry), `.keygate-error`, `.detection-card img`,
`.modal`, `button.ghost` (Lock). API key lives in localStorage under
`cloud-api-key`.

## Gotchas

- The compose volumes persist old smoke-test detections, including one whose
  "image" is the README as image/jpeg (device `local-smoke-test`) — it should
  render as an "image unavailable" placeholder, not break the grid.
- Image responses must carry `Cache-Control: public, max-age=31536000,
  immutable`; on reload, `performance.getEntriesByType('resource')` entries
  for /image URLs should show `transferSize: 0` (browser cache hit).
- Images embed the key as `?key=` (img tags can't send headers); the API
  accepts header or query param on the image endpoint only.
