# Raspberry Pi FastAPI Receiver

This service accepts detection uploads at `POST /detections` using `multipart/form-data`.

## Run locally

```bash
cd raspberry-pi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
fastapi dev app.py --host 127.0.0.1 --port 8000
```

## Send an image and metadata

```bash
curl -X POST http://127.0.0.1:8000/detections \
  -F 'image=@/path/to/image.jpg;type=image/jpeg' \
  -F 'metadata={"device_id":"pi-01","label":"cat","confidence":0.94,"captured_at":"2026-07-02T20:15:00-07:00"}'
```

The server saves images under `raspberry-pi/uploads/` and metadata JSON files under `raspberry-pi/metadata/`. Both files share the same generated `image_id`.
