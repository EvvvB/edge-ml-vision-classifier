# Raspberry Pi FastAPI Receiver

This service accepts detection uploads at `POST /detections` using `multipart/form-data`.

## Run locally

```bash
cd raspberry-pi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
fastapi dev app/main.py --host 127.0.0.1 --port 8000
```

## Send an image and metadata

```bash
curl -X POST http://127.0.0.1:8000/detections \
  -F 'image=@/path/to/image.jpg;type=image/jpeg' \
  -F 'metadata={"device_id":"pi-01","label":"cat","confidence":0.94,"captured_at":"2026-07-02T20:15:00-07:00"}'
```

The receiver also accepts raw Nicla uploads as `application/octet-stream`.
For the current HVGA RGB565 color firmware, metadata includes:

```json
{
  "image_encoding": "rgb565",
  "frame_width": 480,
  "frame_height": 320,
  "image_byte_count": 307200
}
```

Raw uploads are converted to high-quality JPEG files before storage and
inference. The receiver reads the dimensions from metadata, so other raw
frame sizes can work as long as `image_byte_count` matches the encoding and
dimensions.

The server saves images under `raspberry-pi/uploads/` and metadata JSON files under `raspberry-pi/metadata/`. Both files share the same generated `image_id`.

## Model weights

Model weights are stored locally under `app/inference/models/` and ignored by Git.

```bash
mkdir -p app/inference/models
curl -fL https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt \
  -o app/inference/models/yolo26m.pt
```

Run a quick prediction with the local model:

```bash
python app/inference/test_model.py uploads/example.jpg --save
```

Optional model commands:

```bash
python app/inference/test_model.py --val
python app/inference/test_model.py --export-onnx
python app/inference/test_model.py --train-coco8 --epochs 100
```

## Project structure

```text
app/
  main.py                  # FastAPI application setup
  api/                     # HTTP routes
  services/                # Business workflow
  inference/               # Model loading and prediction boundary
  storage/                 # Filesystem persistence
  config.py                # Paths and app settings
```
