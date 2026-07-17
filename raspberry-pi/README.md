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

## Cloud forwarding

After inference finishes, each detection (image plus merged metadata: device
fields, FOMO detections, YOLO detections) is forwarded to the cloud API's
`POST /detections` endpoint. Forwarding is off unless `CLOUD_API_URL` is set,
so the Pi keeps working standalone.

```bash
export CLOUD_API_URL="http://<cloud-host>"
export CLOUD_API_KEY="<key from the server's .env.production>"
fastapi dev app/main.py --host 0.0.0.0 --port 8000
```

Optional tuning:

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLOUD_FORWARD_TIMEOUT_SECONDS` | `30` | Per-request timeout |
| `CLOUD_FORWARD_ATTEMPTS` | `3` | Attempts per detection (retries on network errors and 5xx only) |
| `CLOUD_FORWARD_RETRY_SECONDS` | `2` | First retry delay, doubling each attempt |

The outcome lands in the local metadata JSON as `cloud_sync_status`
(`synced` or `failed`), with `cloud_image_id` holding the cloud's UUID on
success and `cloud_sync_error` holding the reason on failure. A forwarding
failure never blocks or fails the local pipeline.

## Manual capture relay

When `CLOUD_API_URL` is set, the Pi also holds an SSE connection to the cloud
API's `/devices/{device_id}/capture/stream` endpoint. Each dashboard
"Capture photo" press increments a monotonic counter in the cloud; the Pi
relays every increase to the Nicla as a UDP datagram (`snap:<counter>`) on
the LAN, repeated a few times because the firmware dedupes by counter. The
target address is learned from the device's most recent upload and persisted
in `metadata/device_addresses.json`.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CAPTURE_DEVICE_ID` | `nicla-vision-01` | Device whose capture stream to subscribe to |
| `NICLA_UDP_HOST` | last upload's source address | Static trigger target override |
| `NICLA_UDP_PORT` | `5005` | Firmware's trigger listener port |
| `CAPTURE_UDP_REPEATS` | `3` | Datagrams sent per press |
| `CAPTURE_STREAM_READ_TIMEOUT_SECONDS` | `45` | Stream considered dead without traffic for this long |

## Run at boot (systemd)

[deploy/vision-receiver.service](deploy/vision-receiver.service) runs the
receiver as a system service. It reads its environment from
`/etc/vision-receiver.env`, which is `.env.cloud` converted to systemd's
format (no `export` prefix). Install on the Pi:

```bash
sudo sh -c 'sed "s/^export //" ~ev/edge-ml-vision-classifier/raspberry-pi/.env.cloud \
  > /etc/vision-receiver.env && chmod 600 /etc/vision-receiver.env'
sudo cp ~ev/edge-ml-vision-classifier/raspberry-pi/deploy/vision-receiver.service \
  /etc/systemd/system/vision-receiver.service
sudo systemctl daemon-reload
sudo systemctl enable --now vision-receiver
```

Check on it with:

```bash
systemctl status vision-receiver
journalctl -u vision-receiver -f
```

The unit assumes the repo lives at `/home/ev/edge-ml-vision-classifier`;
edit the paths and `User=` if that differs. After changing `.env.cloud`,
regenerate `/etc/vision-receiver.env` (first command above) and
`sudo systemctl restart vision-receiver`.

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
