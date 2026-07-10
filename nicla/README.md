# Nicla Vision

This folder contains the code used on and around the Arduino Nicla Vision.

## Firmware

`firmware/main.py` runs on the Nicla through OpenMV firmware. It continuously
captures camera frames, runs the trained Edge Impulse/OpenMV FOMO model over a
2-by-3 tile grid, and sends detected frames plus JSON metadata over Wi-Fi to a
REST API.

The wireless upload format is `multipart/form-data` sent to `POST /detections`:

1. `metadata`: JSON metadata
2. `image`: raw RGB565 image bytes

The current firmware captures HVGA (`480x320`) RGB565 frames. The Nicla sends
raw bytes over Wi-Fi, and the Raspberry Pi receiver converts them to JPEG.

Copy the example Wi-Fi config and edit it for your local network:

```bash
cp nicla/firmware/wifi_config.example.py nicla/firmware/wifi_config.py
```

For local testing from your MacBook, `API_URL` must use your Mac's Wi-Fi/LAN IP
address, not `127.0.0.1`. On macOS, find it with:

```bash
ipconfig getifaddr en0
```

If that returns nothing, try `en1`:

```bash
ipconfig getifaddr en1
```

Example `wifi_config.py`:

```python
WIFI_SSID = "your-wifi-name"
WIFI_PASSWORD = "your-wifi-password"
API_URL = "http://192.168.1.50:8000/detections"
DEVICE_ID = "nicla-vision-01"
```

Run the API on your Mac so other devices on the same Wi-Fi can reach it:

```bash
cd raspberry-pi
fastapi dev app/main.py --host 0.0.0.0 --port 8000
```

The cloud API uses the same `POST /detections` shape, so you can point
`API_URL` at either local service. The important part is using `--host 0.0.0.0`
and your Mac's LAN IP address.

If the Nicla cannot connect, check that the Mac and Nicla are on the same Wi-Fi
network, macOS firewall allows incoming connections for Python/FastAPI, and the
router does not have client isolation enabled.

Then open `firmware/main.py` in OpenMV IDE and save both `main.py` and
`wifi_config.py` to the Nicla.

## Receiver

`receiver/receive_nicla_images.py` is the older USB dataset-capture receiver.
The wireless firmware does not require this receiver for normal uploads. Keep it
around only if you want to collect a local COCO-style dataset over USB.

Install the local receiver dependency:

```bash
python3 -m pip install -r nicla/receiver/requirements.txt
```

Run the receiver with automatic Nicla serial-port detection:

```bash
python3 nicla/receiver/receive_nicla_images.py --output nicla/datasets/detections
```

Use `--port /dev/cu.usbmodem...` if automatic detection picks the wrong serial
device.

## Generated Data

Generated detection datasets are intentionally ignored by git. Keep source code
and configuration in this folder, and store captured images/annotations under
`nicla/datasets/` or another local output directory.
