# Nicla Vision

This folder contains the code used on and around the Arduino Nicla Vision.

## Firmware

`firmware/main.py` runs on the Nicla through OpenMV firmware. It continuously
captures camera frames, runs the trained Edge Impulse/OpenMV FOMO model over a
2-by-3 tile grid, and sends detected frames plus JSON metadata over USB.

The USB packet format is:

1. `NIMG` magic bytes
2. 4-byte little-endian JSON metadata length
3. 4-byte little-endian JPEG length
4. UTF-8 JSON metadata
5. JPEG image bytes

## Receiver

`receiver/receive_nicla_images.py` runs on the Mac and listens for Nicla USB
packets. For each detection event, it saves the JPEG image and appends the
detections to a COCO-style `annotations.json` file.

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
