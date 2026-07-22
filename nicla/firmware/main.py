# Edge Impulse / OpenMV FOMO motion-gated object detection
#
# Frame differencing against a downscaled copy of the previous frame
# finds up to MOTION_MAX_CROPS changed regions, and the model runs only
# on those crops. A full 2x3 tile sweep still runs on the first frame,
# on manual captures, when the whole frame changed (lighting shift),
# and every FULL_SWEEP_INTERVAL_MS as a safety net, because
# differencing cannot see an object once it stops moving.
#
# Wireless upload format:
#   HTTP POST multipart/form-data to POST /detections
#   form field "metadata": JSON metadata
#   form file "image": raw RGB565 image bytes
#
# Detection and upload use the same HVGA frame so boxes line up exactly.
#
# LED meanings:
#   3 green flashes = main.py started
#   1 short red flash = object detected
#   1 short blue flash = image and metadata sent successfully
#   1 long red light = Wi-Fi/API transfer failed
#
# LEDs remain off during normal operation.

import sensor
import time
import ml
import math
import image
import pyb
import json
import binascii
import hashlib
import network
import socket
import gc

from ml.utils import NMS
from ml.preprocessing import Normalization

try:
    from wifi_config import (
        WIFI_SSID,
        WIFI_PASSWORD,
        API_URL,
        DEVICE_ID
    )
except ImportError:
    WIFI_SSID = ""
    WIFI_PASSWORD = ""
    API_URL = ""
    DEVICE_ID = "nicla-vision"


# ---------------------------------------------------------
# LED setup
# ---------------------------------------------------------

red_led = pyb.LED(1)
green_led = pyb.LED(2)
blue_led = pyb.LED(3)

red_led.off()
green_led.off()
blue_led.off()


def flash_led(led, count=1, on_ms=100, off_ms=100):
    for flash_number in range(count):
        led.on()
        time.sleep_ms(on_ms)
        led.off()

        if flash_number < count - 1:
            time.sleep_ms(off_ms)


# Three green flashes indicate that main.py started.
for _ in range(3):
    green_led.on()
    time.sleep_ms(250)
    green_led.off()
    time.sleep_ms(750)

red_led.off()
green_led.off()
blue_led.off()


# ---------------------------------------------------------
# Camera setup
# ---------------------------------------------------------

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.HVGA)  # 480 x 320 detection frame
sensor.skip_frames(time=2000)

# Frame differencing compares raw pixel values across frames, so the
# sensor's automatic adjustments must stay frozen or every exposure
# step reads as whole-frame motion. Locking after skip_frames keeps
# whatever the warmup converged on. Not every control exists on every
# sensor, so failures are only reported.
for lock_sensor_auto in (
    lambda: sensor.set_auto_gain(False),
    lambda: sensor.set_auto_exposure(False),
    lambda: sensor.set_auto_whitebal(False),
):
    try:
        lock_sensor_auto()
    except Exception as error:
        print("Sensor auto-lock unsupported:", error)


# ---------------------------------------------------------
# Detection settings
# ---------------------------------------------------------

MIN_CONFIDENCE = 0.35

threshold_list = [
    (math.ceil(MIN_CONFIDENCE * 255), 255)
]

# Keep socket writes small. Sending data[offset:] on a large
# framebuffer can allocate a second full-frame copy and fail.
HTTP_SEND_CHUNK_SIZE = 2048

FRAME_WIDTH = 480
FRAME_HEIGHT = 320

GRID_COLUMNS = 3
GRID_ROWS = 2

# Expand each grid tile by this many pixels on each side, clipped
# to the frame edges. With 160x160 base cells, 16 px per side creates
# a 32 px neighbor overlap, or 20% of the base tile width.
TILE_OVERLAP_PIXELS = 16

DRAW_TILE_GRID = True
DRAW_INFERENCE_ROIS = True
DRAW_DETECTIONS = True

FRAME_DELAY_MS = 0


# ---------------------------------------------------------
# Motion-gated inference settings
# ---------------------------------------------------------

# Differencing runs on a grayscale copy downscaled by this factor, so
# the retained previous frame costs 120 x 80 = 9.6 KB instead of a
# second 307 KB RGB565 frame.
MOTION_DOWNSCALE = 4

SMALL_WIDTH = FRAME_WIDTH // MOTION_DOWNSCALE
SMALL_HEIGHT = FRAME_HEIGHT // MOTION_DOWNSCALE

# Minimum per-pixel intensity delta (0-255) that counts as change.
MOTION_DIFF_THRESHOLD = 24

# Measured at the downscaled resolution, where one pixel covers a
# MOTION_DOWNSCALE x MOTION_DOWNSCALE block of the full frame.
MOTION_BLOB_MIN_PIXELS = 12

MOTION_MAX_CROPS = 3

# Crops match the model's 96 x 96 input exactly, so inference sees
# full sensor detail with no downscaling. Training data should be
# prepared with matching 96 x 96 object-centered crops.
CROP_SIZE = 96

# Motion crops may instead be cut at 192 x 192 and downsampled by the
# model preprocessing to the 96 x 96 input: less detail per pixel, but
# four times the field of view around the motion center for scenes
# where objects keep slipping out of the tight crop.
CONFIG_CROP_SIZES = (96, 192)

# When changed pixels cover this fraction of the frame the difference
# is global (lighting shift, exposure step), not an object entering;
# fall back to a full tile sweep instead of trusting the blobs.
MOTION_GLOBAL_CHANGE_FRACTION = 0.4

# Differencing cannot see an object once it stops moving, so a full
# tile sweep still runs at this interval as a safety net. Kept long
# because the stock model over-detects and each sweep is six chances
# for a false positive to become an upload; motion crops carry the
# real signal.
FULL_SWEEP_INTERVAL_MS = 10 * 60 * 1000

CONFIG_MIN_SWEEP_INTERVAL_MS = 5000
CONFIG_MAX_SWEEP_INTERVAL_MS = 24 * 60 * 60 * 1000
CONFIG_MIN_DIFF_THRESHOLD = 5
CONFIG_MAX_DIFF_THRESHOLD = 128


# ---------------------------------------------------------
# Remote-configurable settings
# ---------------------------------------------------------
#
# The dashboard owns these knobs; the Pi pushes them here as
# config:<seq>:<json> datagrams under the same high-water/ack
# discipline as desired mode. The constants above are the boot
# defaults, used until (and unless) a config ever arrives.
#
# model_enabled False turns the camera into a pure motion recorder:
# FOMO never runs, every motion event uploads its crops unfiltered
# (unbiased training data), and the sweep timer uploads a whole-frame
# scene reference instead of running tiles.

full_sweep_interval_ms = FULL_SWEEP_INTERVAL_MS
crop_size = CROP_SIZE
motion_diff_threshold = MOTION_DIFF_THRESHOLD
min_confidence = MIN_CONFIDENCE
model_enabled = True


# ---------------------------------------------------------
# Motion differencing
# ---------------------------------------------------------

current_small = sensor.alloc_extra_fb(
    SMALL_WIDTH,
    SMALL_HEIGHT,
    sensor.GRAYSCALE
)

previous_small = sensor.alloc_extra_fb(
    SMALL_WIDTH,
    SMALL_HEIGHT,
    sensor.GRAYSCALE
)

background_valid = False

motion_diff_thresholds = [
    (motion_diff_threshold, 255)
]


def downscale_into(img, destination):
    # draw_image converts RGB565 to the destination's grayscale format
    # while scaling; AREA averaging keeps small motion from aliasing
    # away entirely.
    destination.draw_image(
        img,
        0,
        0,
        x_scale=SMALL_WIDTH / FRAME_WIDTH,
        y_scale=SMALL_HEIGHT / FRAME_HEIGHT,
        hint=image.AREA
    )


def point_in_roi(x, y, roi):
    roi_x, roi_y, roi_width, roi_height = roi

    return (
        roi_x <= x < roi_x + roi_width
        and roi_y <= y < roi_y + roi_height
    )


def crop_roi_around(center_x, center_y):
    """Fixed-size square crop centered on a point, clamped to the frame."""
    half = crop_size // 2

    x = min(
        max(center_x - half, 0),
        FRAME_WIDTH - crop_size
    )

    y = min(
        max(center_y - half, 0),
        FRAME_HEIGHT - crop_size
    )

    return (x, y, crop_size, crop_size)


def detect_motion_regions(img):
    """Diff against the previous downscaled frame.

    Returns (crop_rois, global_change). Crop coordinates are in full
    frame pixels. global_change is True when the diff cannot be trusted
    to isolate objects: the first frame after boot, or a change so large
    it must be lighting rather than something entering the scene.
    """
    global background_valid

    downscale_into(img, current_small)

    if not background_valid:
        previous_small.replace(current_small)
        background_valid = True
        return [], True

    # difference() mutates in place. Consuming previous_small keeps the
    # buffer count at two: its old contents become the diff, then the
    # current frame is stored into it for the next iteration.
    previous_small.difference(current_small)

    blobs = previous_small.find_blobs(
        motion_diff_thresholds,
        pixels_threshold=MOTION_BLOB_MIN_PIXELS,
        area_threshold=MOTION_BLOB_MIN_PIXELS,
        merge=True,
        margin=4
    )

    previous_small.replace(current_small)

    if not blobs:
        return [], False

    changed_pixels = 0

    for blob in blobs:
        changed_pixels += blob.pixels()

    global_change_pixels = (
        MOTION_GLOBAL_CHANGE_FRACTION * SMALL_WIDTH * SMALL_HEIGHT
    )

    if changed_pixels >= global_change_pixels:
        return [], True

    crop_rois = []

    largest_first = sorted(
        blobs,
        key=lambda candidate: candidate.pixels(),
        reverse=True
    )

    for blob in largest_first:
        if len(crop_rois) >= MOTION_MAX_CROPS:
            break

        center_x = blob.cx() * MOTION_DOWNSCALE
        center_y = blob.cy() * MOTION_DOWNSCALE

        covered = False

        for existing_roi in crop_rois:
            if point_in_roi(center_x, center_y, existing_roi):
                covered = True
                break

        if covered:
            continue

        crop_rois.append(
            crop_roi_around(center_x, center_y)
        )

    return crop_rois, False


# ---------------------------------------------------------
# LED notification settings
# ---------------------------------------------------------

# Prevent the detection light from flashing every frame while
# the same object remains visible.
DETECTION_LED_COOLDOWN_MS = 1000

DETECTION_LED_ON_MS = 100
UPLOAD_SUCCESS_LED_ON_MS = 120
UPLOAD_FAILURE_LED_ON_MS = 1500


# ---------------------------------------------------------
# Manual capture trigger settings
# ---------------------------------------------------------

# The Raspberry Pi relays dashboard capture presses as UDP datagrams of the
# form b"snap:<counter>", where <counter> is the cloud's monotonic press
# count. Comparing counters against a high-water mark makes duplicate and
# stale datagrams harmless, so the Pi can retransmit freely.
#
# Desired-mode changes arrive on the same socket as b"mode:<seq>:<value>",
# with the same high-water-mark discipline on <seq>. The Pi resends a mode
# datagram until this device acknowledges it over HTTP, so delivery relies
# on repetition, never on any single packet.
#
# Remote config arrives as b"config:<seq>:<json>" with the same seq/ack
# scheme; the JSON payload stays far below one WiFi MTU, so a datagram
# never fragments and arrives whole or not at all.
CAPTURE_UDP_PORT = 5005

# Upper bound on frames queued from one counter jump. Protects against a
# runaway backlog if this device reboots and later sees a much larger
# counter than the one it remembers.
CAPTURE_MAX_BURST_FRAMES = 3


# ---------------------------------------------------------
# Device platform settings
# ---------------------------------------------------------

FIRMWARE_BUILD = "2026-07-22.1"

# Boot hello: registers this device with the Pi (and through it the cloud),
# fetches the desired mode, and sets the RTC from the server clock. Retried
# with backoff from the main loop until one succeeds; detection runs in
# automated mode meanwhile.
HELLO_RETRY_BASE_MS = 5000
HELLO_RETRY_MAX_MS = 120000

# An applied mode is confirmed to the Pi over HTTP. The Pi keeps resending
# the mode datagram until the ack lands, so this retry pace only matters
# while the Pi is unreachable.
MODE_ACK_RETRY_MS = 2000

# Positioning mode: automated sweeps and motion inference stop; instead a
# lean JPEG preview is posted at this cadence so the dashboard shows a
# slow live view for aiming. Manual captures still run the full pipeline.
PREVIEW_INTERVAL_MS = 1000
PREVIEW_JPEG_QUALITY = 60


# ---------------------------------------------------------
# Wireless image-transfer settings
# ---------------------------------------------------------

# Prevent repeated uploads of the same continuously visible object.
UPLOAD_COOLDOWN_MS = 3000

WIFI_CONNECT_TIMEOUT_MS = 10000
WIFI_RETRY_COOLDOWN_MS = 30000
HTTP_SOCKET_TIMEOUT_SECONDS = 20
REQUIRE_WIFI_BEFORE_DETECTION = True
PRINT_WIFI_SCAN_ON_FAILURE = True

DEBUG_PRINTS = True

HTTP_BOUNDARY = "----nicla-vision-boundary"

wlan = network.WLAN(network.WLAN.IF_STA)

last_wifi_attempt_ms = time.ticks_add(
    time.ticks_ms(),
    -WIFI_RETRY_COOLDOWN_MS
)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

def debug_print(*args):
    if DEBUG_PRINTS:
        print(*args)


# ---------------------------------------------------------
# Wi-Fi connection helpers
# ---------------------------------------------------------

def wireless_configured():
    return bool(WIFI_SSID and API_URL)


def ensure_wifi_connected():
    global last_wifi_attempt_ms

    if not wireless_configured():
        print(
            "Missing Wi-Fi config. Save wifi_config.py to the Nicla filesystem."
        )
        return False

    try:
        if wlan.isconnected():
            return True

    except Exception as error:
        print("Wi-Fi status check failed:", error)
        return False

    now = time.ticks_ms()

    if (
        time.ticks_diff(
            now,
            last_wifi_attempt_ms
        )
        < WIFI_RETRY_COOLDOWN_MS
    ):
        return False

    last_wifi_attempt_ms = now

    print("Connecting Wi-Fi:", WIFI_SSID)

    try:
        wlan.active(True)
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    except Exception as error:
        print("Wi-Fi connect failed:", error)
        return False

    started_at = time.ticks_ms()

    while True:
        try:
            if wlan.isconnected():
                break

        except Exception as error:
            print("Wi-Fi status check failed while connecting:", error)
            return False

        if (
            time.ticks_diff(
                time.ticks_ms(),
                started_at
            )
            > WIFI_CONNECT_TIMEOUT_MS
        ):
            print("Wi-Fi connection timed out.")
            print("Check SSID/password, 2.4 GHz Wi-Fi, and same network as Mac.")
            print_wifi_scan()
            return False

        time.sleep_ms(250)

    try:
        print("Wi-Fi connected:", wlan.ifconfig())

    except Exception:
        print("Wi-Fi connected.")

    return True


def decode_ssid(raw_ssid):
    try:
        return raw_ssid.decode("utf-8")
    except Exception:
        return str(raw_ssid)


def print_wifi_scan():
    if not PRINT_WIFI_SCAN_ON_FAILURE:
        return

    print("Scanning Wi-Fi networks...")

    try:
        networks = wlan.scan()
    except Exception as error:
        print("Wi-Fi scan failed:", error)
        return

    found_target = False

    for network_info in networks:
        ssid = decode_ssid(network_info[0])
        channel = network_info[2]
        rssi = network_info[3]
        security = network_info[4]

        if ssid == WIFI_SSID:
            found_target = True

        print(
            "  SSID:",
            ssid,
            "| channel:",
            channel,
            "| RSSI:",
            rssi,
            "| security:",
            security
        )

    if found_target:
        print("Target SSID is visible:", WIFI_SSID)
    else:
        print("Target SSID was NOT visible:", WIFI_SSID)


def api_reachable_hint():
    if not API_URL:
        return

    print("API URL:", API_URL)


api_reachable_hint()


def wait_for_initial_wifi():
    if not wireless_configured():
        print("Wi-Fi is not configured; detection loop will run without upload.")
        return

    if not REQUIRE_WIFI_BEFORE_DETECTION:
        ensure_wifi_connected()
        return

    print("Waiting for Wi-Fi before starting detections...")

    while not ensure_wifi_connected():
        red_led.on()
        time.sleep_ms(150)
        red_led.off()
        time.sleep_ms(850)

    print("Wi-Fi ready. Starting detections.")


# ---------------------------------------------------------
# Manual capture trigger
# ---------------------------------------------------------

capture_socket = None
capture_high_water = 0
pending_manual_captures = 0

# Desired-mode state. current_mode gates the main loop; the seq high-water
# mark makes stale/duplicate datagrams harmless; pending_mode_ack holds an
# applied (mode, seq) until the confirmation POST to the Pi succeeds.
current_mode = "automated"
mode_seq_high_water = 0
pending_mode_ack = None

# Remote-config state, same discipline. pending_config_ack holds only the
# seq; the ack body reports the applied values read at send time.
config_seq_high_water = 0
pending_config_ack = None


def apply_mode(mode, seq):
    """Adopt a desired mode if its seq is new; queue the ack.

    A seq equal to the high-water mark re-queues the ack without changing
    state: after a Pi restart the push loop knows nothing of past acks, so
    re-acking the current seq is what lets it go quiet again.
    """
    global current_mode
    global mode_seq_high_water
    global pending_mode_ack
    global background_valid
    global last_full_sweep_ms

    if mode not in ("automated", "positioning"):
        return

    try:
        seq = int(seq)
    except (ValueError, TypeError):
        return

    if seq < mode_seq_high_water:
        return

    if seq == mode_seq_high_water:
        if seq > 0:
            pending_mode_ack = (mode, seq)
        return

    mode_seq_high_water = seq

    if mode != current_mode:
        if current_mode == "positioning":
            # The camera was likely just physically moved; the motion
            # baseline is garbage. Rebuild it and sweep immediately.
            background_valid = False
            last_full_sweep_ms = time.ticks_add(
                time.ticks_ms(),
                -full_sweep_interval_ms
            )

        current_mode = mode
        print("Mode applied:", mode, "| seq:", seq)

    pending_mode_ack = (mode, seq)


def apply_config(config, seq):
    """Adopt desired config if its seq is new; queue the ack.

    Out-of-range values and unknown keys are skipped rather than
    rejected: the seq is acked either way, so a firmware older than the
    cloud converges instead of leaving the Pi retrying forever.
    """
    global full_sweep_interval_ms
    global crop_size
    global motion_diff_threshold
    global motion_diff_thresholds
    global min_confidence
    global threshold_list
    global model_enabled
    global config_seq_high_water
    global pending_config_ack

    if not isinstance(config, dict):
        return

    try:
        seq = int(seq)
    except (ValueError, TypeError):
        return

    if seq < config_seq_high_water:
        return

    if seq == config_seq_high_water:
        if seq > 0:
            pending_config_ack = seq
        return

    config_seq_high_water = seq

    interval = config.get("full_sweep_interval_ms")
    if (
        isinstance(interval, int)
        and CONFIG_MIN_SWEEP_INTERVAL_MS <= interval
        and interval <= CONFIG_MAX_SWEEP_INTERVAL_MS
    ):
        full_sweep_interval_ms = interval

    size = config.get("crop_size")
    if size in CONFIG_CROP_SIZES:
        crop_size = size

    diff = config.get("motion_diff_threshold")
    if (
        isinstance(diff, int)
        and not isinstance(diff, bool)
        and CONFIG_MIN_DIFF_THRESHOLD <= diff
        and diff <= CONFIG_MAX_DIFF_THRESHOLD
    ):
        motion_diff_threshold = diff
        motion_diff_thresholds = [
            (motion_diff_threshold, 255)
        ]

    confidence = config.get("min_confidence")
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0.0 <= confidence
        and confidence <= 1.0
    ):
        min_confidence = float(confidence)
        # The confidence gate lives in the FOMO heatmap blob cutoff,
        # which is precomputed; rebuild it or the change is silent.
        threshold_list = [
            (math.ceil(min_confidence * 255), 255)
        ]

    enabled = config.get("model_enabled")
    if isinstance(enabled, bool):
        model_enabled = enabled

    print(
        "Config applied | sweep_ms:", full_sweep_interval_ms,
        "| crop:", crop_size,
        "| diff:", motion_diff_threshold,
        "| conf:", min_confidence,
        "| model:", model_enabled,
        "| seq:", seq
    )

    pending_config_ack = seq


def ensure_capture_socket():
    global capture_socket

    if capture_socket is not None:
        return True

    try:
        if not wlan.isconnected():
            return False

        new_socket = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )
        new_socket.bind(("0.0.0.0", CAPTURE_UDP_PORT))
        new_socket.setblocking(False)

    except Exception as error:
        debug_print("Capture socket setup failed:", error)
        return False

    capture_socket = new_socket
    print("Capture trigger listening on UDP port", CAPTURE_UDP_PORT)
    return True


def parse_trigger_datagram(payload):
    """b"snap:<n>" -> ("snap", n); b"mode:<seq>:<value>" -> ("mode", seq, value);
    b"config:<seq>:<json>" -> ("config", seq, dict)."""
    try:
        text = payload.decode("utf-8").strip()
    except Exception:
        return None

    if text.startswith("snap:"):
        try:
            return ("snap", int(text[len("snap:"):]))
        except ValueError:
            return None

    if text.startswith("mode:"):
        pieces = text.split(":")

        if len(pieces) != 3:
            return None

        try:
            seq = int(pieces[1])
        except ValueError:
            return None

        if pieces[2] not in ("automated", "positioning"):
            return None

        return ("mode", seq, pieces[2])

    if text.startswith("config:"):
        # maxsplit keeps the JSON intact: it contains colons of its own.
        pieces = text.split(":", 2)

        if len(pieces) != 3:
            return None

        try:
            seq = int(pieces[1])
            config = json.loads(pieces[2])
        except (ValueError, TypeError):
            return None

        if not isinstance(config, dict):
            return None

        return ("config", seq, config)

    return None


def poll_capture_trigger():
    """Drain pending trigger datagrams; queue captures and apply modes.

    Each snap datagram reports the total number of presses so far. Anything
    at or below the high-water mark is a duplicate or a stale straggler and
    contributes nothing; an increase queues the difference, clamped so a
    reboot cannot create a huge backlog. Mode datagrams carry versioned
    desired state; only the newest drained seq is applied.
    """
    global capture_high_water
    global pending_manual_captures

    if not ensure_capture_socket():
        return

    highest_seen = 0
    mode_update = None
    config_update = None

    while True:
        try:
            # Sized for config JSON with headroom; snap/mode datagrams
            # stay tiny. LWIP truncates anything past the buffer.
            payload, _sender = capture_socket.recvfrom(512)
        except OSError:
            # Non-blocking socket with nothing queued.
            break

        parsed = parse_trigger_datagram(payload)

        if parsed is None:
            continue

        if parsed[0] == "snap":
            if parsed[1] > highest_seen:
                highest_seen = parsed[1]

        elif parsed[0] == "mode":
            if mode_update is None or parsed[1] > mode_update[0]:
                mode_update = (parsed[1], parsed[2])

        elif config_update is None or parsed[1] > config_update[0]:
            config_update = (parsed[1], parsed[2])

    if mode_update is not None:
        apply_mode(mode_update[1], mode_update[0])

    if config_update is not None:
        apply_config(config_update[1], config_update[0])

    if highest_seen <= capture_high_water:
        return

    if capture_high_water == 0:
        # First trigger since boot: the true backlog is unknowable, so
        # count it as a single press.
        queued_frames = 1
    else:
        queued_frames = min(
            highest_seen - capture_high_water,
            CAPTURE_MAX_BURST_FRAMES
        )

    capture_high_water = highest_seen
    pending_manual_captures += queued_frames

    debug_print(
        "Manual capture queued:",
        queued_frames,
        "| pending:",
        pending_manual_captures,
        "| counter:",
        capture_high_water
    )


# ---------------------------------------------------------
# HTTP upload helpers
# ---------------------------------------------------------

def infer_image_encoding(frame_width, frame_height, image_byte_count):
    pixel_count = frame_width * frame_height

    if image_byte_count == pixel_count:
        return "grayscale"

    if image_byte_count == pixel_count * 2:
        return "rgb565"

    return "unknown"


def build_detection_metadata(
    detections,
    frame_width,
    frame_height,
    image_encoding,
    image_byte_count,
    trigger,
    inference_mode,
    inference_rois
):
    metadata_detections = []

    for detection in detections:
        (
            x,
            y,
            width,
            height,
            center_x,
            center_y,
            color,
            roi_number,
            label,
            score
        ) = detection

        # Boxes and centers are always in full-frame pixels; the NMS
        # step maps model output through the inference ROI, so crop
        # mode needs no extra offsetting here.
        metadata_detections.append({
            "label": str(label),
            "score": float(score),
            "tile": int(roi_number),

            "box": [
                int(x),
                int(y),
                int(width),
                int(height)
            ],

            "center": [
                int(center_x),
                int(center_y)
            ]
        })

    metadata = {
        "version": 1,
        "device_id": DEVICE_ID,
        "transport": "wifi_http",
        "trigger": trigger,
        "image_encoding": image_encoding,
        "image_content_type": "application/octet-stream",
        "image_byte_count": image_byte_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "inference_mode": inference_mode,

        "inference_rois": [
            list(roi) for roi in inference_rois
        ],

        "minimum_confidence": min_confidence,

        # Remote-config stamp: which knob settings produced this upload,
        # so evals never silently mix capture regimes. model_enabled
        # False tells the eval layer FOMO never ran on this frame.
        "config_seq": config_seq_high_water,
        "crop_size": crop_size,
        "full_sweep_interval_ms": full_sweep_interval_ms,
        "motion_diff_threshold": motion_diff_threshold,
        "model_enabled": model_enabled,

        "model_hash": model_hash,
        "nicla_uptime_ms": time.ticks_ms(),
        "detection_count": len(metadata_detections),
        "detections": metadata_detections
    }

    if model_manifest is not None:
        metadata["model_manifest"] = model_manifest

    if inference_mode == "full_sweep":
        # The Pi's tile dedupe interprets "tile" as a grid position, so
        # only sweep uploads advertise grid geometry; without it the
        # dedupe stays inert for crop uploads, whose ROI indexes are
        # not grid cells.
        metadata["grid_columns"] = GRID_COLUMNS
        metadata["grid_rows"] = GRID_ROWS

    return metadata


def parse_http_url(url):
    if not url.startswith("http://"):
        raise ValueError("Only http:// API_URL values are supported")

    remainder = url[len("http://"):]
    slash_index = remainder.find("/")

    if slash_index < 0:
        host_port = remainder
        path = "/detections"
    else:
        host_port = remainder[:slash_index]
        path = remainder[slash_index:]

    if ":" in host_port:
        host, port_text = host_port.rsplit(":", 1)
        port = int(port_text)
    else:
        host = host_port
        port = 80

    if not path or path == "/":
        path = "/detections"

    return host, port, path


def send_all(sock, data):
    offset = 0
    data_length = len(data)

    while offset < data_length:
        end = min(
            offset + HTTP_SEND_CHUNK_SIZE,
            data_length
        )

        sent = sock.send(data[offset:end])

        if sent <= 0:
            raise OSError("socket connection broken")

        offset += sent


def post_multipart_image(api_url, metadata_bytes, image_bytes):
    host, port, path = parse_http_url(api_url)

    metadata_header = (
        "--" + HTTP_BOUNDARY + "\r\n"
        'Content-Disposition: form-data; name="metadata"\r\n'
        "\r\n"
    ).encode("utf-8")

    image_header = (
        "\r\n"
        "--" + HTTP_BOUNDARY + "\r\n"
        'Content-Disposition: form-data; '
        'name="image"; filename="nicla-frame.raw"\r\n'
        "Content-Type: application/octet-stream\r\n"
        "\r\n"
    ).encode("utf-8")

    closing_boundary = (
        "\r\n"
        "--" + HTTP_BOUNDARY + "--\r\n"
    ).encode("utf-8")

    content_length = (
        len(metadata_header)
        + len(metadata_bytes)
        + len(image_header)
        + len(image_bytes)
        + len(closing_boundary)
    )

    request_header = (
        "POST " + path + " HTTP/1.1\r\n"
        "Host: " + host + ":" + str(port) + "\r\n"
        "Content-Type: multipart/form-data; boundary=" + HTTP_BOUNDARY + "\r\n"
        "Content-Length: " + str(content_length) + "\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("utf-8")

    address = socket.getaddrinfo(host, port)[0][-1]
    sock = socket.socket()
    sock.settimeout(HTTP_SOCKET_TIMEOUT_SECONDS)

    try:
        sock.connect(address)
        send_all(sock, request_header)
        send_all(sock, metadata_header)
        send_all(sock, metadata_bytes)
        send_all(sock, image_header)
        send_all(sock, image_bytes)
        send_all(sock, closing_boundary)

        response = sock.recv(256)
        print("HTTP response:", response)

    finally:
        sock.close()

    return (
        response.startswith(b"HTTP/1.1 2")
        or response.startswith(b"HTTP/1.0 2")
    )


def read_http_response(sock, max_bytes=2048):
    """Read a small response until the server closes the connection."""
    chunks = b""

    while len(chunks) < max_bytes:
        try:
            chunk = sock.recv(512)
        except OSError:
            break

        if not chunk:
            break

        chunks += chunk

    return chunks


def parse_json_response(response_bytes):
    """Split a raw HTTP response into (status_ok, parsed_json_or_None)."""
    if not response_bytes:
        return False, None

    ok = (
        response_bytes.startswith(b"HTTP/1.1 2")
        or response_bytes.startswith(b"HTTP/1.0 2")
    )

    separator = response_bytes.find(b"\r\n\r\n")

    if separator < 0:
        return ok, None

    try:
        return ok, json.loads(response_bytes[separator + 4:])
    except ValueError:
        return ok, None


def post_body(path, content_type, body):
    """POST a small body to the Pi and return the raw HTTP response.

    All device-platform endpoints live on the same host/port as API_URL,
    so the base address is reused and only the path differs.
    """
    host, port, _ = parse_http_url(API_URL)

    request_header = (
        "POST " + path + " HTTP/1.1\r\n"
        "Host: " + host + ":" + str(port) + "\r\n"
        "Content-Type: " + content_type + "\r\n"
        "Content-Length: " + str(len(body)) + "\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("utf-8")

    address = socket.getaddrinfo(host, port)[0][-1]
    sock = socket.socket()
    sock.settimeout(HTTP_SOCKET_TIMEOUT_SECONDS)

    try:
        sock.connect(address)
        send_all(sock, request_header)
        send_all(sock, body)
        return read_http_response(sock)

    finally:
        sock.close()


def post_json(path, payload):
    return post_body(
        path,
        "application/json",
        json.dumps(payload).encode("utf-8")
    )


def raw_image_buffer(img):
    try:
        return img.bytearray()
    except AttributeError:
        return bytes(img)


def image_dimension(img, dimension_name, fallback):
    try:
        dimension = getattr(img, dimension_name)

        if callable(dimension):
            return int(dimension())

        return int(dimension)

    except Exception:
        return fallback


def send_detection_image(
    img,
    detections,
    trigger,
    inference_mode,
    inference_rois
):
    if not ensure_wifi_connected():
        return False

    gc.collect()

    upload_ok = False
    image_bytes = None
    metadata_bytes = None

    try:
        upload_width = image_dimension(
            img,
            "width",
            FRAME_WIDTH
        )

        upload_height = image_dimension(
            img,
            "height",
            FRAME_HEIGHT
        )

        print(
            "Upload frame:",
            upload_width,
            "x",
            upload_height
        )

        image_bytes = raw_image_buffer(img)
        image_byte_count = len(image_bytes)
        image_encoding = infer_image_encoding(
            upload_width,
            upload_height,
            image_byte_count
        )

        print(
            "Raw image encoding:",
            image_encoding,
            "| bytes:",
            image_byte_count
        )

        metadata = build_detection_metadata(
            detections,
            upload_width,
            upload_height,
            image_encoding,
            image_byte_count,
            trigger,
            inference_mode,
            inference_rois
        )
        metadata_bytes = json.dumps(metadata).encode("utf-8")

        upload_ok = post_multipart_image(
            API_URL,
            metadata_bytes,
            image_bytes
        )

    finally:
        image_bytes = None
        metadata_bytes = None
        gc.collect()

    return upload_ok


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

MODEL_FILE_PATH = "trained.tflite"
MODEL_MANIFEST_PATH = "model_manifest.json"

# Truncated SHA-256 still uniquely identifies the handful of models
# this project will ever train, and keeps upload metadata small.
MODEL_HASH_HEX_CHARS = 12


def compute_model_hash(path):
    """Hash of the deployed model file's exact bytes.

    The hash is the ground-truth model identity for analysis: two
    uploads share a hash only when byte-identical models produced
    them, even if the manifest was forgotten or wrong. Reading in
    chunks keeps the boot-time allocation small.
    """
    try:
        sha = hashlib.sha256()

        with open(path, "rb") as model_file:
            while True:
                chunk = model_file.read(1024)

                if not chunk:
                    break

                sha.update(chunk)

        digest_hex = binascii.hexlify(sha.digest()).decode("utf-8")
        return digest_hex[:MODEL_HASH_HEX_CHARS]

    except Exception as error:
        print("Model hash unavailable:", error)
        return None


def load_model_manifest(path):
    """Human-readable model info deployed alongside the model file.

    The manifest carries what the hash cannot: version label, training
    date, notes. It is optional so firmware still runs on devices that
    predate the convention.
    """
    try:
        with open(path) as manifest_file:
            manifest = json.load(manifest_file)

    except OSError:
        print("No model manifest found:", path)
        return None

    except ValueError as error:
        print("Model manifest is not valid JSON:", error)
        return None

    if not isinstance(manifest, dict):
        print("Model manifest ignored: expected a JSON object")
        return None

    return manifest


model = ml.Model("trained")

model_hash = compute_model_hash(MODEL_FILE_PATH)
model_manifest = load_model_manifest(MODEL_MANIFEST_PATH)

debug_print(model)
debug_print("Model input:", model.input_shape)
debug_print("Model output:", model.output_shape)
debug_print("Labels:", model.labels)
debug_print("Model hash:", model_hash)
debug_print("Model manifest:", model_manifest)


# ---------------------------------------------------------
# Device platform: hello, mode ack, tick, preview
# ---------------------------------------------------------

def hardware_id_hex():
    """The MCU's factory serial: ground-truth identity for this board.

    device_id is the human label and can collide or move between boards;
    this cannot, so the registry records both.
    """
    try:
        import machine
        return binascii.hexlify(machine.unique_id()).decode("utf-8")

    except Exception as error:
        print("Hardware id unavailable:", error)
        return None


def parse_iso_datetime(text):
    """(year, month, day, hours, minutes, seconds) from an ISO string.

    The server always sends UTC, so any trailing offset is ignored.
    """
    try:
        date_part, time_part = text.split("T", 1)
        year, month, day = [int(piece) for piece in date_part.split("-")]

        time_part = time_part.replace("Z", "")

        for offset_sign in ("+", "-"):
            offset_index = time_part.find(offset_sign)

            if offset_index > 0:
                time_part = time_part[:offset_index]
                break

        pieces = time_part.split(":")
        hours = int(pieces[0])
        minutes = int(pieces[1])
        seconds = int(float(pieces[2])) if len(pieces) > 2 else 0

        return year, month, day, hours, minutes, seconds

    except Exception:
        return None


def weekday_from_date(year, month, day):
    """Sakamoto's algorithm, shifted to the RTC's 1=Monday..7=Sunday."""
    offsets = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]

    if month < 3:
        year -= 1

    weekday_sunday_zero = (
        year + year // 4 - year // 100 + year // 400
        + offsets[month - 1] + day
    ) % 7

    if weekday_sunday_zero == 0:
        return 7

    return weekday_sunday_zero


def set_rtc_from_iso(text):
    parsed = parse_iso_datetime(text)

    if parsed is None:
        return False

    year, month, day, hours, minutes, seconds = parsed

    try:
        pyb.RTC().datetime((
            year,
            month,
            day,
            weekday_from_date(year, month, day),
            hours,
            minutes,
            seconds,
            0
        ))
        return True

    except Exception as error:
        print("RTC set failed:", error)
        return False


def send_hello():
    """Register with the Pi and adopt the answered mode and clock.

    Blocking is fine here: this runs before the loop starts, or between
    frames on retry, and the Pi answers from cache within a couple of
    seconds even when the cloud is unreachable.
    """
    if not wireless_configured() or not ensure_wifi_connected():
        return False

    payload = {
        "device_id": DEVICE_ID,
        "hardware_id": hardware_id_hex(),
        "firmware_build": FIRMWARE_BUILD,
        "model_hash": model_hash
    }

    if model_manifest is not None:
        payload["model_manifest"] = model_manifest

    try:
        response = post_json("/hello", payload)
    except Exception as error:
        print("Hello failed:", error)
        return False

    ok, body = parse_json_response(response)

    if not ok or not isinstance(body, dict):
        print("Hello rejected:", response)
        return False

    server_time = body.get("server_time")

    if server_time and set_rtc_from_iso(server_time):
        debug_print("RTC set from server time:", server_time)

    apply_mode(body.get("mode"), body.get("seq"))
    apply_config(body.get("config"), body.get("config_seq"))
    print("Hello ok | mode:", current_mode, "| seq:", mode_seq_high_water)
    return True


def send_mode_ack():
    """Confirm the applied mode to the Pi; kept pending until it lands."""
    global pending_mode_ack

    if pending_mode_ack is None:
        return

    mode, seq = pending_mode_ack

    try:
        response = post_json("/mode-ack", {
            "device_id": DEVICE_ID,
            "mode": mode,
            "seq": seq
        })
        ok, _ = parse_json_response(response)

    except Exception as error:
        debug_print("Mode ack failed:", error)
        return

    if ok:
        pending_mode_ack = None
        debug_print("Mode ack sent:", mode, "| seq:", seq)


def send_config_ack():
    """Confirm the applied config to the Pi; kept pending until it lands.

    The body reports the values actually in effect, so the dashboard
    shows what the camera runs, not what the cloud asked for.
    """
    global pending_config_ack

    if pending_config_ack is None:
        return

    seq = pending_config_ack

    try:
        response = post_json("/config-ack", {
            "device_id": DEVICE_ID,
            "seq": seq,
            "config": {
                "full_sweep_interval_ms": full_sweep_interval_ms,
                "crop_size": crop_size,
                "motion_diff_threshold": motion_diff_threshold,
                "min_confidence": min_confidence,
                "model_enabled": model_enabled
            }
        })
        ok, _ = parse_json_response(response)

    except Exception as error:
        debug_print("Config ack failed:", error)
        return

    if ok:
        pending_config_ack = None
        debug_print("Config ack sent | seq:", seq)


def send_tick():
    """Tiny liveness signal for sweeps that found nothing to upload."""
    try:
        response = post_json("/tick", {"device_id": DEVICE_ID})
        ok, _ = parse_json_response(response)
        return ok

    except Exception as error:
        debug_print("Tick failed:", error)
        return False


def send_preview_image(img):
    """Lean positioning preview: JPEG bytes, no metadata envelope.

    compress() mutates the framebuffer in place, which is safe here
    because positioning frames are never used for inference.
    """
    try:
        jpeg = img.compress(quality=PREVIEW_JPEG_QUALITY)
        body = raw_image_buffer(jpeg)
        response = post_body(
            "/preview?device_id=" + DEVICE_ID,
            "image/jpeg",
            body
        )
        ok, _ = parse_json_response(response)
        return ok

    except Exception as error:
        debug_print("Preview send failed:", error)
        return False


colors = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 255),
]


# ---------------------------------------------------------
# FOMO post-processing
# ---------------------------------------------------------

def fomo_post_process(model, inputs, outputs):
    (
        batch_size,
        output_height,
        output_width,
        output_classes
    ) = model.output_shape[0]

    nms = NMS(
        output_width,
        output_height,
        inputs[0].roi
    )

    for class_index in range(output_classes):
        confidence_image = image.Image(
            outputs[0][0, :, :, class_index] * 255
        )

        blobs = confidence_image.find_blobs(
            threshold_list,
            x_stride=1,
            area_threshold=1,
            pixels_threshold=1
        )

        for blob in blobs:
            x, y, width, height = blob.rect()

            score = (
                confidence_image.get_statistics(
                    thresholds=threshold_list,
                    roi=blob.rect()
                ).l_mean() / 255.0
            )

            nms.add_bounding_box(
                x,
                y,
                x + width,
                y + height,
                score,
                class_index
            )

    return nms.get_bounding_boxes()


# ---------------------------------------------------------
# Build tile regions
# ---------------------------------------------------------

def create_tile_rois(columns, rows, overlap_pixels):
    rois = []

    for row in range(rows):
        y1 = (row * FRAME_HEIGHT) // rows
        y2 = ((row + 1) * FRAME_HEIGHT) // rows

        for column in range(columns):
            x1 = (column * FRAME_WIDTH) // columns
            x2 = ((column + 1) * FRAME_WIDTH) // columns

            roi_x1 = max(
                0,
                x1 - overlap_pixels
            )

            roi_y1 = max(
                0,
                y1 - overlap_pixels
            )

            roi_x2 = min(
                FRAME_WIDTH,
                x2 + overlap_pixels
            )

            roi_y2 = min(
                FRAME_HEIGHT,
                y2 + overlap_pixels
            )

            rois.append(
                (
                    roi_x1,
                    roi_y1,
                    roi_x2 - roi_x1,
                    roi_y2 - roi_y1
                )
            )

    return rois


tile_rois = create_tile_rois(
    GRID_COLUMNS,
    GRID_ROWS,
    TILE_OVERLAP_PIXELS
)

debug_print(
    "Grid:",
    GRID_ROWS,
    "rows x",
    GRID_COLUMNS,
    "columns"
)

debug_print(
    "Number of inferences:",
    len(tile_rois)
)

debug_print(
    "Tile overlap pixels:",
    TILE_OVERLAP_PIXELS
)

debug_print(
    "Tiles:",
    tile_rois
)

wait_for_initial_wifi()


# ---------------------------------------------------------
# Main loop setup
# ---------------------------------------------------------

clock = time.clock()

last_upload_ms = time.ticks_add(
    time.ticks_ms(),
    -UPLOAD_COOLDOWN_MS
)

last_detection_led_ms = time.ticks_add(
    time.ticks_ms(),
    -DETECTION_LED_COOLDOWN_MS
)

last_full_sweep_ms = time.ticks_add(
    time.ticks_ms(),
    -full_sweep_interval_ms
)

last_preview_ms = time.ticks_add(
    time.ticks_ms(),
    -PREVIEW_INTERVAL_MS
)

last_mode_ack_attempt_ms = time.ticks_add(
    time.ticks_ms(),
    -MODE_ACK_RETRY_MS
)

last_config_ack_attempt_ms = time.ticks_add(
    time.ticks_ms(),
    -MODE_ACK_RETRY_MS
)

# The boot hello runs before the first frame; on failure the loop keeps
# detecting in automated mode and retries with backoff. A camera that
# cannot reach its Pi should still be a camera.
hello_retry_delay_ms = HELLO_RETRY_BASE_MS

last_hello_attempt_ms = time.ticks_ms()

hello_pending = wireless_configured() and not send_hello()


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:
    clock.tick()

    img = sensor.snapshot()

    now = time.ticks_ms()

    # Polled before mode selection so a pending manual capture can
    # force a full sweep, giving user-requested frames complete
    # annotations rather than only motion crops.
    poll_capture_trigger()

    if hello_pending and (
        time.ticks_diff(
            now,
            last_hello_attempt_ms
        )
        >= hello_retry_delay_ms
    ):
        last_hello_attempt_ms = now

        if send_hello():
            hello_pending = False
            hello_retry_delay_ms = HELLO_RETRY_BASE_MS
        else:
            hello_retry_delay_ms = min(
                hello_retry_delay_ms * 2,
                HELLO_RETRY_MAX_MS
            )

    if pending_mode_ack is not None and (
        time.ticks_diff(
            now,
            last_mode_ack_attempt_ms
        )
        >= MODE_ACK_RETRY_MS
    ):
        last_mode_ack_attempt_ms = now
        send_mode_ack()

    if pending_config_ack is not None and (
        time.ticks_diff(
            now,
            last_config_ack_attempt_ms
        )
        >= MODE_ACK_RETRY_MS
    ):
        last_config_ack_attempt_ms = now
        send_config_ack()

    manual_capture_due = pending_manual_captures > 0

    # -----------------------------------------------------
    # Positioning mode: preview instead of inference
    # -----------------------------------------------------
    #
    # Sweeps and motion inference stop; a manual capture still falls
    # through to the full pipeline as the aimed "confirmation shot".

    if current_mode == "positioning" and not manual_capture_due:
        preview_due = (
            time.ticks_diff(
                now,
                last_preview_ms
            )
            >= PREVIEW_INTERVAL_MS
        )

        if preview_due and wireless_configured():
            last_preview_ms = now
            send_preview_image(img)

        if FRAME_DELAY_MS > 0:
            time.sleep_ms(FRAME_DELAY_MS)

        continue

    total_detections = 0
    detections_to_draw = []

    # -----------------------------------------------------
    # Choose inference regions for this frame
    # -----------------------------------------------------

    motion_rois, global_change = detect_motion_regions(img)

    full_sweep_due = (
        time.ticks_diff(
            now,
            last_full_sweep_ms
        )
        >= full_sweep_interval_ms
    )

    if manual_capture_due or global_change or full_sweep_due:
        # With the model off, the sweep timer still fires but uploads
        # one whole frame as a scene reference instead of running tiles.
        inference_mode = "full_sweep" if model_enabled else "reference_frame"
        inference_rois = tile_rois if model_enabled else []
        last_full_sweep_ms = now
    elif motion_rois:
        # With the model off, motion crops upload unfiltered: the ROIs
        # are stamped so the dashboard shows what moved, nothing infers.
        inference_mode = "motion_crops" if model_enabled else "motion_only"
        inference_rois = motion_rois
    else:
        # Nothing changed since the previous frame; skip inference
        # entirely.
        inference_mode = "idle"
        inference_rois = []

    # -----------------------------------------------------
    # Run inference on the chosen regions
    # -----------------------------------------------------

    for roi_number, inference_roi in enumerate(
        inference_rois if model_enabled else []
    ):
        roi_input = Normalization(
            roi=inference_roi
        )(img)

        results = model.predict(
            [roi_input],
            callback=fomo_post_process
        )

        for class_index, detection_list in enumerate(results):
            # FOMO class zero is normally the background class.
            if class_index == 0:
                continue

            if len(detection_list) == 0:
                continue

            label = model.labels[class_index]

            color = colors[
                (class_index - 1) % len(colors)
            ]

            for rect, score in detection_list:
                x, y, width, height = rect

                center_x = math.floor(
                    x + width / 2
                )

                center_y = math.floor(
                    y + height / 2
                )

                detections_to_draw.append(
                    (
                        x,
                        y,
                        width,
                        height,
                        center_x,
                        center_y,
                        color,
                        roi_number,
                        label,
                        score
                    )
                )

                total_detections += 1

                debug_print(
                    inference_mode,
                    "roi",
                    roi_number,
                    "class",
                    label,
                    "x",
                    center_x,
                    "y",
                    center_y,
                    "score",
                    score
                )

    # -----------------------------------------------------
    # Detection LED
    # -----------------------------------------------------

    detection_led_ready = (
        time.ticks_diff(
            now,
            last_detection_led_ms
        )
        >= DETECTION_LED_COOLDOWN_MS
    )

    if total_detections > 0 and detection_led_ready:
        flash_led(
            red_led,
            count=1,
            on_ms=DETECTION_LED_ON_MS
        )

        last_detection_led_ms = now

    # -----------------------------------------------------
    # Wireless upload
    # -----------------------------------------------------

    upload_cooldown_complete = (
        time.ticks_diff(
            now,
            last_upload_ms
        )
        >= UPLOAD_COOLDOWN_MS
    )

    # With the model running, only frames with detections are worth the
    # bandwidth. With it off, motion itself is the event (unfiltered
    # training data) and reference frames ride the sweep timer.
    if model_enabled:
        upload_worthy = total_detections > 0
    else:
        upload_worthy = inference_mode in ("motion_only", "reference_frame")

    # Manual captures bypass the detection gate and the cooldown: the
    # user asked for this frame, so it uploads even when nothing was
    # detected. One frame is consumed per queued press.
    should_upload = wireless_configured() and (
        manual_capture_due
        or (upload_worthy and upload_cooldown_complete)
    )

    if should_upload:
        upload_succeeded = False

        if manual_capture_due:
            # Consume the press even if the upload fails; retrying
            # forever would starve detection uploads, and the failure
            # LED tells the user to press again.
            pending_manual_captures -= 1

        try:
            upload_succeeded = send_detection_image(
                img,
                detections_to_draw,
                "manual" if manual_capture_due else "detection",
                inference_mode,
                inference_rois
            )

        except Exception as error:
            debug_print(
                "Wireless image error:",
                error
            )

        # Apply the cooldown even when transmission fails.
        last_upload_ms = now

        if upload_succeeded:
            # Short blue flash means the packet was transmitted.
            flash_led(
                blue_led,
                count=1,
                on_ms=UPLOAD_SUCCESS_LED_ON_MS
            )

        else:
            # Long red light means the Wi-Fi/API transfer failed.
            red_led.on()
            time.sleep_ms(UPLOAD_FAILURE_LED_ON_MS)
            red_led.off()

        debug_print(
            "Wireless image sent:",
            upload_succeeded,
            "| detections:",
            total_detections
        )

        if FRAME_DELAY_MS > 0:
            time.sleep_ms(FRAME_DELAY_MS)

        continue

    # -----------------------------------------------------
    # Presence tick
    # -----------------------------------------------------
    #
    # A sweep that uploads nothing still proves this camera is alive and
    # looking. Piggybacking on the sweep cadence means a healthy camera
    # always emits something, so silence has exactly one meaning.

    if (
        current_mode == "automated"
        and inference_mode == "full_sweep"
        and wireless_configured()
    ):
        send_tick()

    # -----------------------------------------------------
    # Draw detections on non-upload frames
    # -----------------------------------------------------

    try:
        if DRAW_DETECTIONS:
            for detection in detections_to_draw:
                (
                    x,
                    y,
                    width,
                    height,
                    center_x,
                    center_y,
                    color,
                    tile_number,
                    label,
                    score
                ) = detection

                img.draw_circle(
                    center_x,
                    center_y,
                    12,
                    color=color,
                    thickness=2
                )

                img.draw_rectangle(
                    x,
                    y,
                    width,
                    height,
                    color=color,
                    thickness=2
                )

    # -----------------------------------------------------
    # Draw tile grid
    # -----------------------------------------------------

        if DRAW_TILE_GRID and inference_mode == "full_sweep":
            for column in range(1, GRID_COLUMNS):
                grid_x = (
                    column * FRAME_WIDTH
                ) // GRID_COLUMNS

                img.draw_line(
                    grid_x,
                    0,
                    grid_x,
                    FRAME_HEIGHT - 1,
                    color=(255, 255, 255)
                )

            for row in range(1, GRID_ROWS):
                grid_y = (
                    row * FRAME_HEIGHT
                ) // GRID_ROWS

                img.draw_line(
                    0,
                    grid_y,
                    FRAME_WIDTH - 1,
                    grid_y,
                    color=(255, 255, 255)
                )

        if DRAW_INFERENCE_ROIS:
            for inference_roi in inference_rois:
                (
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height
                ) = inference_roi

                img.draw_rectangle(
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height,
                    color=(0, 255, 255),
                    thickness=1
                )

    except ValueError as error:
        debug_print("Skipping annotation draw:", error)

    debug_print(
        "mode:",
        inference_mode,
        "| rois:",
        len(inference_rois),
        "| detections:",
        total_detections,
        "| frames/sec:",
        clock.fps()
    )

    if FRAME_DELAY_MS > 0:
        time.sleep_ms(FRAME_DELAY_MS)
