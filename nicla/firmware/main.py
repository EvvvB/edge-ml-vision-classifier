# Edge Impulse / OpenMV FOMO tiled object detection
#
# 2 rows x 3 columns = 6 inferences per frame
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
DRAW_TILE_ROIS = True
DRAW_DETECTIONS = True

FRAME_DELAY_MS = 0


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
    image_byte_count
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
            tile_number,
            label,
            score
        ) = detection

        metadata_detections.append({
            "label": str(label),
            "score": float(score),
            "tile": int(tile_number),

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

    return {
        "version": 1,
        "device_id": DEVICE_ID,
        "transport": "wifi_http",
        "image_encoding": image_encoding,
        "image_content_type": "application/octet-stream",
        "image_byte_count": image_byte_count,
        "frame_width": frame_width,
        "frame_height": frame_height,
        "grid_columns": GRID_COLUMNS,
        "grid_rows": GRID_ROWS,
        "minimum_confidence": MIN_CONFIDENCE,
        "nicla_uptime_ms": time.ticks_ms(),
        "detection_count": len(metadata_detections),
        "detections": metadata_detections
    }


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


def send_detection_image(img, detections):
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
            image_byte_count
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

model = ml.Model("trained")

debug_print(model)
debug_print("Model input:", model.input_shape)
debug_print("Model output:", model.output_shape)
debug_print("Labels:", model.labels)


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


# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------

while True:
    clock.tick()

    img = sensor.snapshot()

    total_detections = 0
    detections_to_draw = []

    # -----------------------------------------------------
    # Run inference on all six tiles
    # -----------------------------------------------------

    for tile_number, tile_roi in enumerate(tile_rois):
        tile_input = Normalization(
            roi=tile_roi
        )(img)

        results = model.predict(
            [tile_input],
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
                        tile_number,
                        label,
                        score
                    )
                )

                total_detections += 1

                debug_print(
                    "tile",
                    tile_number,
                    "class",
                    label,
                    "x",
                    center_x,
                    "y",
                    center_y,
                    "score",
                    score
                )

    now = time.ticks_ms()

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

    should_upload = (
        total_detections > 0
        and upload_cooldown_complete
        and wireless_configured()
    )

    if should_upload:
        upload_succeeded = False

        try:
            upload_succeeded = send_detection_image(
                img,
                detections_to_draw
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

        if DRAW_TILE_GRID:
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

        if DRAW_TILE_ROIS:
            for tile_roi in tile_rois:
                (
                    roi_x,
                    roi_y,
                    roi_width,
                    roi_height
                ) = tile_roi

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
        "detections:",
        total_detections,
        "| tiled frames/sec:",
        clock.fps()
    )

    if FRAME_DELAY_MS > 0:
        time.sleep_ms(FRAME_DELAY_MS)
