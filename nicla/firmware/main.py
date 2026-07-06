# Edge Impulse / OpenMV FOMO tiled object detection
#
# 2 rows x 3 columns = 6 inferences per frame
#
# USB packet format:
#   4 bytes: b"NIMG"
#   4 bytes: JSON metadata length
#   4 bytes: JPEG length
#   N bytes: JSON metadata
#   N bytes: JPEG image
#
# LED meanings:
#   3 green flashes = main.py started
#   1 short red flash = object detected
#   1 short blue flash = image and metadata sent successfully
#   1 long red light = USB transfer failed
#
# LEDs remain off during normal operation.

import sensor
import time
import ml
import math
import image
import struct
import pyb
import json

from ml.utils import NMS
from ml.preprocessing import Normalization


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
sensor.set_framesize(sensor.QVGA)  # 320 x 240
sensor.skip_frames(time=2000)


# ---------------------------------------------------------
# Detection settings
# ---------------------------------------------------------

MIN_CONFIDENCE = 0.35

threshold_list = [
    (math.ceil(MIN_CONFIDENCE * 255), 255)
]

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

GRID_COLUMNS = 3
GRID_ROWS = 2

DRAW_TILE_GRID = True
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
# USB image-transfer settings
# ---------------------------------------------------------

USB_JPEG_QUALITY = 75

# Prevent repeated uploads of the same continuously visible object.
USB_UPLOAD_COOLDOWN_MS = 3000

USB_SEND_TIMEOUT_MS = 10000

# Keep False during binary USB transfers.
DEBUG_PRINTS = False

USB_MAGIC = b"NIMG"

usb = pyb.USB_VCP()

# JPEG data may contain byte 0x03, which normally means Ctrl+C.
usb.setinterrupt(-1)


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

def debug_print(*args):
    if DEBUG_PRINTS:
        print(*args)


# ---------------------------------------------------------
# USB connection check
# ---------------------------------------------------------

def usb_receiver_connected():
    if not usb.isconnected():
        return False

    # Avoid sending binary packets into the OpenMV IDE terminal.
    try:
        if usb.debug_mode_enabled():
            return False
    except AttributeError:
        # Older firmware might not have debug_mode_enabled().
        pass

    return True


# ---------------------------------------------------------
# Send image and metadata over USB
# ---------------------------------------------------------

def send_detection_image(img, detections):
    """
    Packet format:

        4 bytes: b"NIMG"
        4 bytes: JSON metadata length
        4 bytes: JPEG length
        N bytes: JSON metadata
        N bytes: JPEG image
    """

    if not usb_receiver_connected():
        return False

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

    metadata = {
        "version": 1,
        "frame_width": FRAME_WIDTH,
        "frame_height": FRAME_HEIGHT,
        "grid_columns": GRID_COLUMNS,
        "grid_rows": GRID_ROWS,
        "minimum_confidence": MIN_CONFIDENCE,
        "nicla_uptime_ms": time.ticks_ms(),
        "detection_count": len(metadata_detections),
        "detections": metadata_detections
    }

    metadata_bytes = json.dumps(metadata).encode("utf-8")
    metadata_size = len(metadata_bytes)

    # Compress the original, unannotated frame in place.
    img.compress(quality=USB_JPEG_QUALITY)

    jpeg_size = img.size()

    header = struct.pack(
        "<4sII",
        USB_MAGIC,
        metadata_size,
        jpeg_size
    )

    header_sent = usb.send(
        header,
        timeout=USB_SEND_TIMEOUT_MS
    )

    if header_sent != len(header):
        return False

    metadata_sent = usb.send(
        metadata_bytes,
        timeout=USB_SEND_TIMEOUT_MS
    )

    if metadata_sent != metadata_size:
        return False

    image_sent = usb.send(
        img,
        timeout=USB_SEND_TIMEOUT_MS
    )

    return image_sent == jpeg_size


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

def create_tile_rois(columns, rows):
    rois = []

    for row in range(rows):
        y1 = (row * FRAME_HEIGHT) // rows
        y2 = ((row + 1) * FRAME_HEIGHT) // rows

        for column in range(columns):
            x1 = (column * FRAME_WIDTH) // columns
            x2 = ((column + 1) * FRAME_WIDTH) // columns

            rois.append(
                (
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                )
            )

    return rois


tile_rois = create_tile_rois(
    GRID_COLUMNS,
    GRID_ROWS
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
    "Tiles:",
    tile_rois
)


# ---------------------------------------------------------
# Main loop setup
# ---------------------------------------------------------

clock = time.clock()

last_upload_ms = time.ticks_add(
    time.ticks_ms(),
    -USB_UPLOAD_COOLDOWN_MS
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
    # USB upload
    # -----------------------------------------------------

    upload_cooldown_complete = (
        time.ticks_diff(
            now,
            last_upload_ms
        )
        >= USB_UPLOAD_COOLDOWN_MS
    )

    should_upload = (
        total_detections > 0
        and upload_cooldown_complete
        and usb_receiver_connected()
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
                "USB image error:",
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
            # Long red light means the USB transfer failed.
            red_led.on()
            time.sleep_ms(UPLOAD_FAILURE_LED_ON_MS)
            red_led.off()

        debug_print(
            "USB image sent:",
            upload_succeeded,
            "| detections:",
            total_detections
        )

        # img was JPEG-compressed in place, so capture a new frame
        # instead of trying to draw onto the compressed image.
        if FRAME_DELAY_MS > 0:
            time.sleep_ms(FRAME_DELAY_MS)

        continue

    # -----------------------------------------------------
    # Draw detections on non-upload frames
    # -----------------------------------------------------

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

    debug_print(
        "detections:",
        total_detections,
        "| tiled frames/sec:",
        clock.fps()
    )

    if FRAME_DELAY_MS > 0:
        time.sleep_ms(FRAME_DELAY_MS)
