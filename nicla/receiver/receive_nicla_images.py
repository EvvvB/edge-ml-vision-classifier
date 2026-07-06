#!/usr/bin/env python3

import argparse
import glob
import json
import struct
import time

from datetime import datetime
from pathlib import Path

import serial


# ---------------------------------------------------------
# Nicla USB packet format
# ---------------------------------------------------------

MAGIC = b"NIMG"

# After the four-byte NIMG marker:
#
# 4 bytes: JSON metadata length
# 4 bytes: JPEG length
HEADER_REMAINDER = struct.Struct("<II")

MAX_METADATA_SIZE = 100_000
MAX_JPEG_SIZE = 1_000_000


# ---------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------

def read_exact(port, byte_count):
    output = bytearray()

    while len(output) < byte_count:
        chunk = port.read(byte_count - len(output))

        if chunk:
            output.extend(chunk)
            continue

        if not port.is_open:
            raise serial.SerialException(
                "Serial port closed while reading."
            )

    return bytes(output)


def wait_for_magic(port):
    matched = 0

    while matched < len(MAGIC):
        value = port.read(1)

        if not value:
            continue

        if value[0] == MAGIC[matched]:
            matched += 1

        elif value == MAGIC[:1]:
            matched = 1

        else:
            matched = 0


def find_nicla_port(requested_port):
    if requested_port != "auto":
        return requested_port

    ports = sorted(
        glob.glob("/dev/cu.usbmodem*")
    )

    if not ports:
        return None

    return ports[0]


def open_nicla(port_name):
    """
    Open the Nicla USB VCP and assert DTR so the Nicla's
    usb.isconnected() call recognizes the receiver.
    """

    nicla = serial.Serial(
        port=None,
        baudrate=115200,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        write_timeout=2,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False
    )

    nicla.port = port_name
    nicla.dtr = True
    nicla.rts = False

    nicla.open()

    # Reassert after opening because the operating system may
    # briefly alter the control-line state.
    nicla.dtr = True
    nicla.rts = False

    return nicla


# ---------------------------------------------------------
# COCO dataset helpers
# ---------------------------------------------------------

def new_coco_dataset():
    return {
        "info": {
            "description": "Nicla Vision FOMO detections",
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }


def load_coco_dataset(annotation_path):
    if not annotation_path.exists():
        return new_coco_dataset()

    try:
        with annotation_path.open(
            "r",
            encoding="utf-8"
        ) as file:
            coco = json.load(file)

    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "Could not load existing COCO file {}: {}".format(
                annotation_path,
                error
            )
        )

    # Ensure all expected top-level fields exist.
    coco.setdefault("info", {})
    coco.setdefault("licenses", [])
    coco.setdefault("images", [])
    coco.setdefault("annotations", [])
    coco.setdefault("categories", [])

    return coco


def next_available_id(items):
    if not items:
        return 1

    return max(
        int(item.get("id", 0))
        for item in items
    ) + 1


def get_or_create_category_id(coco, label):
    normalized_label = str(label).strip()

    if not normalized_label:
        normalized_label = "object"

    for category in coco["categories"]:
        if category.get("name") == normalized_label:
            return int(category["id"])

    category_id = next_available_id(
        coco["categories"]
    )

    coco["categories"].append({
        "id": category_id,
        "name": normalized_label,
        "supercategory": "object"
    })

    print(
        "Added COCO category {}: {}".format(
            category_id,
            normalized_label
        )
    )

    return category_id


def write_coco_atomically(coco, annotation_path):
    """
    Write to a temporary file first, then replace annotations.json.

    This reduces the chance of corrupting the dataset if the receiver
    is interrupted during a write.
    """

    temporary_path = annotation_path.with_suffix(
        ".json.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            coco,
            file,
            indent=2
        )

        file.write("\n")

    temporary_path.replace(annotation_path)


def clamp_bbox(x, y, width, height, frame_width, frame_height):
    """
    Keep the bounding box inside the saved image.
    """

    x = max(0, min(int(x), frame_width - 1))
    y = max(0, min(int(y), frame_height - 1))

    width = max(
        1,
        min(int(width), frame_width - x)
    )

    height = max(
        1,
        min(int(height), frame_height - y)
    )

    return x, y, width, height


# ---------------------------------------------------------
# Save one detection event into the COCO dataset
# ---------------------------------------------------------

def save_coco_event(
    coco,
    dataset_directory,
    images_directory,
    annotation_path,
    jpeg_data,
    metadata
):
    received_at = datetime.now().astimezone()

    timestamp = received_at.strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = "detection_{}.jpg".format(
        timestamp
    )

    image_path = images_directory / filename

    frame_width = int(
        metadata.get("frame_width", 320)
    )

    frame_height = int(
        metadata.get("frame_height", 240)
    )

    detections = metadata.get(
        "detections",
        []
    )

    # Save the original unannotated image.
    image_path.write_bytes(jpeg_data)

    image_id = next_available_id(
        coco["images"]
    )

    coco["images"].append({
        "id": image_id,
        "file_name": (
            Path("images") / filename
        ).as_posix(),
        "width": frame_width,
        "height": frame_height,
        "date_captured": received_at.isoformat()
    })

    annotation_id = next_available_id(
        coco["annotations"]
    )

    saved_annotation_count = 0

    for detection in detections:
        label = detection.get(
            "label",
            "object"
        )

        category_id = get_or_create_category_id(
            coco,
            label
        )

        box = detection.get(
            "box",
            [0, 0, 1, 1]
        )

        if len(box) != 4:
            print(
                "Skipping detection with invalid box: {}".format(
                    box
                )
            )
            continue

        x, y, width, height = clamp_bbox(
            box[0],
            box[1],
            box[2],
            box[3],
            frame_width,
            frame_height
        )

        score = float(
            detection.get("score", 0.0)
        )

        tile_number = int(
            detection.get("tile", -1)
        )

        center = detection.get(
            "center",
            [
                x + width // 2,
                y + height // 2
            ]
        )

        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,

            # COCO bounding-box order:
            # [left, top, width, height]
            "bbox": [
                x,
                y,
                width,
                height
            ],

            "area": width * height,
            "iscrowd": 0,

            # Extra fields are retained for reviewing pseudo-labels.
            # Most COCO importers ignore unknown fields.
            "attributes": {
                "confidence": score,
                "tile": tile_number,
                "center": center,
                "source": "nicla_fomo"
            }
        }

        coco["annotations"].append(
            annotation
        )

        annotation_id += 1
        saved_annotation_count += 1

    try:
        write_coco_atomically(
            coco,
            annotation_path
        )

    except Exception:
        # Remove the in-memory image entry if writing fails.
        coco["images"] = [
            image
            for image in coco["images"]
            if image.get("id") != image_id
        ]

        coco["annotations"] = [
            annotation
            for annotation in coco["annotations"]
            if annotation.get("image_id") != image_id
        ]

        try:
            image_path.unlink()
        except OSError:
            pass

        raise

    print("")
    print("Saved COCO detection event")
    print("  Image:       {}".format(image_path.name))
    print("  Image ID:    {}".format(image_id))
    print("  Annotations: {}".format(saved_annotation_count))
    print("  COCO file:   {}".format(annotation_path.name))


# ---------------------------------------------------------
# Receive packets
# ---------------------------------------------------------

def receive_packets(
    nicla,
    coco,
    dataset_directory,
    images_directory,
    annotation_path
):
    while True:
        wait_for_magic(nicla)

        print("Receiving detection packet...")

        header = read_exact(
            nicla,
            HEADER_REMAINDER.size
        )

        metadata_size, jpeg_size = (
            HEADER_REMAINDER.unpack(header)
        )

        print(
            "Packet sizes: metadata={} bytes, JPEG={} bytes".format(
                metadata_size,
                jpeg_size
            )
        )

        if (
            metadata_size <= 0
            or metadata_size > MAX_METADATA_SIZE
        ):
            print(
                "Invalid metadata size: {}".format(
                    metadata_size
                )
            )
            continue

        if (
            jpeg_size <= 0
            or jpeg_size > MAX_JPEG_SIZE
        ):
            print(
                "Invalid JPEG size: {}".format(
                    jpeg_size
                )
            )
            continue

        metadata_bytes = read_exact(
            nicla,
            metadata_size
        )

        jpeg_data = read_exact(
            nicla,
            jpeg_size
        )

        try:
            metadata = json.loads(
                metadata_bytes.decode("utf-8")
            )

        except Exception as error:
            print(
                "Invalid metadata JSON: {}".format(
                    error
                )
            )
            continue

        if not jpeg_data.startswith(b"\xff\xd8"):
            print(
                "Received data does not start with a JPEG header."
            )
            continue

        if not jpeg_data.endswith(b"\xff\xd9"):
            print(
                "Warning: JPEG does not end with the expected trailer."
            )

        save_coco_event(
            coco,
            dataset_directory,
            images_directory,
            annotation_path,
            jpeg_data,
            metadata
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Receive Nicla Vision images and append detections "
            "to a COCO dataset."
        )
    )

    parser.add_argument(
        "--port",
        default="auto",
        help="Serial port or 'auto'. Default: auto"
    )

    parser.add_argument(
        "--output",
        default="detections",
        help="COCO dataset directory."
    )

    args = parser.parse_args()

    dataset_directory = Path(
        args.output
    )

    images_directory = (
        dataset_directory / "images"
    )

    annotation_path = (
        dataset_directory / "annotations.json"
    )

    dataset_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    images_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    coco = load_coco_dataset(
        annotation_path
    )

    print(
        "COCO dataset: {}".format(
            dataset_directory.resolve()
        )
    )

    print(
        "Existing images: {}".format(
            len(coco["images"])
        )
    )

    print(
        "Existing annotations: {}".format(
            len(coco["annotations"])
        )
    )

    print(
        "Existing categories: {}".format(
            len(coco["categories"])
        )
    )

    while True:
        nicla = None

        port_name = find_nicla_port(
            args.port
        )

        if port_name is None:
            print(
                "Nicla USB port not found. Retrying..."
            )

            time.sleep(1)
            continue

        try:
            print(
                "Opening {}...".format(
                    port_name
                )
            )

            nicla = open_nicla(
                port_name
            )

            print(
                "Listening on {} with DTR enabled".format(
                    port_name
                )
            )

            time.sleep(0.5)

            nicla.reset_input_buffer()

            receive_packets(
                nicla,
                coco,
                dataset_directory,
                images_directory,
                annotation_path
            )

        except serial.SerialException as error:
            print(
                "USB connection lost or busy: {}".format(
                    error
                )
            )

            print("Retrying...")

        except OSError as error:
            print(
                "USB operating-system error: {}".format(
                    error
                )
            )

            print("Retrying...")

        except KeyboardInterrupt:
            print("\nReceiver stopped.")
            return

        finally:
            if nicla is not None and nicla.is_open:
                nicla.close()

        time.sleep(1)


if __name__ == "__main__":
    main()
