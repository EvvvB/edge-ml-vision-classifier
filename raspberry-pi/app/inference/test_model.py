from __future__ import annotations

import argparse
import os
from pathlib import Path


INFERENCE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = INFERENCE_DIR / "models" / "yolo26m.pt"
RUNS_DIR = INFERENCE_DIR / "runs"

os.environ.setdefault("YOLO_CONFIG_DIR", str(INFERENCE_DIR / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(INFERENCE_DIR / ".matplotlib"))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test the local YOLO26m model.")
    parser.add_argument(
        "image",
        nargs="?",
        help="Path to an image to run detection on.",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the YOLO model weights.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the prediction result image window.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save prediction results under the Ultralytics runs directory.",
    )
    parser.add_argument(
        "--train-coco8",
        action="store_true",
        help="Train the model on the COCO8 dataset.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs when using --train-coco8.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training, validation, and prediction.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to use, such as 'cpu', '0', or 'mps'.",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Run validation after loading the model.",
    )
    parser.add_argument(
        "--export-onnx",
        action="store_true",
        help="Export the model to ONNX after loading it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    if args.train_coco8:
        train_results = model.train(
            data="coco8.yaml",
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=args.device,
            project=str(RUNS_DIR),
        )
        print(train_results)

    if args.image:
        results = model(
            args.image,
            imgsz=args.imgsz,
            device=args.device,
            save=args.save,
            project=str(RUNS_DIR),
        )
        print(results[0].summary())
        if args.show:
            results[0].show()

    if args.val:
        metrics = model.val(imgsz=args.imgsz, device=args.device)
        print(metrics)

    if args.export_onnx:
        path = model.export(format="onnx")
        print(f"Exported ONNX model to: {path}")


if __name__ == "__main__":
    main()
