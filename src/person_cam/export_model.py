from __future__ import annotations

import argparse
from pathlib import Path

from person_cam.runtime import configure_runtime_dirs

configure_runtime_dirs()

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO pose model to NCNN for local inference."
    )
    parser.add_argument(
        "--model",
        default="yolo11n-pose.pt",
        help="Ultralytics model name or local .pt path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Export image size. Smaller values trade accuracy for speed.",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export with FP16 weights when supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    print(f"Loading pose model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Exporting NCNN model at imgsz={args.imgsz}, half={args.half}...")
    exported = model.export(format="ncnn", imgsz=args.imgsz, half=args.half)
    print(f"Export complete: {exported}")


if __name__ == "__main__":
    main()
