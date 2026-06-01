from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import win32api

from person_cam.runtime import configure_runtime_dirs

configure_runtime_dirs()

import cv2

from ultralytics import YOLO


DEFAULT_MODEL = "yolo11n-pose_ncnn_model"
DEFAULT_SOURCE = "test_street_video.mp4"


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local YOLO11 pose detector through the NCNN runtime."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to exported NCNN model directory.",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video path, image-stream URL, or webcam index such as 0.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.50,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--display",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show the annotated OpenCV preview window.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for an annotated output video.",
    )
    parser.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print detected boxes and confidence values.",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Process every Nth frame. 0=process all. Higher = less load, higher FPS.",
    )
    return parser.parse_args()


def open_writer(
    output: Path | None,
    fps: float,
    width: int,
    height: int,
) -> cv2.VideoWriter | None:
    if output is None:
        return None

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(output), fourcc, fps or 30.0, (width, height))


def log_telemetry(result: Any) -> None:
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or len(boxes) == 0:
        return

    keypoint_count = len(keypoints) if keypoints is not None else 0
    for index, box in enumerate(boxes):
        bbox_coords = [round(value, 2) for value in box.xyxy[0].tolist()]
        confidence = box.conf[0].item()
        print(
            "Detected Target | "
            f"Box: {bbox_coords} | "
            f"Conf: {confidence:.2f} | "
            f"Keypoints: {keypoint_count}"
        )


def run_detector(args: argparse.Namespace) -> int:
    model_path = Path(args.model)
    if not model_path.exists():
        print(
            f"Error: NCNN model directory not found: {model_path}\n"
            "Run `python export.py` or `person-cam-export` first."
        )
        return 2

    print(f"Loading NCNN model: {model_path}")
    ncnn_model = YOLO(str(model_path))

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {args.source}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(args.output, fps, width, height)

    step = args.skip + 1

    target_w = win32api.GetSystemMetrics(0) // 2
    target_h = win32api.GetSystemMetrics(1) // 2
    display_scale = min(target_w / width, target_h / height)
    display_w = int(width * display_scale)
    display_h = int(height * display_scale)

    print("NCNN model loaded successfully. Beginning inference loop...")
    print("Press 'q' in the preview window to quit.")

    frame_count = 0
    processed_count = 0
    annotated_frame = None
    started_at = time.perf_counter()

    try:
        if args.display:
            cv2.namedWindow("Local NCNN Live Broadcast Simulation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Local NCNN Live Broadcast Simulation", display_w, display_h)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            should_process = (frame_count - 1) % step == 0

            if should_process:
                results = ncnn_model(
                    frame,
                    imgsz=args.imgsz,
                    verbose=False,
                    conf=args.conf,
                )
                result = results[0]
                annotated_frame = result.plot()
                processed_count += 1

                if args.telemetry:
                    log_telemetry(result)

                if writer is not None:
                    writer.write(annotated_frame)

            if args.display:
                elapsed = max(time.perf_counter() - started_at, 1e-9)
                if annotated_frame is not None:
                    vis_frame = annotated_frame.copy()
                else:
                    vis_frame = frame.copy()
                cv2.putText(
                    vis_frame,
                    f"FPS: {processed_count / elapsed:.1f}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (50, 255, 50),
                    2,
                    cv2.LINE_AA,
                )
                display_frame = cv2.resize(vis_frame, (display_w, display_h))
                cv2.imshow("Local NCNN Live Broadcast Simulation", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Processed {processed_count} frames ({frame_count} total read).")
    return 0


def main() -> None:
    raise SystemExit(run_detector(parse_args()))


if __name__ == "__main__":
    main()
