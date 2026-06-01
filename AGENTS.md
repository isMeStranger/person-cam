# person-cam — Agent Documentation

## Overview

Local YOLO11 pose detection using Ultralytics' native NCNN runtime. Takes webcam or video file input, runs real-time person pose inference, and displays an annotated OpenCV preview window with bounding boxes, keypoints, and FPS overlay.

## Project State

- **Python**: 3.14.3
- **Virtual env**: `.venv/` — fully set up with all dependencies
- **Dependencies**: ultralytics 8.4.59, ncnn 1.0.20260526, opencv-python 4.13.0.92, torch 2.12.0
- **Git**: single commit (`init commit`); working directory clean
- **Model**: NOT exported yet — must run `python export.py` first
- **Test video**: NOT present — provide your own `.mp4` or use webcam index `0`
- **Gitignore** covers `.venv/`, `.runtime/`, `Ultralytics/`, `__pycache__/`, `*.pt`, `ncnn_model/`, `runs/`, `*.mp4/avi/mov/mkv`

## Architecture

### File Map

```
person-cam/
├── export.py                        # CLI entrypoint: export PT model → NCNN
├── test_ncnn.py                     # CLI entrypoint: run inference loop
├── pyproject.toml                   # Package config, entry points
├── requirements.txt                 # Flat dep list
├── AGENTS.md                        # ← YOU ARE HERE
│
└── src/person_cam/
    ├── __init__.py                  # Package marker, exports __version__
    ├── runtime.py                   # Runtime dir bootstrap (creates .runtime/ subdirs, sets env vars)
    ├── export_model.py              # Downloads & exports YOLO pose model to NCNN format
    └── detector.py                  # Main detection loop (VideoCapture → YOLO → annotated display)
```

### Data Flow (Inference)

```
Video source (file/webcam)
  → cv2.VideoCapture.read()          # frame-by-frame
  → ncnn_model(frame, imgsz, conf)   # YOLO inference
  → result.plot()                    # draw boxes + keypoints on frame
  → cv2.imshow()                     # display preview window
  → optional: cv2.VideoWriter.write() # save annotated output
```

### Data Flow (Export)

```
Ultralytics PT model (e.g. yolo11n-pose.pt)
  → YOLO.export(format="ncnn", imgsz, half)
  → yolo11n-pose_ncnn_model/
      ├── model.ncnn.param
      └── model.ncnn.bin
```

## File-by-File Breakdown

### `export.py` — Top-level export entrypoint
- Adds `src/` to `sys.path`, then delegates to `person_cam.export_model.main()`.
- Run: `python export.py`
- Equivalent installed command: `person-cam-export`

### `test_ncnn.py` — Top-level inference entrypoint
- Adds `src/` to `sys.path`, then delegates to `person_cam.detector.main()`.
- Run: `python test_ncnn.py --source video.mp4` or `--source 0` for webcam.

### `src/person_cam/runtime.py` — Bootstrap
- Creates `.runtime/ultralytics/` and `.runtime/matplotlib/` dirs.
- Sets `YOLO_CONFIG_DIR` and `MPLCONFIGDIR` to those dirs (if not already set).
- **Must be called before any ultralytics import** — both `export_model.py` and `detector.py` do this at module level.

### `src/person_cam/export_model.py` — Model export logic
- Loads a YOLO pose model by name (default `yolo11n-pose.pt`) or local path.
- Exports to NCNN format with configurable `imgsz` (default 320) and `half` (default True).
- Output dir: `yolo11n-pose_ncnn_model/` in CWD.

### `src/person_cam/detector.py` — Inference loop
- **Args** (via argparse):
  - `--model`: path to NCNN model dir (default `yolo11n-pose_ncnn_model`)
  - `--source`: video file path, RTSP URL, or integer webcam index (default `test_street_video.mp4`)
  - `--imgsz`: inference size (default 320)
  - `--conf`: confidence threshold (default 0.50)
  - `--display`/`--no-display`: show preview window (default True)
  - `--output`: save annotated video to path
  - `--telemetry`/`--no-telemetry`: print per-frame box/conf/keypoint data (default True)
  - `--skip N`: process every Nth frame (0=all, 1=every other, etc.). Higher = less load, higher FPS.
- `parse_source()`: converts `"0"` → `0` (int for webcam), else leaves as str.
- `run_detector()`: main loop — opens capture, loads NCNN model, reads frames, calls `ncnn_model()`, annotates via `result.plot()`, draws FPS overlay, optionally writes to file and/or displays window.
  - Display window is auto-sized to half the screen dimensions (via `win32api.GetSystemMetrics` + `cv2.resize`), preserving aspect ratio.
- Press `q` to quit the preview window.
- Uses `cv2.VideoWriter_fourcc(*"mp4v")` for output.

### `src/person_cam/__init__.py`
- Exports `__version__ = "0.1.0"`.

## Key Technical Details

### Model
- **Type**: YOLO11 nano pose (keypoint detection — 17 body keypoints)
- **Format**: NCNN (`.param` + `.bin` files) — no PyTorch at inference time
- **Default inference size**: 320px (small → fast, but lower accuracy for small people)
- **Keypoint model**: outputs 17 keypoints per detected person (COCO skeleton)

### Runtime dirs
- `.runtime/ultralytics/` — Ultralytics config, cache, settings
- `.runtime/matplotlib/` — Matplotlib config cache
- Both are gitignored and auto-created on first import of any `person_cam` module.

### Inference conventions
- Model is called with `verbose=False` to suppress ultralytics logging per frame.
- FPS is computed as `processed_count / elapsed_since_start` (not rolling average).
- `result.plot()` returns a BGR numpy array ready for OpenCV display.
- Telemetry prints: `Detected Target | Box: [x1,y1,x2,y2] | Conf: 0.XX | Keypoints: N`.

## How to Run (from scratch)

```powershell
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Export NCNN model (downloads yolo11n-pose.pt if needed)
python export.py

# 3. Run on a video file
python test_ncnn.py --source path/to/video.mp4

# Or use webcam:
python test_ncnn.py --source 0

# With custom settings:
python test_ncnn.py --source 0 --imgsz 320 --conf 0.50 --output runs/annotated.mp4
```

## Agent Task Guidance

### Common tasks for this codebase:

1. **Switch to a different YOLO model**: change `DEFAULT_MODEL` in `detector.py` and/or `--model` in `export_model.py`. Must export that model to NCNN first. Support for classification/seg/OBB models is straightforward.

2. **Add custom post-processing**: modify `run_detector()` after `results = ncnn_model(...)`. Access `result.boxes`, `result.keypoints`, `result.masks` etc.

3. **Add a different visualizer**: replace `result.plot()` with custom OpenCV drawing. `annotated_frame` must be a BGR numpy array.

4. **Add streaming output (RTSP/RTMP)**: replace `cv2.VideoWriter` with a streaming encoder, or push frames via FFmpeg subprocess.

5. **Switch to async/fast inference**: the model call is synchronous — consider a thread pool or async pipeline for higher throughput.

6. **Multi-camera support**: spawn per-camera processes or threads, each running `run_detector()` with a different `--source`.

7. **Add tracking (e.g. ByteTrack, DeepSORT)**: available via `ultralytics` `model.track()` method — swap `ncnn_model(frame, ...)` for `ncnn_model.track(frame, ...)`.

### Code conventions
- No comments in source code.
- `from __future__ import annotations` at top of each module.
- Type hints throughout.
- `configure_runtime_dirs()` called at module import time before any ultralytics import.
- Error handling: early returns with exit codes (1 = source error, 2 = model missing).
- `finally` block in inference loop ensures cleanup of `cap`, `writer`, and OpenCV windows.

### Environment notes
- Windows (PowerShell 5.1) — paths use backslashes, activate with `.\.venv\Scripts\Activate.ps1`
- NCNN on Windows is prebuilt (ncnn wheel 1.0.20260526) — no manual compilation needed
- OpenCV highgui backend is the default Windows one (`cv2.imshow` works natively)
