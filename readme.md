# Detect People in Live Video

Local Python scaffold for testing YOLO11 pose detection through Ultralytics'
native NCNN runtime.

## Setup

Use Python 3.10+ in a virtual environment, then install the project:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

If editable install is not needed, this also works:

```powershell
pip install -r requirements.txt
```

## Export YOLO11 Pose to NCNN

```powershell
python export.py
```

This downloads `yolo11n-pose.pt` if needed and exports:

```text
yolo11n-pose_ncnn_model/
  model.ncnn.param
  model.ncnn.bin
```

Equivalent installed command:

```powershell
person-cam-export --model yolo11n-pose.pt --imgsz 320 --half
```

## Run the Detector

Use a local video file:

```powershell
python test_ncnn.py --source test_street_video.mp4
```

Use a webcam:

```powershell
python test_ncnn.py --source 0
```

Useful options:

```powershell
python test_ncnn.py --source 0 --imgsz 320 --conf 0.50 --output runs/annotated.mp4
python test_ncnn.py --source test_street_video.mp4 --no-display
```

Watch for detection stability in shadows, frame-rate drops when multiple people
enter the frame, and whether side-profile or angled people keep stable
keypoints.
