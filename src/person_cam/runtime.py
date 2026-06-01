from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_dirs() -> None:
    runtime_dir = Path.cwd() / ".runtime"
    yolo_dir = runtime_dir / "ultralytics"
    matplotlib_dir = runtime_dir / "matplotlib"

    yolo_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

