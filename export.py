from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from person_cam.export_model import main


if __name__ == "__main__":
    main()
