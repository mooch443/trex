#!/usr/bin/env python3
"""Validate the installed Conda minimal dependency profile."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


REQUIRED_CONDA_PACKAGES = {
    "ffmpeg",
    "libopencv",
    "libpng",
    "libzip",
    "py-opencv",
    "zlib",
}
PYPI_OPENCV_NAMES = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}


def main() -> int:
    prefix = Path(sys.prefix).resolve()
    records = []
    for record_path in (prefix / "conda-meta").glob("*.json"):
        records.append(json.loads(record_path.read_text(encoding="utf-8")))

    owned_names = {record.get("name") for record in records}
    missing = sorted(REQUIRED_CONDA_PACKAGES - owned_names)
    if missing:
        raise RuntimeError(f"Conda does not own required minimal packages: {missing}")

    conda_list = json.loads(
        subprocess.check_output(
            ["conda", "list", "--json", "--prefix", str(prefix)], text=True
        )
    )
    pypi_opencv = [
        item
        for item in conda_list
        if item.get("name") in PYPI_OPENCV_NAMES
        and str(item.get("channel", "")).lower() == "pypi"
    ]
    if pypi_opencv:
        raise RuntimeError(f"A PyPI OpenCV wheel was installed: {pypi_opencv}")

    import cv2  # type: ignore[import-not-found]
    import ultralytics  # type: ignore[import-not-found]
    from importlib.metadata import version

    if cv2.__version__.split(".", 1)[0] != "4":
        raise RuntimeError(f"Expected OpenCV 4.x, found {cv2.__version__}")
    if version("opencv-python").split(".", 1)[0] != "4":
        raise RuntimeError("Conda py-opencv does not provide OpenCV 4 pip metadata")

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(f"cv2 {cv2.__version__}: {cv2.__file__}")
    print(f"Ultralytics {ultralytics.__version__}: {ultralytics.__file__}")
    print("Conda minimal dependency ownership and Python metadata checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
