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

    if "py-opencv" in owned_names:
        raise RuntimeError(
            "Conda py-opencv is present alongside the required pip cv2 provider"
        )

    import cv2  # type: ignore[import-not-found]
    import numpy  # type: ignore[import-not-found]
    import ultralytics  # type: ignore[import-not-found]
    from importlib.metadata import PackageNotFoundError, version

    installed_opencv = []
    for name in sorted(PYPI_OPENCV_NAMES):
        try:
            installed_opencv.append((name, version(name)))
        except PackageNotFoundError:
            pass
    if [name for name, _ in installed_opencv] != ["opencv-python"]:
        raise RuntimeError(
            "Exactly one pip OpenCV provider is required: "
            f"expected opencv-python, found {installed_opencv}"
        )

    if cv2.__version__.split(".", 1)[0] != "4":
        raise RuntimeError(f"Expected OpenCV 4.x, found {cv2.__version__}")
    if version("opencv-python").split(".", 1)[0] != "4":
        raise RuntimeError("Expected opencv-python 4.x pip metadata")
    if version("numpy") != numpy.__version__:
        raise RuntimeError("Imported NumPy does not match its pip distribution metadata")

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(f"cv2 {cv2.__version__}: {cv2.__file__}")
    print(f"Ultralytics {ultralytics.__version__}: {ultralytics.__file__}")
    numpy_owner = "Conda (preserved exactly)" if "numpy" in owned_names else "pip"
    print(f"NumPy {numpy.__version__} [{numpy_owner}]: {numpy.__file__}")
    print("Conda native-library and pip Python-package ownership checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
