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
    "numpy",
    "py-opencv",
    "zlib",
}
PYPI_OPENCV_NAMES = {
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
}


def normalized_record_path(path: str) -> str:
    """Normalize a Conda record path for comparison on every platform."""
    return path.replace("\\", "/").casefold()


def opencv_api_version(raw_version: str) -> tuple[str, ...]:
    """Return the OpenCV API version, excluding a pip wheel build suffix."""
    return tuple(raw_version.split(".")[:3])


def validate_required_conda_packages(owned_names: set[str | None]) -> None:
    """Require every native and Python package owned by the minimal recipe."""
    missing = sorted(REQUIRED_CONDA_PACKAGES - owned_names)
    if missing:
        raise RuntimeError(f"Conda does not own required minimal packages: {missing}")


def validate_opencv_providers(
    conda_cv2_owners: list[str],
    installed_opencv: list[tuple[str, str, str]],
) -> list[tuple[str, str, str]]:
    """Validate that exactly one implementation owns the imported cv2 module."""
    external_opencv = [item for item in installed_opencv if item[2] != "conda"]
    if len(conda_cv2_owners) > 1:
        raise RuntimeError(
            f"Multiple Conda packages own the imported cv2: {conda_cv2_owners}"
        )
    if conda_cv2_owners:
        if external_opencv:
            raise RuntimeError(
                "cv2 is owned by Conda and an external OpenCV wheel is also "
                f"present: Conda={conda_cv2_owners}, external={external_opencv}"
            )
    else:
        if len(external_opencv) != 1:
            raise RuntimeError(
                "Exactly one external cv2 provider is required when Conda does "
                f"not own cv2: found {external_opencv}"
            )
        if external_opencv[0][0] != "opencv-python":
            raise RuntimeError(
                "The only permitted external cv2 provider is opencv-python: "
                f"found {external_opencv}"
            )
    return external_opencv


def main() -> int:
    prefix = Path(sys.prefix).resolve()
    records = []
    for record_path in (prefix / "conda-meta").glob("*.json"):
        records.append(json.loads(record_path.read_text(encoding="utf-8")))

    owned_names = {record.get("name") for record in records}
    validate_required_conda_packages(owned_names)

    import cv2  # type: ignore[import-not-found]
    import numpy  # type: ignore[import-not-found]
    import ultralytics  # type: ignore[import-not-found]
    from importlib.metadata import PackageNotFoundError, distribution, version

    installed_opencv = []
    for name in sorted(PYPI_OPENCV_NAMES):
        try:
            metadata = distribution(name)
        except PackageNotFoundError:
            continue
        installed_opencv.append(
            (
                name,
                metadata.version,
                (metadata.read_text("INSTALLER") or "unknown").strip().casefold(),
            )
        )

    # Conda-forge's OpenCV 4 packages deliberately publish both
    # opencv-python and opencv-python-headless compatibility metadata. Count
    # actual cv2 file ownership and non-Conda installers instead of treating
    # those two aliases as two Python implementations.
    cv2_path = Path(cv2.__file__).resolve()
    try:
        cv2_record_path = normalized_record_path(cv2_path.relative_to(prefix).as_posix())
    except ValueError as error:
        raise RuntimeError(
            f"cv2 was imported from outside the environment: {cv2_path}"
        ) from error

    conda_cv2_owners = sorted(
        str(record.get("name"))
        for record in records
        if cv2_record_path
        in {
            normalized_record_path(str(path))
            for path in record.get("files", [])
        }
    )
    external_opencv = validate_opencv_providers(conda_cv2_owners, installed_opencv)

    if cv2.__version__.split(".", 1)[0] != "4":
        raise RuntimeError(f"Expected OpenCV 4.x, found {cv2.__version__}")
    mismatched_opencv = [
        item
        for item in installed_opencv
        if opencv_api_version(item[1]) != opencv_api_version(cv2.__version__)
    ]
    if mismatched_opencv:
        raise RuntimeError(
            f"OpenCV metadata does not match cv2 {cv2.__version__}: "
            f"{mismatched_opencv}"
        )
    if version("numpy") != numpy.__version__:
        raise RuntimeError("Imported NumPy does not match its pip distribution metadata")

    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    print(f"cv2 {cv2.__version__}: {cv2.__file__}")
    if conda_cv2_owners:
        print(
            f"OpenCV provider: Conda {conda_cv2_owners}; "
            f"metadata aliases: {installed_opencv}"
        )
    else:
        print(f"OpenCV provider: pip {external_opencv}")
    print(f"Ultralytics {ultralytics.__version__}: {ultralytics.__file__}")
    numpy_owner = "Conda (preserved exactly)" if "numpy" in owned_names else "pip"
    print(f"NumPy {numpy.__version__} [{numpy_owner}]: {numpy.__file__}")
    print("Conda native-library and pip Python-package ownership checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
