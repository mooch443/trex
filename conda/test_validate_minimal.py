#!/usr/bin/env python3
"""Targeted tests for minimal-package ownership validation."""

from __future__ import annotations

import unittest

from validate_minimal import (
    REQUIRED_CONDA_PACKAGES,
    validate_opencv_providers,
    validate_required_conda_packages,
)


class OpenCVOwnershipTests(unittest.TestCase):
    def test_minimal_recipe_requires_conda_py_opencv(self) -> None:
        owned_names = {
            "ffmpeg",
            "libopencv",
            "libpng",
            "libzip",
            "numpy",
            "py-opencv",
            "zlib",
        }

        validate_required_conda_packages(owned_names)

        self.assertIn("py-opencv", REQUIRED_CONDA_PACKAGES)

    def test_conda_py_opencv_metadata_is_not_an_external_wheel(self) -> None:
        installed_opencv = [
            ("opencv-python", "4.12.0", "conda"),
            ("opencv-python-headless", "4.12.0", "conda"),
        ]

        external = validate_opencv_providers(["libopencv"], installed_opencv)

        self.assertEqual(external, [])

    def test_external_wheel_is_rejected_when_conda_owns_cv2(self) -> None:
        installed_opencv = [
            ("opencv-python", "4.12.0", "conda"),
            ("opencv-python-headless", "4.12.0.88", "pip"),
        ]

        with self.assertRaisesRegex(RuntimeError, "external OpenCV wheel"):
            validate_opencv_providers(["libopencv"], installed_opencv)


if __name__ == "__main__":
    unittest.main(verbosity=2)
