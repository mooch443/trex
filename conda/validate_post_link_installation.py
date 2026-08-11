#!/usr/bin/env python3
"""Validate the real post-link result inside a Conda CI installation."""

from __future__ import annotations

import argparse
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
import subprocess
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--post-link-output",
    required=True,
    type=Path,
    help="Output captured while Conda ran the post-link script.",
)
options = parser.parse_args()

prefix = Path(sys.prefix)
require(
    options.post_link_output.is_file(),
    f"post-link output was not captured at {options.post_link_output}",
)
output = options.post_link_output.read_text(encoding="utf-8", errors="replace")

install_commands = [
    line.strip()
    for line in output.splitlines()
    if line.strip().startswith("[post-link] Running:") and " -m pip install " in line
]
require(
    len(install_commands) == 1,
    "post-link must execute exactly one pip install transaction; got:\n"
    + "\n".join(install_commands),
)
for forbidden in (
    "trying the next",
    "fallback",
    "pip index versions",
    "Attempting uninstall: torch",
    "Attempting uninstall: torchvision",
):
    require(forbidden.lower() not in output.lower(), f"forbidden retry behavior: {forbidden}")
require(
    "installation transaction completed successfully" in output,
    "post-link did not report a successful transaction:\n" + output,
)
require(
    "WARNING: YOLO runtime warm-up failed" not in output,
    "the production warm-up failed:\n" + output,
)

py_opencv_records = list((prefix / "conda-meta").glob("py-opencv-*.json"))
require(py_opencv_records, "minimal installations must be Conda-owned by py-opencv")
install_arguments = install_commands[0]
require("opencv-python" not in install_arguments, "pip must not install an OpenCV wheel in minimal builds")

try:
    opencv_distribution = distribution("opencv-python")
except PackageNotFoundError as error:
    raise AssertionError("py-opencv did not expose opencv-python metadata to pip") from error

import clip  # noqa: F401,E402
import cv2  # noqa: E402
import dill  # noqa: F401,E402
import numpy as np  # noqa: E402
from rfdetr import RFDETR  # noqa: F401,E402
import sklearn  # noqa: F401,E402
import timm  # noqa: F401,E402
import torch  # noqa: E402
import torchmetrics  # noqa: F401,E402
import torchvision  # noqa: F401,E402
from torchvision.ops import nms  # noqa: E402
import tqdm  # noqa: F401,E402
from ultralytics import YOLO  # noqa: E402

require(int(cv2.__version__.split(".")[0]) >= 4, f"unexpected cv2 version {cv2.__version__}")
require(
    opencv_distribution.version == cv2.__version__,
    "opencv-python metadata and the imported Conda cv2 binding disagree",
)
require(
    nms(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), torch.tensor([1.0]), 0.5).tolist() == [0],
    "torchvision NMS smoke test failed",
)

# The production post-link already ran the cache warm-up exactly once. Requiring
# no warning above verifies that result without downloading or priming it again.
subprocess.run([sys.executable, "-m", "pip", "check"], check=True)

print(
    "Validated one post-link install: "
    f"torch {version('torch')}, torchvision {version('torchvision')}, cv2 {cv2.__version__}."
)
