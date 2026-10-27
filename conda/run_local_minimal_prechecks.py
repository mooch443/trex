#!/usr/bin/env python3
"""Run all locally available prechecks for the minimal Conda package flow."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CONDA_DIR = ROOT / "conda"


UNIX_REAL_RESOLVER_CASES = (
    [
        ("Linux", "x86_64", "", "minimal"),
        ("Linux", "x86_64", "", "buildall"),
    ]
    + [
        ("Linux", "x86_64", cuda, "minimal")
        for cuda in (
            "11.7",
            "11.8",
            "12.1",
            "12.4",
            "12.6",
            "12.8",
            "12.9",
            "13.0",
            "13.3",
        )
    ]
    + [
        ("Darwin", "arm64", "", "minimal"),
        ("Darwin", "x86_64", "", "minimal"),
    ]
)
WINDOWS_REAL_RESOLVER_CASES = [
    ("Windows", "AMD64", "", "minimal"),
    ("Windows", "AMD64", "13.3", "minimal"),
]
REAL_RESOLVER_CASES = (
    WINDOWS_REAL_RESOLVER_CASES if os.name == "nt" else UNIX_REAL_RESOLVER_CASES
)


def run(
    label: str,
    command: list[str],
    *,
    expected_status: int | None = 0,
    environment: dict[str, str] | None = None,
) -> None:
    print(f"\n==> {label}", flush=True)
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        env=environment,
        capture_output=True,
        text=True,
    )
    failed = (
        result.returncode == 0
        if expected_status is None
        else result.returncode != expected_status
    )
    if failed:
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        expectation = (
            "a nonzero status" if expected_status is None else str(expected_status)
        )
        raise RuntimeError(
            f"{label} returned {result.returncode}; expected {expectation}"
        )
    print("PASS")


def run_real_resolver_matrix() -> None:
    for system, machine, cuda, profile in REAL_RESOLVER_CASES:
        case = f"{system}/{machine}/cuda={cuda or 'none'}/{profile}"
        with tempfile.TemporaryDirectory(prefix="trex-post-link-precheck-") as temporary:
            environment_dir = Path(temporary) / "environment"
            run(
                f"create isolated resolver environment for {case}",
                [sys.executable, "-m", "venv", str(environment_dir)],
            )
            if os.name == "nt":
                python = environment_dir / "Scripts" / "python.exe"
            else:
                python = environment_dir / "bin" / "python"
            environment = os.environ.copy()
            environment["GITHUB_ACTIONS"] = "true"
            environment["PATH"] = str(python.parent) + os.pathsep + environment["PATH"]
            environment["PYTHONNOUSERSITE"] = "1"
            environment["VIRTUAL_ENV"] = str(environment_dir)
            run(
                f"real offline resolver {case}",
                [
                    str(python),
                    str(CONDA_DIR / "test_post_link_real_resolver.py"),
                    "--system",
                    system,
                    "--machine",
                    machine,
                    "--cuda",
                    cuda,
                    "--profile",
                    profile,
                ],
                environment=environment,
            )


def check_opencv_features() -> None:
    pvinfo = shutil.which("pvinfo")
    if pvinfo is not None:
        run("OpenCV FFmpeg support", [pvinfo, "-opencv_ffmpeg_support"])
        run(
            "minimal OpenCV excludes OpenCL",
            [pvinfo, "-opencv_opencl_support"],
            expected_status=None,
        )
        return

    print(
        "\n==> pvinfo is unavailable; checking the same OpenCV build flags "
        "through cv2",
        flush=True,
    )
    import cv2  # type: ignore[import-not-found]

    build_lines = cv2.getBuildInformation().splitlines()
    ffmpeg_lines = [line for line in build_lines if "FFMPEG:" in line]
    opencl_lines = [line for line in build_lines if "OpenCL:" in line]
    if not ffmpeg_lines or not any("YES" in line for line in ffmpeg_lines):
        raise RuntimeError(f"OpenCV lacks FFmpeg support: {ffmpeg_lines}")
    # pvinfo returns nonzero both when OpenCL is explicitly disabled and when
    # the build information omits the feature line entirely.
    if any("YES" in line for line in opencl_lines):
        raise RuntimeError(f"Minimal OpenCV unexpectedly has OpenCL: {opencl_lines}")
    print(f"FFmpeg: {ffmpeg_lines}; OpenCL: {opencl_lines}")


def main() -> int:
    if not (Path(sys.prefix) / "conda-meta").is_dir():
        raise RuntimeError(
            "Run this preflight inside the target Conda environment, for example: "
            "KMP_DUPLICATE_LIB_OK=TRUE conda run -n trex python "
            "conda/run_local_minimal_prechecks.py"
        )

    run(
        "workflow runner configuration tests",
        [sys.executable, str(CONDA_DIR / "test_workflow_configuration.py")],
    )
    run(
        "minimal validator unit tests",
        [sys.executable, str(CONDA_DIR / "test_validate_minimal.py")],
    )
    run(
        "post-link simulations",
        [sys.executable, str(CONDA_DIR / "test_post_link_simulation.py")],
    )
    run_real_resolver_matrix()
    check_opencv_features()
    print("\nAll local minimal-package prechecks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
