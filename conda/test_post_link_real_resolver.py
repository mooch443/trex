#!/usr/bin/env python3
"""Verify the native post-link hook against live pip metadata without installing."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


REPOSITORY = Path(__file__).resolve().parents[1]
POST_LINK_SH = REPOSITORY / "conda" / "post-link.sh"
POST_LINK_BAT = REPOSITORY / "conda" / "post-link.bat"
PYPI = "https://pypi.org/simple"
TORCH_ROOT = "https://download.pytorch.org/whl"
PUBLISHED_CHANNEL_SNAPSHOT = {
    "cu118", "cu121", "cu124", "cu126", "cu128", "cu129", "cu130", "cu132"
}

EXPECTED_PAIRS = {
    "pypi": ("2.13.0", "0.28.0"),
    "cu118": ("2.7.1+cu118", "0.22.1+cu118"),
    "cu121": ("2.5.1+cu121", "0.20.1+cu121"),
    "cu124": ("2.6.0+cu124", "0.21.0+cu124"),
    "cu126": ("2.13.0+cu126", "0.28.0+cu126"),
    "cu128": ("2.11.0+cu128", "0.26.0+cu128"),
    "cu129": ("2.13.0+cu129", "0.28.0+cu129"),
    "cu130": ("2.13.0+cu130", "0.28.0+cu130"),
    "cu132": ("2.13.0+cu132", "0.28.0+cu132"),
}
INTEL_MACOS_PYPI_PAIR = ("2.2.2", "0.17.2")

# These compact snapshots intentionally contain only Torch/Torchvision and the
# CUDA/NVIDIA closure. Unrelated resolver movement does not create alert noise.
LINUX_SNAPSHOT_LINES = {
    "pypi": "cuda-bindings==13.3.1 cuda-pathfinder==1.6.1 cuda-toolkit==13.0.3.0 nvidia-cublas==13.1.1.3 nvidia-cuda-cupti==13.0.85 nvidia-cuda-nvrtc==13.0.88 nvidia-cuda-runtime==13.0.96 nvidia-cudnn-cu13==9.20.0.48 nvidia-cufft==12.0.0.61 nvidia-cufile==1.15.1.6 nvidia-curand==10.4.0.35 nvidia-cusolver==12.0.4.66 nvidia-cusparse==12.6.3.3 nvidia-cusparselt-cu13==0.8.1 nvidia-nccl-cu13==2.29.7 nvidia-nvjitlink==13.3.33 nvidia-nvshmem-cu13==3.4.5 nvidia-nvtx==13.0.85 torch==2.13.0 torchvision==0.28.0",
    "cu118": "nvidia-cublas-cu11==11.11.3.6 nvidia-cuda-cupti-cu11==11.8.87 nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 nvidia-cudnn-cu11==9.1.0.70 nvidia-cufft-cu11==10.9.0.58 nvidia-curand-cu11==10.3.0.86 nvidia-cusolver-cu11==11.4.1.48 nvidia-cusparse-cu11==11.7.5.86 nvidia-nccl-cu11==2.21.5 nvidia-nvtx-cu11==11.8.86 torch==2.7.1+cu118 torchvision==0.22.1+cu118",
    "cu121": "nvidia-cublas-cu12==12.1.3.1 nvidia-cuda-cupti-cu12==12.1.105 nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.9.86 nvidia-nvtx-cu12==12.1.105 torch==2.5.1+cu121 torchvision==0.20.1+cu121",
    "cu124": "nvidia-cublas-cu12==12.4.5.8 nvidia-cuda-cupti-cu12==12.4.127 nvidia-cuda-nvrtc-cu12==12.4.127 nvidia-cuda-runtime-cu12==12.4.127 nvidia-cudnn-cu12==9.1.0.70 nvidia-cufft-cu12==11.2.1.3 nvidia-curand-cu12==10.3.5.147 nvidia-cusolver-cu12==11.6.1.9 nvidia-cusparse-cu12==12.3.1.170 nvidia-cusparselt-cu12==0.6.2 nvidia-nccl-cu12==2.21.5 nvidia-nvjitlink-cu12==12.4.127 nvidia-nvtx-cu12==12.4.127 torch==2.6.0+cu124 torchvision==0.21.0+cu124",
    "cu126": "cuda-bindings==12.9.7 cuda-pathfinder==1.6.1 cuda-toolkit==12.6.3 nvidia-cublas-cu12==12.6.4.1 nvidia-cuda-cupti-cu12==12.6.80 nvidia-cuda-nvrtc-cu12==12.6.85 nvidia-cuda-runtime-cu12==12.6.77 nvidia-cudnn-cu12==9.10.2.21 nvidia-cufft-cu12==11.3.0.4 nvidia-cufile-cu12==1.11.1.6 nvidia-curand-cu12==10.3.7.77 nvidia-cusolver-cu12==11.7.1.2 nvidia-cusparse-cu12==12.5.4.2 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.29.3 nvidia-nvjitlink-cu12==12.6.85 nvidia-nvshmem-cu12==3.4.5 nvidia-nvtx-cu12==12.6.77 torch==2.13.0+cu126 torchvision==0.28.0+cu126",
    "cu128": "cuda-bindings==12.9.7 cuda-pathfinder==1.6.1 cuda-toolkit==12.8.1 nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 nvidia-cudnn-cu12==9.19.0.56 nvidia-cufft-cu12==11.3.3.83 nvidia-cufile-cu12==1.13.1.3 nvidia-curand-cu12==10.3.9.90 nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.28.9 nvidia-nvjitlink-cu12==12.8.93 nvidia-nvshmem-cu12==3.4.5 nvidia-nvtx-cu12==12.8.90 torch==2.11.0+cu128 torchvision==0.26.0+cu128",
    "cu129": "cuda-bindings==12.9.7 cuda-pathfinder==1.6.1 cuda-toolkit==12.9.1 nvidia-cublas-cu12==12.9.1.4 nvidia-cuda-cupti-cu12==12.9.79 nvidia-cuda-nvrtc-cu12==12.9.86 nvidia-cuda-runtime-cu12==12.9.79 nvidia-cudnn-cu12==9.20.0.48 nvidia-cufft-cu12==11.4.1.4 nvidia-cufile-cu12==1.14.1.1 nvidia-curand-cu12==10.3.10.19 nvidia-cusolver-cu12==11.7.5.82 nvidia-cusparse-cu12==12.5.10.65 nvidia-cusparselt-cu12==0.8.1 nvidia-nccl-cu12==2.29.7 nvidia-nvjitlink-cu12==12.9.86 nvidia-nvshmem-cu12==3.4.5 nvidia-nvtx-cu12==12.9.79 torch==2.13.0+cu129 torchvision==0.28.0+cu129",
    "cu130": "cuda-bindings==13.3.1 cuda-pathfinder==1.6.1 cuda-toolkit==13.0.3.0 nvidia-cublas==13.1.1.3 nvidia-cuda-cupti==13.0.85 nvidia-cuda-nvrtc==13.0.88 nvidia-cuda-runtime==13.0.96 nvidia-cudnn-cu13==9.20.0.48 nvidia-cufft==12.0.0.61 nvidia-cufile==1.15.1.6 nvidia-curand==10.4.0.35 nvidia-cusolver==12.0.4.66 nvidia-cusparse==12.6.3.3 nvidia-cusparselt-cu13==0.8.1 nvidia-nccl-cu13==2.29.7 nvidia-nvjitlink==13.3.33 nvidia-nvshmem-cu13==3.4.5 nvidia-nvtx==13.0.85 torch==2.13.0+cu130 torchvision==0.28.0+cu130",
    "cu132": "cuda-bindings==13.3.1 cuda-pathfinder==1.6.1 cuda-toolkit==13.2.1 nvidia-cublas==13.4.0.1 nvidia-cuda-cupti==13.2.75 nvidia-cuda-nvrtc==13.2.78 nvidia-cuda-runtime==13.2.75 nvidia-cudnn-cu13==9.20.0.48 nvidia-cufft==12.2.0.46 nvidia-cufile==1.17.1.22 nvidia-curand==10.4.2.55 nvidia-cusolver==12.2.0.1 nvidia-cusparse==12.7.10.1 nvidia-cusparselt-cu13==0.8.1 nvidia-nccl-cu13==2.29.7 nvidia-nvjitlink==13.3.33 nvidia-nvshmem-cu13==3.4.5 nvidia-nvtx==13.2.75 torch==2.13.0+cu132 torchvision==0.28.0+cu132",
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_snapshot(line: str) -> dict[str, str]:
    return dict(item.rsplit("==", 1) for item in line.split())


def channel_code(channel: str) -> int:
    digits = channel[2:]
    return int(digits[:2]) * 100 + int(digits[2:])


def report_diff(expected: dict[str, str], actual: dict[str, str]) -> str:
    before = json.dumps(expected, indent=2, sort_keys=True).splitlines()
    after = json.dumps(actual, indent=2, sort_keys=True).splitlines()
    return "\n".join(difflib.unified_diff(before, after, "snapshot", "resolved", lineterm=""))


def dispatch_python(arguments: list[str]) -> int:
    """Intercept the hook's sole install; delegate all metadata-only commands."""
    real_python = os.environ["TREX_REAL_PYTHON"]
    is_pip = "-m" in arguments and arguments[arguments.index("-m") + 1 :][:1] == ["pip"]
    if is_pip and "install" in arguments:
        requirements = [
            value for value in arguments
            if re.fullmatch(r"torch(?:vision)?(?:[<>=!~].*)?", value)
        ]
        sources: list[str] = []
        for option in ("--index-url", "--extra-index-url"):
            if option in arguments:
                offset = arguments.index(option)
                sources.extend((option, arguments[offset + 1]))
        if len(requirements) != 2 or "--index-url" not in sources:
            raise SystemExit(f"could not isolate Torch request: {arguments}")

        cache = Path(os.environ["TREX_RESOLVER_CACHE"])
        key = hashlib.sha256(json.dumps([sources, requirements]).encode()).hexdigest()[:16]
        report_path = cache / f"{key}.json"
        output_path = cache / f"{key}.txt"
        status_path = cache / f"{key}.status"
        if not status_path.exists():
            pending = report_path.with_suffix(".pending.json")
            command = [
                real_python, "-m", "pip", "install", "--dry-run",
                "--ignore-installed", "--only-binary=:all:",
                "--disable-pip-version-check", "--no-input", "--no-color",
                "--progress-bar", "off", "--report", str(pending),
                *sources, *requirements,
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=900)
            output_path.write_text(result.stdout + result.stderr, encoding="utf-8")
            status_path.write_text(str(result.returncode), encoding="ascii")
            if result.returncode == 0:
                pending.replace(report_path)

        status = int(status_path.read_text(encoding="ascii"))
        output = output_path.read_text(encoding="utf-8")
        event = {
            "arguments": arguments,
            "case": os.environ["TREX_RESOLVER_CASE"],
            "report": str(report_path),
            "requirements": requirements,
            "sources": {
                name: os.environ.get(name, "")
                for name in ("PIP_CONFIG_FILE", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS")
            },
            "status": status,
        }
        with Path(os.environ["TREX_RESOLVER_EVENTS"]).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        sys.stdout.write(output)
        return status

    if "-c" in arguments:
        code = arguments[arguments.index("-c") + 1]
        if "import torch" in code or "from ultralytics" in code:
            return 0
    return subprocess.call([real_python, *arguments])


def discover_published_channels() -> set[str]:
    request = Request(TORCH_ROOT + "/", headers={"User-Agent": "TRex resolver CI"})
    text = urlopen(request, timeout=30).read().decode("utf-8", errors="replace")
    found = set()
    for href in re.findall(r"href=[\"']([^\"']+)", text, re.I):
        name = urlparse(urljoin(TORCH_ROOT + "/", href)).path.rstrip("/").rsplit("/", 1)[-1]
        if re.fullmatch(r"cu[0-9]{3,}", name) and channel_code(name) >= 1108:
            found.add(name)
    return found


def make_shims(root: Path, cuda: str) -> Path:
    fake_bin = root / "bin"
    fake_bin.mkdir()
    dispatcher = Path(__file__).resolve()
    if os.name == "nt":
        (fake_bin / "python.cmd").write_text(
            f'@"%TREX_REAL_PYTHON%" "{dispatcher}" --dispatch %*\n', encoding="utf-8"
        )
        smi = "@echo off\n"
        smi += (
            f"echo NVIDIA-SMI 999 Driver Version: 999 CUDA Version: {cuda}\n"
            if cuda else "echo NVIDIA-SMI unavailable 1>&2\nexit /b 1\n"
        )
        (fake_bin / "nvidia-smi.cmd").write_text(smi, encoding="utf-8")
    else:
        (fake_bin / "python").write_text(
            f'#!/bin/sh\nexec "$TREX_REAL_PYTHON" "{dispatcher}" --dispatch "$@"\n', encoding="utf-8"
        )
        (fake_bin / "python").chmod(0o755)
        smi = "#!/bin/sh\n"
        smi += (
            f"echo 'NVIDIA-SMI 999 Driver Version: 999 CUDA Version: {cuda}'\n"
            if cuda else "echo 'NVIDIA-SMI unavailable' >&2\nexit 1\n"
        )
        (fake_bin / "nvidia-smi").write_text(smi, encoding="utf-8")
        (fake_bin / "nvidia-smi").chmod(0o755)
    return fake_bin


def expected_snapshot(system: str, machine: str, channel: str) -> dict[str, str]:
    if system == "Linux":
        return parse_snapshot(LINUX_SNAPSHOT_LINES[channel])
    if system == "Darwin" and machine.casefold() in {"x86_64", "amd64"} and channel == "pypi":
        torch_version, vision_version = INTEL_MACOS_PYPI_PAIR
    else:
        torch_version, vision_version = EXPECTED_PAIRS[channel]
    return {"torch": torch_version, "torchvision": vision_version}


def resolved_snapshot(report_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    packages: dict[str, str] = {}
    urls: dict[str, str] = {}
    for item in report["install"]:
        name = canonical_name(item["metadata"]["name"])
        if name in {"torch", "torchvision"} or name.startswith(("cuda-", "nvidia-")):
            packages[name] = item["metadata"]["version"]
            urls[name] = item["download_info"]["url"]
    return dict(sorted(packages.items())), urls


def version_code(version: str) -> int | None:
    match = re.match(r"(\d+)\.(\d+)", version)
    return int(match.group(1)) * 100 + int(match.group(2)) if match else None


def run_case(root: Path, cuda: str, expected_channel: str) -> list[str]:
    label = cuda or "no-driver"
    case_root = root / label.replace(".", "-")
    prefix = case_root / "prefix"
    metadata = prefix / "conda-meta"
    metadata.mkdir(parents=True)
    (metadata / "numpy-2.4.6-ci.json").write_text(
        json.dumps({"name": "numpy", "version": "2.4.6", "files": []}), encoding="utf-8"
    )
    (metadata / "py-opencv-4.12-ci.json").write_text(
        json.dumps({"name": "py-opencv", "version": "4.12", "files": []}), encoding="utf-8"
    )
    fake_bin = make_shims(case_root, cuda)
    events_path = case_root / "events.jsonl"
    ambient = case_root / "ambient-pip.conf"
    ambient.write_text(
        "[global]\nindex-url = https://ambient.invalid/simple\n"
        "extra-index-url = https://ambient-extra.invalid/simple\n"
        "find-links = https://ambient-links.invalid/\n",
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment.update({
        "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
        "PREFIX": str(prefix),
        "TREX_POST_LINK_OUTPUT": "stdout",
        "TREX_PYPI_INDEX_URL": PYPI,
        "TREX_TORCH_INDEX_ROOT": TORCH_ROOT,
        "TREX_REAL_PYTHON": sys.executable,
        "TREX_RESOLVER_CACHE": str(root / "cache"),
        "TREX_RESOLVER_CASE": label,
        "TREX_RESOLVER_EVENTS": str(events_path),
        "PIP_CONFIG_FILE": str(ambient),
        "PIP_INDEX_URL": "https://ambient.invalid/simple",
        "PIP_EXTRA_INDEX_URL": "https://ambient-extra.invalid/simple",
        "PIP_FIND_LINKS": "https://ambient-links.invalid/",
    })
    if os.name == "nt":
        command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(POST_LINK_BAT)]
    else:
        command = ["bash", str(POST_LINK_SH)]
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=1200)
    output = result.stdout + result.stderr
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()] if events_path.exists() else []
    errors: list[str] = []
    if result.returncode != 0:
        errors.append(f"{label}: hook returned {result.returncode}\n{output}")
    if len(events) != 1:
        errors.append(f"{label}: expected one pip install, recorded {len(events)}\n{output}")
        return errors
    event = events[0]
    if event["status"] != 0:
        errors.append(f"{label}: pip dry-run failed\n{output}")
        return errors
    sources = event["sources"]
    null_config = {"nul"} if os.name == "nt" else {"/dev/null"}
    if sources["PIP_CONFIG_FILE"].casefold() not in null_config or any(
        sources[name] for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS")
    ):
        errors.append(f"{label}: ambient pip configuration leaked: {sources}")

    requirements = event["requirements"]
    torch_requirement = next(
        value for value in requirements if re.match(r"^torch(?:[<>=!~]|$)", value)
    )
    flavor_match = re.search(r"\+(cu\d+)$", torch_requirement)
    selected = flavor_match.group(1) if flavor_match else "pypi"
    if selected != expected_channel:
        errors.append(f"{label}: expected {expected_channel}, selected {selected}: {requirements}")
    arguments = event["arguments"]
    primary_index = arguments[arguments.index("--index-url") + 1]
    expected_index = PYPI if selected == "pypi" else f"{TORCH_ROOT}/{selected}"
    if primary_index != expected_index:
        errors.append(f"{label}: expected source {expected_index}, selected {primary_index}")
    if selected != "pypi":
        if "--extra-index-url" not in arguments or arguments[arguments.index("--extra-index-url") + 1] != PYPI:
            errors.append(f"{label}: CUDA solve did not retain PyPI for non-Torch dependencies")
    if selected != "pypi" and not all(value.endswith("+" + selected) for value in requirements):
        errors.append(f"{label}: CUDA requirements were not exact flavored pins: {requirements}")

    snapshot, urls = resolved_snapshot(Path(event["report"]))
    expected = expected_snapshot(platform.system(), platform.machine(), expected_channel)
    if snapshot != expected:
        errors.append(f"{label}/{expected_channel}: resolver snapshot changed\n{report_diff(expected, snapshot)}")
    if selected != "pypi" and "+" + selected not in snapshot.get("torch", ""):
        errors.append(f"{label}: generic PyPI Torch competed with {selected}: {snapshot.get('torch')} {urls.get('torch')}")
    if selected != "pypi" and cuda:
        driver_code = version_code(cuda)
        for name, version in snapshot.items():
            if name == "cuda-toolkit" or "cuda-runtime" in name:
                runtime_code = version_code(version)
                if driver_code is not None and runtime_code is not None and runtime_code > driver_code:
                    errors.append(f"{label}: resolved {name} {version} newer than simulated driver {cuda}")
    print(f"{label}: {selected} -> {snapshot.get('torch')} / {snapshot.get('torchvision')}")
    return errors


def run_live_matrix() -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true" and os.environ.get("TREX_ALLOW_LIVE_RESOLVER") != "1":
        raise SystemExit("live resolver is restricted to CI; set TREX_ALLOW_LIVE_RESOLVER=1 to run manually")
    system = platform.system()
    published = discover_published_channels()
    failures: list[str] = []
    if published != PUBLISHED_CHANNEL_SNAPSHOT:
        failures.append(
            "official CUDA channel set changed\n"
            + report_diff(
                {name: "published" for name in sorted(PUBLISHED_CHANNEL_SNAPSHOT)},
                {name: "published" for name in sorted(published)},
            )
        )
    newest = max(published or PUBLISHED_CHANNEL_SNAPSHOT, key=channel_code)
    if system in {"Linux", "Windows"}:
        cases = [
            ("", "pypi"), ("11.7", "pypi"), ("11.8", "cu118"), ("12.0", "cu118"),
            ("12.1", "cu121"), ("12.2", "cu121"), ("12.3", "cu121"),
            ("12.4", "cu124"), ("12.6", "cu126"), ("12.8", "cu128"),
            ("12.9", "cu129"), ("13.0", "cu130"), ("13.2", "cu132"),
            ("13.3", "cu132"), ("99.0", newest),
        ]
    elif system == "Darwin":
        cases = [("", "pypi")]
    else:
        raise SystemExit(f"unsupported native CI platform: {system}")

    with tempfile.TemporaryDirectory(prefix="trex-live-resolver-") as temporary:
        root = Path(temporary)
        (root / "cache").mkdir()
        for cuda, channel in cases:
            try:
                failures.extend(run_case(root, cuda, channel))
            except Exception as error:
                failures.append(f"{cuda or 'no-driver'}: {type(error).__name__}: {error}")
    if failures:
        raise AssertionError("\n\n".join(failures))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dispatch":
        raise SystemExit(dispatch_python(sys.argv[2:]))
    run_live_matrix()
