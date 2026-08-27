#!/usr/bin/env python3
"""Verify the native post-link hook against live pip metadata without installing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
POST_LINK_SH = REPOSITORY / "conda" / "post-link.sh"
POST_LINK_BAT = REPOSITORY / "conda" / "post-link.bat"
PYPI = "https://pypi.org/simple"
TORCH_ROOT = "https://download.pytorch.org/whl"
MINIMUM_PACKAGE_VERSIONS = {
    "torch": (2, 2),
    "torchvision": (0, 17),
}


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def channel_code(channel: str) -> int:
    digits = channel[2:]
    return int(digits[:2]) * 100 + int(digits[2:])


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


def windows_dispatch_sitecustomize(dispatcher: Path) -> str:
    return f'''\
import os
import runpy
import sys
import traceback

if os.environ.get("TREX_RESOLVER_DISPATCH") == "1":
    os.environ["TREX_RESOLVER_DISPATCH"] = "0"
    dispatcher = {str(dispatcher)!r}
    sys.argv = [dispatcher, "--dispatch", *sys.orig_argv[1:]]
    try:
        runpy.run_path(dispatcher, run_name="__main__")
    except SystemExit as error:
        status = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
    except BaseException:
        traceback.print_exc()
        status = 1
    else:
        status = 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
'''


def make_shims(root: Path, cuda: str) -> Path:
    fake_bin = root / "bin"
    fake_bin.mkdir()
    dispatcher = Path(__file__).resolve()
    if os.name == "nt":
        (fake_bin / "sitecustomize.py").write_text(
            windows_dispatch_sitecustomize(dispatcher),
            encoding="utf-8",
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


def resolved_packages(report_path: Path) -> tuple[dict[str, str], dict[str, str]]:
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


def release_prefix(version: str) -> tuple[int, int] | None:
    match = re.match(r"(\d+)\.(\d+)", version)
    return (int(match.group(1)), int(match.group(2))) if match else None


def compatibility_errors(
    *,
    system: str,
    label: str,
    cuda: str,
    minimum_channel: str,
    selected: str,
    requirements: list[str],
    packages: dict[str, str],
    urls: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    if minimum_channel == "pypi":
        if selected != "pypi":
            errors.append(f"{label}: expected the normal PyPI fallback, selected {selected}")
    elif selected == "pypi":
        errors.append(
            f"{label}: lost the guaranteed {minimum_channel} GPU path on {system}; "
            "review the supported driver mapping and documentation"
        )
    elif channel_code(selected) < channel_code(minimum_channel):
        errors.append(
            f"{label}: selected {selected}, below TRex's guaranteed {minimum_channel} path on "
            f"{system}; review the supported driver mapping and documentation"
        )

    for package, minimum in MINIMUM_PACKAGE_VERSIONS.items():
        version = packages.get(package)
        release = release_prefix(version or "")
        if release is None or release < minimum:
            errors.append(
                f"{label}: resolved {package} {version or 'missing'}, below TRex's supported "
                f"minimum {minimum[0]}.{minimum[1]}"
            )

    if selected != "pypi":
        requested = {
            match.group(1): match.group(2)
            for value in requirements
            if (match := re.fullmatch(r"(torch(?:vision)?)===([^ ]+)", value))
        }
        for package in ("torch", "torchvision"):
            if packages.get(package) != requested.get(package):
                errors.append(
                    f"{label}: exact {package} request {requested.get(package)} resolved as "
                    f"{packages.get(package)}"
                )
        if "+" + selected not in packages.get("torch", ""):
            errors.append(
                f"{label}: generic PyPI Torch competed with {selected}: "
                f"{packages.get('torch')} {urls.get('torch')}"
            )
        if f"/{selected}/" not in urls.get("torch", ""):
            errors.append(f"{label}: Torch did not resolve from the selected {selected} index")

    if selected != "pypi" and cuda:
        driver_code = version_code(cuda)
        cuda_requirements = [(f"PyTorch flavor {selected}", channel_code(selected))]
        for name, version in packages.items():
            if name == "cuda-toolkit" or "cuda-runtime" in name:
                runtime_code = version_code(version)
                if runtime_code is not None:
                    cuda_requirements.append((f"{name} {version}", runtime_code))
        for requirement, required_code in cuda_requirements:
            if driver_code is not None and required_code > driver_code:
                errors.append(
                    f"{label}: {requirement} now requires CUDA "
                    f"{required_code // 100}.{required_code % 100}, newer than the simulated "
                    f"driver's CUDA {cuda}; review the supported driver mapping and documentation"
                )

    if system == "Darwin":
        cuda_packages = {
            name: version
            for name, version in packages.items()
            if name.startswith(("cuda-", "nvidia-"))
        }
        if cuda_packages:
            errors.append(f"{label}: macOS unexpectedly resolved a CUDA/NVIDIA closure: {cuda_packages}")
    return errors


def run_case(root: Path, cuda: str, minimum_channel: str) -> list[str]:
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
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(fake_bin) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
        )
        environment["TREX_RESOLVER_DISPATCH"] = "1"
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

    packages, urls = resolved_packages(Path(event["report"]))
    errors.extend(
        compatibility_errors(
            system=platform.system(),
            label=label,
            cuda=cuda,
            minimum_channel=minimum_channel,
            selected=selected,
            requirements=requirements,
            packages=packages,
            urls=urls,
        )
    )

    print(f"{label}: {selected} -> {packages.get('torch')} / {packages.get('torchvision')}")
    return errors


def run_live_matrix() -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true" and os.environ.get("TREX_ALLOW_LIVE_RESOLVER") != "1":
        raise SystemExit("live resolver is restricted to CI; set TREX_ALLOW_LIVE_RESOLVER=1 to run manually")
    system = platform.system()
    failures: list[str] = []
    if system in {"Linux", "Windows"}:
        cases = [
            ("", "pypi"), ("11.7", "pypi"), ("11.8", "cu118"), ("12.0", "cu118"),
            ("12.1", "cu121"), ("12.2", "cu121"), ("12.3", "cu121"),
            ("12.4", "cu124"), ("12.6", "cu126"), ("12.8", "cu128"),
            ("12.9", "cu129"), ("13.0", "cu130"), ("13.2", "cu132"),
            ("13.3", "cu132"), ("99.0", "cu132"),
        ]
    elif system == "Darwin":
        cases = [("", "pypi")]
    else:
        raise SystemExit(f"unsupported native CI platform: {system}")

    with tempfile.TemporaryDirectory(prefix="trex-live-resolver-") as temporary:
        root = Path(temporary)
        (root / "cache").mkdir()
        for cuda, minimum_channel in cases:
            try:
                failures.extend(run_case(root, cuda, minimum_channel))
            except Exception as error:
                failures.append(f"{cuda or 'no-driver'}: {type(error).__name__}: {error}")
    if failures:
        raise AssertionError("\n\n".join(failures))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dispatch":
        raise SystemExit(dispatch_python(sys.argv[2:]))
    run_live_matrix()
