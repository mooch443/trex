#!/usr/bin/env python3
"""Run post-link with real pip against tiny offline package indexes.

This intentionally installs into the ephemeral GitHub Actions interpreter. It
does not create a second Conda prefix, contact a package server, or use pip's
dry-run mode.
"""

from __future__ import annotations

import argparse
from email.message import Message
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
from urllib.parse import quote
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
POST_LINK_SH = REPOSITORY / "conda" / "post-link.sh"
POST_LINK_BAT = REPOSITORY / "conda" / "post-link.bat"


def make_wheel(
    directory: Path,
    distribution: str,
    version: str,
    *,
    requirements: tuple[str, ...] = (),
    files: dict[str, str] | None = None,
) -> Path:
    normalized = distribution.replace("-", "_")
    wheel = directory / f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = Message()
    metadata["Metadata-Version"] = "2.1"
    metadata["Name"] = distribution
    metadata["Version"] = version
    for requirement in requirements:
        metadata["Requires-Dist"] = requirement
    package_files = files or {f"{normalized}/__init__.py": f'__version__ = "{version}"\n'}
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, contents in package_files.items():
            archive.writestr(name, contents)
        archive.writestr(f"{dist_info}/METADATA", metadata.as_string())
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: trex-post-link-ci\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel


def add_to_index(root: Path, wheel: Path, distribution: str) -> None:
    package = distribution.lower().replace("_", "-").replace(".", "-")
    package_dir = root / package
    package_dir.mkdir(parents=True, exist_ok=True)
    destination = package_dir / wheel.name
    shutil.copy2(wheel, destination)
    (package_dir / "index.html").write_text(
        f'<a href="{quote(destination.name)}">{destination.name}</a>\n',
        encoding="utf-8",
    )


def selected_channel(system: str, machine: str, cuda: str) -> str:
    if system == "Darwin" or machine in {"arm", "arm64", "aarch64"}:
        return "pypi"
    if system != "Linux" and os.name != "nt":
        return "pypi"
    if cuda:
        major, minor = cuda.split(".", 1)
        code = int(major) * 100 + int(minor)
    else:
        code = 0
    for minimum, channel in (
        (1302, "cu132"),
        (1300, "cu130"),
        (1209, "cu129"),
        (1208, "cu128"),
        (1206, "cu126"),
        (1204, "cu124"),
        (1201, "cu121"),
        (1108, "cu118"),
    ):
        if code >= minimum:
            return channel
    return "cpu"


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True)
    parser.add_argument("--machine", required=True)
    parser.add_argument("--cuda", default="")
    parser.add_argument("--profile", choices=("minimal", "buildall"), default="minimal")
    options = parser.parse_args()
    if os.environ.get("GITHUB_ACTIONS") != "true":
        raise SystemExit("This integration test mutates its interpreter and is restricted to GitHub Actions.")

    with tempfile.TemporaryDirectory(prefix="trex-real-post-link-") as temporary:
        root = Path(temporary)
        wheels = root / "wheels"
        pypi = root / "pypi"
        torch_root = root / "torch"
        state = root / "pip-events.jsonl"
        wheels.mkdir()
        pypi.mkdir()
        torch_root.mkdir()

        channel = selected_channel(options.system, options.machine, options.cuda)
        torch_version = "2.6.0" if options.system == "Darwin" else f"2.7.1+{channel}"
        vision_version = "0.21.0" if options.system == "Darwin" else f"0.22.1+{channel}"
        torch_files = {
            "torch/__init__.py": textwrap.dedent(
                f'''\
                __version__ = "{torch_version}"
                class Tensor:
                    def __init__(self, value): self.value = value
                    def tolist(self): return self.value
                def tensor(value): return Tensor(value)
                class cuda:
                    @staticmethod
                    def is_available(): return False
                '''
            )
        }
        vision_files = {
            "torchvision/__init__.py": f'__version__ = "{vision_version}"\n',
            "torchvision/ops.py": "from torch import Tensor\ndef nms(*args): return Tensor([0])\n",
        }
        numpy_files = {
            "numpy/__init__.py": '__version__ = "2.4.6"\nuint8 = int\ndef zeros(shape, dtype=None): return 0\n'
        }
        opencv_files = {"cv2/__init__.py": '__version__ = "4.12.0"\n'}
        ultralytics_files = {
            "ultralytics/__init__.py": (
                "class YOLO:\n"
                "    def __init__(self, *args): pass\n"
                "    def to(self, *args): return self\n"
                "    def predict(self, *args, **kwargs): return []\n"
            )
        }

        created: dict[str, Path] = {}

        def wheel(name: str, version: str, **kwargs: object) -> Path:
            result = make_wheel(wheels, name, version, **kwargs)
            created[name] = result
            return result

        wheel("numpy", "2.4.6", files=numpy_files)
        wheel("opencv-python", "4.12.0", requirements=("numpy>=1.26",), files=opencv_files)
        wheel("torch", torch_version, requirements=("numpy>=1.26",), files=torch_files)
        wheel(
            "torchvision",
            vision_version,
            requirements=(f"torch=={torch_version}", "numpy>=1.26"),
            files=vision_files,
        )
        wheel("torchmetrics", "1.0", requirements=("torch>=2.2",))
        wheel("tqdm", "1.0")
        wheel(
            "ultralytics",
            "8.3.0",
            requirements=("torch>=2.2", "torchvision>=0.17", "opencv-python>=4.6", "numpy>=1.26"),
            files=ultralytics_files,
        )
        wheel("rfdetr", "1.8.3", requirements=("torch>=2.2",), files={"rfdetr/__init__.py": "class RFDETR: pass\n"})
        wheel("dill", "1.0")
        wheel("timm", "1.0", requirements=("torch>=2.2", "torchvision>=0.17"))
        wheel("scikit-learn", "1.0", requirements=("numpy>=1.26",), files={"sklearn/__init__.py": ""})
        wheel("clip", "1.0", requirements=("torch>=2.2",))

        for name, path in created.items():
            if name not in {"torch", "torchvision"} or channel == "pypi":
                add_to_index(pypi, path, name)
        if channel != "pypi":
            channel_index = torch_root / channel
            add_to_index(channel_index, created["torch"], "torch")
            add_to_index(channel_index, created["torchvision"], "torchvision")

        conda_meta = Path(sys.prefix) / "conda-meta"
        conda_meta.mkdir(exist_ok=True)
        if options.profile == "minimal":
            # Simulate packages linked by the minimal Conda recipe. The wheels
            # are genuinely installed once, and post-link must leave them alone.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    "--no-deps",
                    str(created["numpy"]),
                    str(created["opencv-python"]),
                ],
                check=True,
            )
            (conda_meta / "numpy-2.4.6-ci.json").write_text(
                '{"version":"2.4.6"}', encoding="utf-8"
            )
            (conda_meta / "py-opencv-4.12.0-ci.json").write_text("{}", encoding="utf-8")

        sitecustomize = root / "sitecustomize"
        sitecustomize.mkdir()
        (sitecustomize / "sitecustomize.py").write_text(
            textwrap.dedent(
                """\
                import json, os, sys
                args = list(sys.orig_argv)
                if '-m' in args and 'pip' in args:
                    with open(os.environ['TREX_REAL_PIP_EVENTS'], 'a', encoding='utf-8') as stream:
                        stream.write(json.dumps(args) + '\\n')
                """
            ),
            encoding="utf-8",
        )
        fake_bin = root / "bin"
        fake_bin.mkdir()
        if os.name == "nt":
            if options.cuda:
                (fake_bin / "nvidia-smi.cmd").write_text(
                    f"@echo off\necho NVIDIA-SMI 999 Driver Version: 999 CUDA Version: {options.cuda}\n",
                    encoding="utf-8",
                )
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", str(POST_LINK_BAT)]
        else:
            (fake_bin / "uname").write_text(
                "#!/bin/sh\nif [ \"$1\" = \"-m\" ]; then echo \"$TREX_REAL_MACHINE\"; else echo \"$TREX_REAL_SYSTEM\"; fi\n",
                encoding="utf-8",
            )
            (fake_bin / "uname").chmod(0o755)
            (fake_bin / "nvidia-smi").write_text(
                "#!/bin/sh\n"
                "if [ -z \"$TREX_REAL_CUDA\" ]; then exit 1; fi\n"
                "echo \"NVIDIA-SMI 999 Driver Version: 999 CUDA Version: $TREX_REAL_CUDA\"\n",
                encoding="utf-8",
            )
            (fake_bin / "nvidia-smi").chmod(0o755)
            command = ["bash", str(POST_LINK_SH)]

        environment = os.environ.copy()
        environment.update(
            {
                "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
                "PREFIX": sys.prefix,
                "PYTHONPATH": str(sitecustomize),
                "TREX_REAL_PIP_EVENTS": str(state),
                "TREX_REAL_SYSTEM": options.system,
                "TREX_REAL_MACHINE": options.machine,
                "TREX_REAL_CUDA": options.cuda,
                "TREX_PYPI_INDEX_URL": pypi.as_uri(),
                "TREX_TORCH_INDEX_ROOT": torch_root.as_uri(),
                "TREX_CLIP_REQUIREMENT": "clip==1.0",
                "PIP_CONFIG_FILE": os.devnull,
            }
        )
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
        messages = (Path(sys.prefix) / ".messages.txt").read_text(encoding="utf-8", errors="replace")
        output = result.stdout + result.stderr + messages
        if result.returncode:
            raise AssertionError(output)
        events = [json.loads(line) for line in state.read_text(encoding="utf-8").splitlines()]
        installs = [event for event in events if "install" in event]
        if len(installs) != 1:
            raise AssertionError(f"expected one real pip install, got {installs}\n{output}")
        install = installs[0]
        expected_index = pypi.as_uri() if channel == "pypi" else (torch_root / channel).as_uri()
        actual_index = install[install.index("--index-url") + 1]
        if actual_index != expected_index:
            raise AssertionError(f"expected {expected_index}, got {actual_index}")
        requested_opencv = any("opencv-python" in argument for argument in install)
        if options.profile == "minimal" and requested_opencv:
            raise AssertionError("minimal post-link explicitly requested opencv-python")
        if options.profile == "buildall" and not requested_opencv:
            raise AssertionError("buildall post-link did not request its pip OpenCV binding")
        if "Attempting uninstall" in output or "no retry was attempted" in output:
            raise AssertionError(output)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import clip,cv2,dill,numpy,rfdetr,sklearn,timm,torch,torchmetrics,torchvision,tqdm,ultralytics; "
                "from torchvision.ops import nms; assert nms(torch.tensor([]),torch.tensor([]),0.5).tolist()==[0]",
            ],
            check=True,
        )
        if "installation transaction completed successfully" not in messages:
            raise AssertionError(messages)
        print(
            "Real offline pip transaction passed for "
            f"{options.system}/{options.machine}/{channel}/{options.profile}."
        )


if __name__ == "__main__":
    run()
