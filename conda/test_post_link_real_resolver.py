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
import sysconfig
import tempfile
import textwrap
from urllib.parse import quote
import zipfile


REPOSITORY = Path(__file__).resolve().parents[1]
POST_LINK_SH = REPOSITORY / "conda" / "post-link.sh"
POST_LINK_BAT = REPOSITORY / "conda" / "post-link.bat"
VALIDATE_MINIMAL = REPOSITORY / "conda" / "validate_minimal.py"
VALIDATE_POST_LINK = REPOSITORY / "conda" / "validate_post_link_installation.py"


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
    index = package_dir / "index.html"
    existing = index.read_text(encoding="utf-8") if index.exists() else ""
    index.write_text(
        existing + f'<a href="{quote(destination.name)}">{destination.name}</a>\n',
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
    return "pypi"


def emit_resolver_diagnostics(
    *,
    system: str,
    machine: str,
    cuda: str,
    profile: str,
    channel: str,
    wheels: Path,
    installs: list[dict[str, object]],
    result: subprocess.CompletedProcess[str],
) -> None:
    """Expose the exact offline resolver inputs and pip results in CI logs."""
    case = f"{system}/{machine}/cuda={cuda or 'none'}/{profile}/{channel}"
    print(f"Resolver case: {case}")
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::group::Offline pip resolver details ({case})")
    print("Recorded pip invocation(s):")
    if installs:
        for install in installs:
            print(json.dumps(install))
    else:
        print("<none>")
    print("Offline wheel set:")
    for wheel in sorted(wheels.glob("*.whl")):
        print(f"  {wheel.name}")
    print("Captured post-link stdout:")
    print(result.stdout.rstrip() or "<empty>")
    print("Captured post-link stderr:")
    print(result.stderr.rstrip() or "<empty>")
    installed = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    print("Installed distributions after the resolver transaction:")
    print(installed.stdout.rstrip() or "<empty>")
    if installed.stderr:
        print(installed.stderr.rstrip())
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print("::endgroup::")
    sys.stdout.flush()


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
        selected_torch_version = f"2.7.1+{channel}" if channel.startswith("cu") else "2.13.0"
        selected_vision_version = f"0.22.1+{channel}" if channel.startswith("cu") else "0.28.0"

        def torch_files(version: str) -> dict[str, str]:
            return {
                "torch/__init__.py": textwrap.dedent(
                    f'''\
                __version__ = "{version}"
                class Tensor:
                    def __init__(self, value): self.value = value
                    def tolist(self): return self.value
                def tensor(value): return Tensor(value)
                class cuda:
                    @staticmethod
                    def is_available(): return False
                class version:
                    cuda = None
                '''
                )
            }

        def vision_files(version: str) -> dict[str, str]:
            return {
                "torchvision/__init__.py": f'__version__ = "{version}"\n',
                "torchvision/ops.py": "from torch import Tensor\ndef nms(*args): return Tensor([0])\n",
            }
        numpy_files = {
            "numpy/__init__.py": '__version__ = "2.4.6"\nuint8 = int\ndef zeros(shape, dtype=None): return 0\n'
        }
        opencv_files = {"cv2/__init__.py": '__version__ = "4.12.0"\n'}
        ultralytics_files = {
            "ultralytics/__init__.py": (
                '__version__ = "8.3.0"\n'
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
            add_to_index(pypi, path, name)

        generic_torch = make_wheel(
            wheels,
            "torch",
            "2.13.0",
            requirements=("numpy>=1.26",),
            files=torch_files("2.13.0"),
        )
        generic_vision = make_wheel(
            wheels,
            "torchvision",
            "0.28.0",
            requirements=("torch==2.13.0", "numpy>=1.26"),
            files=vision_files("0.28.0"),
        )
        add_to_index(pypi, generic_torch, "torch")
        add_to_index(pypi, generic_vision, "torchvision")

        if channel.startswith("cu"):
            channel_index = torch_root / channel
            for torch_version, vision_version in (
                (f"2.6.0+{channel}", f"0.21.0+{channel}"),
                (selected_torch_version, selected_vision_version),
            ):
                cuda_torch = make_wheel(
                    wheels,
                    "torch",
                    torch_version,
                    requirements=("numpy>=1.26",),
                    files=torch_files(torch_version),
                )
                cuda_vision = make_wheel(
                    wheels,
                    "torchvision",
                    vision_version,
                    requirements=(f"torch=={torch_version}", "numpy>=1.26"),
                    files=vision_files(vision_version),
                )
                add_to_index(channel_index, cuda_torch, "torch")
                add_to_index(channel_index, cuda_vision, "torchvision")

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
            site_packages = Path(sysconfig.get_paths()["purelib"])
            numpy_file = site_packages / "numpy" / "__init__.py"
            cv2_file = site_packages / "cv2" / "__init__.py"
            for installer in (
                *site_packages.glob("numpy-*.dist-info/INSTALLER"),
                *site_packages.glob("opencv_python-*.dist-info/INSTALLER"),
            ):
                installer.write_text("conda\n", encoding="utf-8")

            def write_conda_record(
                name: str,
                version: str,
                files: tuple[Path, ...] = (),
            ) -> None:
                relative_files = [
                    path.resolve().relative_to(Path(sys.prefix).resolve()).as_posix()
                    for path in files
                ]
                (conda_meta / f"{name}-{version}-ci.json").write_text(
                    json.dumps(
                        {"name": name, "version": version, "files": relative_files}
                    ),
                    encoding="utf-8",
                )

            write_conda_record("numpy", "2.4.6", (numpy_file,))
            write_conda_record("py-opencv", "4.12.0")
            for native_package in ("ffmpeg", "libopencv", "libpng", "libzip", "zlib"):
                files = (cv2_file,) if native_package == "libopencv" else ()
                write_conda_record(native_package, "1.0", files)

        sitecustomize = root / "sitecustomize"
        sitecustomize.mkdir()
        (sitecustomize / "sitecustomize.py").write_text(
            textwrap.dedent(
                """\
                import json, os, sys
                args = list(sys.orig_argv)
                if '-m' in args and 'pip' in args:
                    with open(os.environ['TREX_REAL_PIP_EVENTS'], 'a', encoding='utf-8') as stream:
                        stream.write(json.dumps({
                            'args': args,
                            'sources': {
                                name: os.environ.get(name, '')
                                for name in (
                                    'PIP_CONFIG_FILE',
                                    'PIP_INDEX_URL',
                                    'PIP_EXTRA_INDEX_URL',
                                    'PIP_FIND_LINKS',
                                )
                            },
                        }) + '\\n')
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

        ambient = root / "ambient"
        ambient.mkdir()
        ambient_config = root / "ambient-pip.conf"
        ambient_config.write_text(
            "[global]\n"
            f"index-url = {ambient.as_uri()}\n"
            f"extra-index-url = {ambient.as_uri()}\n"
            f"find-links = {ambient.as_uri()}\n",
            encoding="utf-8",
        )

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
                "TREX_POST_LINK_OUTPUT": "stdout",
                "TREX_PYPI_INDEX_URL": pypi.as_uri(),
                "TREX_TORCH_INDEX_ROOT": torch_root.as_uri(),
                "TREX_CLIP_REQUIREMENT": "clip==1.0",
                "PIP_CONFIG_FILE": str(ambient_config),
                "PIP_INDEX_URL": ambient.as_uri(),
                "PIP_EXTRA_INDEX_URL": ambient.as_uri(),
                "PIP_FIND_LINKS": ambient.as_uri(),
            }
        )
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=120)
        output = result.stdout + result.stderr
        events = (
            [json.loads(line) for line in state.read_text(encoding="utf-8").splitlines()]
            if state.exists()
            else []
        )
        installs = [event for event in events if "install" in event["args"]]
        emit_resolver_diagnostics(
            system=options.system,
            machine=options.machine,
            cuda=options.cuda,
            profile=options.profile,
            channel=channel,
            wheels=wheels,
            installs=installs,
            result=result,
        )
        if result.returncode:
            raise AssertionError(output)
        if len(installs) != 1:
            raise AssertionError(f"expected one real pip install, got {installs}\n{output}")
        index_queries = [event for event in events if "index" in event["args"]]
        expected_queries = 2 if channel.startswith("cu") else 0
        if len(index_queries) != expected_queries:
            raise AssertionError(
                f"expected {expected_queries} metadata queries, got {index_queries}\n{output}"
            )
        for event in events:
            sources = event["sources"]
            if sources["PIP_CONFIG_FILE"].casefold() not in {"/dev/null", "nul"}:
                raise AssertionError(f"pip configuration leaked into post-link: {event}")
            if any(sources[name] for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_FIND_LINKS")):
                raise AssertionError(f"pip source environment leaked into post-link: {event}")

        install = installs[0]["args"]
        expected_index = pypi.as_uri() if channel == "pypi" else (torch_root / channel).as_uri()
        actual_index = install[install.index("--index-url") + 1]
        if actual_index != expected_index:
            raise AssertionError(f"expected {expected_index}, got {actual_index}")
        if channel.startswith("cu"):
            expected_requirements = {
                f"torch==={selected_torch_version}",
                f"torchvision==={selected_vision_version}",
            }
            missing_requirements = expected_requirements - set(install)
            if missing_requirements:
                raise AssertionError(
                    f"CUDA install did not pin the newest flavored pair: {missing_requirements}\n{output}"
                )
        elif not {"torch>=2.2", "torchvision>=0.17"}.issubset(install):
            raise AssertionError(f"PyPI fallback was unexpectedly pinned: {install}")
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
                "from torchvision.ops import nms; "
                "assert nms(torch.tensor([]),torch.tensor([]),0.5).tolist()==[0]; "
                f"assert torch.__version__ == {selected_torch_version!r}; "
                f"assert torchvision.__version__ == {selected_vision_version!r}",
            ],
            check=True,
        )
        if options.profile == "minimal":
            subprocess.run([sys.executable, str(VALIDATE_POST_LINK)], check=True)
            subprocess.run([sys.executable, str(VALIDATE_MINIMAL)], check=True)
        if "installation transaction completed successfully" not in output:
            raise AssertionError(output)
        print(
            "Real offline pip transaction passed for "
            f"{options.system}/{options.machine}/{channel}/{options.profile}."
        )


if __name__ == "__main__":
    run()
