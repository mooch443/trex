#!/usr/bin/env python3
"""Simulate post-link platform and resolver outcomes without downloading wheels."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
POST_LINK_SH = REPOSITORY / "conda" / "post-link.sh"
POST_LINK_BAT = REPOSITORY / "conda" / "post-link.bat"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYPI_INDEX = "https://pypi.org/simple"


FAKE_PYTHON = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys


state = Path(os.environ["TREX_FAKE_STATE"])
events_path = state / "events.jsonl"
args = sys.argv[1:]
if args[:2] == ["-X", "utf8"]:
    args = args[2:]


def event(kind, **values):
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"kind": kind, **values}) + "\n")


def index_value(arguments):
    try:
        return arguments[arguments.index("--index-url") + 1]
    except (ValueError, IndexError):
        return ""


if args[:3] == ["-m", "pip", "--version"]:
    print("pip 99.0 (simulated)")
    raise SystemExit(0)

if args[:3] == ["-m", "pip", "install"]:
    counter_path = state / "install-count"
    count = int(counter_path.read_text(encoding="utf-8")) if counter_path.exists() else 0
    counter_path.write_text(str(count + 1), encoding="utf-8")
    outcomes = list(filter(None, os.environ.get("TREX_FAKE_INSTALL_OUTCOMES", "success").split(",")))
    outcome = outcomes[count] if count < len(outcomes) else outcomes[-1]
    constraint = ""
    if "--constraint" in args:
        constraint_path = Path(args[args.index("--constraint") + 1])
        constraint = constraint_path.read_text(encoding="utf-8").strip()
    event(
        "install",
        args=args,
        index=index_value(args),
        constraint=constraint,
        outcome=outcome,
    )
    if outcome == "resolution":
        print("ERROR: ResolutionImpossible: simulated dependency conflict", file=sys.stderr)
        raise SystemExit(1)
    if outcome == "network":
        print("ERROR: simulated connection failure", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

if args and args[0] == "-c":
    code = args[1] if len(args) > 1 else ""
    if "json.load" in code and "['version']" in code:
        print("2.4.6", end="")
    elif "CUDA(?:" in code:
        print(os.environ.get("TREX_FAKE_CUDA_VERSION", ""))
    elif "from ultralytics import YOLO" in code and os.environ.get("TREX_FAKE_WARM_OUTCOME") == "failure":
        print("simulated cache warm-up failure", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(0)

# The Windows progress helper is launched through python as a temporary .py file.
if args and args[0].lower().endswith(".py"):
    raise SystemExit(0)

event("unhandled-python", args=args)
raise SystemExit(0)
'''


FAKE_UNAME = r'''#!/bin/sh
case "$1" in
    -m|-p) printf '%s\n' "$TREX_FAKE_MACHINE" ;;
    *) printf '%s\n' "$TREX_FAKE_SYSTEM" ;;
esac
'''


FAKE_NVIDIA_SMI_SH = r'''#!/bin/sh
if [ -z "$TREX_FAKE_CUDA_VERSION" ]; then
    echo "NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver." >&2
    exit 1
fi
case "$1" in
    --query-gpu=name) printf '%s\n' "Simulated NVIDIA GPU" ;;
    *) printf '%s\n' "NVIDIA-SMI 999.0  Driver Version: 999.0  CUDA Version: $TREX_FAKE_CUDA_VERSION" ;;
esac
'''


def _write_executable(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(0o755)


def _fake_python_sitecustomize() -> str:
    return textwrap.dedent(
        f'''\
        import os
        import sys

        if os.environ.get("TREX_FAKE_PYTHON_SITE") == "1":
            sys.argv = list(sys.orig_argv)
            try:
                exec({FAKE_PYTHON!r}, {{"__name__": "__main__"}})
            except SystemExit as error:
                sys.stdout.flush()
                sys.stderr.flush()
                status = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
                os._exit(status)
        '''
    )


def _events(state: Path) -> list[dict[str, object]]:
    path = state / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _install_events(state: Path) -> list[dict[str, object]]:
    return [item for item in _events(state) if item["kind"] == "install"]


class PostLinkSimulationMixin:
    def assert_safe_installs(
        self,
        installs: list[dict[str, object]],
        *,
        conda_numpy: bool,
        conda_opencv: bool,
    ) -> None:
        self.assertTrue(installs, "the simulation did not attempt a Torch installation")
        for install in installs:
            arguments = list(install["args"])
            self.assertEqual(
                install["constraint"],
                "numpy==2.4.6" if conda_numpy else "",
            )
            self.assertNotIn("--no-deps", arguments)
            self.assertNotIn("--force-reinstall", arguments)
            self.assertNotIn("uninstall", arguments)
            self.assertEqual(arguments.count("--index-url"), 1)
            self.assertTrue(any(str(value).startswith("torch") for value in arguments))
            self.assertTrue(any(str(value).startswith("torchvision") for value in arguments))
            for requirement in (
                "torchmetrics",
                "tqdm",
                "ultralytics>=8.3.0,<9",
                "rfdetr==1.8.3",
                "dill",
                "scikit-learn",
                "timm",
                "git+https://github.com/ultralytics/CLIP.git",
            ):
                self.assertIn(requirement, arguments)
            if conda_numpy:
                self.assertNotIn("numpy>=1.26,<3", arguments)
            else:
                self.assertIn("numpy>=1.26,<3", arguments)
            if conda_opencv:
                self.assertNotIn("opencv-python>=4.6,<5", arguments)
            else:
                self.assertIn("opencv-python>=4.6,<5", arguments)
            pytorch_extra_indexes = [
                arguments[position + 1]
                for position, value in enumerate(arguments[:-1])
                if value == "--extra-index-url"
                and "download.pytorch.org" in str(arguments[position + 1])
            ]
            self.assertEqual(pytorch_extra_indexes, [])


@unittest.skipIf(os.name == "nt", "Unix post-link simulation")
class UnixPostLinkSimulation(PostLinkSimulationMixin, unittest.TestCase):
    def run_scenario(
        self,
        *,
        system: str,
        machine: str,
        cuda: str = "",
        outcomes: tuple[str, ...] = ("success",),
        conda_numpy: bool = True,
        conda_opencv: bool = True,
        warm_outcome: str = "success",
        expect_installs: bool = True,
    ) -> tuple[list[dict[str, object]], str]:
        with tempfile.TemporaryDirectory(prefix="trex-post-link-") as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            state = root / "state"
            prefix = root / "prefix"
            fake_bin.mkdir()
            state.mkdir()
            (prefix / "conda-meta").mkdir(parents=True)
            if conda_numpy:
                (prefix / "conda-meta" / "numpy-2.4.6-simulated.json").write_text(
                    '{"version":"2.4.6"}', encoding="utf-8"
                )
            if conda_opencv:
                (prefix / "conda-meta" / "py-opencv-4.12-simulated.json").write_text(
                    "{}", encoding="utf-8"
                )

            driver = root / "fake_python.py"
            _write_executable(driver, FAKE_PYTHON)
            _write_executable(
                fake_bin / "python",
                "#!/bin/sh\nexec "
                + shlex.quote(sys.executable)
                + " "
                + shlex.quote(str(driver))
                + ' "$@"\n',
            )
            _write_executable(fake_bin / "uname", FAKE_UNAME)
            _write_executable(fake_bin / "nvidia-smi", FAKE_NVIDIA_SMI_SH)

            environment = os.environ.copy()
            environment.pop("GITHUB_WORKSPACE", None)
            environment.update(
                {
                    "PATH": str(fake_bin) + os.pathsep + environment["PATH"],
                    "PREFIX": str(prefix),
                    "TMPDIR": str(root),
                    "TREX_FAKE_STATE": str(state),
                    "TREX_FAKE_SYSTEM": system,
                    "TREX_FAKE_MACHINE": machine,
                    "TREX_FAKE_CUDA_VERSION": cuda,
                    "TREX_FAKE_INSTALL_OUTCOMES": ",".join(outcomes),
                    "TREX_FAKE_WARM_OUTCOME": warm_outcome,
                    "TREX_POST_LINK_OUTPUT": "stdout",
                }
            )
            result = subprocess.run(
                ["bash", str(POST_LINK_SH)],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            output = result.stdout + result.stderr
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            self.assertEqual(result.returncode, 0, output)
            installs = _install_events(state)
            if expect_installs and not installs:
                self.fail(output)
            if expect_installs:
                self.assert_safe_installs(
                    installs,
                    conda_numpy=conda_numpy,
                    conda_opencv=conda_opencv,
                )
            else:
                self.assertEqual(installs, [])
            return installs, output

    def test_macos_uses_pypi_and_other_cpu_platforms_use_cpu_index(self) -> None:
        cases = (
            ("Darwin", "arm64", "13.2", PYPI_INDEX),
            ("Darwin", "x86_64", "13.2", PYPI_INDEX),
            ("Linux", "aarch64", "13.2", PYPI_INDEX),
            ("Linux", "arm64", "13.2", PYPI_INDEX),
            ("FreeBSD", "amd64", "13.2", PYPI_INDEX),
        )
        for system, machine, cuda, expected_index in cases:
            with self.subTest(system=system, machine=machine):
                installs, _ = self.run_scenario(system=system, machine=machine, cuda=cuda)
                self.assertEqual([item["index"] for item in installs], [expected_index])
                self.assertFalse(any("+cu" in str(value) for value in installs[0]["args"]))
                if system == "Darwin":
                    self.assertIn("torch==2.6.0", installs[0]["args"])
                    self.assertIn("torchvision==0.21.0", installs[0]["args"])

    def test_linux_without_a_usable_nvidia_driver_uses_cpu_index(self) -> None:
        installs, output = self.run_scenario(system="Linux", machine="x86_64")
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])
        self.assertIn("No usable NVIDIA driver detected", output)

    def test_resolution_failure_is_not_retried(self) -> None:
        installs, output = self.run_scenario(
            system="Linux",
            machine="x86_64",
            outcomes=("resolution",),
        )
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])
        self.assertIn("torch>=2.2", installs[0]["args"])
        self.assertIn("torchvision>=0.17", installs[0]["args"])
        self.assertIn("no retry was attempted", output)

    def test_linux_selects_exactly_one_compatible_cuda_index(self) -> None:
        cases = {
            "11.7": CPU_INDEX,
            "11.8": "https://download.pytorch.org/whl/cu118",
            "12.0": "https://download.pytorch.org/whl/cu118",
            "12.1": "https://download.pytorch.org/whl/cu121",
            "12.2": "https://download.pytorch.org/whl/cu121",
            "12.4": "https://download.pytorch.org/whl/cu124",
            "12.6": "https://download.pytorch.org/whl/cu126",
            "12.8": "https://download.pytorch.org/whl/cu128",
            "12.9": "https://download.pytorch.org/whl/cu129",
            "13.0": "https://download.pytorch.org/whl/cu130",
            "13.2": "https://download.pytorch.org/whl/cu132",
            "13.3": "https://download.pytorch.org/whl/cu132",
        }
        for cuda, expected_index in cases.items():
            with self.subTest(cuda=cuda):
                installs, _ = self.run_scenario(system="Linux", machine="x86_64", cuda=cuda)
                self.assertEqual([item["index"] for item in installs], [expected_index])

    def test_failed_cuda_install_is_not_retried_or_downgraded(self) -> None:
        installs, output = self.run_scenario(
            system="Linux",
            machine="x86_64",
            cuda="12.4",
            outcomes=("network",),
        )
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu124"],
        )
        self.assertIn("no retry was attempted", output)

    def test_pip_owns_and_solves_numpy_when_conda_does_not(self) -> None:
        installs, output = self.run_scenario(
            system="Linux",
            machine="x86_64",
            conda_numpy=False,
        )
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])
        self.assertEqual(installs[0]["constraint"], "")
        self.assertIn("numpy>=1.26,<3", installs[0]["args"])
        self.assertIn("Conda does not own NumPy", output)

    def test_buildall_uses_one_pip_opencv_provider(self) -> None:
        installs, output = self.run_scenario(
            system="Linux",
            machine="x86_64",
            conda_opencv=False,
        )
        self.assertIn("opencv-python>=4.6,<5", installs[0]["args"])
        self.assertIn("non-minimal profile", output)

    def test_yolo_cache_warm_failure_is_warning_only(self) -> None:
        installs, output = self.run_scenario(
            system="Darwin",
            machine="arm64",
            warm_outcome="failure",
        )
        self.assertEqual(len(installs), 1)
        self.assertIn("WARNING: YOLO runtime warm-up failed", output)
        self.assertNotIn("TRex PYTHON ML SETUP IS INCOMPLETE", output)

@unittest.skipUnless(os.name == "nt", "native Windows batch simulation")
class WindowsPostLinkSimulation(PostLinkSimulationMixin, unittest.TestCase):
    def run_scenario(
        self,
        *,
        cuda: str = "",
        outcomes: tuple[str, ...] = ("success",),
        conda_numpy: bool = True,
        conda_opencv: bool = True,
        warm_outcome: str = "success",
        expect_installs: bool = True,
    ) -> tuple[list[dict[str, object]], str]:
        with tempfile.TemporaryDirectory(prefix="trex-post-link-") as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            state = root / "state"
            prefix = root / "prefix"
            fake_bin.mkdir()
            state.mkdir()
            (prefix / "conda-meta").mkdir(parents=True)
            if conda_numpy:
                (prefix / "conda-meta" / "numpy-2.4.6-simulated.json").write_text(
                    '{"version":"2.4.6"}', encoding="utf-8"
                )
            if conda_opencv:
                (prefix / "conda-meta" / "py-opencv-4.12-simulated.json").write_text(
                    "{}", encoding="utf-8"
                )

            fake_site = root / "fake-site"
            fake_site.mkdir()
            (fake_site / "sitecustomize.py").write_text(
                _fake_python_sitecustomize(),
                encoding="utf-8",
            )
            if cuda:
                (fake_bin / "nvidia-smi.cmd").write_text(
                    textwrap.dedent(
                        """\
                        @echo off
                        if "%~1"=="--query-gpu=name" (
                          echo Simulated NVIDIA GPU
                          exit /b 0
                        )
                        echo NVIDIA-SMI 999.0  Driver Version: 999.0  CUDA Version: %TREX_FAKE_CUDA_VERSION%
                        """
                    ).replace("\n", "\r\n"),
                    encoding="utf-8",
                )

            system32 = Path(os.environ["SystemRoot"]) / "System32"
            environment = os.environ.copy()
            environment.pop("GITHUB_WORKSPACE", None)
            environment.update(
                {
                    "PATH": os.pathsep.join((str(fake_bin), str(Path(sys.executable).parent), str(system32))),
                    "PYTHONPATH": str(fake_site),
                    "PREFIX": str(prefix),
                    "TEMP": str(root),
                    "TMP": str(root),
                    "TREX_FAKE_PYTHON_SITE": "1",
                    "TREX_FAKE_STATE": str(state),
                    "TREX_FAKE_CUDA_VERSION": cuda,
                    "TREX_FAKE_INSTALL_OUTCOMES": ",".join(outcomes),
                    "TREX_FAKE_WARM_OUTCOME": warm_outcome,
                    "TREX_POST_LINK_OUTPUT": "stdout",
                }
            )
            result = subprocess.run(
                [os.environ.get("COMSPEC", str(system32 / "cmd.exe")), "/d", "/c", str(POST_LINK_BAT)],
                cwd=REPOSITORY,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            output = result.stdout + result.stderr
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            self.assertEqual(result.returncode, 0, output)
            installs = _install_events(state)
            if expect_installs and not installs:
                self.fail(output)
            if expect_installs:
                self.assert_safe_installs(
                    installs,
                    conda_numpy=conda_numpy,
                    conda_opencv=conda_opencv,
                )
            else:
                self.assertEqual(installs, [])
            return installs, output

    def test_windows_without_nvidia_uses_cpu_index(self) -> None:
        installs, _ = self.run_scenario()
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])

    def test_windows_resolution_failure_is_not_retried(self) -> None:
        installs, output = self.run_scenario(outcomes=("resolution",))
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])
        self.assertIn("no retry was attempted", output)

    def test_windows_selects_exactly_one_compatible_cuda_index(self) -> None:
        cases = {
            "11.7": CPU_INDEX,
            "11.8": "https://download.pytorch.org/whl/cu118",
            "12.0": "https://download.pytorch.org/whl/cu118",
            "12.1": "https://download.pytorch.org/whl/cu121",
            "12.4": "https://download.pytorch.org/whl/cu124",
            "12.6": "https://download.pytorch.org/whl/cu126",
            "12.8": "https://download.pytorch.org/whl/cu128",
            "12.9": "https://download.pytorch.org/whl/cu129",
            "13.0": "https://download.pytorch.org/whl/cu130",
            "13.2": "https://download.pytorch.org/whl/cu132",
            "13.3": "https://download.pytorch.org/whl/cu132",
        }
        for cuda, expected_index in cases.items():
            with self.subTest(cuda=cuda):
                installs, _ = self.run_scenario(cuda=cuda)
                self.assertEqual([item["index"] for item in installs], [expected_index])

    def test_windows_failed_cuda_install_is_not_retried(self) -> None:
        installs, output = self.run_scenario(cuda="12.4", outcomes=("network",))
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu124"],
        )
        self.assertIn("no retry was attempted", output)

    def test_windows_pip_owns_numpy_when_conda_does_not(self) -> None:
        installs, _ = self.run_scenario(conda_numpy=False)
        self.assertEqual(installs[0]["constraint"], "")
        self.assertIn("numpy>=1.26,<3", installs[0]["args"])

    def test_windows_buildall_uses_one_pip_opencv_provider(self) -> None:
        installs, output = self.run_scenario(conda_opencv=False)
        self.assertIn("opencv-python>=4.6,<5", installs[0]["args"])
        self.assertIn("non-minimal profile", output)

    def test_windows_yolo_cache_warm_failure_is_warning_only(self) -> None:
        installs, output = self.run_scenario(warm_outcome="failure")
        self.assertEqual(len(installs), 1)
        self.assertIn("WARNING: YOLO runtime warm-up failed", output)
        self.assertNotIn("TRex PYTHON ML SETUP IS INCOMPLETE", output)

    def test_windows_total_failure_warns_but_returns_success(self) -> None:
        installs, output = self.run_scenario(
            cuda="12.4",
            outcomes=("network",),
        )
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu124"],
        )
        self.assertIn("WARNING: TRex PYTHON ML SETUP IS INCOMPLETE", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
