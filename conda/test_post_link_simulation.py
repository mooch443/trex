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
import zipfile


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


def candidate_versions(index_url):
    flavor = index_url.rstrip("/").rsplit("/", 1)[-1]
    missing = set(filter(None, os.environ.get("TREX_FAKE_MISSING_INDEXES", "").split(",")))
    if flavor in missing or index_url in missing:
        return 1
    if flavor.startswith("cu"):
        print(f"2.7.1+{flavor}|0.22.1+{flavor}")
        print(f"2.7.0+{flavor}|0.22.0+{flavor}")
    elif flavor == "cpu":
        print("2.7.1+cpu|0.22.1+cpu")
        print("2.7.0+cpu|0.22.0+cpu")
    else:
        print("2.7.1|0.22.1")
        print("2.7.0|0.22.0")
    return 0


if args[:3] == ["-m", "pip", "--version"]:
    print("pip 99.0 (simulated)")
    raise SystemExit(0)

if args and args[0] == "-":
    raise SystemExit(candidate_versions(args[1]))

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

if args[:3] == ["-m", "pip", "check"]:
    event("pip-check")
    raise SystemExit(0)

if args and args[0] == "-c":
    code = args[1] if len(args) > 1 else ""
    if "numpy.__version__" in code:
        print("2.4.6", end="")
    elif "version('torchvision')" in code:
        print("0.22.1")
    elif "version('torch')" in code:
        print("2.7.1")
    elif "torch.cuda.is_available" in code:
        print("False")
    elif "pip','index','versions'" in code or 'pip", "index", "versions' in code:
        raise SystemExit(candidate_versions(args[-1]))
    elif "CUDA(?:" in code:
        print(os.environ.get("TREX_FAKE_CUDA_VERSION", ""))
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


def _make_wheel(
    wheelhouse: Path,
    *,
    distribution: str,
    version: str,
    requirements: tuple[str, ...] = (),
) -> Path:
    normalized = distribution.replace("-", "_")
    wheel = wheelhouse / f"{normalized}-{version}-py3-none-any.whl"
    dist_info = f"{normalized}-{version}.dist-info"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {distribution}",
        f"Version: {version}",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{normalized}/__init__.py", f'__version__ = "{version}"\n')
        archive.writestr(f"{dist_info}/METADATA", "\n".join(metadata) + "\n")
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\n"
            "Generator: trex-post-link-simulation\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    return wheel


class PostLinkSimulationMixin:
    def assert_safe_installs(self, installs: list[dict[str, object]]) -> None:
        self.assertTrue(installs, "the simulation did not attempt a Torch installation")
        for install in installs:
            arguments = list(install["args"])
            self.assertEqual(install["constraint"], "numpy==2.4.6")
            self.assertNotIn("--no-deps", arguments)
            self.assertNotIn("--force-reinstall", arguments)
            self.assertNotIn("uninstall", arguments)
            self.assertEqual(arguments.count("--index-url"), 1)
            self.assertTrue(any(str(value).startswith("torch===") for value in arguments))
            self.assertTrue(any(str(value).startswith("torchvision===") for value in arguments))
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
            pytorch_extra_indexes = [
                arguments[position + 1]
                for position, value in enumerate(arguments[:-1])
                if value == "--extra-index-url"
                and "download.pytorch.org" in str(arguments[position + 1])
            ]
            self.assertEqual(pytorch_extra_indexes, [])


class RealPipResolverSimulation(unittest.TestCase):
    def test_sitecustomize_intercepts_the_real_python_executable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="trex-python-shim-") as temporary:
            root = Path(temporary)
            state = root / "state"
            fake_site = root / "fake-site"
            state.mkdir()
            fake_site.mkdir()
            (fake_site / "sitecustomize.py").write_text(
                _fake_python_sitecustomize(),
                encoding="utf-8",
            )
            constraint = root / "constraints.txt"
            constraint.write_text("numpy==2.4.6\n", encoding="utf-8")
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONPATH": str(fake_site),
                    "TREX_FAKE_PYTHON_SITE": "1",
                    "TREX_FAKE_STATE": str(state),
                    "TREX_FAKE_INSTALL_OUTCOMES": "success",
                }
            )

            version_result = subprocess.run(
                [sys.executable, "-X", "utf8", "-m", "pip", "--version"],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(version_result.returncode, 0, version_result.stdout + version_result.stderr)
            self.assertIn("pip 99.0 (simulated)", version_result.stdout)

            install_result = subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-m",
                    "pip",
                    "install",
                    "--constraint",
                    str(constraint),
                    "--index-url",
                    CPU_INDEX,
                    "torch===2.7.1+cpu",
                    "torchvision===0.22.1+cpu",
                ],
                env=environment,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(install_result.returncode, 0, install_result.stdout + install_result.stderr)
            installs = _install_events(state)
            self.assertEqual(len(installs), 1)
            self.assertEqual(installs[0]["index"], CPU_INDEX)
            self.assertEqual(installs[0]["constraint"], "numpy==2.4.6")

    def test_pip_retries_a_pair_without_relaxing_numpy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="trex-pip-resolver-") as temporary:
            root = Path(temporary)
            wheelhouse = root / "wheels"
            wheelhouse.mkdir()
            constraint = root / "constraints.txt"
            constraint.write_text("numpy==2.4.6\n", encoding="utf-8")

            _make_wheel(wheelhouse, distribution="numpy", version="2.4.6")
            _make_wheel(
                wheelhouse,
                distribution="torch",
                version="2.7.1+cpu",
                requirements=("numpy>=3",),
            )
            _make_wheel(
                wheelhouse,
                distribution="torchvision",
                version="0.22.1+cpu",
                requirements=("torch==2.7.1",),
            )
            _make_wheel(
                wheelhouse,
                distribution="torch",
                version="2.7.0+cpu",
                requirements=("numpy>=2",),
            )
            _make_wheel(
                wheelhouse,
                distribution="torchvision",
                version="0.22.0+cpu",
                requirements=("torch==2.7.0",),
            )
            _make_wheel(
                wheelhouse,
                distribution="trex-ml-stack",
                version="1.0",
                requirements=("numpy>=2",),
            )

            results: list[subprocess.CompletedProcess[str]] = []
            for torch_version, vision_version in (
                ("2.7.1+cpu", "0.22.1+cpu"),
                ("2.7.0+cpu", "0.22.0+cpu"),
            ):
                results.append(
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--dry-run",
                            "--ignore-installed",
                            "--no-index",
                            "--find-links",
                            str(wheelhouse),
                            "--constraint",
                            str(constraint),
                            f"torch==={torch_version}",
                            f"torchvision==={vision_version}",
                            "trex-ml-stack==1.0",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=20,
                        check=False,
                    )
                )

            first_output = results[0].stdout + results[0].stderr
            second_output = results[1].stdout + results[1].stderr
            self.assertNotEqual(results[0].returncode, 0, first_output)
            self.assertIn("ResolutionImpossible", first_output)
            self.assertEqual(results[1].returncode, 0, second_output)
            self.assertIn("numpy-2.4.6", second_output)
            self.assertEqual(constraint.read_text(encoding="utf-8"), "numpy==2.4.6\n")


@unittest.skipIf(os.name == "nt", "Unix post-link simulation")
class UnixPostLinkSimulation(PostLinkSimulationMixin, unittest.TestCase):
    def run_scenario(
        self,
        *,
        system: str,
        machine: str,
        cuda: str = "",
        missing_indexes: tuple[str, ...] = (),
        outcomes: tuple[str, ...] = ("success",),
    ) -> tuple[list[dict[str, object]], str]:
        with tempfile.TemporaryDirectory(prefix="trex-post-link-") as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            state = root / "state"
            prefix = root / "prefix"
            fake_bin.mkdir()
            state.mkdir()
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta" / "py-opencv-simulated.json").write_text("{}", encoding="utf-8")

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
                    "TREX_FAKE_MISSING_INDEXES": ",".join(missing_indexes),
                    "TREX_FAKE_INSTALL_OUTCOMES": ",".join(outcomes),
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
            messages_path = prefix / ".messages.txt"
            messages = messages_path.read_text(encoding="utf-8") if messages_path.exists() else ""
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + messages)
            installs = _install_events(state)
            if not installs:
                self.fail(result.stdout + result.stderr + messages)
            self.assert_safe_installs(installs)
            return installs, messages

    def test_native_cpu_platforms_use_only_pypi(self) -> None:
        cases = (
            ("Darwin", "arm64", "13.2"),
            ("Darwin", "x86_64", "13.2"),
            ("Linux", "aarch64", "13.2"),
            ("Linux", "arm64", "13.2"),
            ("FreeBSD", "amd64", "13.2"),
        )
        for system, machine, cuda in cases:
            with self.subTest(system=system, machine=machine):
                installs, _ = self.run_scenario(system=system, machine=machine, cuda=cuda)
                self.assertEqual([item["index"] for item in installs], [PYPI_INDEX])
                self.assertFalse(any("+cu" in str(value) for value in installs[0]["args"]))

    def test_linux_without_a_usable_nvidia_driver_uses_cpu_index(self) -> None:
        installs, messages = self.run_scenario(system="Linux", machine="x86_64")
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])
        self.assertIn("No usable NVIDIA driver detected", messages)

    def test_original_no_nvidia_cpu_numpy_conflict_retries_cpu_pair(self) -> None:
        installs, messages = self.run_scenario(
            system="Linux",
            machine="x86_64",
            outcomes=("resolution", "success"),
        )
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX, CPU_INDEX])
        self.assertIn("torch===2.7.1+cpu", installs[0]["args"])
        self.assertIn("torchvision===0.22.1+cpu", installs[0]["args"])
        self.assertIn("torch===2.7.0+cpu", installs[1]["args"])
        self.assertIn("torchvision===0.22.0+cpu", installs[1]["args"])
        self.assertEqual({item["constraint"] for item in installs}, {"numpy==2.4.6"})
        self.assertNotIn("Completed with issues", messages)

    def test_linux_selects_exactly_one_compatible_cuda_index(self) -> None:
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
        }
        for cuda, expected_index in cases.items():
            with self.subTest(cuda=cuda):
                installs, _ = self.run_scenario(system="Linux", machine="x86_64", cuda=cuda)
                self.assertEqual([item["index"] for item in installs], [expected_index])

    def test_resolution_conflict_retries_older_pair_with_same_constraints(self) -> None:
        installs, _ = self.run_scenario(
            system="Linux",
            machine="x86_64",
            cuda="12.4",
            outcomes=("resolution", "success"),
        )
        expected_index = "https://download.pytorch.org/whl/cu124"
        self.assertEqual([item["index"] for item in installs], [expected_index, expected_index])
        self.assertIn("torch===2.7.1+cu124", installs[0]["args"])
        self.assertIn("torch===2.7.0+cu124", installs[1]["args"])
        self.assertEqual({item["constraint"] for item in installs}, {"numpy==2.4.6"})

    def test_missing_cuda_index_falls_back_before_installing(self) -> None:
        installs, _ = self.run_scenario(
            system="Linux",
            machine="x86_64",
            cuda="12.4",
            missing_indexes=("cu124",),
        )
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])

    def test_failed_cuda_install_falls_back_to_cpu_not_another_cuda(self) -> None:
        installs, _ = self.run_scenario(
            system="Linux",
            machine="x86_64",
            cuda="11.8",
            outcomes=("network", "success"),
        )
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu118", CPU_INDEX],
        )

    def test_missing_cpu_index_falls_back_to_default_pypi(self) -> None:
        installs, _ = self.run_scenario(
            system="Linux",
            machine="x86_64",
            missing_indexes=("cpu",),
        )
        self.assertEqual([item["index"] for item in installs], [PYPI_INDEX])

    def test_no_attempt_ever_uninstalls_the_existing_torch(self) -> None:
        installs, messages = self.run_scenario(
            system="Linux",
            machine="x86_64",
            cuda="12.4",
            outcomes=("network", "network", "network"),
        )
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu124", CPU_INDEX, PYPI_INDEX],
        )
        self.assertNotIn("pip uninstall", messages)


@unittest.skipUnless(os.name == "nt", "native Windows batch simulation")
class WindowsPostLinkSimulation(PostLinkSimulationMixin, unittest.TestCase):
    def run_scenario(
        self,
        *,
        cuda: str = "",
        missing_indexes: tuple[str, ...] = (),
        outcomes: tuple[str, ...] = ("success",),
    ) -> tuple[list[dict[str, object]], str]:
        with tempfile.TemporaryDirectory(prefix="trex-post-link-") as temporary:
            root = Path(temporary)
            fake_bin = root / "bin"
            state = root / "state"
            prefix = root / "prefix"
            fake_bin.mkdir()
            state.mkdir()
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta" / "py-opencv-simulated.json").write_text("{}", encoding="utf-8")

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
                    "TREX_FAKE_MISSING_INDEXES": ",".join(missing_indexes),
                    "TREX_FAKE_INSTALL_OUTCOMES": ",".join(outcomes),
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
            messages_path = prefix / ".messages.txt"
            messages = messages_path.read_text(encoding="utf-8", errors="replace") if messages_path.exists() else ""
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr + messages)
            installs = _install_events(state)
            if not installs:
                self.fail(result.stdout + result.stderr + messages)
            self.assert_safe_installs(installs)
            return installs, messages

    def test_windows_without_nvidia_uses_cpu_index(self) -> None:
        installs, _ = self.run_scenario()
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])

    def test_original_windows_no_nvidia_cpu_numpy_conflict_retries_cpu_pair(self) -> None:
        installs, messages = self.run_scenario(outcomes=("resolution", "success"))
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX, CPU_INDEX])
        self.assertIn("torch===2.7.1+cpu", installs[0]["args"])
        self.assertIn("torchvision===0.22.1+cpu", installs[0]["args"])
        self.assertIn("torch===2.7.0+cpu", installs[1]["args"])
        self.assertIn("torchvision===0.22.0+cpu", installs[1]["args"])
        self.assertEqual({item["constraint"] for item in installs}, {"numpy==2.4.6"})
        self.assertNotIn("Completed with issues", messages)

    def test_windows_uses_one_driver_compatible_cuda_index(self) -> None:
        installs, _ = self.run_scenario(cuda="12.4")
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu124"],
        )

    def test_windows_retries_pair_without_changing_numpy_or_cuda(self) -> None:
        installs, _ = self.run_scenario(cuda="12.4", outcomes=("resolution", "success"))
        expected_index = "https://download.pytorch.org/whl/cu124"
        self.assertEqual([item["index"] for item in installs], [expected_index, expected_index])
        self.assertEqual({item["constraint"] for item in installs}, {"numpy==2.4.6"})

    def test_windows_missing_cuda_index_falls_back_to_cpu(self) -> None:
        installs, _ = self.run_scenario(cuda="12.4", missing_indexes=("cu124",))
        self.assertEqual([item["index"] for item in installs], [CPU_INDEX])

    def test_windows_failed_cuda_install_falls_back_to_cpu(self) -> None:
        installs, _ = self.run_scenario(cuda="11.8", outcomes=("network", "success"))
        self.assertEqual(
            [item["index"] for item in installs],
            ["https://download.pytorch.org/whl/cu118", CPU_INDEX],
        )

    def test_windows_missing_cpu_index_falls_back_to_pypi(self) -> None:
        installs, _ = self.run_scenario(missing_indexes=("cpu",))
        self.assertEqual([item["index"] for item in installs], [PYPI_INDEX])


if __name__ == "__main__":
    unittest.main(verbosity=2)
