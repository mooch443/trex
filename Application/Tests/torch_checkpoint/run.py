"""Run the packaged checkpoint probes in separate processes and report crashes."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Directory containing the probe executables")
    parser.add_argument("--checkpoint", type=Path,
                        help="Load an existing TorchScript checkpoint instead of generated fixtures")
    args = parser.parse_args()
    probe = Path(__file__).resolve().with_name("probe.py")
    binary_dir = args.bin_dir.resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="", PYTHONHOME=sys.prefix,
               TREX_TEST_PYTHON_EXECUTABLE=sys.executable)
    results = []

    def run(label, command):
        print(f"\n=== {label} ===", flush=True)
        try:
            result = subprocess.run(command, env=env, timeout=120)
            if result.returncode < 0:
                status = f"FAIL: {signal.Signals(-result.returncode).name}"
            elif result.returncode:
                status = f"FAIL: exit {result.returncode}"
            else:
                status = "PASS"
        except (OSError, subprocess.TimeoutExpired) as error:
            status = f"FAIL: {error}"
        results.append((label, status))
        print(f"{label}: {status}", flush=True)
        return status == "PASS"

    with tempfile.TemporaryDirectory(prefix="trex-torch-checkpoint-", dir=".") as temporary:
        if args.checkpoint:
            load_args = ["--load-jit", str(args.checkpoint.resolve())]
        else:
            fixture = str(Path(temporary).resolve())
            if not run("prepare", [sys.executable, "-B", str(probe), "--create", fixture]):
                return 1
            load_args = ["--load", fixture]

        run("python", [sys.executable, "-B", str(probe), *load_args])
        run("embedded", [str(binary_dir / "test_torch_checkpoint"), str(probe), *load_args])
        shared = binary_dir / "test_torch_checkpoint_shared"
        if shared.is_file():
            run("shared", [str(shared), str(probe), *load_args])
        else:
            print("Shared-libstdc++ comparison is not available in this build.", flush=True)

    print("\nResults:", flush=True)
    for label, status in results:
        print(f"  {label}: {status}", flush=True)
    return 0 if all(status == "PASS" for _, status in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
