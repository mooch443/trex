"""Compare Linux CPU batched matmul with native OpenCV and Torch preload orders.

Run with the affected environment's Python; no model or compiled test is needed.
--native-library can select the liblapack.so.3 seen in a crash instead of OpenCV.
"""

import argparse
import faulthandler
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys


def probe():
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()
    print(f"LD_PRELOAD={os.environ.get('LD_PRELOAD', '')}", flush=True)
    import torch

    print(f"Torch {torch.__version__}: {torch.__file__}", flush=True)
    libraries = sorted({
        line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines()
        if any(name in line for name in (
            "libopencv_core", "liblapack", "libblas", "libopenblas", "libmkl",
            "libtorch_cpu", "libgomp", "libiomp", "libstdc++",
        ))
    })
    for library in libraries:
        print(f"Loaded: {library}", flush=True)
    if not torch.backends.mkl.is_available():
        raise RuntimeError("This probe requires the MKL batched-GEMM path in PyTorch.")

    # Exercise BLAS directly instead of taking a oneDNN matmul alternative.
    # 16**3 exceeds PyTorch's small scalar-loop threshold for batched matmul.
    torch.backends.mkldnn.enabled = False
    for dtype in (torch.float32, torch.float64):
        print(f"CPU batched matmul: {dtype}, shape=(16, 16, 16)", flush=True)
        left = torch.arange(16**3, dtype=dtype, device="cpu").reshape(16, 16, 16)
        right = torch.eye(16, dtype=dtype, device="cpu").repeat(16, 1, 1)
        result = torch.bmm(left, right)
        torch.testing.assert_close(result, left, rtol=0, atol=0)
        print(f"PASS: {dtype}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-library", type=Path,
                        default=Path(sys.prefix) / "lib" / "libopencv_core.so")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("This probe is for the Linux BLAS loading failure.")
    if args.child:
        probe()
        return 0

    native = args.native_library.resolve()
    if not native.is_file():
        parser.error(f"Native library not found: {native}; use --native-library PATH")
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        parser.error("PyTorch is not installed in this Python environment.")
    torch_cpu = Path(spec.origin).parent / "lib" / "libtorch_cpu.so"
    if not torch_cpu.is_file():
        parser.error(f"Torch CPU library not found: {torch_cpu}")
    if any(character in str(path) for path in (native, torch_cpu)
           for character in " :\t\n"):
        parser.error("LD_PRELOAD paths cannot contain spaces, colons, tabs, or newlines.")

    # Each subprocess starts with its own loader order, including the baseline.
    # CUDA visibility and thread limits apply only to these tiny CPU probes.
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="2",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    env.pop("LD_PRELOAD", None)
    cases = {
        "python": "",
        "native-first": str(native),
        "torch-first": f"{torch_cpu}:{native}",
    }
    results = {}
    for name, preload in cases.items():
        print(f"\n=== {name} ===", flush=True)
        child_env = dict(env, LD_PRELOAD=preload)
        try:
            completed = subprocess.run(
                [sys.executable, "-B", str(Path(__file__).resolve()), "--child"],
                env=child_env, timeout=45, check=False,
            )
            results[name] = completed.returncode
            status = (signal.Signals(-completed.returncode).name
                      if completed.returncode < 0 else f"exit {completed.returncode}")
        except (OSError, subprocess.TimeoutExpired) as error:
            results[name] = None
            status = str(error)
        print(f"{name}: {status}", flush=True)

    print(f"\nResults: {results}", flush=True)
    if (results["python"] == 0 and results["native-first"] == -signal.SIGSEGV
            and results["torch-first"] == 0):
        print("Reproduced a native-library load-order-dependent SIGSEGV.", flush=True)
    else:
        print("The expected pass / SIGSEGV / pass pattern was not reproduced.", flush=True)
    return 0 if all(code == 0 for code in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
