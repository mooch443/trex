"""CPU-only checkpoint probe, usable in standalone or embedded Python."""

import argparse
import faulthandler
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", type=Path, metavar="FIXTURE_DIR")
    mode.add_argument("--load", type=Path, metavar="FIXTURE_DIR")
    mode.add_argument("--load-jit", type=Path, metavar="CHECKPOINT")
    args = parser.parse_args()

    faulthandler.enable()
    print(f"Python: {sys.version}; prefix: {sys.prefix}", flush=True)
    print("Importing torch...", flush=True)
    import torch

    print(f"Torch: {torch.__version__}; file: {torch.__file__}; "
          f"C++11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}", flush=True)
    maps = Path("/proc/self/maps")
    if maps.exists():
        runtimes = sorted({line.split()[-1] for line in maps.read_text().splitlines()
                           if "libstdc++.so" in line})
        print(f"Shared C++ runtimes: {runtimes}", flush=True)
    if sys.platform == "linux":
        torch.set_num_threads(2)
        # Batched GEMM must stay in Torch when the host already loaded OpenCV/BLAS.
        with torch.backends.mkldnn.flags(enabled=False):
            for dtype in (torch.float32, torch.float64):
                print(f"CPU batched matmul: {dtype}, shape=(16, 16, 16)", flush=True)
                left = torch.arange(16**3, dtype=dtype, device="cpu").reshape(16, 16, 16)
                right = torch.eye(16, dtype=dtype, device="cpu").repeat(16, 1, 1)
                torch.testing.assert_close(torch.bmm(left, right), left, rtol=0, atol=0)
                print(f"PASS: CPU batched matmul {dtype}", flush=True)
    torch.set_num_threads(1)

    if args.create:
        args.create.mkdir(parents=True, exist_ok=True)
        model = torch.nn.Linear(1, 1).eval()
        with torch.no_grad():
            model.weight.fill_(2)
            model.bias.fill_(1)
        metadata = {"probe": "torch-checkpoint"}
        torch.save({"state_dict": model.state_dict(), "metadata": metadata},
                   args.create / "weights_dict.pth")
        traced = torch.jit.trace(model, torch.tensor([[3.0]]), check_trace=False)
        torch.jit.save(traced, str(args.create / "weights_model.pth"),
                       _extra_files={"metadata": json.dumps(metadata)})
        print("PASS: created dictionary and TorchScript checkpoints.", flush=True)
        return

    if args.load:
        dictionary = args.load / "weights_dict.pth"
        print(f"Trying dictionary as TorchScript: {dictionary}", flush=True)
        try:
            torch.jit.load(str(dictionary), map_location="cpu")
        except RuntimeError:
            print("Expected JIT rejection; loading with torch.load...", flush=True)
        else:
            raise AssertionError("A state dictionary unexpectedly loaded as TorchScript")
        checkpoint = torch.load(dictionary, map_location="cpu", weights_only=True)
        assert checkpoint["metadata"] == {"probe": "torch-checkpoint"}
        torch.testing.assert_close(checkpoint["state_dict"]["weight"], torch.tensor([[2.0]]))

    path = args.load_jit if args.load_jit else args.load / "weights_model.pth"
    print(f"Loading TorchScript: {path}", flush=True)
    extra = {"metadata": ""}
    loaded = torch.jit.load(str(path), map_location="cpu", _extra_files=extra)
    if args.load:
        assert json.loads(extra["metadata"]) == {"probe": "torch-checkpoint"}
        with torch.no_grad():
            torch.testing.assert_close(loaded(torch.tensor([[3.0]])), torch.tensor([[7.0]]))
    print("PASS: checkpoint loaded successfully.", flush=True)


if __name__ == "__main__":
    main()
