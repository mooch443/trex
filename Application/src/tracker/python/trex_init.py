import locale
import os
import sys
import warnings

def enable_utf8() -> None:
    """
    Enable UTF-8 encoding for both locale-aware operations and stdio streams.
    Tries a prioritized list of UTF-8 locales on POSIX, and uses the PYTHONUTF8 env var on Windows.
    Should be called at the very start of `if __name__ == '__main__'`.
    """
    # Ensure Python runs in UTF-8 mode (Python 3.7+)
    os.environ.setdefault('PYTHONUTF8', '1')

    # Attempt to set a UTF-8 locale on POSIX
    if os.name != 'nt':
        candidates = ['C.UTF-8', '', 'en_US.UTF-8']
        for name in candidates:
            try:
                locale.setlocale(locale.LC_ALL, name)
                break
            except locale.Error:
                continue
    else:
        # On Windows, rely on PYTHONUTF8=1 or -X utf8
        # Optionally, adjust console code page (requires ctypes):
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleCP(65001)
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass

    # Reconfigure stdio to UTF-8 if supported
    if hasattr(sys.stdin, 'reconfigure'):
        sys.stdin.reconfigure(encoding='utf-8')
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

enable_utf8()

import numpy as np
import TRex

#torch.cuda.empty_cache()
#total_memory = torch.cuda.get_device_properties(0).total_memory
#torch.cuda.set_per_process_memory_fraction(0.01)

#print("total_memory = ", total_memory)

#if not hasattr(sys, "argv") or not sys.argv or len(sys.argv) == 0:
#    sys.argv = [""]
#    print("avoiding tensorflow bug")

#import tensorflow as tf
#print("TensorFlow version:", tf.__version__)

try:
    import torch
    TRex.log("PyTorch version:"+str(torch.__version__))

    try:
        import platform;
        if platform.system() == "Darwin" and platform.processor() == "i386":
            TRex.log("Disabling multi-threading on Intel Macs to work around an OpenMP crash.")
            torch.set_num_threads(1)
    except Exception as e:
        TRex.log("Error when disabling multi-threading: "+str(e))
    
    cuda_available = False
    cuda_problem = None
    try:
        with warnings.catch_warnings(record=True) as cuda_warnings:
            warnings.simplefilter("always")
            cuda_available = torch.cuda.is_available()

        for warning in cuda_warnings:
            message = str(warning.message)
            TRex.warn(message)
            lowered = message.lower()
            if cuda_problem is None and (
                "cuda initialization" in lowered
                or "nvidia driver" in lowered
                or "cuda driver version is insufficient" in lowered
            ):
                cuda_problem = message
    except Exception as e:
        cuda_problem = "The CUDA availability check failed: " + str(e)
        TRex.warn(cuda_problem)

    if cuda_problem is not None:
        cuda_runtime = getattr(getattr(torch, "version", None), "cuda", None)
        version_details = f"PyTorch {torch.__version__}"
        if cuda_runtime is not None:
            version_details += f" was built for CUDA {cuda_runtime}"

        TRex.show_runtime_warning(
            "PyTorch could not initialize CUDA, so TRex will continue using the CPU. "
            "Machine-learning predictions may be much slower.\n\n"
            f"{version_details}.\n{cuda_problem}\n\n"
            "Update the NVIDIA driver or install a PyTorch build compatible with the installed driver."
        )

    if cuda_available:
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            TRex.log(f"PyTorch CUDA Device {i}: {torch.cuda.get_device_name(i)}")
        TRex.log(f"Choosing device {torch.cuda.current_device()}")
        set_version(sys.version, True, torch.cuda.get_device_name())

    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            TRex.log('PyTorch is not built with MPS support')
        else:
            TRex.log("PyTorch was built with MPS support, but it's not available. Please check whether all necessary libraries are installed and the correct version is used.")
    else:
        TRex.log("Using Apple Metal for PyTorch")
        set_version(sys.version, True, 'METAL')

except ImportError:
    TRex.log("PyTorch is not installed")
