import sys
import numpy as np

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
    print("PyTorch version:", torch.__version__)
except ImportError:
    print("PyTorch is not installed")

if "tf" in locals():
    # Your testing code here
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    found = False
    physical = ''

    if int(sys.version[0]) >= 3:
        from tensorflow.python.client import device_lib
        gpus = [x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        found = len(gpus) > 0
        if found:
            for device in gpus:
                physical = device.split(',')[1].split(': ')[1]
    else:
        from tensorflow.python.client import device_lib
        found = len([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0

    print('setting version',sys.version,found,physical)
    set_version(sys.version, found, physical)

elif "torch" in locals():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"PyTorch CUDA Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"Choosing device {torch.cuda.current_device()}")
        set_version(sys.version, True, torch.cuda.get_device_name())

    elif not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print('PyTorch is not built with MPS support')
        else:
            print("PyTorch was built with MPS support, but it's not available. Please check whether all necessary libraries are installed and the correct version is used.")
    else:
        print("Using Apple Metal for PyTorch")
        set_version(sys.version, True, 'METAL')