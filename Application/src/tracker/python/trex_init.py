import locale
locale.setlocale(locale.LC_ALL, 'C')

import sys
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
except ImportError:
    TRex.log("PyTorch is not installed")

if "tf" in locals():
    # Your testing code here
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                TRex.log(str(len(gpus))+ " Physical GPUs, "+str( len(logical_gpus))+" Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            TRex.log(str(e))

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

    TRex.log('setting version '+str(sys.version)+" "+str(found)+" "+str(physical))
    set_version(sys.version, found, physical)

elif "torch" in locals():
    if torch.cuda.is_available():
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
