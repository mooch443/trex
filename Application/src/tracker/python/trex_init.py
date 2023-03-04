import sys
import numpy as np
import torch

#torch.cuda.empty_cache()
#total_memory = torch.cuda.get_device_properties(0).total_memory
#torch.cuda.set_per_process_memory_fraction(0.01)

#print("total_memory = ", total_memory)

#if not hasattr(sys, "argv") or not sys.argv or len(sys.argv) == 0:
#    sys.argv = [""]
#    print("avoiding tensorflow bug")

import tensorflow as tf
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
    import importlib
    try:
        importlib.import_module('tensorflow')
        import tensorflow

        from tensorflow.python.client import device_lib
        gpus = [x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']
        found = len(gpus) > 0
        if found:
            for device in gpus:
                physical = device.split(',')[1].split(': ')[1]

    except ImportError:
        found = False
else:
    try:
        imp.find_module('tensorflow')
        from tensorflow.python.client import device_lib
        found = len([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0
    except ImportError:
        pass

print('setting version',sys.version,found,physical)
set_version(sys.version, found, physical)
