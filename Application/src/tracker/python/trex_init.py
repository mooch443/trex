import sys
import numpy as np

if not hasattr(sys, "argv") or not sys.argv or len(sys.argv) == 0:
    sys.argv = [""]
    print("avoiding tensorflow bug")

found = False
physical = ''
if int(sys.version[0]) >= 3:
    import importlib
    try:
        importlib.import_module('tensorflow')
        import tensorflow

        import platform
        if platform.system() == "Darwin":
            from tensorflow.python.compiler.mlcompute import mlcompute
            if mlcompute.is_apple_mlc_enabled():
                found = True
                physical = 'MLC'
            else:
                physical = ''

        if not found:
            from tensorflow.compat.v1 import ConfigProto, InteractiveSession
            config = ConfigProto()
            config.gpu_options.allow_growth=True
            sess = InteractiveSession(config=config)
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