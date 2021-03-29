print('import sys')
import sys
found = True
physical = ''
print('sys.version = ',sys.version)
if int(sys.version[0]) >= 3:
        print('import importlib')
        import importlib
        try:
                print('importlib.import_module(tensorflow)')
                importlib.import_module('tensorflow')
                import tensorflow
                if True:
                        print('tensorflow.compat.v1')
                        from tensorflow.compat.v1 import ConfigProto, InteractiveSession
                        config = ConfigProto()
                        print('config.gpu_options')
                        config.gpu_options.allow_growth=True
                        sess = InteractiveSession(config=config)
                        from tensorflow.python.client import device_lib
                        print('before gpus = ...')
                        gpus = [x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']
                        print('result:', gpus)
                        found = len(gpus) > 0
                        if found:
                                for device in gpus:
                                        physical = device.split(',')[1].split(': ')[1]
                        print('after')
        except ImportError:
                found = False
else:
        try:
                imp.find_module('tensorflow')
                from tensorflow.python.client import device_lib
                found = len([x.physical_device_desc for x in device_lib.list_local_devices() if x.device_type == 'GPU']) > 0
        except ImportError:
                found = False
print('setting version',sys.version,found,physical)
