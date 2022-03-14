#ifndef _PYLON_CAMERA_H
#define _PYLON_CAMERA_H

#include "Camera.h"

#if WITH_PYLON
//#include <fake_pylon.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"

#if __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wall"
#pragma clang diagnostic ignored "-Wpedantic"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wdelete-non-virtual-dtor"
#endif

#include <pylon/PylonIncludes.h>

#include <pylon/usb/BaslerUsbInstantCamera.h>
typedef Pylon::CBaslerUsbInstantCamera Camera_t;
using namespace Basler_UsbCameraParams;

#if __clang__
#pragma clang diagnostic pop
#endif
#pragma GCC diagnostic pop

namespace fg {
    class PylonCamera : public Camera {
        Camera_t *_camera;
        Size2 _size;
        std::recursive_mutex _mutex;
        
    public:
        PylonCamera();
        virtual ~PylonCamera();
        
        virtual bool open() override {
            std::unique_lock<std::recursive_mutex> lock(_mutex);
            return _camera->IsOpen();
        }
        
        virtual void close() override {
            std::unique_lock<std::recursive_mutex> lock(_mutex);
            _camera->Close();
        }
        
        virtual bool next(Image& image) override;
        virtual Size2 size() const override { return _size; }
        
    private:
        
    };
}

#endif
#endif
