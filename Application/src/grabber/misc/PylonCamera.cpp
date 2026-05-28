#if WITH_PYLON

#include "PylonCamera.h"

#include <misc/GlobalSettings.h>

namespace fg {

using namespace cmn;

namespace {
BaslerCameraRequest request_from_settings() {
    BaslerCameraRequest request;
    if(GlobalSettings::has_value("basler_runtime_root")) {
        request.runtime_root = READ_SETTING(basler_runtime_root, file::Path);
    }
    if(GlobalSettings::has_value("cam_serial_number")) {
        const auto serial = READ_SETTING(cam_serial_number, std::string);
        if(!serial.empty()) {
            request.serial_number = serial;
        }
    }
    if(GlobalSettings::has_value("cam_resolution")) {
        request.requested_size = READ_SETTING(cam_resolution, Size2);
    }
    if(GlobalSettings::has_value("cam_limit_exposure")) {
        request.exposure_us = READ_SETTING(cam_limit_exposure, int);
    }
    if(GlobalSettings::has_value("cam_framerate")) {
        request.frame_rate = READ_SETTING(cam_framerate, int);
    }
    return request;
}
}

PylonCamera::PylonCamera(std::unique_ptr<BaslerBackend>&& backend)
    : Camera(),
      _backend(std::move(backend))
{
}

PylonCamera::~PylonCamera() {
    close();
}

PylonCamera& PylonCamera::operator=(PylonCamera &&other) noexcept {
    std::scoped_lock g{other._mutex, _mutex};
    _backend = std::move(other._backend);
    return *this;
}

PylonCamera::PylonCamera(PylonCamera&& other) noexcept {
    *this = std::move(other);
}

std::expected<PylonCamera, BaslerRuntimeStatus> PylonCamera::create(const Request& request) {
    Request effective = request;
    if(effective.runtime_root.empty()
       && !effective.serial_number.has_value()
       && !effective.requested_size.has_value()
       && effective.exposure_us == 5500
       && effective.frame_rate == -1)
    {
        effective = request_from_settings();
    } else {
        if(effective.runtime_root.empty() && GlobalSettings::has_value("basler_runtime_root")) {
            effective.runtime_root = READ_SETTING(basler_runtime_root, file::Path);
        }
        if(!effective.serial_number && GlobalSettings::has_value("cam_serial_number")) {
            const auto serial = READ_SETTING(cam_serial_number, std::string);
            if(!serial.empty()) {
                effective.serial_number = serial;
            }
        }
        if(!effective.requested_size && GlobalSettings::has_value("cam_resolution")) {
            effective.requested_size = READ_SETTING(cam_resolution, Size2);
        }
        if(effective.exposure_us == 5500 && GlobalSettings::has_value("cam_limit_exposure")) {
            effective.exposure_us = READ_SETTING(cam_limit_exposure, int);
        }
        if(effective.frame_rate == -1 && GlobalSettings::has_value("cam_framerate")) {
            effective.frame_rate = READ_SETTING(cam_framerate, int);
        }
    }

    auto backend = BaslerRuntimeLoader::create_backend(effective);
    if(!backend) {
        return std::unexpected(backend.error());
    }

    auto camera = PylonCamera{std::move(*backend)};
    if(auto opened = camera._backend->open(effective); !opened) {
        return std::unexpected(opened.error());
    }
    return std::expected<PylonCamera, BaslerRuntimeStatus>{std::move(camera)};
}

bool PylonCamera::open() const {
    return _backend && _backend->is_open();
}

void PylonCamera::close() {
    if(_backend) {
        _backend->close();
    }
}

bool PylonCamera::next(Image& image) {
    if(!_backend) {
        return false;
    }

    try {
        auto grabbed = _backend->grab(image);
        if(!grabbed) {
            FormatWarning("Basler grab failed: ", grabbed.error().user_message, " / ", grabbed.error().diagnostic);
            return false;
        }
        return true;
    } catch(const std::exception& ex) {
        FormatWarning("Basler grab threw: ", ex.what());
        return false;
    }
}

Size2 PylonCamera::size() const {
    return _backend ? _backend->size() : Size2();
}

ImageMode PylonCamera::colors() const {
    return _backend ? _backend->colors() : ImageMode::GRAY;
}

std::string PylonCamera::toStr() const {
    if(_backend) {
        return "PylonCamera<" + _backend->camera_name() + ">";
    }
    return "PylonCamera<uninitialized>";
}

}

#endif
