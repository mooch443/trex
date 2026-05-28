#ifndef _PYLON_CAMERA_H
#define _PYLON_CAMERA_H

#include "Camera.h"

#if WITH_PYLON
#include "BaslerRuntimeLoader.h"

namespace fg {

class PylonCamera final : public Camera {
    std::unique_ptr<BaslerBackend> _backend;
    mutable std::recursive_mutex _mutex;

    explicit PylonCamera(std::unique_ptr<BaslerBackend>&& backend);

public:
    using Request = BaslerCameraRequest;

    PylonCamera(const PylonCamera&) = delete;
    PylonCamera& operator=(const PylonCamera&) = delete;
    PylonCamera(PylonCamera&&) noexcept;
    PylonCamera& operator=(PylonCamera&&) noexcept;
    ~PylonCamera() override;

    static std::expected<PylonCamera, BaslerRuntimeStatus> create(const Request& request = {});

    [[nodiscard]] bool open() const override;
    void close() override;
    bool next(cmn::Image& image) override;
    [[nodiscard]] cmn::Size2 size() const override;
    [[nodiscard]] cmn::ImageMode colors() const override;

    [[nodiscard]] std::string toStr() const override;
};

}

#endif
#endif
