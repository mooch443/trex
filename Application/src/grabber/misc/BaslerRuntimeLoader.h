#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/Path.h>

namespace fg {

enum class BaslerRuntimeState {
    not_compiled,
    compiled_but_not_requested,
    runtime_missing,
    runtime_found,
    runtime_incompatible,
    symbol_resolution_failed,
    transport_unavailable,
    device_enumeration_failed,
    ready
};

struct BaslerRuntimeStatus {
    BaslerRuntimeState state{BaslerRuntimeState::compiled_but_not_requested};
    std::string code;
    std::string user_message;
    std::string diagnostic;
    std::vector<std::string> candidate_roots;
    std::vector<std::string> attempted_libraries;
    std::string runtime_version;
};

struct BaslerCameraRequest {
    cmn::file::Path runtime_root;
    std::optional<std::string> serial_number;
    std::optional<cmn::Size2> requested_size;
    int exposure_us{5500};
    int frame_rate{-1};
};

class BaslerBackend {
public:
    virtual ~BaslerBackend() = default;
    virtual std::expected<void, BaslerRuntimeStatus> open(const BaslerCameraRequest& request) = 0;
    virtual bool is_open() const = 0;
    virtual void close() = 0;
    virtual std::expected<void, BaslerRuntimeStatus> grab(cmn::Image& image) = 0;
    virtual cmn::Size2 size() const = 0;
    virtual cmn::ImageMode colors() const = 0;
    virtual std::string camera_name() const = 0;
};

class BaslerRuntimeLoader {
public:
    static BaslerRuntimeStatus probe();
    static std::expected<std::unique_ptr<BaslerBackend>, BaslerRuntimeStatus> create_backend(const BaslerCameraRequest& request);
};

}
