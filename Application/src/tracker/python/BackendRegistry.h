#pragma once

#include <commons.pc.h>
#include <core/DetectionTypes.h>
#include <core/TileImage.h>
#include <misc/Image.h>

extern "C" {
namespace track::detect {

struct TREX_EXPORT BackendHooks {
    std::function<void()> init;
    std::function<void()> deinit;
    std::function<bool()> is_initializing;
    std::function<double()> fps;
    std::function<void(std::vector<TileImage>&&)> apply;
    std::function<void(const cmn::Image::Ptr&)> set_background;
};

TREX_EXPORT void register_backend(ObjectDetectionType::Class type, BackendHooks hooks);
TREX_EXPORT void unregister_backend(ObjectDetectionType::Class type);
TREX_EXPORT const BackendHooks* backend(ObjectDetectionType::Class type);
TREX_EXPORT const BackendHooks* ensure_backend(ObjectDetectionType::Class type);

} // namespace track::detect
}
