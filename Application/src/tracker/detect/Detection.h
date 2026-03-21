#pragma once

#include <commons.pc.h>
#include <core/DetectionTypes.h>
#include <core/TaskPipeline.h>
#include <core/TileImage.h>

namespace track::detect {

struct TREX_EXPORT BackendHooks {
    std::function<void()> init;
    std::function<void()> deinit;
    std::function<bool()> is_initializing;
    std::function<double()> fps;
    std::function<void(std::vector<TileImage>&&)> apply;
};

TREX_EXPORT void register_backend(ObjectDetectionType::Class type, BackendHooks hooks);
TREX_EXPORT void unregister_backend(ObjectDetectionType::Class type);
TREX_EXPORT const BackendHooks* backend(ObjectDetectionType::Class type);

} // namespace track::detect

namespace track {

template<typename T>
concept MultiObjectDetection = requires (std::vector<TileImage> tiles) {
    { T::apply(std::move(tiles)) };
};

template<typename T>
concept SingleObjectDetection = requires (TileImage tiles) {
    { T::apply(std::move(tiles)) } -> std::convertible_to<std::future<SegmentationData>>;
};

template<typename T>
concept ObjectDetection = MultiObjectDetection<T> || SingleObjectDetection<T>;

struct TREX_EXPORT Detection {
    Detection();

    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void deinit();
    static bool is_initializing();
    static double fps();

    static PipelineManager<TileImage, true>& manager();

private:
    static void apply(std::vector<TileImage>&& tiled);
};

} // namespace track
