#pragma once

#include <commons.pc.h>
#include <core/DetectionTypes.h>
#include <core/TaskPipeline.h>
#include <core/TileImage.h>
#include <misc/Image.h>
#include <python/BackendRegistry.h>
#include <python/PipelineRegistry.h>

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
    Detection() { init(); }

    static void init();
    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void deinit();
    static bool is_initializing();
    static double fps();
    static void set_background(const cmn::Image::Ptr& image);

    static PipelineManager<TileImage>& manager();

private:
    static void apply(std::vector<TileImage>&& tiled);
};

} // namespace track
