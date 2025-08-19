#pragma once
#include <commons.pc.h>
#include <misc/TileImage.h>
#include <misc/TaskPipeline.h>
#include <misc/DetectionTypes.h>

namespace track {

using namespace cmn;

template<typename T>
concept MultiObjectDetection = requires (std::vector<TileImage> tiles) {
    { T::apply(std::move(tiles)) };
    //{ T::receive(data, Vec2{}, {}) };
};

template<typename T>
concept SingleObjectDetection = requires (TileImage tiles) {
    { T::apply(std::move(tiles)) } -> std::convertible_to<std::future<SegmentationData>>;
    //{ T::receive(data, Vec2{}, {}) };
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
