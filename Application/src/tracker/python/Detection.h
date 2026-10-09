#pragma once

#include <commons.pc.h>
#include <misc/Image.h>

struct TileImage;

namespace cmn {
struct SegmentationData;
template<typename Data>
class PipelineManager;
}

namespace track {

template<typename T>
concept MultiObjectDetection = requires {
    { T::apply(std::declval<std::vector<TileImage>&&>()) };
};

template<typename T>
concept SingleObjectDetection = requires {
    { T::apply(std::declval<TileImage&&>()) } -> std::convertible_to<std::future<cmn::SegmentationData>>;
};

template<typename T>
concept ObjectDetection = MultiObjectDetection<T> || SingleObjectDetection<T>;

struct TREX_EXPORT Detection {
    Detection() { init(); }

    static void init();
    static std::future<cmn::SegmentationData> apply(TileImage&& tiled);
    static void deinit();
    static bool is_initializing();
    static double fps();
    static void set_background(const cmn::Image::Ptr& image);

    static cmn::PipelineManager<TileImage>& manager();

private:
    static void apply(std::vector<TileImage>&& tiled);
};

} // namespace track
