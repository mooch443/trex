#pragma once

#include <python/Detection.h>
#include <python/ModuleProxy.h>

namespace track {

struct TREX_EXPORT Yolo7InstanceSegmentation {
    Yolo7InstanceSegmentation() = delete;

    static void reinit(ModuleProxy& proxy);
    static void init();
    static void receive(std::vector<Vec2> offsets, SegmentationData& data, Vec2 scale_factor, std::vector<float>& masks, const std::vector<float>& vector, const std::vector<int>& meta);
    static tl::expected<SegmentationData, const char*> apply(TileImage&& tiled);
};

} // namespace track
