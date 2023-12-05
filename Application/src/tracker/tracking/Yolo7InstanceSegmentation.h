#pragma once

#include <tracking/Detection.h>
#include <python/GPURecognition.h>

namespace track {

struct Yolo7InstanceSegmentation {
    Yolo7InstanceSegmentation() = delete;

    static void reinit(track::PythonIntegration::ModuleProxy& proxy);
    static void init();
    static void receive(std::vector<Vec2> offsets, SegmentationData& data, Vec2 scale_factor, std::vector<float>& masks, const std::vector<float>& vector, const std::vector<int>& meta);
    static tl::expected<SegmentationData, const char*> apply(TileImage&& tiled);
};

} // namespace track
