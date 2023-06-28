#pragma once

#include <tracking/Detection.h>
#include <python/GPURecognition.h>

namespace track {

struct Yolo8 {
    Yolo8() = delete;
    
    static void reinit(track::PythonIntegration::ModuleProxy& proxy);
    
    static void init();
    static void deinit();

    static void receive(SegmentationData& data, Vec2 scale_factor, track::detect::Result&& result);
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector,
        const std::span<float>& mask_points, const std::span<uint64_t>& mask_Ns);
    
    static void apply(std::vector<TileImage>&& tiles);
};

} // namespace track
