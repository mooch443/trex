#pragma once

#include <python/Detection.h>
#include <python/ModuleProxy.h>

namespace track {

struct Yolo7ObjectDetection {
    Yolo7ObjectDetection() = delete;
    
    static void reinit(ModuleProxy& proxy);
    
    static void init();
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector);
    
    static void apply(std::vector<TileImage>&& tiles);
};

} // namespace track
