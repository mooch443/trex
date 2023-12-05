#pragma once

#include <tracking/Detection.h>
#include <python/GPURecognition.h>

namespace track {

struct Yolo7ObjectDetection {
    Yolo7ObjectDetection() = delete;
    
    static void reinit(track::PythonIntegration::ModuleProxy& proxy);
    
    static void init();
    
    static void receive(SegmentationData& data, Vec2 scale_factor, const std::span<float>& vector);
    
    static void apply(std::vector<TileImage>&& tiles);
};

} // namespace track
