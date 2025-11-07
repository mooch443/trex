#pragma once
#include <commons.pc.h>
#include <misc/DetectionTypes.h>
#include <misc/TileImage.h>
#include <misc/Image.h>

namespace track {

struct TREX_EXPORT NoDetection {
public:
    static detect::ObjectDetectionType_t type() { return detect::ObjectDetectionType::none; }
    
private:
    friend struct Detection;
    
    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void apply(std::vector<TileImage>&& tiled);
};

}
