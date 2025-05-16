#pragma once
#include <commons.pc.h>
#include <misc/DetectionTypes.h>
#include <misc/TileImage.h>
#include <misc/Image.h>

namespace track {

struct TREX_EXPORT PrecomputedDetection {
    PrecomputedDetection(file::PathArray&& filename, Image::Ptr&&, meta_encoding_t::Class);
    static void set_background(Image::Ptr&&, meta_encoding_t::Class);
    
    static detect::ObjectDetectionType_t type() { return detect::ObjectDetectionType::precomputed; }
    static std::future<SegmentationData> apply(TileImage&& tiled);
    static void deinit();
    static double fps();
    
    static PipelineManager<TileImage, true>& manager();
    
private:
    static void apply(std::vector<TileImage>&& tiled);
    friend struct Detection;
    
    struct Data;
    
    static Data& data();
};

}
