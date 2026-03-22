#pragma once

#include <commons.pc.h>
#include <detect/Detection.h>
#include <misc/Image.h>
#include <core/TileImage.h>

namespace track {

struct TREX_EXPORT BackgroundSubtraction {
    BackgroundSubtraction(cmn::Image::Ptr&& = nullptr);
    static void set_background(cmn::Image::Ptr&&);
    
    //static detect::ObjectDetectionType_t type() { return detect::ObjectDetectionType::background_subtraction; }
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
