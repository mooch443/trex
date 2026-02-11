#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/TileImage.h>
#include <python/ModuleProxy.h>

namespace track {

struct Detection;

struct TREX_EXPORT SAM3 {
    SAM3(cmn::Image::Ptr&& = nullptr);
    static void set_background(cmn::Image::Ptr&&);
    
    //static detect::ObjectDetectionType_t type() { return detect::ObjectDetectionType::background_subtraction; }
    static std::future<SegmentationData> apply(TileImage&& tiled);
    
    static void reinit(track::ModuleProxy& proxy);
    
    static void init();
    static void deinit();
    
    static bool is_initializing();
    static double fps();

    static PipelineManager<TileImage, false>& manager();
    
private:
    static void apply(std::vector<TileImage>&& tiled);
    friend struct Detection;
    
    struct Data;
    
    static Data& data();
};


}
