#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <core/TileImage.h>
#include <python/ModuleProxy.h>

namespace track {

struct Detection;
void register_sam3_backend();

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

    
private:
    static void apply(std::vector<TileImage>&& tiled);
    friend struct Detection;
    friend void register_sam3_backend();
    
    struct Data;
    
    static Data& data();
};


}
