#pragma once

#include <commons.pc.h>
#include <misc/PVBlob.h>
#include <misc/frame_t.h>
#include <misc/BlobSizeRange.h>
#include <processing/CPULabeling.h>

namespace cmn {
class Background;

}

namespace track {
struct PrefilterBlobs {
    std::vector<pv::BlobPtr> filtered;
    std::vector<pv::BlobPtr> filtered_out;
    std::vector<pv::BlobPtr> big_blobs;
    
    CPULabeling::ListCache_t cache;
    
    Frame_t frame_index;
    BlobSizeRange fish_size;
    const Background* background;
    int threshold;
    
    size_t overall_pixels = 0;
    size_t samples = 0;
    
    PrefilterBlobs(Frame_t index, int threshold, const BlobSizeRange& fish_size, const Background& background);
    
    void commit(const pv::BlobPtr& b);
    void commit(const std::vector<pv::BlobPtr>& v);
    
    void filter_out(const pv::BlobPtr& b);
    void filter_out(const std::vector<pv::BlobPtr>& v);
};

}
