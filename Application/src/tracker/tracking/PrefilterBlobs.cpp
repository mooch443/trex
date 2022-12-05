#include "PrefilterBlobs.h"
#include <tracking/StaticBackground.h>

namespace track {
using namespace cmn;

PrefilterBlobs::PrefilterBlobs(Frame_t index, int threshold, const BlobSizeRange& fish_size, const Background& background)
: frame_index(index), fish_size(fish_size), background(&background), threshold(threshold)
{
    
}

void PrefilterBlobs::commit(const pv::BlobPtr& b) {
    overall_pixels += b->num_pixels();
    ++samples;
    filtered.push_back(b);
}

void PrefilterBlobs::commit(const std::vector<pv::BlobPtr>& v) {
    for(auto &b:v)
        overall_pixels += b->num_pixels();
    samples += v.size();
    filtered.insert(filtered.end(), v.begin(), v.end());
}

void PrefilterBlobs::filter_out(const pv::BlobPtr& b) {
    overall_pixels += b->num_pixels();
    ++samples;
    filtered_out.push_back(b);
}

void PrefilterBlobs::filter_out(const std::vector<pv::BlobPtr>& v) {
    for(auto &b:v)
        overall_pixels += b->num_pixels();
    samples += v.size();
    filtered_out.insert(filtered_out.end(), v.begin(), v.end());
}

}
