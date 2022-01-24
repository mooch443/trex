#pragma once

#include <misc/Image.h>
#include <misc/PVBlob.h>

namespace track {
    using namespace cmn;
    
    namespace tags {
        using blob_pixel = pv::BlobPtr;
        
        struct Tag {
            float variance;
            uint32_t blob_id;
            Image::Ptr image;
            long_t frame;
            
            Tag(float v, uint32_t bid, Image::Ptr img, long_t frame = -1)
                : variance(v), blob_id(bid), image(img), frame(frame)
            {}
            
            bool operator>(const Tag& other) const {
                return variance > other.variance;
            }
            
            bool operator<(const Tag& other) const {
                return variance < other.variance;
            }
        };
        
        struct result_t {
            const pv::BlobPtr blob;
            Image::UPtr grey;
            Image::UPtr mask;
        };
        
        std::vector<result_t> prettify_blobs(const std::vector<blob_pixel>& fish, const std::vector<blob_pixel>& noise, const std::vector<blob_pixel>& original, const Image& average);
        Tag is_good_image(const result_t& result, const Image& average);
    }
}
