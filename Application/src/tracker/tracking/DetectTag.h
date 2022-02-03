#pragma once

#include <misc/Image.h>
#include <misc/PVBlob.h>
#include <misc/ranges.h>

namespace track {
    using namespace cmn;
    
    namespace tags {
        using blob_pixel = pv::BlobPtr;
        
        struct Tag {
            float variance;
            pv::bid blob_id;
            Image::Ptr image;
            Frame_t frame;
            
            Tag(float v, pv::bid bdx, Image::Ptr img, Frame_t frame = {})
                : variance(v), blob_id(bdx), image(img), frame(frame)
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
        Tag is_good_image(const result_t& result);
    }
}
