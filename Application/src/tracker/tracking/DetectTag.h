#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/bid.h>
#include <misc/ranges.h>

namespace track {
    
    
    namespace tags {
        using blob_pixel = pv::BlobPtr;
        
        struct Tag {
            float variance;
            pv::bid blob_id;
            cmn::Image::Ptr image;
            cmn::Frame_t frame;
            
            Tag(float v, pv::bid bdx, cmn::Image::Ptr&& img, cmn::Frame_t frame = {})
                : variance(v), blob_id(bdx), image(std::move(img)), frame(frame)
            {}
            
            bool operator>(const Tag& other) const {
                return variance > other.variance;
            }
            
            bool operator<(const Tag& other) const {
                return variance < other.variance;
            }
        };
        
        struct result_t {
            const pv::bid bdx;
            cmn::Image::Ptr grey;
            cmn::Image::Ptr mask;
        };
        
    std::vector<result_t> prettify_blobs(const std::vector<blob_pixel>& fish, const std::vector<blob_pixel>& noise, const std::vector<blob_pixel>& original, const cmn::Image& average);
        Tag is_good_image(const result_t& result);
    }
}
