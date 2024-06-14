#pragma once

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/bid.h>
#include <gui/Transform.h>
#include <tracker/misc/default_config.h>
#include <misc/frame_t.h>
#include <misc/ranges.h>

namespace track {
class Individual;

namespace image {

std::tuple<cmn::Image::Ptr, cmn::Vec2>
calculate_normalized_image(const cmn::gui::Transform &midline_transform,
                           const pv::BlobWeakPtr& blob,
                           float midline_length,
                           const cmn::Size2 &output_size,
                           bool use_legacy,
                           const cmn::Image* background);

std::tuple<cmn::Image::Ptr, cmn::Vec2>
calculate_normalized_diff_image(const cmn::gui::Transform &midline_transform,
                                const pv::BlobWeakPtr& blob,
                                float midline_length,
                                const cmn::Size2 &output_size,
                                bool use_legacy,
                                const cmn::Image* background);

std::tuple<cmn::Image::Ptr, cmn::Vec2>
calculate_diff_image(pv::BlobWeakPtr blob,
                     const cmn::Size2& output_size,
                     const cmn::Image* background);

}

namespace constraints {

struct FilterCache {
    float median_midline_length_px{-1};
    float median_number_outline_pts{-1};
    float midline_length_px_std{-1}, outline_pts_std{-1};
    float median_angle_diff{-1};
    
    bool empty() const { return median_midline_length_px == -1; }
    bool has_std() const { return midline_length_px_std != -1; }
    
    std::string toStr() const;
    static std::string class_name() {
        return "FilterCache";
    }
    
    static void clear();
};

std::tuple<cmn::Image::Ptr, cmn::Vec2> diff_image(const default_config::individual_image_normalization_t::Class &normalize,
                                         pv::BlobWeakPtr blob,
                                         const cmn::gui::Transform& midline_transform,
                                         float median_midline_length_px,
                                         const cmn::Size2& output_shape,
                                         const cmn::Image* background);

std::shared_ptr<FilterCache> local_midline_length(const Individual *fish,
                                                  const cmn::Range<cmn::Frame_t>& segment,
                                                  const bool calculate_std = false);

std::shared_ptr<FilterCache>
local_midline_length(const Individual *fish,
                     cmn::Frame_t frame,
                     const bool calculate_std = false);

}


}
