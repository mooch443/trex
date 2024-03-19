#pragma once

#include <misc/frame_t.h>
#include <misc/PVBlob.h>
#include <tracking/MotionRecord.h>
#include <tracking/Outline.h>

namespace track {

//! Stuff that belongs together and is definitely
//! present in every frame
struct BasicStuff {
    Frame_t frame;
    
    MotionRecord centroid;
    //MotionRecord* weighted_centroid;
    uint64_t thresholded_size;
    pv::CompressedBlob blob;
};

//! Stuff that is only present if postures are
//! calculated and present in the given frame.
//! (There are no frame_segments available for pre-sorting requests)
struct PostureStuff {
    static constexpr float infinity = cmn::infinity<float>();
    Frame_t frame;
    
    MotionRecord* head{nullptr};
    MotionRecord* centroid_posture{nullptr};
    Midline::Ptr cached_pp_midline;
    MinimalOutline::Ptr outline;
    float posture_original_angle{infinity};
    float midline_angle{infinity}, midline_length{infinity};
    //!TODO: consider adding processed midline_angle and length
    //!TODO: need to fix copy operations for head etc
    PostureStuff() noexcept = default;
    PostureStuff(PostureStuff&&) = delete;
    PostureStuff(const PostureStuff& other) : frame(other.frame) {
        if (other.head) head = new MotionRecord(*other.head);
        if (other.centroid_posture) centroid_posture = new MotionRecord(*other.centroid_posture);
        if (other.cached_pp_midline) cached_pp_midline = other.cached_pp_midline;
        if (other.outline) outline = other.outline;
        posture_original_angle = other.posture_original_angle;
        midline_angle = other.midline_angle;
        midline_length = other.midline_length;
    }
    PostureStuff& operator=(const PostureStuff& other) {
        if (this != &other) {
            frame = other.frame;
            if (other.head) head = new MotionRecord(*other.head);
            if (other.centroid_posture) centroid_posture = new MotionRecord(*other.centroid_posture);
            if (other.cached_pp_midline) cached_pp_midline = other.cached_pp_midline;
            if (other.outline) outline = other.outline;
            posture_original_angle = other.posture_original_angle;
            midline_angle = other.midline_angle;
            midline_length = other.midline_length;
        }
        return *this;
    }
    PostureStuff& operator=(PostureStuff&&) = delete;
    ~PostureStuff();
    bool cached() const { return posture_original_angle != infinity; }
};

}
