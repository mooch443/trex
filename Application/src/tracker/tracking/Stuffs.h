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
    
    ~PostureStuff();
    bool cached() const { return posture_original_angle != infinity; }
};

}
