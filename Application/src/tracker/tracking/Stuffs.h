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
    using ml_t = TrivialOptional<Float2_t, TrivialIllegalValueType::Infinity>;
    
    //static constexpr Float2_t infinity = cmn::infinity<Float2_t>();
    Frame_t frame;
    
    std::unique_ptr<MotionRecord> head;
    std::unique_ptr<MotionRecord> centroid_posture;
    
    Midline::Ptr cached_pp_midline;
    MinimalOutline outline;
    
    ml_t posture_original_angle;
    ml_t midline_angle, midline_length;
    
    //!TODO: consider adding processed midline_angle and length
    //!TODO: need to fix copy operations for head etc
    PostureStuff() noexcept = default;
    PostureStuff(PostureStuff&& other) noexcept
          : frame(other.frame), 
            head(std::move(other.head)), 
            centroid_posture(std::move(other.centroid_posture)),
            cached_pp_midline(std::move(other.cached_pp_midline)), 
            outline(std::move(other.outline)),
            posture_original_angle(other.posture_original_angle), 
            midline_angle(other.midline_angle),
            midline_length(other.midline_length) 
    {
        assert(posture_original_angle.has_value() == midline_angle.has_value()
               && midline_length.has_value() == midline_angle.has_value());
    }

    PostureStuff& operator=(PostureStuff&& other) noexcept {
        if (this != &other) {
            frame = other.frame;
            head = std::move(other.head);
            centroid_posture = std::move(other.centroid_posture);
            cached_pp_midline = std::move(other.cached_pp_midline);
            outline = std::move(other.outline);
            posture_original_angle = other.posture_original_angle;
            midline_angle = other.midline_angle;
            midline_length = other.midline_length;
            
            assert(posture_original_angle.has_value() == midline_angle.has_value()
                   && midline_length.has_value() == midline_angle.has_value());
        }
        return *this;
    }

    /// Perform a deep copy of the object
    PostureStuff clone() const noexcept {
        PostureStuff copy;
        copy.frame = frame;
        if (head) {
            copy.head = std::make_unique<MotionRecord>(*head);
        }
        if (centroid_posture) {
            copy.centroid_posture = std::make_unique<MotionRecord>(*centroid_posture);
        }
        if (cached_pp_midline) {
            copy.cached_pp_midline = std::make_unique<Midline>(*cached_pp_midline);
        }
        if (outline) {
            copy.outline = outline;
        }
        copy.posture_original_angle = posture_original_angle;
        copy.midline_angle = midline_angle;
        copy.midline_length = midline_length;
        return copy;
    }

    ~PostureStuff() = default;
    bool cached() const {
        assert(posture_original_angle.has_value() == midline_angle.has_value()
               && midline_length.has_value() == midline_angle.has_value());
        return posture_original_angle.has_value();
    }
};

}
