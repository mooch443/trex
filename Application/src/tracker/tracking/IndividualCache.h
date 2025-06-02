#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>

namespace track {

using MaybeLabel = cmn::TrivialOptional<uint16_t>;

static_assert(sizeof(uint8_t) == sizeof(bool), "Why would this be different? Which language is this?");

struct IndividualCache {
    bool valid_frame;
    uint8_t valid_frame_streak;
    MaybeLabel current_category;
    
    cmn::Frame_t previous_frame;
    
    float local_tdelta;
    float time_probability;
    
    cmn::Vec2 last_seen_px;
    cmn::Vec2 estimated_px;
};

}
