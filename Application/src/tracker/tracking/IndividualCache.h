#pragma once

#include <misc/vec2.h>
#include <tracking/PhysicalProperties.h>
#include <tracking/PairingGraph.h>

namespace track {

using prob_t = track::Match::prob_t;

struct IndividualCache {
    Idx_t _idx;
    const PhysicalProperties* h;
    Vec2 last_seen_px;
    Vec2 estimated_px;
    bool last_frame_manual;
    bool valid = false;
    float tdelta;
    float local_tdelta;
    Frame_t previous_frame;
    int current_category;
    bool consistent_categories;
    float cm_per_pixel, track_max_speed;
    
    Match::prob_t speed;
    Match::prob_t time_probability;
    
    bool operator<(const IndividualCache& other) const {
        return _idx < other._idx;
    }
    bool operator==(Idx_t idx) const {
        return _idx == idx;
    }
};

}
