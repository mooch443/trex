#pragma once

#include <commons.pc.h>
#include <tracking/MotionRecord.h>
#include <tracking/PairingGraph.h>

namespace track {

using prob_t = track::Match::prob_t;

struct IndividualCache {
    //Idx_t _idx;
    //const MotionRecord* h;
    //bool last_frame_manual;
    bool valid;
    bool individual_empty;
    bool valid_frame;
    /*enum class Flag {
        valid,
        individual_empty,
        valid_frame
    };

    std::bitset<3> flags{0};
    void set_flag(Flag flag, bool value) {
        flags.set(static_cast<size_t>(flag), value);
    }
    bool get_flag(Flag flag) const {
        return flags.test(static_cast<size_t>(flag));
    }*/

    //float tdelta;
    int current_category;
    //bool consistent_categories;
    //float track_max_speed_px;
    
    //Match::prob_t speed;
    
    float local_tdelta;
    
    float time_probability;//, position_probability;
    Frame_t previous_frame;
    
    Vec2 last_seen_px;
    Vec2 estimated_px;
    /*bool operator<(const IndividualCache& other) const {
        return _idx < other._idx;
    }
    bool operator==(Idx_t idx) const {
        return _idx == idx;
    }*/
};

}
