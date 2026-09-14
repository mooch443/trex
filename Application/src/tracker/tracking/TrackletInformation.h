#pragma once

#include <misc/ranges.h>

namespace track {

class Individual;
struct PostureStuff;

struct TrackletInformation : public cmn::FrameRange {
    std::vector<long_t> basic_index;
    std::vector<long_t> posture_index;
    uint32_t error_code = std::numeric_limits<uint32_t>::max();
    
    TrackletInformation(
        const cmn::Range<cmn::Frame_t>& range = cmn::Range<cmn::Frame_t>(cmn::Frame_t(), cmn::Frame_t()),
        cmn::Frame_t first_usable = cmn::Frame_t())
      : FrameRange(range, first_usable)
    {}
    
    void add_basic_at(cmn::Frame_t frame, long_t gdx);
    void add_posture_at(std::unique_ptr<PostureStuff>&& stuff, Individual* fish); //long_t gdx);
    //void remove_frame(long_t);
    
    long_t basic_stuff(cmn::Frame_t frame) const;
    long_t posture_stuff(cmn::Frame_t frame) const;
    
    constexpr bool overlaps(const TrackletInformation& v) const {
        return contains(v.start()) || contains(v.end())
            || v.contains(start()) || v.contains(end())
            || v.start() == end() || start() == v.end();
    }
    
    constexpr bool operator<(const TrackletInformation& other) const {
        return range < other.range;
    }
    
    constexpr bool operator<(cmn::Frame_t frame) const {
        return range.start < frame;
    }
};

}

inline bool operator<(const std::shared_ptr<track::TrackletInformation>& ptr, cmn::Frame_t frame) {
    assert(ptr != nullptr);
    return ptr->start() < frame;
}
