#pragma once

#include <misc/ranges.h>
#include <misc/frame_t.h>
#include <tracking/Stuffs.h>

namespace track {

struct SegmentInformation : public cmn::FrameRange {
    std::vector<long_t> basic_index;
    std::vector<long_t> posture_index;
    uint32_t error_code = std::numeric_limits<uint32_t>::max();
    
    SegmentInformation(
        const Range<Frame_t>& range = Range<Frame_t>(Frame_t(), Frame_t()),
        Frame_t first_usable = Frame_t())
      : FrameRange(range, first_usable)
    {}
    
    void add_basic_at(Frame_t frame, long_t gdx);
    void add_posture_at(std::unique_ptr<PostureStuff>&& stuff, Individual* fish); //long_t gdx);
    //void remove_frame(long_t);
    
    long_t basic_stuff(Frame_t frame) const;
    long_t posture_stuff(Frame_t frame) const;
    
    constexpr bool overlaps(const SegmentInformation& v) const {
        return contains(v.start()) || contains(v.end())
            || v.contains(start()) || v.contains(end())
            || v.start() == end() || start() == v.end();
    }
    
    constexpr bool operator<(const SegmentInformation& other) const {
        return range < other.range;
    }
    
    constexpr bool operator<(Frame_t frame) const {
        return range.start < frame;
    }
};

}
