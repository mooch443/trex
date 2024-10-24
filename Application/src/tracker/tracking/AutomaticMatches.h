#pragma once

#include <misc/bid.h>
#include <misc/idx_t.h>
#include <misc/ranges.h>

namespace track {

namespace AutoAssign {

using namespace cmn;

struct RangesForID {
    struct AutomaticRange {
        cmn::Range<Frame_t> range;
        std::vector<pv::bid> bids;
        
        bool operator==(const cmn::Range<Frame_t>& range) const {
            return this->range.start == range.start && this->range.end == range.end;
        }
    };
    
    Idx_t id;
    std::vector<AutomaticRange> ranges;
    
    bool operator==(const Idx_t& idx) const {
        return id == idx;
    }
};

void clear_automatic_ranges();
void set_automatic_ranges(std::vector<RangesForID>&&);
std::map<pv::bid,Idx_t> automatically_assigned(Frame_t frame);
void delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range);
void add_assigned_range(std::vector<RangesForID>& assigned, Idx_t fdx, const Range<Frame_t>& range, std::vector<pv::bid>&& bids);
}

}
