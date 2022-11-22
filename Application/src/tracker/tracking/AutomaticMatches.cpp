#include "AutomaticMatches.h"
#include <tracking/Tracker.h>

namespace track {

namespace AutoAssign {

std::vector<RangesForID> _automatically_assigned_ranges;

void clear_automatic_ranges() {
    _automatically_assigned_ranges.clear();
}

void set_automatic_ranges(decltype(_automatically_assigned_ranges)&& tmp_ranges)
{
    _automatically_assigned_ranges = std::move(tmp_ranges);
}

void add_assigned_range(std::vector<RangesForID>& assigned, Idx_t fdx, const Range<Frame_t>& range, std::vector<pv::bid>&& bids) {
    auto it = std::find(assigned.begin(), assigned.end(), fdx);
    if(it == assigned.end()) {
        assigned.push_back(RangesForID{ fdx, { RangesForID::AutomaticRange{range, std::move(bids)} } });
        
    } else {
        it->ranges.push_back(RangesForID::AutomaticRange{ range, std::move(bids) });
    }
}

std::map<Idx_t, pv::bid> automatically_assigned(Frame_t frame) {
    //LockGuard guard;
    std::map<Idx_t, pv::bid> blob_for_fish;
    
    for(auto && [fdx, bff] : _automatically_assigned_ranges) {
        blob_for_fish[fdx] = pv::bid::invalid;
        
        for(auto & assign : bff) {
            if(assign.range.contains(frame)) {
                assert(frame >= assign.range.start && assign.range.end >= frame);
                blob_for_fish[fdx] = assign.bids.at(sign_cast<size_t>((frame - assign.range.start).get()));
                break;
            }
        }
    }
    
    return blob_for_fish;
}

void delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range) {
    Tracker::LockGuard guard(w_t{}, "delete_automatic_assignments");
    
    auto it = std::find(_automatically_assigned_ranges.begin(), _automatically_assigned_ranges.end(), fish_id);
    if(it == _automatically_assigned_ranges.end()) {
        FormatExcept("Cannot find fish ",fish_id," in automatic assignments");
        return;
    }
    
    std::set<Range<Frame_t>> ranges_to_remove;
    for(auto && [range, blob_ids] : it->ranges) {
        if(frame_range.overlaps(range)) {
            ranges_to_remove.insert(range);
        }
    }
    for(auto range : ranges_to_remove) {
        std::erase_if(it->ranges, [&](auto &assign){
            return assign.range == range;
        });
    }
}


}

}
