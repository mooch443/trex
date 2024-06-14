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

std::map<pv::bid, Idx_t> automatically_assigned(Frame_t frame) {
    //LockGuard guard;
    std::map<pv::bid, Idx_t> blob_for_fish;
    
    for(auto && [fdx, bff] : _automatically_assigned_ranges) {
        //blob_for_fish[fdx] = pv::bid::invalid;
        
        for(auto & assign : bff) {
            if(assign.range.contains(frame)) {
                assert(frame >= assign.range.start && assign.range.end >= frame);
                blob_for_fish[assign.bids.at(sign_cast<size_t>((frame - assign.range.start).get()))] = fdx;
                break;
            }
        }
    }
    
    return blob_for_fish;
}

void delete_automatic_assignments(Idx_t fish_id, const FrameRange& search_range) {
    LockGuard guard(w_t{}, "delete_automatic_assignments");
    
    auto it = std::find(_automatically_assigned_ranges.begin(), _automatically_assigned_ranges.end(), fish_id);
    if(it == _automatically_assigned_ranges.end()) {
        FormatExcept("Cannot find fish ",fish_id," in automatic assignments");
        return;
    }
    
    /*std::set<Range<Frame_t>> ranges_to_remove;
    for(auto && [range, blob_ids] : it->ranges) {
        if(frame_range.overlaps(range)) {
            ranges_to_remove.insert(range);
        }
    }*/
    
    for(auto rit = it->ranges.begin(); rit != it->ranges.end();) {
        auto &range = (*rit).range;
        if(not search_range.overlaps(range)) {
            ++rit;
            continue;
        }
        
        if(search_range.start() <= range.start) {
            /// the search range begins before or at the current range
            if(search_range.end() < range.end) {
                /// need to remove the start of this range
                auto difference = range.start - search_range.start();
                print("Removing the first ", difference, " items from ", rit->bids.size(), "(",search_range," vs. ",range,")");
                rit->bids.erase(rit->bids.begin(), rit->bids.begin() + difference.get());
                print("\t=> ", rit->bids.size());
                range = Range<Frame_t>{search_range.end() + 1_f, range.end};
                
            } else {
                /// need to remove the entire range
                rit = it->ranges.erase(rit);
                continue;
            }
            
        } else if(range.end >= search_range.start()
                  && range.start <= search_range.start())
        {
            /// the search begins within the current segment
            if(search_range.end() < range.end) {
                /// the range is completely contained
                /// => we need to split our range?
                print("Splitting ", rit->bids.size(), "(",search_range," vs. ",range,")");
                auto difference = search_range.start() - range.start;
                auto difference_end = range.end - search_range.end();
                print("Keeping the first ", difference, " items and the last ", difference_end, " items");
                print("Middle: ", std::vector<pv::bid>(rit->bids.begin() + difference.get(), rit->bids.end() - difference_end.get()));
                auto new_range = Range<Frame_t>{
                    range.end - difference_end + 1_f,
                    range.end
                };
                auto new_bids = std::vector<pv::bid>(rit->bids.end() - difference_end.get(), rit->bids.end());

                rit->bids.erase(rit->bids.begin() + difference.get(), rit->bids.end());
                range = Range<Frame_t>{range.start, search_range.start() - 1_f};
                print("\t=> ", rit->bids.size(), " ",range);

                print("Adding ", new_bids.size(), " ",new_range,".");
                it->ranges.insert(rit, RangesForID::AutomaticRange{new_range, std::move(new_bids)});

            } else {
                /// the range starts within the range and continues
                /// after the range
                /// => we need to shorten this range
                auto difference = range.end - search_range.start() + 1_f;
                print("Removing the last ", difference, " items from ", rit->bids.size(), "(",search_range," vs. ",range,")");
                rit->bids.erase(rit->bids.end() - difference.get(), rit->bids.end());
                range = Range<Frame_t>{range.start, search_range.start() - 1_f};
                print("\t=> ", rit->bids.size(), " ", range);
            }
        }
        /*std::erase_if(it->ranges, [&](auto &assign){
            return assign.range == range;
        });*/
        
        ++rit;
    }
}


}

}
