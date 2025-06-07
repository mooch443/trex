#include "AutomaticMatches.h"
#include <tracking/Tracker.h>

namespace track::AutoAssign {

std::shared_mutex _mutex;
std::vector<RangesForID> _automatically_assigned_ranges;

void clear_automatic_ranges() {
    std::unique_lock guard{_mutex};
    _automatically_assigned_ranges.clear();
}

void set_automatic_ranges(decltype(_automatically_assigned_ranges)&& tmp_ranges)
{
    std::unique_lock guard{_mutex};
    _automatically_assigned_ranges = std::move(tmp_ranges);
}

bool have_assignments() {
    std::unique_lock guard{_mutex};
    return not _automatically_assigned_ranges.empty();
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
    
    std::shared_lock guard{_mutex};
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
    std::unique_lock guard{_mutex};
    
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
                Print("Removing the first ", difference, " items from ", rit->bids.size(), "(",search_range," vs. ",range,")");
                rit->bids.erase(rit->bids.begin(), rit->bids.begin() + difference.get());
                Print("\t=> ", rit->bids.size());
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
                Print("Splitting ", rit->bids.size(), "(",search_range," vs. ",range,")");
                auto difference = search_range.start() - range.start;
                auto difference_end = range.end - search_range.end();
                Print("Keeping the first ", difference, " items and the last ", difference_end, " items");
                Print("Middle: ", std::vector<pv::bid>(rit->bids.begin() + difference.get(), rit->bids.end() - difference_end.get()));
                auto new_range = Range<Frame_t>{
                    range.end - difference_end + 1_f,
                    range.end
                };
                auto new_bids = std::vector<pv::bid>(rit->bids.end() - difference_end.get(), rit->bids.end());

                rit->bids.erase(rit->bids.begin() + difference.get(), rit->bids.end());
                range = Range<Frame_t>{range.start, search_range.start() - 1_f};
                Print("\t=> ", rit->bids.size(), " ",range);

                Print("Adding ", new_bids.size(), " ",new_range,".");
                it->ranges.insert(rit, RangesForID::AutomaticRange{new_range, std::move(new_bids)});

            } else {
                /// the range starts within the range and continues
                /// after the range
                /// => we need to shorten this range
                auto difference = range.end - search_range.start() + 1_f;
                Print("Removing the last ", difference, " items from ", rit->bids.size(), "(",search_range," vs. ",range,")");
                rit->bids.erase(rit->bids.end() - difference.get(), rit->bids.end());
                range = Range<Frame_t>{range.start, search_range.start() - 1_f};
                Print("\t=> ", rit->bids.size(), " ", range);
            }
        }
        /*std::erase_if(it->ranges, [&](auto &assign){
            return assign.range == range;
        });*/
        
        ++rit;
    }
}

void write(cmn::DataFormat& ref) {
    /// Check if there are no automatic assignments; if so, write 0 and exit.
    if(not have_assignments()) {
        ref.write<uint64_t>(0);
        return;
    }
    
    /// Acquire a shared lock to safely read the automatic assignments.
    std::shared_lock guard{_mutex};
    /// Write the total number of automatic assignments.
    ref.write<uint64_t>(_automatically_assigned_ranges.size());
    
    for(auto &[id, ranges] : _automatically_assigned_ranges) {
        assert(id.valid());
        /// Write the valid fish identifier as a 32-bit unsigned integer.
        ref.write<uint32_t>(id.get());
        /// Write the number of ranges associated with this fish.
        ref.write<uint64_t>(ranges.size());
        
        for(uint64_t i = 0; i < ranges.size(); ++i) {
            // If the range's start or end is not valid, write default values and skip this range.
            if(not ranges[i].range.start.valid()
               || not ranges[i].range.end.valid())
            {
                ref.write<uint32_t>(0);
                ref.write<uint32_t>(0);
                ref.write<uint64_t>(0);
                continue;
            }
            
            /// Write the starting frame of the range.
            ref.write<uint32_t>(ranges[i].range.start.get());
            /// Write the ending frame of the range.
            ref.write<uint32_t>(ranges[i].range.end.get());
            
            /// Write the number of bid IDs. The count must equal the length of the range plus one.
            ref.write<uint64_t>(ranges[i].bids.size());
            assert(ranges[i].bids.size() == ranges[i].range.length().get() + 1);
            
            /// Iterate over each bid and write its identifier.
            for(auto& bdx : ranges[i].bids) {
                assert(bdx.valid());
                ref.write<uint32_t>(bdx._id);
            }
        }
    }
}

void read(cmn::DataFormat& ref) {
    uint64_t N;
    /// Read the total number of automatic assignments from the binary stream.
    ref.read<uint64_t>(N);
    
    /// If there are no assignments, exit early.
    if(N == 0)
        return;
    
    using namespace AutoAssign;
    
    /// Temporary container to accumulate the read assignments.
    std::vector<RangesForID> ranges_for_id(N);
    for(auto& [id, ranges] : ranges_for_id) {
        /// Read the fish identifier.
        ref.read<uint32_t>(id._identity);
        assert(id.valid());
        
        uint64_t Nranges;
        /// Read the number of ranges associated with the current fish.
        ref.read<uint64_t>(Nranges);
        ranges.resize(Nranges);
        
        for(auto& range : ranges) {
            uint32_t start, end;
            /// Read the starting and ending frames for the range.
            ref.read<uint32_t>(start);
            ref.read<uint32_t>(end);
            
            /// Construct the frame range from the read start and end values.
            range.range = Range<Frame_t>{
                Frame_t(start),
                Frame_t(end)
            };
            
            uint64_t n;
            /// Read the number of bid IDs, which should equal (end - start + 1).
            ref.read<uint64_t>(n);
            assert(n == end - start + 1);
            
            /// Read each bid ID and store them in a vector.
            range.bids.resize(n);
            for (auto &bid : range.bids) {
                uint32_t bdx;
                ref.read<uint32_t>(bdx);
                bid = pv::bid(bdx);
            }
        }
    }
    
    /// Update the global automatic assignments with the newly read data.
    set_automatic_ranges(std::move(ranges_for_id));
}

}
