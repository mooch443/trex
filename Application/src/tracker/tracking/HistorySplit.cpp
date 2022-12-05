#include "HistorySplit.h"
#include <tracking/SplitBlob.h>
#include <tracking/Tracker.h>
#include <tracking/BlobReceiver.h>

namespace track {

Settings::manual_splits_t::mapped_type HistorySplit::apply_manual_matches(PPFrame& frame) {
    auto manual_splits = FAST_SETTING(manual_splits);
    auto manual_splits_frame =
        (manual_splits.empty() || manual_splits.count(frame.index()) == 0)
            ? decltype(manual_splits)::mapped_type()
            : manual_splits.at(frame.index());
    
    PPFrame::Log("manual_splits = ", manual_splits);
    
    if(!manual_splits_frame.empty()) {
        for(auto bdx : manual_splits_frame) {
            if(!bdx.valid())
                continue;
            
            auto it = frame.blob_mappings.find(bdx);
            if(it == frame.blob_mappings.end()) {
                frame.blob_mappings[bdx] = { };
            } else{
                //it->second.insert(Idx_t());
            }
            
            PPFrame::Log("\t\tManually splitting ", (uint32_t)bdx);
            auto ptr = frame.erase_anywhere(bdx);
            if(ptr) {
                big_blobs.insert(ptr);
                
                expect[bdx].number = 2;
                expect[bdx].allow_less_than = false;
                
                already_walked.insert(bdx);
            }
        }
        
    } else
        PPFrame::Log("\t\tNo manual splits for frame ", frame.index());
    
    PPFrame::Log("fish_mappings ", frame.fish_mappings);
    PPFrame::Log("blob_mappings ", frame.blob_mappings);
    PPFrame::Log("Paired ", frame.paired);
    
    return manual_splits_frame;
}

HistorySplit::HistorySplit(PPFrame &frame, const set_of_individuals_t& active, GenericThreadPool* pool)
{
    //! Finalize the cache and this frame:
    frame.init_cache(frame, active, pool);
    
    PPFrame::Log("");
    PPFrame::Log("------------------------");
    PPFrame::Log("HISTORY MATCHING for frame ", frame.index(), ": ", active);
    
    apply_manual_matches(frame);
    
    /**
     * Now we have found all the mappings from fish->blob and vice-a-versa,
     * lets do the actual splitting (if enabled).
     * -------------------------------------------------------------------------------------
     */

    if(!FAST_SETTING(track_do_history_split)) {
        frame.finalize();
        return;
    }
    
    for(auto && [bdx, set] : frame.blob_mappings) {
        if(already_walked.contains(bdx)) {
            PPFrame::Log("\tblob ", bdx," already walked");
            continue;
        }
        PPFrame::Log("\tblob ", bdx," has ", set.size()," fish mapped to it");
        
        if(set.size() <= 1)
            continue;
        PPFrame::Log("\tFinding clique of this blob:");
        
        UnorderedVectorSet<Idx_t> clique;
        UnorderedVectorSet<pv::bid> others;
        std::queue<pv::bid> q;
        q.push(bdx);
        
        while(!q.empty()) {
            auto current = q.front();
            q.pop();
            
            for(auto fdx: frame.blob_mappings.at(current)) {
                // ignore manually forced splits
                if(!fdx.valid())
                    continue;
                
                for(auto &b : frame.fish_mappings.at(fdx)) {
                    if(!others.contains(b)) {
                        q.push(b);
                        others.insert(b);
                        already_walked.insert(b);
                    }
                }
                
                clique.insert(fdx);
            }
        }
        
        assert(bdx.valid());
        //frame.clique_for_blob[bdx] = clique;
        //frame.clique_second_order[bdx] = others;
    
        PPFrame::Log("\t\t", clique, " ", others);
        
        if(clique.size() <= others.size())
            continue;
        
        using namespace Match;
        std::unordered_map<pv::bid, std::pair<Idx_t, Match::prob_t>> assign_blob; // blob: individual
        std::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> all_probs_per_fish;
        std::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> probs_per_fish;
        
        PPFrame::Log("\t\tMismatch between blobs and number of fish assigned to them.");
        if(clique.size() > others.size() + 1)
            PPFrame::Log("\t\tSizes: ", clique.size()," != ",others.size());
        
        bool allow_less_than = false;
        
        auto check_combinations =
            [&assign_blob](Idx_t c, decltype(probs_per_fish)::mapped_type& combinations, std::queue<Idx_t>& q)
          -> bool
        {
            if(combinations.empty())
                return false; // nothing to do
            
            auto b = std::get<1>(*combinations.begin());
            
            if(assign_blob.count(b) == 0) {
                // great! this blob has not been assigned at all (yet)
                // so just assign it to this fish
                assign_blob[b] = {c, std::get<0>(*combinations.begin())};
                PPFrame::Log("\t\t",b,"(",c,"): ", std::get<0>(*combinations.begin()));
                return true;
                
            } else if(assign_blob[b].first != c) {
                // this blob has been assigned to a different fish!
                // check for validity (which one is closer)
                if(assign_blob[b].second <= std::get<0>(*combinations.begin())) {
                    PPFrame::Log("\t\tBlob ", b," is already assigned to individual ", assign_blob[b], " (", c,")...");
                } else {
                    auto oid = assign_blob[b].first;
                    
                    PPFrame::Log("\t\tBlob ", b," is already assigned to ", assign_blob[b],", but fish ", c," is closer (need to check combinations of fish ", oid," again)");
                    PPFrame::Log("\t\t", b,"(", c,"): ", std::get<0>(*combinations.begin()));
                    
                    assign_blob[b] = {c, std::get<0>(*combinations.begin())};
                    q.push(oid);
                    
                    return true;
                }
            }
            
            combinations.erase(combinations.begin());
            return false;
        };
        
        // 1. assign best matches (remove if better one is found)
        // 2. assign 2. best matches... until nothing is free
        std::queue<Idx_t> checks;
        for(auto c : clique) {
            decltype(probs_per_fish)::mapped_type combinations;
            for(auto && [bdx, d] : frame.paired.at(c)) {
                combinations.insert({Match::prob_t(d), bdx});
            }
            
            probs_per_fish[c] = combinations;
            all_probs_per_fish[c] = combinations;
            
            checks.push(c);
        }
        
         while(!checks.empty()) {
            auto c = checks.front();
            checks.pop();
            
            auto &combinations = all_probs_per_fish.at(c);
            if(!combinations.empty() && !check_combinations(c, combinations, checks))
                checks.push(c);
        }
        
        //UnorderedVectorSet<pv::bid> to_delete;
        //std::map<pv::bid, std::vector<Vec2>> centers;
        size_t counter = 0;
        PPFrame::Log("Final assign blob:", assign_blob);

        for(auto && [fdx, set] : all_probs_per_fish) {
            PPFrame::Log("Combinations ", fdx,": ", set);
            if(!set.empty())
                continue;

            auto last_pos = frame.last_positions.at(fdx);
            
            ++counter;
            PPFrame::Log("No more alternatives for ", fdx);
            if(probs_per_fish.at(fdx).empty())
                continue;
            
            for(auto && [d, bdx] : probs_per_fish.at(fdx)) {
                PPFrame::Log("\t", bdx,": ", d);
            }
            
            pv::bid max_id = std::get<1>(*probs_per_fish.at(fdx).begin());
            if(max_id.valid()) {
                //frame.split_blobs.insert(max_id);

                auto ptr = frame.find_bdx(max_id);
                if(ptr) {
                    if(assign_blob.count(max_id)) {
                        ++expect[max_id].number;
                        expect[max_id].centers.push_back(frame.last_positions.at(assign_blob.at(max_id).first) - ptr->bounds().pos());
                        assign_blob.erase(max_id);
                    }

                    ++expect[max_id].number;
                    big_blobs.insert(ptr);
                    expect[max_id].centers.push_back(last_pos - ptr->bounds().pos());
                    PPFrame::Log("Increasing split number in ", *ptr, " to ", expect[max_id]);
                } else
                    PPFrame::Log("Cannot split blob ", max_id, " since it cannot be found.");

                if(allow_less_than)
                    expect[max_id].allow_less_than = allow_less_than;
            }
        }
        
        PPFrame::Log("expect: ", expect);
        if(counter > 1) {
            PPFrame::Log("Lost ", counter," fish (", expect, ")");
        }
    }
    
    for(auto& b : big_blobs)
        frame.erase_regular(b->blob_id());
    
    auto big_filtered = Tracker::split_big(BlobReceiver(frame, BlobReceiver::noise), big_blobs._data, expect, true, nullptr, pool);
    
    if(!big_filtered.empty())
        frame.add_regular(std::move(big_filtered));
    
    for(size_t i=0; i<frame.blobs().size(); ) {
        if(!FAST_SETTING(blob_size_ranges).in_range_of_one(frame.blobs()[i]->recount(-1))) {
            frame.move_to_noise(i);
        } else
            ++i;
    }
    
    frame.finalize();
}


}
