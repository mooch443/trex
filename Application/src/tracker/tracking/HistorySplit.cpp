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
            
            /*auto it = frame.blob_mappings.find(bdx);
            if(it == frame.blob_mappings.end()) {
                frame.blob_mappings[bdx] = { };
            } else{
                //it->second.insert(Idx_t());
            }*/
            
            PPFrame::Log("\t\tManually splitting ", (uint32_t)bdx);
            if(frame.has_bdx(bdx)) {
                big_blobs.insert(bdx);
                
                expect[bdx].number = 2;
                expect[bdx].allow_less_than = false;
                
                already_walked.insert(bdx);
            }
        }
        
    } else
        PPFrame::Log("\t\tNo manual splits for frame ", frame.index());
    
    //PPFrame::Log("fish_mappings ", frame.fish_mappings);
    //PPFrame::Log("blob_mappings ", frame.blob_mappings);
    //PPFrame::Log("Paired ", frame.paired);
    
    return manual_splits_frame;
}

HistorySplit::HistorySplit(PPFrame &frame, PPFrame::NeedGrid need, GenericThreadPool* pool)
{
    PPFrame::Log("FRAME ", frame.index());
    
    //! Finalize the cache and this frame:
    frame.init_cache(pool, need);
    
    apply_manual_matches(frame);
    
    /**
     * Now we have found all the mappings from fish->blob and vice-a-versa,
     * lets do the actual splitting (if enabled).
     * -------------------------------------------------------------------------------------
     */
    if(!FAST_SETTING(track_do_history_split)) {
        //! history split is disabled, so we can skip this step
        frame.finalize();
        return;
    }
    
    for(auto && [bdx, set] : frame.blob_mappings) {
        PPFrame::Log("\tblob ", bdx," has ", set.size()," fish mapped to it");
        
        if(set.size() <= 1)
            continue;
        
        if(already_walked.contains(bdx)) {
            PPFrame::Log("\tblob ", bdx," already walked");
            continue;
        }
        
        PPFrame::Log("\tFinding clique of this blob:");
        
        UnorderedVectorSet<Idx_t> available_fdx;
        UnorderedVectorSet<pv::bid> available_bdx;
        std::queue<pv::bid> q;
        q.push(bdx);
        
        while(!q.empty()) {
            auto current = q.front();
            q.pop();
            
            for(auto fdx: frame.blob_mappings.at(current)) {
                // ignore manually forced splits
                if(!fdx.valid())
                    continue;
                
                for(auto &[bdx, d] : frame.paired.at(fdx)) {
                    if(!available_bdx.contains(bdx)) {
                        q.push(bdx);
                        available_bdx.insert(bdx);
                        already_walked.insert(bdx);
                    }
                }
                
                available_fdx.insert(fdx);
            }
        }
        
        assert(bdx.valid());
        //frame.clique_for_blob[bdx] = clique;
        //frame.clique_second_order[bdx] = others;
    
        PPFrame::Log("\t\t", available_fdx, " ", available_bdx);
        
        //! do we have more individuals than blobs?
        //! otherwise we can quit here, since no splitting required.
        if(available_fdx.size() <= available_bdx.size())
            continue;
        
        using namespace Match;
        std::unordered_map<pv::bid, std::pair<Idx_t, Match::prob_t>> assign_blob; // blob: individual
        //std::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> all_probs_per_fish;
        std::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> probs_per_fish;
        std::unordered_map<Idx_t, std::tuple<Match::prob_t, pv::bid>> assign_fish;
        
        PPFrame::Log("\t\tMismatch between blobs and number of fish assigned to them.");
        if(available_fdx.size() > available_bdx.size() + 1)
            PPFrame::Log("\t\tSizes: ", available_fdx.size()," != ",available_bdx.size());
        
        bool allow_less_than = false;
        
        //! Check
        auto check_combinations =
            [&assign_blob](Idx_t fdx, decltype(probs_per_fish)::mapped_type& combinations, std::queue<Idx_t>& q)
          -> bool
        {
            if(combinations.empty())
                return false; // nothing to do
            
            auto b = std::get<1>(*combinations.begin());
            
            if(assign_blob.count(b) == 0) {
                // great! this blob has not been assigned at all (yet)
                // so just assign it to this fish
                assign_blob[b] = {fdx, std::get<0>(*combinations.begin())};
                PPFrame::Log("\t\t",b,"(",fdx,"): ", std::get<0>(*combinations.begin()));
                return true;
                
            } else if(assign_blob[b].first != fdx) {
                // this blob has been assigned to a different fish!
                // check for validity (which one is closer)
                if(assign_blob[b].second <= std::get<0>(*combinations.begin())) {
                    PPFrame::Log("\t\tBlob ", b," is already assigned to individual ", assign_blob[b], " (", fdx,")...");
                } else {
                    auto oid = assign_blob[b].first;
                    
                    PPFrame::Log("\t\tBlob ", b," is already assigned to ", assign_blob[b],", but fish ", fdx," is closer (need to check combinations of fish ", oid," again)");
                    PPFrame::Log("\t\t", b,"(", fdx,"): ", std::get<0>(*combinations.begin()));
                    
                    assign_blob[b] = {fdx, std::get<0>(*combinations.begin())};
                    q.push(oid);
                    return true;
                }
            }
            
            combinations.erase(combinations.begin());
            //if(combinations.empty()) assign_fish[fdx] = {-1, pv::bid()};
            return false;
        };
        
        // 1. assign best matches (remove if better one is found)
        // 2. assign 2. best matches... until nothing is free
        std::queue<Idx_t> checks;
        for(auto c : available_fdx) {
            const auto& pairs = frame.paired.at(c);
            if(pairs.empty())
                continue;
            
            decltype(probs_per_fish)::mapped_type combinations;
            for(auto && [bdx, d] : pairs) {
                combinations.insert({Match::prob_t(d), bdx});
            }
            
            //if(combinations.empty())
            //    assign_fish[c] = {-1, pv::bid()};
            //else
            assign_fish[c] = *combinations.begin();
            probs_per_fish[c] = combinations;
            
            checks.push(c);
        }
        
         while(!checks.empty()) {
            auto c = checks.front();
            checks.pop();
            
            auto &combinations = probs_per_fish.at(c);
             
            //! probs_per_fish is modified in check_combinations:
            if(!combinations.empty() && !check_combinations(c, combinations, checks))
                checks.push(c);
        }
        
        //! Counting how many individuals have no matching
        //! alternatives after resolving probabilities.
        //! These are the ones that we need to investigate,
        //! since that means we have fewer objects than we need.
        //! -> split?
        size_t count_unassignable = 0;
        PPFrame::Log("Final assign blob:", assign_blob);

        for(auto && [fdx, set] : probs_per_fish) {
            PPFrame::Log("Combinations ", fdx,": ", set);
            
            //! There are alternatives left for this fish.
            //! We are interested in individuals that have no more
            //! alternatives for assignments. Skip.
            if(!set.empty())
                continue;
            
            ++count_unassignable;
            PPFrame::Log("No more alternatives for ", fdx);
            
#ifndef NDEBUG
            if(frame.paired.at(fdx).empty())
                throw U_EXCEPTION(fdx," has no edges to any objects.");
                //continue;
#endif
            
            pv::bid max_id = std::get<1>(assign_fish.at(fdx));
            if(!max_id.valid())
                continue; // no more blobs assignable
            
            auto ptr = frame.bdx_to_ptr(max_id);
            if(ptr) {
                if(assign_blob.count(max_id)) {
                    ++expect[max_id].number;
                    expect[max_id].centers.push_back(frame.last_positions.at(assign_blob.at(max_id).first) - ptr->bounds().pos());
                    assign_blob.erase(max_id);
                }
                
                auto last_pos = frame.last_positions.at(fdx);

                ++expect[max_id].number;
                expect[max_id].centers.push_back(last_pos - ptr->bounds().pos());
                PPFrame::Log("Increasing split number in ", *ptr, " to ", expect[max_id]);
                
                big_blobs.insert(max_id);
                
            } else
                PPFrame::Log("Cannot split blob ", max_id, " since it cannot be found.");

            if(allow_less_than)
                expect[max_id].allow_less_than = true;
        }
        
        PPFrame::Log("expect: ", expect);
        if(count_unassignable > 1) {
            PPFrame::Log("Lost ", count_unassignable," fish (", expect, ")");
        }
    }
    
    //for(auto& b : big_blobs)
    //    frame.erase_regular(b->blob_id());
    PPFrame::Log("big_blobs = ", big_blobs);
    PPFrame::Log("blob_grid = ", frame.blob_grid().value_where());
    PPFrame::Log("blob_mappings = ", frame.blob_mappings);
    PPFrame::Log("blobs = ",frame.unsafe_access_all_blobs());
    
    std::vector<pv::bid> bids;
    frame.transform_noise([&](const pv::Blob& blob){
        bids.push_back(blob.blob_id());
    });
    PPFrame::Log("noise = ", bids);
    
    auto collection = frame.extract_from_all<PPFrame::VectorHandling::Compress, PPFrame::RemoveHandling::Leave>(big_blobs);
    assert(frame.extract_from_noise(big_blobs).empty());
    //assert(collection.size() == big_blobs.size());
    big_blobs.clear();
    
    PPFrame::Log("&nbsp;Collected ", collection, " from frame.");
    for(auto &b: collection)
        frame._split_pixels += b->num_pixels();
    frame._split_objects += collection.size();
    
    PrefilterBlobs::split_big(
           std::move(collection),
           BlobReceiver(frame, BlobReceiver::noise, FilterReason::SplitFailed),
           BlobReceiver(frame, BlobReceiver::regular),
           expect, true, nullptr, pool);
    
    PPFrame::Log("&nbsp;collection = ", collection);
    
    //for(auto &&b : collection) {
    //    if(b)
    //        frame.add_regular(std::move(b));
    //}
        
    
    //! final filtering step that filters out small blobs
    //! from the split_big TODO: (which might not be possible?)
    frame.move_to_noise_if([size = FAST_SETTING(blob_size_ranges)](const pv::Blob& blob) {
        if(!size.in_range_of_one(blob.recount(-1))) {
            PPFrame::Log("&nbsp;Filtering out ", blob, " not in ", size);
            return true;
        }
        return false;
    });
    
    frame.finalize();
}


}
