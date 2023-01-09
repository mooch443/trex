#include "TrackingHelper.h"
#include <tracking/Tracker.h>
#include <tracker/misc/default_config.h>
#include <misc/pretty.h>
#include <tracking/AutomaticMatches.h>
#include <tracking/IndividualManager.h>

namespace track {

struct CachedSettings {
    const bool do_posture{FAST_SETTING(calculate_posture)};
    const bool save_tags{!FAST_SETTING(tags_path).empty()};
    const uint32_t number_fish{(uint32_t)FAST_SETTING(track_max_individuals)};
    const Frame_t approximation_delay_time{Frame_t(max(1, SLOW_SETTING(frame_rate) * 0.25))};
};

inline auto& blob_grid() {
    static grid::ProximityGrid grid{Tracker::average().bounds().size()};
    return grid;
}

void TrackingHelper::blob_assign(pv::bid bdx) {
    std::scoped_lock guard(blob_mutex);
    _blob_assigned.insert(bdx);
}

void TrackingHelper::clear_blob_assigned() noexcept {
    std::scoped_lock guard(blob_mutex);
    _blob_assigned.clear();
}

bool TrackingHelper::blob_assigned(pv::bid blob) const {
    std::shared_lock guard(blob_mutex);
    return _blob_assigned.contains(blob);
}

bool TrackingHelper::fish_assigned(Individual* fish) const {
    std::shared_lock guard(fish_mutex);
    return _fish_assigned.contains(fish);
}

void TrackingHelper::clear_fish_assigned() noexcept {
    std::scoped_lock guard(fish_mutex);
    _fish_assigned.clear();
}

void TrackingHelper::fish_assign(Individual* fish) {
    std::scoped_lock guard(fish_mutex);
    _fish_assigned.insert(fish);
}

bool TrackingHelper::save_tags() const {
    return cache->save_tags;
}

TrackingHelper::~TrackingHelper() {
    delete cache;
}

TrackingHelper::TrackingHelper(PPFrame& f, const std::vector<FrameProperties::Ptr>& added_frames)
      : cache(new CachedSettings), frame(f), _manager(frame)
{
    const BlobSizeRange minmax = FAST_SETTING(blob_size_ranges);
    double time(double(frame.timestamp) / double(1000*1000));
    props = Tracker::add_next_frame(FrameProperties(frame.index(), time, frame.timestamp));
    
    {
        auto it = --added_frames.end();
        if(it != added_frames.begin()) {
            --it;
            if((*it)->frame == frame.index() - 1_f)
                prev_props = (*it).get();
        }
    }
    
    if(save_tags()) {
        frame.transform_noise([this, &minmax](const pv::Blob& blob){
            if(blob.recount(-1) <= minmax.max_range().start)
                noise.emplace_back(pv::Blob::Make(blob));
        });
    }
    
    clear_blob_assigned();
    
    //! TODO: Can probably reuse frame.blob_grid here, but need to add noise() as well
    blob_grid().clear();
    
    frame.transform_all([](const pv::Blob& blob){
        blob_grid().insert(blob.bounds().x + blob.bounds().width  * 0.5f,
                           blob.bounds().y + blob.bounds().height * 0.5f,
                           blob.blob_id());
    });
    
    using namespace default_config;
    const auto frameIndex = frame.index();
    frame_uses_approximate = (_approximative_enabled_in_frame.valid() && frameIndex - _approximative_enabled_in_frame < cache->approximation_delay_time);
    
    match_mode = frame_uses_approximate
                    ? default_config::matching_mode_t::hungarian
                    : FAST_SETTING(match_mode);
    
    // see if there are manually fixed matches for this frame
    apply_manual_matches();
    apply_automatic_matches();
}

void TrackingHelper::assign_blob_individual(Individual* fish, pv::BlobPtr&& blob, default_config::matching_mode_t::Class match_mode)
{
    // transfer ownership of blob to individual
    // delete the copied objects from the original array.
    // otherwise they would be deleted after the RawProcessing
    // object gets deleted (ownership of blobs is gonna be
    // transferred to Individuals)
    /*auto it = std::find(frame.blobs.begin(), frame.blobs.end(), blob);
    if(it != frame.blobs.end())
        frame.blobs.erase(it);
    else if((it = std::find(frame.filtered_out.begin(), frame.filtered_out.end(), blob)) != frame.filtered_out.end()) {
        frame.filtered_out.erase(it);
    }
#ifndef NDEBUG
    else
        throw U_EXCEPTION("Cannot find blob in frame.");
#endif*/
#ifndef NDEBUG
    /*if(!contains(frame.blobs(), blob)
       && !contains(frame.noise(), blob))
    {
        FormatExcept("Cannot find blob ", blob->blob_id()," in frame ", frame.index(),".");
    }*/
#endif
    
#ifdef TREX_DEBUG_MATCHING
    for(auto &[i, b] : pairs) {
        if(i == fish) {
            if(b != &blob) {
                FormatWarning("Frame ",frameIndex,": Assigning individual ",i->identity().ID()," to ",blob ? blob->blob_id() : 0," instead of ", b ? (*b)->blob_id() : 0);
            }
            break;
        }
    }
#endif
    
    //auto &pixels = *blob_to_pixels.at(blob);
    assert(blob->properties_ready());
    auto bdx = blob->blob_id();
    if(!blob->moments().ready) {
        blob->calculate_moments();
    }
    
    if(save_tags()) {
        if(!blob->split()){
            std::scoped_lock guard(blob_fish_mutex);
            blob_fish_map[bdx] = fish;
            if(blob->parent_id().valid())
                blob_fish_map[blob->parent_id()] = fish;
            
            //pv::BlobPtr copy = pv::Blob::Make((Blob*)blob.get(), std::make_shared<std::vector<uchar>>(*blob->pixels()));
            tagged_fish.push_back(
                pv::Blob::Make(
                    *blob->lines(),
                    *blob->pixels(),
                    blob->flags())
            );
        }
    }
    
    auto index = fish->add(*this, std::move(blob), -1);
    if(index == -1) {
#ifndef NDEBUG
        FormatExcept("Was not able to assign individual ", fish->identity().ID()," with blob ", bdx," in frame ", frame.index());
#endif
        return;
    }
    
    auto &basic = fish->basic_stuff()[size_t(index)];
    fish_assign(fish);
    blob_assign(bdx);
    
    if (cache->do_posture)
        need_postures.push({fish, basic.get()});
    else {
        basic->pixels = nullptr;
    }
    
    ++assigned_count;
}

void TrackingHelper::apply_manual_matches()
{
    const auto frameIndex = frame.index();
    
    //! document the blobs that were requested but could not be found in the
    //! current frame
    ska::bytell_hash_map<pv::bid, std::set<Idx_t>> cannot_find;
    
    {
        //! blobs that were assigned to more than one individual
        ska::bytell_hash_map<pv::bid, std::set<Idx_t>> double_find;
        ska::bytell_hash_map<pv::bid, Idx_t> actually_assign;
        
        IndividualManager::transform_ids_with_error(frame.fixed_matches, [&](Idx_t fdx, pv::bid bdx, Individual* fish)
        {
            if(not bdx.valid()) {
                // dont assign this fish! (bdx == -1)
                return;
            }
            
            if(not frame.has_bdx(bdx)) {
                cannot_find[bdx].insert(fdx);
                return;
            }
            
            if(actually_assign.count(bdx) > 0) {
                FormatError("(fixed matches) Trying to assign blob ",bdx," twice in frame ",frameIndex," (fish ",fdx," and ",actually_assign.at(bdx),").");
                double_find[bdx].insert(fdx);
                
            } else if(blob_assigned(bdx)) {
                FormatError("(fixed matches, blob_assigned) Trying to assign blob ", bdx," twice in frame ", frameIndex," (fish ",fdx,").");
                // TODO: remove assignment from the other fish as well and add it to cannot_find
                double_find[bdx].insert(fdx);
                
            } else if(fish_assigned(fish)) {
                FormatError("Trying to assign fish ", fish->identity().ID()," twice in frame ",frameIndex,".");
            } else {
                actually_assign[bdx] = fdx;
            }
            
        }, [&](Idx_t fdx, pv::bid bdx) {
#ifndef NDEBUG
           if(frameIndex != Tracker::start_frame())
               FormatWarning("Individual number ", fdx," out of range in frame ",frameIndex,". Creating new one.");
#endif
           
           if(!frame.has_bdx(bdx)) {
               //FormatWarning("Cannot find blob ", it->second," in frame ",frameIndex,". Fallback to normal assignment behavior.");
               cannot_find[bdx].insert(fdx);
               return;
           }
           
           if(actually_assign.count(bdx) > 0) {
               FormatError("(fixed matches) Trying to assign blob ",bdx," twice in frame ",frameIndex," (fish ",fdx," and ",actually_assign.at(bdx),").");
               double_find[bdx].insert(fdx);
           } else
               actually_assign[bdx] = fdx;
        });
        
        for(auto && [bdx, fdxs] : double_find) {
            if(actually_assign.count(bdx) > 0) {
                fdxs.insert(actually_assign.at(bdx));
                actually_assign.erase(bdx);
            }
            
            cannot_find[bdx].insert(fdxs.begin(), fdxs.end());
        }
        
        auto blobs = frame.extract_from_all(actually_assign);
        assert(blobs.size() == actually_assign.size());
        
        for(auto&& blob: blobs) {
            auto bdx = blob->blob_id();
            auto fdx = actually_assign.at(bdx);
            
            auto result = _manager.retrieve_globally(fdx);
            if(result) {
                result.value()->add_manual_match(frameIndex);
                assign_blob_individual(result.value(), std::move(blob), default_config::matching_mode_t::benchmark);
            } else {
                FormatExcept("Cannot assign ", fdx, " to ", bdx, " in frame ", frameIndex, " reporting: ", result.error());
            }
        }
    }
    
    /**
     * Now resolve the rest of the objects that have not been
     * assigned yet.
     */
    if(not cannot_find.empty()) {
        struct Blaze {
            PPFrame *_frame;
            Blaze(PPFrame& frame) : _frame(&frame) {
                _frame->_finalized = false;
            }
            
            ~Blaze() {
                _frame->finalize();
            }
        } blaze(frame);
        
        std::map<pv::bid, std::vector<std::tuple<Idx_t, Vec2, pv::bid>>> assign_blobs;
        const auto max_speed_px = SLOW_SETTING(track_max_speed) / SLOW_SETTING(cm_per_pixel);
        
        for(auto && [bdx, fdxs] : cannot_find) {
            assert(bdx.valid());
            auto pos = bdx.calc_position();
            auto list = blob_grid().query(pos, max_speed_px);
            //auto str = Meta::toStr(list);
            
            if(!list.empty()) {
                // blob ids will not be < 0, as they have been inserted into the
                // grid before directly from the file. so we can assume (uint32_t)
                for(auto fdx: fdxs)
                    assign_blobs[std::get<1>(*list.begin())].push_back({fdx, pos, bdx});
            }
        }
        
        robin_hood::unordered_map<Idx_t, pv::bid> actual_assignments;
        
        for(const auto & [bdx, clique] : assign_blobs) {
            // have to split blob...
            
            robin_hood::unordered_map<pv::bid, split_expectation> expect;
            expect[bdx] = split_expectation(clique.size() == 1 ? 2 : clique.size(), false);
            //! TODO: this is broken right now (manual_matches)
            std::vector<pv::BlobPtr> big_filtered, single;
            single.emplace_back(frame.extract(bdx));
            
            PrefilterBlobs::split_big(
                      std::move(single),
                      BlobReceiver(frame,  BlobReceiver::noise),
                      BlobReceiver(big_filtered),
                      expect);
            //split_objects++;
            
            if(!big_filtered.empty()) {
                size_t found_perfect = 0;
                for(auto & [fdx, pos, original_bdx] : clique) {
                    for(auto &b : big_filtered) {
                        if(b->blob_id() == original_bdx) {
#ifndef NDEBUG
                            print("frame ",frame.index(),": Found perfect match for individual ",fdx,", bdx ",b->blob_id()," after splitting ",b->parent_id());
#endif
                            actual_assignments[fdx] = original_bdx;
                            //frame.blobs.insert(frame.blobs.end(), b);
                            ++found_perfect;
                            
                            break;
                        }
                    }
                }
                
                if(found_perfect) {
                    frame.add_regular(std::move(big_filtered));
                    // remove the blob thats to be split from all arrays
                    //frame.erase_anywhere(bdx);
                }
                
                if(found_perfect == clique.size()) {
#ifndef NDEBUG
                    print("frame ", frame.index(),": All missing manual matches perfectly matched.");
#endif
                } else {
                    FormatError("frame ",frame.index(),": Missing some matches (",found_perfect,"/",clique.size(),") for blob ",bdx," (identities ", clique,").");
                }
            }
        }
        
        if(!actual_assignments.empty()) {
            auto str = prettify_array(Meta::toStr(actual_assignments));
            print("frame ", frame.index(),": actually assigning:\n",str.c_str());
        }
        
        std::set<FOI::fdx_t> identities;
        
        IndividualManager::transform_ids(actual_assignments, [&](Idx_t fdx, pv::bid bdx, Individual* fish)
        {
            auto && blob = frame.extract(bdx);
            
            if(blob_assigned(bdx)) {
                print("Trying to assign blob ",bdx," twice.");
            } else if(fish_assigned(fish)) {
                print("Trying to assign fish ",fdx," twice.");
            } else {
                fish->add_manual_match(frameIndex);
                assign_blob_individual(fish, std::move(blob), default_config::matching_mode_t::benchmark);
                
                identities.insert(FOI::fdx_t(fdx));
            }
        });
        
        FOI::add(FOI(frameIndex, identities, "failed matches"));
    }
}


void TrackingHelper::apply_automatic_matches() {
    const auto frameIndex = frame.index();
    auto automatic_assignments = AutoAssign::automatically_assigned(frameIndex);
    for(auto && [fdx, bdx] : automatic_assignments) {
        if(!bdx.valid())
            continue; // dont assign this fish
        
        auto result = _manager.retrieve_globally(fdx);
        
        if(not result) {
#ifndef NDEBUG
            FormatError("frame ", frameIndex, ": Automatic assignment cannot be executed with fdx ", fdx, " -> ", bdx, " reporting: ", result.error());
#endif
        } else if(frame.has_bdx(bdx)
           && not fish_assigned(result.value())
           && not blob_assigned(bdx) )
        {
            assign_blob_individual(result.value(), frame.extract(bdx), default_config::matching_mode_t::benchmark);
            //frame.erase_anywhere(blob);
            result.value()->add_automatic_match(frameIndex);
            
        } else {
#ifndef NDEBUG
            FormatError("frame ",frameIndex,": Automatic assignment cannot be executed with fdx ",fdx,"(",result ? (fish_assigned(result.value()) ? "assigned" : "unassigned") : result.error(),") and bdx ",bdx,"(",bdx.valid() ? (blob_assigned(bdx) ? "assigned" : "unassigned") : "no blob",")");
#endif
        }
    }
}

void TrackingHelper::apply_matching() {
    // calculate optimal permutation of blob assignments
    static Timing perm_timing("PairingGraph", 30);
    TakeTiming take(perm_timing);
    
    PPFrame::Log(paired);
    
    using namespace Match;
    const auto frameIndex = frame.index();
    PairingGraph graph(*props, frameIndex, std::move(paired));
    
#if defined(PAIRING_PRINT_STATS)
    size_t nedges = 0;
    size_t max_edges_per_fish = 0, max_edges_per_blob = 0;
    double mean_edges_per_blob = 0, mean_edges_per_fish = 0;
    size_t fish_with_one_edge = 0, fish_with_more_edges = 0;
    
    std::map<long_t, size_t> edges_per_blob;
    
    double average_probability = 0;
    double one_edge_probability = double(fish_with_one_edge) / double(fish_with_one_edge + fish_with_more_edges);
    double blob_one_edge = double(blobs_with_one_edge) / double(blobs_with_one_edge + blobs_with_more_edges);
    double one_to_one = double(one_to_ones) / double(graph.edges().size());
    
    //graph.print_summary();
    
    auto print_statistics = [&](const PairingGraph::Result& optimal, bool force = false){
        std::lock_guard<std::mutex> guard(_statistics_mutex);
        
        static double average_improvements = 0, samples = 0, average_leafs = 0, average_objects = 0, average_bad_probabilities = 0;
        static Timer timer;
        
        size_t bad_probs = 0;
        for(auto && [fish, mp] : max_probs) {
            if(mp <= 0.5)
                ++bad_probs;
        }
        
        average_bad_probabilities += bad_probs;
        average_improvements += optimal.improvements_made;
        average_leafs += optimal.leafs_visited;
        average_objects += optimal.objects_looked_at;
        ++samples;
        
        if(size_t(samples) % 50 == 0 || force) {
            print("frame ",frameIndex,": ",optimal.improvements_made," of ",optimal.leafs_visited," / ",optimal.objects_looked_at," objects. ",average_improvements / samples," improvements on average, ",average_leafs / samples," leafs visited on average, ",average_objects / samples," objects on average (",mean_edges_per_fish," mean edges per fish and ",mean_edges_per_blob," mean edges per blob). On average we encounter ",average_bad_probabilities / samples," bad probabilities below 0.5 (currently ",bad_probs,").");
            print("g fish_has_one_edge * mean_edges_per_fish = ", one_edge_probability," * ", mean_edges_per_fish," = ",one_edge_probability * (mean_edges_per_fish));
            print("g fish_has_one_edge * mean_edges_per_blob = ", one_edge_probability," * ", mean_edges_per_blob," = ",one_edge_probability * (mean_edges_per_blob));
            print("g blob_has_one_edge * mean_edges_per_fish = ", blob_one_edge," * ", mean_edges_per_fish," = ",blob_one_edge * mean_edges_per_fish);
            print("g blob_has_one_edge * mean_edges_per_blob = ", blob_one_edge," * ", mean_edges_per_blob," = ",blob_one_edge * mean_edges_per_blob);
            print("g mean_edges_per_fish / mean_edges_per_blob = ", mean_edges_per_fish / mean_edges_per_blob);
            print("g one_to_one = ",one_to_one,", one_to_one * mean_edges_per_fish = ",one_to_one * mean_edges_per_fish," / blob: ",one_to_one * mean_edges_per_blob," /// ",average_probability,", ",average_probability * mean_edges_per_fish);
            print("g --");
            timer.reset();
        }
    };
    
    if(average_probability * mean_edges_per_fish <= 1) {
        FormatWarning("(", frameIndex,") Warning: ",average_probability * mean_edges_per_fish);
    }
#endif
    
    try {
        auto &optimal = graph.get_optimal_pairing(false, match_mode);
        
        if(match_mode != default_config::matching_mode_t::approximate) {
            /*
             NOT SAFE BECAUSE PAIRED HAS BEEN MOVED FROM
             std::lock_guard<std::mutex> guard(_statistics_mutex);
            _statistics[frameIndex].match_number_blob = (Match::index_t)paired.n_cols();
            _statistics[frameIndex].match_number_fish = (Match::index_t)paired.n_rows();
            _statistics[frameIndex].match_stack_objects = optimal.objects_looked_at;
            _statistics[frameIndex].match_improvements_made = optimal.improvements_made;
            _statistics[frameIndex].match_leafs_visited = optimal.leafs_visited;
            _statistics[frameIndex].method_used = (int)match_mode.value();*/
        }
        
#if defined(PAIRING_PRINT_STATS)
        print_statistics(optimal);
#endif
        
        const auto& paired = optimal.pairings;
        auto blobs = frame.extract_from_blobs_unsafe(paired);
        for(auto &blob : blobs) {
            /*auto it = std::find_if(paired.begin(), paired.end(), [&](auto& tuple){
                auto& [bdx, fish] = tuple;
                return bdx == blob->blob_id();
            });
            auto fish = std::get<1>(*it);*/
            auto fish = paired.at(blob->blob_id());
#ifdef TREX_DEBUG_MATCHING
            for(auto &[i, b] : pairs) {
                if(i == p.first) {
                    if(b != p.second) {
                        FormatWarning("Frame ",frameIndex,": Assigning individual ",i->identity().ID()," to ",p.second ? (*p.second)->blob_id() : 0," instead of ", b ? (*b)->blob_id() : 0);
                    }
                    break;
                }
            }
#endif
            
            _manager.become_active(fish);
            assign_blob_individual(fish, std::move(blob), match_mode);
        }
        
    } catch (const UtilsException&) {
#if !defined(NDEBUG) && defined(PAIRING_PRINT_STATS)
        if(graph.optimal_pairing())
            print_statistics(*graph.optimal_pairing(), true);
        else
            FormatWarning("No optimal pairing object.");
        
        graph.print_summary();
#endif
                    
#if defined(PAIRING_PRINT_STATS)
        // matching did not work
        FormatWarning("Falling back to approximative matching in frame ",frameIndex,". (p=",one_edge_probability,",",mean_edges_per_fish,", ",one_edge_probability * (mean_edges_per_fish),", ",one_edge_probability * mean_edges_per_blob,")");
        FormatWarning("frame ",frameIndex,": (",mean_edges_per_fish," mean edges per fish and ",mean_edges_per_blob," mean edges per blob).");
        
        print("gw Probabilities: fish_has_one_edge=", one_edge_probability," blob_has_one_edge=",blob_one_edge);
        print("gw fish_has_one_edge * mean_edges_per_fish = ", one_edge_probability," * ", mean_edges_per_fish," = ",one_edge_probability * (mean_edges_per_fish));
        print("gw fish_has_one_edge * mean_edges_per_blob = ", one_edge_probability," * ", mean_edges_per_blob," = ",one_edge_probability * (mean_edges_per_blob));
        print("gw blob_has_one_edge * mean_edges_per_fish = ", blob_one_edge," * ", mean_edges_per_fish," = ",blob_one_edge * mean_edges_per_fish);
        print("gw blob_has_one_edge * mean_edges_per_blob = ", blob_one_edge," * ", mean_edges_per_blob," = ",blob_one_edge * mean_edges_per_blob);
        print("gw one_to_one = ",one_to_one,", one_to_one * mean_edges_per_fish = ",one_to_one * mean_edges_per_fish," / blob: ",one_to_one * mean_edges_per_blob," /// ",average_probability,", ",average_probability * mean_edges_per_fish);
        print("gw mean_edges_per_fish / mean_edges_per_blob = ", mean_edges_per_fish / mean_edges_per_blob);
        print("gw ---");
#endif
        
        auto &optimal = graph.get_optimal_pairing(false, default_config::matching_mode_t::hungarian);
        for (auto &[bdx, fdx]: optimal.pairings) {
            _manager.become_active(fdx);
            assign_blob_individual(fdx, frame.extract(bdx), default_config::matching_mode_t::hungarian);
        }
        
        _approximative_enabled_in_frame = frameIndex;
        
        FOI::add(FOI(Range<Frame_t>(frameIndex, frameIndex + cache->approximation_delay_time - 1_f), "apprx matching"));
    }
}


double TrackingHelper::process_postures() {
    const auto frameIndex = frame.index();
    
    static Timing timing("Tracker::need_postures", 30);
    TakeTiming take(timing);
    
    double combined_posture_seconds = 0;
    static std::mutex _statistics_mutex;
    
    if(cache->do_posture && !need_postures.empty()) {
        static std::vector<std::tuple<Individual*, BasicStuff*>> all;
        
        while(!need_postures.empty()) {
            all.emplace_back(std::move(need_postures.front()));
            need_postures.pop();
        }
        
        distribute_indexes([frameIndex, &combined_posture_seconds](auto, auto start, auto end, auto){
            Timer t;
            double collected = 0;
            
            for(auto it = start; it != end; ++it) {
                t.reset();
                
                auto fish = std::get<0>(*it);
                auto basic = std::get<1>(*it);
                fish->save_posture(*basic, frameIndex);
                basic->pixels = nullptr;
                
                collected += t.elapsed();
            }
            
            std::lock_guard guard((_statistics_mutex));
            combined_posture_seconds += narrow_cast<float>(collected);
            
        }, Tracker::instance()->thread_pool(), all.begin(), all.end());
        
        all.clear();
        assert(need_postures.empty());
    }
    
    return combined_posture_seconds;
}

}
