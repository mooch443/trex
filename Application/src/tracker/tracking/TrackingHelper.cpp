#include "TrackingHelper.h"
#include <tracking/Tracker.h>
#include <tracker/misc/default_config.h>
#include <misc/pretty.h>
#include <tracking/AutomaticMatches.h>

namespace track {

struct CachedSettings {
    const bool do_posture{FAST_SETTINGS(calculate_posture)};
    const bool save_tags{!FAST_SETTINGS(tags_path).empty()};
    const uint32_t number_fish{(uint32_t)FAST_SETTINGS(track_max_individuals)};
    const Frame_t approximation_delay_time{Frame_t(max(1, FAST_SETTINGS(frame_rate) * 0.25))};
};

inline auto& blob_grid() {
    static grid::ProximityGrid grid{Tracker::average().bounds().size()};
    return grid;
}

bool TrackingHelper::blob_assigned(const pv::BlobPtr& blob) const {
    return _blob_assigned.contains(blob.get());
}

bool TrackingHelper::fish_assigned(Individual* fish) const {
    return _fish_assigned.contains(fish);
}

bool TrackingHelper::save_tags() const {
    return cache->save_tags;
}

TrackingHelper::~TrackingHelper() {
    delete cache;
}

TrackingHelper::TrackingHelper(PPFrame& frame, const std::vector<std::unique_ptr<FrameProperties>>& added_frames)
      : cache(new CachedSettings), frame(frame)
{
    const BlobSizeRange minmax = FAST_SETTINGS(blob_size_ranges);
    double time(double(frame.frame().timestamp()) / double(1000*1000));
    props = Tracker::add_next_frame(FrameProperties(frame.index(), time, frame.frame().timestamp()));
    
    {
        auto it = --added_frames.end();
        if(it != added_frames.begin()) {
            --it;
            if((*it)->frame == frame.index() - 1_f)
                prev_props = (*it).get();
        }
    }
    
    if(save_tags()) {
        for(auto &blob : frame.noise()) {
            if(blob->recount(-1) <= minmax.max_range().start) {
                pv::BlobPtr copy = std::make_shared<pv::Blob>(*blob);
                noise.emplace_back(std::move(copy));
            }
        }
    }
    
    _blob_assigned.clear();
    /*for(auto &blob: frame.blobs()) {
        auto it = _blob_assigned.find(blob.get());
        
    }
        _blob_assigned[blob.get()] = false;*/
    
    //! TODO: Can probably reuse frame.blob_grid here, but need to add noise() as well
    blob_grid().clear();
    
    for(auto &b : frame.blobs()) {
        //id_to_blob[b->blob_id()] = b;
        blob_grid().insert(b->bounds().x + b->bounds().width * 0.5f, b->bounds().y + b->bounds().height * 0.5f, b->blob_id());
    }
    for(auto &b : frame.noise()) {
        //id_to_blob[b->blob_id()] = b;
        blob_grid().insert(b->bounds().x + b->bounds().width * 0.5f, b->bounds().y + b->bounds().height * 0.5f, b->blob_id());
    }
    
    using namespace default_config;
    const auto frameIndex = frame.index();
    frame_uses_approximate = (_approximative_enabled_in_frame.valid() && frameIndex - _approximative_enabled_in_frame < cache->approximation_delay_time);
    
    match_mode = frame_uses_approximate
                    ? default_config::matching_mode_t::hungarian
                    : FAST_SETTINGS(match_mode);
}

void TrackingHelper::assign_blob_individual(Individual* fish, const pv::BlobPtr& blob, default_config::matching_mode_t::Class match_mode)
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
    if(!contains(frame.blobs(), blob)
       && !contains(frame.noise(), blob))
    {
        FormatExcept("Cannot find blob ", blob->blob_id()," in frame ", frame.index(),".");
    }
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
    if(!blob->moments().ready) {
        blob->calculate_moments();
    }
    auto index = fish->add(props, frame, blob, -1, match_mode);
    if(index == -1) {
#ifndef NDEBUG
        FormatExcept("Was not able to assign individual ", fish->identity().ID()," with blob ", blob->blob_id()," in frame ", frame.index());
#endif
        return;
    }
    
    auto &basic = fish->basic_stuff()[size_t(index)];
    _fish_assigned.insert(fish);
    _blob_assigned.insert(blob.get());
    //_fish_assigned[fish] = true;
    //_blob_assigned[blob.get()] = true;
    
    if(save_tags()) {
        if(!blob->split()){
            blob_fish_map[blob->blob_id()] = fish;
            if(blob->parent_id().valid())
                blob_fish_map[blob->parent_id()] = fish;
            
            //pv::BlobPtr copy = std::make_shared<pv::Blob>((Blob*)blob.get(), std::make_shared<std::vector<uchar>>(*blob->pixels()));
            tagged_fish.push_back(
                std::make_shared<pv::Blob>(
                    *blob->lines(),
                    *blob->pixels(),
                    blob->flags())
            );
        }
    }
    
    if (cache->do_posture)
        need_postures.push({fish, basic.get()});
    else {
        basic->pixels = nullptr;
    }
    
    ++assigned_count;
}

void TrackingHelper::apply_manual_matches(typename std::invoke_result_t<decltype(Tracker::individuals)> individuals)
{
    const auto frameIndex = frame.index();
    
    //! document the blobs that were requested but could not be found in the
    //! current frame
    ska::bytell_hash_map<pv::bid, std::set<Idx_t>> cannot_find;
    
    {
        //! blobs that were assigned to more than one individual
        ska::bytell_hash_map<pv::bid, std::set<Idx_t>> double_find;
        
        ska::bytell_hash_map<pv::bid, Idx_t> actually_assign;
        
        Settings::manual_matches_t::mapped_type current_fixed_matches;
        {
            auto manual_matches = Settings::get<Settings::manual_matches>();
            auto it = manual_matches->find(frameIndex);
            if (it != manual_matches->end())
                current_fixed_matches = it->second;
        }
        
        for(auto && [fdx, bdx] : current_fixed_matches) {
            auto it = individuals.find(fdx);
            if(it != individuals.end()) { //&& size_t(fm.second) < blobs.size()) {
                auto fish = it->second;
                
                if(!bdx.valid()) {
                    // dont assign this fish! (bdx == -1)
                    continue;
                }
                
                auto blob = frame.find_bdx(bdx);
                if(!blob) {
                    cannot_find[bdx].insert(fdx);
                    continue;
                }
                
                if(actually_assign.count(bdx) > 0) {
                    FormatError("(fixed matches) Trying to assign blob ",bdx," twice in frame ",frameIndex," (fish ",fdx," and ",actually_assign.at(bdx),").");
                    double_find[bdx].insert(fdx);
                    
                } else if(blob_assigned(blob)) {
                    FormatError("(fixed matches, blob_assigned) Trying to assign blob ", bdx," twice in frame ", frameIndex," (fish ",fdx,").");
                    // TODO: remove assignment from the other fish as well and add it to cannot_find
                    double_find[bdx].insert(fdx);
                    
                } else if(fish_assigned(fish)) {
                    FormatError("Trying to assign fish ", fish->identity().ID()," twice in frame ",frameIndex,".");
                } else {
                    actually_assign[blob->blob_id()] = fdx;
                }
                
            } else {
#ifndef NDEBUG
                if(frameIndex != Tracker::start_frame())
                    FormatWarning("Individual number ", fdx," out of range in frame ",frameIndex,". Creating new one.");
#endif
                
                auto blob = frame.find_bdx(bdx);
                if(!blob) {
                    //FormatWarning("Cannot find blob ", it->second," in frame ",frameIndex,". Fallback to normal assignment behavior.");
                    cannot_find[bdx].insert(fdx);
                    continue;
                }
                
                if(actually_assign.count(bdx) > 0) {
                    FormatError("(fixed matches) Trying to assign blob ",bdx," twice in frame ",frameIndex," (fish ",fdx," and ",actually_assign.at(bdx),").");
                    double_find[bdx].insert(fdx);
                } else
                    actually_assign[bdx] = fdx;
            }
        }
        
        for(auto && [bdx, fdxs] : double_find) {
            if(actually_assign.count(bdx) > 0) {
                fdxs.insert(actually_assign.at(bdx));
                actually_assign.erase(bdx);
            }
            
            cannot_find[bdx].insert(fdxs.begin(), fdxs.end());
        }
        
        for(auto && [bdx, fdx] : actually_assign) {
            auto &blob = frame.bdx_to_ptr(bdx);
            Individual *fish = NULL;
            
            auto it = individuals.find(fdx);
            if(it == individuals.end()) {
                fish = Tracker::create_individual(fdx, active_individuals);
            } else {
                fish = it->second;
                active_individuals.insert(fish);
            }
            
            fish->add_manual_match(frameIndex);
            assign_blob_individual(fish, blob, default_config::matching_mode_t::benchmark);
        }
    }
    
    /**
     * Now resolve the rest of the objects that have not been
     * assigned yet.
     */
    if(!cannot_find.empty()) {
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
        const auto max_speed_px = FAST_SETTINGS(track_max_speed) / FAST_SETTINGS(cm_per_pixel);
        
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
            auto &blob = frame.bdx_to_ptr(bdx);
            
            robin_hood::unordered_map<pv::bid, split_expectation> expect;
            expect[bdx] = split_expectation(clique.size() == 1 ? 2 : clique.size(), false);
            
            auto big_filtered = Tracker::instance()->split_big(Tracker::BlobReceiver(frame,  Tracker::BlobReceiver::noise), {blob}, expect);
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
                    frame.erase_anywhere(blob);
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
        
        for(auto && [fdx, bdx] : actual_assignments) {
            auto &blob = frame.bdx_to_ptr(bdx);
            
            Individual *fish = nullptr;
            auto it = individuals.find(fdx);
            
            // individual doesnt exist yet. create it
            if(it == individuals.end()) {
                throw U_EXCEPTION("Should have created it already.");
            } else
                fish = it->second;
            
            if(blob_assigned(blob)) {
                print("Trying to assign blob ",bdx," twice.");
            } else if(fish_assigned(fish)) {
                print("Trying to assign fish ",fdx," twice.");
            } else {
                fish->add_manual_match(frameIndex);
                assign_blob_individual(fish, blob, default_config::matching_mode_t::benchmark);
                
                frame.erase_anywhere(blob);
                active_individuals.insert(fish);
                
                identities.insert(FOI::fdx_t(fdx));
            }
        }
        
        FOI::add(FOI(frameIndex, identities, "failed matches"));
    }
}


void TrackingHelper::apply_automatic_matches() {
    const auto frameIndex = frame.index();
    auto &individuals = Tracker::individuals();
    
    auto automatic_assignments = AutoAssign::automatically_assigned(frameIndex);
    for(auto && [fdx, bdx] : automatic_assignments) {
        if(!bdx.valid())
            continue; // dont assign this fish
        
        Individual *fish = nullptr;
        if(individuals.find(fdx) != individuals.end())
            fish = individuals.at(fdx);
        
        pv::BlobPtr blob = frame.find_bdx((uint32_t)bdx);
        if(fish
           && blob
           && !fish_assigned(fish)
           && !blob_assigned(blob))
        {
            assign_blob_individual(fish, blob, default_config::matching_mode_t::benchmark);
            //frame.erase_anywhere(blob);
            fish->add_automatic_match(frameIndex);
            active_individuals.insert(fish);
            
        } else {
#ifndef NDEBUG
            FormatError("frame ",frameIndex,": Automatic assignment cannot be executed with fdx ",fdx,"(",fish ? (fish_assigned(fish) ? "assigned" : "unassigned") : "no fish",") and bdx ",bdx,"(",blob ? (blob_assigned(blob)] ? "assigned" : "unassigned") : "no blob",")");
#endif
        }
    }
}

void TrackingHelper::apply_matching() {
    // calculate optimal permutation of blob assignments
    static Timing perm_timing("PairingGraph", 30);
    TakeTiming take(perm_timing);
    
    using namespace Match;
    const auto frameIndex = frame.index();
    PairingGraph graph(*props, frameIndex, paired);
    
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
            /*std::lock_guard<std::mutex> guard(_statistics_mutex);
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
        
        for (auto &p: optimal.pairings) {
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
            
            assign_blob_individual(p.first, *p.second, match_mode);
            active_individuals.insert(p.first);
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
        for (auto &p: optimal.pairings) {
            assign_blob_individual(p.first, *p.second, default_config::matching_mode_t::hungarian);
            active_individuals.insert(p.first);
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
        
        distribute_vector([frameIndex, &combined_posture_seconds](auto, auto start, auto end, auto){
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
