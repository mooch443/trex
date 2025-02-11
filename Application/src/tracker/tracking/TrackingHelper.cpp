#include "TrackingHelper.h"
#include <tracking/Tracker.h>
#include <tracker/misc/default_config.h>
#include <misc/pretty.h>
#include <tracking/AutomaticMatches.h>
#include <tracking/IndividualManager.h>
#include <misc/FOI.h>

namespace track {

struct CachedSettings {
    const bool do_posture{FAST_SETTING(calculate_posture)};
    const bool save_tags{!FAST_SETTING(tags_path).empty()};
    const uint32_t number_fish{(uint32_t)FAST_SETTING(track_max_individuals)};
    const Frame_t approximation_delay_time{Frame_t(max(1u, SLOW_SETTING(frame_rate) / 4u))};
};

inline auto& blob_grid() {
    static grid::ProximityGrid grid{Tracker::average().bounds().size()};
    return grid;
}

bool TrackingHelper::save_tags() const {
    return cache->save_tags;
}

TrackingHelper::~TrackingHelper() {
    delete cache;
}

TrackingHelper::TrackingHelper(
   PPFrame& f,
   const std::vector<FrameProperties::Ptr>& added_frames,
   Frame_t approximative_enabled_in_frame)
    : cache(new CachedSettings),
      _approximative_enabled_in_frame(approximative_enabled_in_frame),
      frame(f),
      _manager(frame)
{
    const SizeFilters track_size_filter = FAST_SETTING(track_size_filter);
    double time(double(frame.timestamp) / double(1000*1000));
    props = Tracker::add_next_frame(FrameProperties(frame.index(), time, frame.timestamp));
    
    {
        auto it = --added_frames.end();
        if(it != added_frames.begin()) {
            --it;
            if((*it)->frame() == frame.index() - 1_f)
                prev_props = (*it).get();
        }
    }
    
    if(save_tags() && track_size_filter) {
        frame.transform_noise([this, max_range = track_size_filter.max_range()](const pv::Blob& blob){
            if(blob.recount(-1) <= max_range.start)
                noise.emplace_back(pv::Blob::Make(blob));
        });
    }
    
    _manager.clear_blob_assigned();
    
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
        
        IndividualManager::transform_ids_with_error(frame.fixed_matches, [&](Idx_t fdx, pv::bid bdx, Individual* )
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
                
            } else if(_manager.blob_assigned(bdx)) {
                FormatError("(fixed matches, blob_assigned) Trying to assign blob ", bdx," twice in frame ", frameIndex," (fish ",fdx,").");
                // TODO: remove assignment from the other fish as well and add it to cannot_find
                double_find[bdx].insert(fdx);
                
            } else if(_manager.fish_assigned(fdx)) {
                FormatError("Trying to assign fish ", fdx," twice in frame ",frameIndex,".");
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
        
        _manager.assign(AssignInfo{
            .frame = &frame,
            .f_prop = props,
            .f_prev_prop = prev_props,
            .match_mode = match_mode
        }, std::move(actually_assign), [frameIndex](pv::bid, Idx_t, Individual* fish) {
            fish->add_manual_match(frameIndex);
            
        }, [frameIndex](pv::bid bdx, Idx_t fdx, Individual*, const char* error) {
            FormatExcept("Cannot assign ", fdx, " to ", bdx, " in frame ", frameIndex, " reporting: ", error);
        });
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
        
        std::unordered_map<pv::bid, Idx_t> actual_assignments;
        
        for(const auto & [bdx, clique] : assign_blobs) {
            // have to split blob...
            
            robin_hood::unordered_map<pv::bid, split_expectation> expect;
            expect[bdx] = split_expectation(clique.size() == 1 ? 2 : clique.size(), false);
            //! TODO: this is broken right now (manual_matches)
            std::vector<pv::BlobPtr> big_filtered, single;
            single.emplace_back(frame.extract(bdx));
            
            PrefilterBlobs::split_big(
                      frameIndex,
                      std::move(single),
                      BlobReceiver(frame,  BlobReceiver::noise, FilterReason::SplitFailed),
                      BlobReceiver(big_filtered),
                      expect);
            //split_objects++;
            
            if(!big_filtered.empty()) {
                size_t found_perfect = 0;
                for(auto & [fdx, pos, original_bdx] : clique) {
                    for(auto &b : big_filtered) {
                        if(b->blob_id() == original_bdx) {
#ifndef NDEBUG
                            Print("frame ",frame.index(),": Found perfect match for individual ",fdx,", bdx ",b->blob_id()," after splitting ",b->parent_id());
#endif
                            actual_assignments[original_bdx] = fdx;
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
                    Print("frame ", frame.index(),": All missing manual matches perfectly matched.");
#endif
                } else {
                    FormatError("frame ",frame.index(),": Missing some matches (",found_perfect,"/",clique.size(),") for blob ",bdx," (identities ", clique,"). big_filtered=", big_filtered);
                }
            }
        }
        
        if(!actual_assignments.empty()) {
            auto str = prettify_array(Meta::toStr(actual_assignments));
            Print("frame ", frame.index(),": actually assigning:\n",str.c_str());
        }
        
        std::set<FOI::fdx_t> identities;
        _manager.assign(AssignInfo{
            .frame = &frame,
            .f_prop = props,
            .f_prev_prop = prev_props,
            .match_mode = match_mode
        }, std::move(actual_assignments), [frameIndex, &identities](pv::bid, Idx_t fdx, Individual* fish)
        {
            fish->add_manual_match(frameIndex);
            identities.insert(FOI::fdx_t(fdx));
            
        }, [frameIndex](pv::bid bdx, Idx_t fdx, Individual*, const char* error) {
            FormatExcept("Cannot assign ", fdx, " to ", bdx, " in frame ", frameIndex, " reporting: ", error);
        });
        
        FOI::add(FOI(frameIndex, identities, "failed matches"));
    }
}


void TrackingHelper::apply_automatic_matches() {
    const auto frameIndex = frame.index();
    auto automatic_assignments = AutoAssign::automatically_assigned(frameIndex);
    /*_manager.transform_ids(automatic_assignments, [](Idx_t fdx, pv::bid bdx, Individual* fish)
    {
        assert(bdx.valid());
        //if(!bdx.valid())
        //    continue; // dont assign this fish
        
        auto result = _manager.retrieve_globally(fdx);
        
        if(not result) {
#ifndef NDEBUG
            FormatError("frame ", frameIndex, ": Automatic assignment cannot be executed with fdx ", fdx, " -> ", bdx, " reporting: ", result.error());
#endif
        } else if(frame.has_bdx(bdx)
           && not _manager.fish_assigned(fdx)
           && not _manager.blob_assigned(bdx) )
        {
            assign_blob_individual(result.value(), frame.extract(bdx), default_config::matching_mode_t::benchmark);
            //frame.erase_anywhere(blob);
            result.value()->add_automatic_match(frameIndex);
            
        } else {
#ifndef NDEBUG
            FormatError("frame ",frameIndex,": Automatic assignment cannot be executed with fdx ",fdx,"(",result ? (_manager.fish_assigned(fdx) ? "assigned" : "unassigned") : result.error(),") and bdx ",bdx,"(",bdx.valid() ? (_manager.blob_assigned(bdx) ? "assigned" : "unassigned") : "no blob",")");
#endif
        }
    });
    for(auto && [fdx, bdx] : automatic_assignments) {
        
    }*/
    _manager.assign(AssignInfo{
        .frame = &frame,
        .f_prop = props,
        .f_prev_prop = prev_props,
        .match_mode = default_config::matching_mode_t::none
        
    }, std::move(automatic_assignments), [frameIndex](pv::bid, Idx_t, Individual* fish) {
        fish->add_automatic_match(frameIndex);
    }
#ifndef NDEBUG
    , [frameIndex, this](pv::bid bdx, Idx_t fdx, Individual*, const char* error) {

            FormatError("frame ",frameIndex,": Automatic assignment cannot be executed with fdx ",fdx,"(", _manager.fish_assigned(fdx) ? "assigned" : "unassigned",") and bdx ",bdx,"(",bdx.valid() ? (_manager.blob_assigned(bdx) ? "assigned" : "unassigned") : "no blob","): ", error);
      }
#endif
    );
}

void TrackingHelper::apply_matching() {
    // calculate optimal permutation of blob assignments
    static Timing perm_timing("PairingGraph", 30);
    TakeTiming take(perm_timing);
    
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
            Print("frame ",frameIndex,": ",optimal.improvements_made," of ",optimal.leafs_visited," / ",optimal.objects_looked_at," objects. ",average_improvements / samples," improvements on average, ",average_leafs / samples," leafs visited on average, ",average_objects / samples," objects on average (",mean_edges_per_fish," mean edges per fish and ",mean_edges_per_blob," mean edges per blob). On average we encounter ",average_bad_probabilities / samples," bad probabilities below 0.5 (currently ",bad_probs,").");
            Print("g fish_has_one_edge * mean_edges_per_fish = ", one_edge_probability," * ", mean_edges_per_fish," = ",one_edge_probability * (mean_edges_per_fish));
            Print("g fish_has_one_edge * mean_edges_per_blob = ", one_edge_probability," * ", mean_edges_per_blob," = ",one_edge_probability * (mean_edges_per_blob));
            Print("g blob_has_one_edge * mean_edges_per_fish = ", blob_one_edge," * ", mean_edges_per_fish," = ",blob_one_edge * mean_edges_per_fish);
            Print("g blob_has_one_edge * mean_edges_per_blob = ", blob_one_edge," * ", mean_edges_per_blob," = ",blob_one_edge * mean_edges_per_blob);
            Print("g mean_edges_per_fish / mean_edges_per_blob = ", mean_edges_per_fish / mean_edges_per_blob);
            Print("g one_to_one = ",one_to_one,", one_to_one * mean_edges_per_fish = ",one_to_one * mean_edges_per_fish," / blob: ",one_to_one * mean_edges_per_blob," /// ",average_probability,", ",average_probability * mean_edges_per_fish);
            Print("g --");
            timer.reset();
        }
    };
    
    if(average_probability * mean_edges_per_fish <= 1) {
        FormatWarning("(", frameIndex,") Warning: ",average_probability * mean_edges_per_fish);
    }
#endif
    
    try {
        auto &optimal = graph.get_optimal_pairing(false, match_mode);
        PPFrame::Log("Got pairing = ", optimal.pairings);
        
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
        
        _manager.assign<false>(AssignInfo{
            .frame = &frame,
            .f_prop = props,
            .f_prev_prop = prev_props,
            .match_mode = match_mode
        }, std::move(optimal.pairings), [&](pv::bid, Idx_t, Individual*)
        {
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
        }, [frameIndex](pv::bid bdx, Idx_t fdx, Individual*, const char* error) {
            FormatExcept("Cannot assign ", fdx, " to ", bdx, " in frame ", frameIndex, " reporting: ", error);
        });
        
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
        
        Print("gw Probabilities: fish_has_one_edge=", one_edge_probability," blob_has_one_edge=",blob_one_edge);
        Print("gw fish_has_one_edge * mean_edges_per_fish = ", one_edge_probability," * ", mean_edges_per_fish," = ",one_edge_probability * (mean_edges_per_fish));
        Print("gw fish_has_one_edge * mean_edges_per_blob = ", one_edge_probability," * ", mean_edges_per_blob," = ",one_edge_probability * (mean_edges_per_blob));
        Print("gw blob_has_one_edge * mean_edges_per_fish = ", blob_one_edge," * ", mean_edges_per_fish," = ",blob_one_edge * mean_edges_per_fish);
        Print("gw blob_has_one_edge * mean_edges_per_blob = ", blob_one_edge," * ", mean_edges_per_blob," = ",blob_one_edge * mean_edges_per_blob);
        Print("gw one_to_one = ",one_to_one,", one_to_one * mean_edges_per_fish = ",one_to_one * mean_edges_per_fish," / blob: ",one_to_one * mean_edges_per_blob," /// ",average_probability,", ",average_probability * mean_edges_per_fish);
        Print("gw mean_edges_per_fish / mean_edges_per_blob = ", mean_edges_per_fish / mean_edges_per_blob);
        Print("gw ---");
#endif
        
        auto &optimal = graph.get_optimal_pairing(false, default_config::matching_mode_t::hungarian);
        PPFrame::Log("Got backup pairing = ", optimal.pairings);
        
        _manager.assign<false>(AssignInfo{
            .frame = &frame,
            .f_prop = props,
            .f_prev_prop = prev_props,
            .match_mode = default_config::matching_mode_t::hungarian
        }, std::move(optimal.pairings), [&](pv::bid, Idx_t, Individual*)
        {
            
        }, [frameIndex](pv::bid bdx, Idx_t fdx, Individual*, const char* error) {
            FormatExcept("Cannot assign ", fdx, " to ", bdx, " in frame ", frameIndex, " reporting: ", error);
        });
        
        _approximative_enabled_in_frame = frameIndex;
        
        FOI::add(FOI(Range<Frame_t>(frameIndex, frameIndex + cache->approximation_delay_time - 1_f), "apprx matching"));
    }
}


double TrackingHelper::process_postures() {
    const auto frameIndex = frame.index();
    
    static Timing timing("Tracker::need_postures", 100);
    TakeTiming take(timing);
    
    double combined_posture_seconds = 0;
    static auto _statistics_mutex = LOGGED_MUTEX("TrackingHelper::statistics_mutex");
    
    if(cache->do_posture && !_manager.need_postures.empty()) {
        const auto pose_midline_indexes = SETTING(pose_midline_indexes).value<PoseMidlineIndexes>();
        
        std::vector<std::tuple<Individual*, BasicStuff*, pv::BlobPtr>> posture_store;
        posture_store.reserve(_manager.need_postures.size());
        while(!_manager.need_postures.empty()) {
            posture_store.emplace_back(std::move(_manager.need_postures.front()));
            _manager.need_postures.pop();
        }
        
        IndividualManager::Protect{};
        
        distribute_indexes([frameIndex, &combined_posture_seconds, &pose_midline_indexes](auto, auto start, auto end, auto) {
            Timer t;
            double collected = 0;
            
            for(auto it = start; it != end; ++it) {
                t.reset();
                
                auto &&[fish, basic, pixels] = *it;
                fish->save_posture(*basic, pose_midline_indexes, frameIndex, std::move(pixels));
                collected += t.elapsed();
            }
            
            auto guard = LOGGED_LOCK(_statistics_mutex);
            combined_posture_seconds += collected;
            
        }, Tracker::thread_pool(), posture_store.begin(), posture_store.end());
        
        assert(_manager.need_postures.empty());
    }
    
    return combined_posture_seconds;
}

}
