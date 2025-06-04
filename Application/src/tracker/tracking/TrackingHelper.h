#pragma once

#include <tracking/Individual.h>
#include <tracking/Stuffs.h>
#include <misc/PVBlob.h>
#include <tracker/misc/default_config.h>
#include <misc/TrackingSettings.h>
#include <tracking/IndividualManager.h>

namespace track {

class IndividualManager;

#define DEFINE_CACHE_SETTING(NAME) const Settings:: NAME ## _t NAME = SLOW_SETTING(NAME)

struct CachedSettings {
    DEFINE_CACHE_SETTING(calculate_posture);
    const bool save_tags = not FAST_SETTING(tags_path).empty();
    DEFINE_CACHE_SETTING(track_max_individuals);
    DEFINE_CACHE_SETTING(frame_rate);
    const Frame_t approximation_delay_time = Frame_t(max(1u, frame_rate / 4u));
    DEFINE_CACHE_SETTING(track_size_filter);
    DEFINE_CACHE_SETTING(match_mode);
    DEFINE_CACHE_SETTING(track_max_speed);
    DEFINE_CACHE_SETTING(cm_per_pixel);
    DEFINE_CACHE_SETTING(match_min_probability);
    DEFINE_CACHE_SETTING(huge_timestamp_seconds);
};

struct TrackingHelper {
    GETTER(CachedSettings, cache);
    
public:
    Frame_t _approximative_enabled_in_frame;
    
public:
    bool save_tags() const;
    
    PPFrame& frame;
    IndividualManager _manager;
    
    // ------------------------------------
    // filter and calculate blob properties
    // ------------------------------------
    
public:
    std::vector<tags::blob_pixel> tagged_fish, noise;
    
    ska::bytell_hash_map<pv::bid, Individual*> blob_fish_map;
    std::shared_mutex blob_fish_mutex;
    
//#define TREX_DEBUG_MATCHING
#ifdef TREX_DEBUG_MATCHING
    std::vector<std::pair<Individual*, Match::Blob_t>> pairs;
#endif
    
    //! current frame properties (e.g. time)
    //! as well as previous frame
    const FrameProperties* props = nullptr;
    const FrameProperties* prev_props = nullptr;
    
    bool frame_uses_approximate{false};
    
    Match::PairedProbabilities paired;
    default_config::matching_mode_t::Class match_mode{default_config::matching_mode_t::automatic};
    
    TrackingHelper(PPFrame& frame, const std::vector<FrameProperties::Ptr>& added_frames, Frame_t approximative_enabled_in_frame);
    ~TrackingHelper();
    
    void apply_manual_matches();
    void apply_automatic_matches();
    
    void apply_matching();
    
    double process_postures();
};

}
