#pragma once

#include <tracking/Individual.h>
#include <tracking/Stuffs.h>
#include <misc/PVBlob.h>
#include <tracker/misc/default_config.h>
#include <misc/TrackingSettings.h>
#include <tracking/IndividualManager.h>

namespace track {

class IndividualManager;
struct CachedSettings;

struct TrackingHelper {
    const CachedSettings* cache;
    
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
