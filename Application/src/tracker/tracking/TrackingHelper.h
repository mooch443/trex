#pragma once

#include <tracking/Individual.h>
#include <tracking/Stuffs.h>
#include <misc/PVBlob.h>
#include <tracker/misc/default_config.h>

namespace track {

struct CachedSettings;

struct TrackingHelper {
    const CachedSettings* cache;
    
public:
    inline static Frame_t _approximative_enabled_in_frame;
    
    bool blob_assigned(const pv::BlobPtr&) const;
    bool fish_assigned(Individual*) const;
    
public:
    bool save_tags() const;
    
    PPFrame& frame;
    
    // ------------------------------------
    // filter and calculate blob properties
    // ------------------------------------
    std::queue<std::tuple<Individual*, BasicStuff*>> need_postures;
    
private:
    robin_hood::unordered_flat_set<pv::Blob*> _blob_assigned;
    robin_hood::unordered_flat_set<Individual*> _fish_assigned;
    //ska::bytell_hash_map<pv::Blob*, bool> _blob_assigned;
    //ska::bytell_hash_map<Individual*, bool> _fish_assigned;
    
public:
    size_t assigned_count = 0;
    
    std::vector<tags::blob_pixel> tagged_fish, noise;
    ska::bytell_hash_map<pv::bid, Individual*> blob_fish_map;
//#define TREX_DEBUG_MATCHING
#ifdef TREX_DEBUG_MATCHING
    std::vector<std::pair<Individual*, Match::Blob_t>> pairs;
#endif
    using set_of_individuals_t = UnorderedVectorSet<Individual*>;
    
    // collect all the currently active individuals
    set_of_individuals_t active_individuals;
    
    //! current frame properties (e.g. time)
    //! as well as previous frame
    const FrameProperties* props = nullptr;
    const FrameProperties* prev_props = nullptr;
    
    
    bool frame_uses_approximate{false};
    
    Match::PairedProbabilities paired;
    default_config::matching_mode_t::Class match_mode{default_config::matching_mode_t::automatic};
    
    TrackingHelper(PPFrame& frame, const std::vector<std::unique_ptr<FrameProperties>>& added_frames);
    ~TrackingHelper();
    
    void assign_blob_individual(Individual* fish, const pv::BlobPtr& blob, default_config::matching_mode_t::Class match_mode);
    
    void apply_manual_matches(const ska::bytell_hash_map<Idx_t, Individual*>& individuals);
    void apply_automatic_matches();
    
    void apply_matching();
    
    double process_postures();
};

}
