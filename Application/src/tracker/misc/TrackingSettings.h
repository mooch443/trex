#pragma once

#include <commons.pc.h>
#include <misc/GlobalSettings.h>
#include <file/Path.h>
#include <tracker/misc/default_config.h>
#include <misc/idx_t.h>
#include <misc/bid.h>
#include <misc/frame_t.h>
#include <misc/create_struct.h>
#include <misc/SizeFilters.h>
#include <processing/encoding.h>

namespace track {
using namespace cmn;
class Individual;

using mmatches_t = std::map<Frame_t, std::map<Idx_t, pv::bid>>;
using msplits_t = std::map<Frame_t, std::set<pv::bid>>;
using inames_t = std::map<Idx_t, std::string>;
using mapproved_t = std::map<long_t,long_t>;
using analrange_t = Range<long_t>;

//! Stable references arent technically needed, but the speed difference
//! here is negligible.
using set_of_individuals_t = robin_hood::unordered_node_set<Individual*>;

using inactive_individuals_t = robin_hood::unordered_node_map<Idx_t, Individual*>;

//! A std::unordered_map turns out to be the fastest container for this
//! purpose (sparse container for frame to individuals association).
using active_individuals_map_t = std::unordered_map<Frame_t, std::unique_ptr<set_of_individuals_t>>;

//! The global map of individual ids -> Individual*
using individuals_map_t = robin_hood::unordered_flat_map<Idx_t, std::unique_ptr<Individual>>;

//! Where is the external identity information sourced from?
//! 1. Visual identification
//! 2. QRCode recognition
enum IdentitySource {
    VisualIdent,
    QRCodes
};

//! Information needed to assign an individual to a blob
class PPFrame;
struct FrameProperties;
struct CachedSettings;

struct AssignInfo {
    PPFrame* frame;
    const FrameProperties* f_prop;
    const FrameProperties* f_prev_prop;
    default_config::matching_mode_t::Class match_mode;
    const CachedSettings* settings;
};

//! Given a pose (a set of points) this array indicates the order and indexes
//! that shall contribute to the midline of animal poses. This is mainly used
//! for `individual_image_normalization` right now.
struct PoseMidlineIndexes {
    std::vector<uint8_t> indexes;
    
    static PoseMidlineIndexes fromStr(cmn::StringLike auto&& str) {
        return PoseMidlineIndexes{
            .indexes = Meta::fromStr<std::vector<uint8_t>>(str)
        };
    }
    std::string toStr() const;
    glz::json_t to_json() const;
    static std::string class_name() { return "PoseMidlineIndexes"; }

    bool operator==(const PoseMidlineIndexes& other) const {
        return indexes == other.indexes;
    }
};

//! A global settings cache used across the application by
//! calling `FAST_SETTING(name)`.
CREATE_STRUCT(Settings,
  (uint32_t, smooth_window),
  (Float2_t, cm_per_pixel),
  (uint32_t, frame_rate),
  (bool, track_enforce_frame_rate),
  (float, track_max_reassign_time),
  (float, speed_extrapolation),
  (bool, calculate_posture),
  (Float2_t, track_max_speed),
  (SizeFilters, track_size_filter),
  (int, track_threshold),
  (int, track_threshold_2),
  (Rangef, threshold_ratio_range),
  (uint32_t, track_max_individuals),
  (int, track_posture_threshold),
  (uint8_t, outline_smooth_step),
  (uint8_t, outline_smooth_samples),
  (float, outline_resample),
  (mmatches_t, manual_matches),
  (msplits_t, manual_splits),
  (msplits_t, track_ignore_bdx),
  (uint32_t, midline_resolution),
  (float, meta_mass_mg),
  (inames_t, individual_names),
  (float, midline_stiff_percentage),
  (float, match_min_probability),
  (uint16_t, posture_direction_smoothing),
  (file::Path, tags_path),
  (std::vector<Vec2>, grid_points),
  (std::vector<std::vector<Vec2>>, recognition_shapes),
  (float, grid_points_scaling),
  (std::vector<std::vector<Vec2>>, track_ignore),
  (std::vector<std::vector<Vec2>>, track_include),
  (bool, tracklet_punish_timedelta),
  (double, huge_timestamp_seconds),
  (mapproved_t, manually_approved),
  (float, track_speed_decay),
  (bool, midline_invert),
  (bool, track_time_probability_enabled),
  (float, posture_head_percentage),
  (bool, track_threshold_is_absolute),
  (bool, track_background_subtraction),
  (float, blobs_per_thread),
  (std::string, individual_prefix),
  (uint64_t, video_length),
  (analrange_t, analysis_range),
  (float, visual_field_eye_offset),
  (float, visual_field_eye_separation),
  (uint8_t, visual_field_history_smoothing),
  (bool, tracklet_punish_speeding),
  (default_config::matching_mode_t::Class, match_mode),
  (bool, track_do_history_split),
  (uint8_t, posture_closing_steps),
  (uint8_t, posture_closing_size),
  (float, individual_image_scale),
  (bool, track_pause),
  (Float2_t, track_trusted_probability),
  (Float2_t, accumulation_tracklet_add_factor),
  (bool, output_interpolate_positions),
  (bool, track_consistent_categories),
  (std::vector<std::string>, categories_ordered),
  (std::vector<std::string>, track_only_categories),
  (std::vector<std::string>, track_only_classes),
  (Float2_t, track_conf_threshold),
  (float, tracklet_max_length),
  (Size2, individual_image_size),
  (uint32_t, categories_train_min_tracklet_length),
  (uint32_t, categories_apply_min_tracklet_length),
  (cmn::meta_encoding_t::Class, meta_encoding),
  (float, outline_compression),
  (bool, image_invert),
  (Frame_t, track_history_split_threshold)
)

//! Shorthand for defining slow settings cache entries:
#define DEF_SLOW_SETTINGS(X) inline static Settings:: X##_t X

//! Parameters that are only saved once per frame,
//! but have faster access than the settings cache.
//! Slower update, but faster access.
struct slow {
    DEF_SLOW_SETTINGS(frame_rate);
    DEF_SLOW_SETTINGS(track_max_speed);
    DEF_SLOW_SETTINGS(track_consistent_categories);
    DEF_SLOW_SETTINGS(cm_per_pixel);
    DEF_SLOW_SETTINGS(analysis_range);
    DEF_SLOW_SETTINGS(track_threshold);
    DEF_SLOW_SETTINGS(track_max_reassign_time);
    DEF_SLOW_SETTINGS(calculate_posture);
    DEF_SLOW_SETTINGS(track_threshold_is_absolute);
    DEF_SLOW_SETTINGS(track_background_subtraction);
    DEF_SLOW_SETTINGS(track_enforce_frame_rate);
    DEF_SLOW_SETTINGS(track_max_individuals);
    DEF_SLOW_SETTINGS(track_size_filter);
    DEF_SLOW_SETTINGS(match_mode);
    DEF_SLOW_SETTINGS(track_time_probability_enabled);
    DEF_SLOW_SETTINGS(track_speed_decay);
    DEF_SLOW_SETTINGS(match_min_probability);
    
    DEF_SLOW_SETTINGS(track_include);
    DEF_SLOW_SETTINGS(track_ignore);
    DEF_SLOW_SETTINGS(track_ignore_bdx);
    DEF_SLOW_SETTINGS(manual_matches);
    DEF_SLOW_SETTINGS(manual_splits);
    
    DEF_SLOW_SETTINGS(track_trusted_probability);
    DEF_SLOW_SETTINGS(tracklet_punish_timedelta);
    DEF_SLOW_SETTINGS(huge_timestamp_seconds);
    DEF_SLOW_SETTINGS(tracklet_punish_speeding);
    DEF_SLOW_SETTINGS(tracklet_max_length);
    
    DEF_SLOW_SETTINGS(posture_direction_smoothing);
};

#undef DEF_SLOW_SETTINGS

//! Fast updates, but slower access:
#define FAST_SETTING(NAME) (track::Settings::copy<track::Settings:: NAME>())

//#define DEBUG_TRACKING_THREADS
#if defined(DEBUG_TRACKING_THREADS)

// Global variable to hold the tracking thread's id.
inline std::shared_mutex tracking_thread_mutex;
inline std::unordered_set<std::thread::id> tracking_thread_ids;

// Call this at the start of your tracking thread.
inline void add_tracking_thread_id(std::thread::id id) {
    std::unique_lock guard(tracking_thread_mutex);
    tracking_thread_ids.insert(id);
}

inline void remove_tracking_thread_id(std::thread::id id) {
    std::unique_lock guard(tracking_thread_mutex);
    tracking_thread_ids.erase(id);
}

// Assert that the current thread is the tracking thread.
void assert_tracking_thread();

inline void clear_tracking_ids() {
    std::unique_lock guard(tracking_thread_mutex);
    tracking_thread_ids.clear();
}

struct TrackingThreadG {
    TrackingThreadG();
    ~TrackingThreadG();
};

// The debug version: wrap the thread check and the value access in a lambda.
// This ensures that the entire macro expands to an expression that can be used safely in all contexts.
#define SLOW_SETTING(NAME) ([&]() -> decltype(track::slow::NAME)& {                     \
    track::assert_tracking_thread();                   \
    return track::slow::NAME;                          \
}())

#else

// Release version: no thread check.
#define SLOW_SETTING(NAME) (track::slow:: NAME)

struct TrackingThreadG {};
inline void assert_tracking_thread() {}
inline void add_tracking_thread_id(std::thread::id) {}
inline void remove_tracking_thread_id(std::thread::id) {}
inline void clear_tracking_ids() {}

#endif

struct Clique {
    UnorderedVectorSet<pv::bid> bids;  // index of blob, not blob id
    UnorderedVectorSet<Idx_t> fishs; // index of fish
};


namespace Match {
using prob_t = double;
using Blob_t = pv::bid;
using Fish_t = Idx_t;
}

using Probability = Match::prob_t;
struct DetailProbability {
    Match::prob_t p, p_time, p_pos, p_angle;
    std::string toStr() const;
    static std::string class_name() { return "DetailProbability"; }
};

struct Statistics {
    float adding_seconds;
    float combined_posture_seconds;
    float number_fish;
    float loading_seconds;
    float posture_seconds;
    float match_number_fish;
    float match_number_blob;
    float match_number_edges;
    float match_stack_objects;
    float match_max_edges_per_blob;
    float match_max_edges_per_fish;
    float match_mean_edges_per_blob;
    float match_mean_edges_per_fish;
    float match_improvements_made;
    float match_leafs_visited;
    float method_used;
    
    Statistics() {
        std::fill((float*)this, (float*)this + sizeof(Statistics) / sizeof(float), infinity<float>());
    }
};

struct IDaverage {
    int64_t best_id;
    float p;
    uint32_t samples;

    std::string toStr() const {
        return "Pred<" + std::to_string(best_id) + ","+std::to_string(p) + ">";
    }
    static std::string class_name() { return "IDaverage"; }
};

std::map<Idx_t, float> prediction2map(const std::vector<float>& pred);

}
