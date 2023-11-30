#pragma once

#include <commons.pc.h>
#include <misc/GlobalSettings.h>
#include <file/Path.h>
#include <tracker/misc/default_config.h>
#include <misc/idx_t.h>
#include <misc/bid.h>
#include <misc/frame_t.h>
#include <misc/create_struct.h>
#include <misc/BlobSizeRange.h>

namespace track {
using namespace cmn;
class Individual;

using mmatches_t = std::map<Frame_t, std::map<Idx_t, pv::bid>>;
using msplits_t = std::map<Frame_t, std::set<pv::bid>>;
using inames_t = std::map<uint32_t, std::string>;
using mapproved_t = std::map<long_t,long_t>;
using analrange_t = std::pair<long_t,long_t>;

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

struct AssignInfo {
    PPFrame* frame;
    const FrameProperties* f_prop;
    const FrameProperties* f_prev_prop;
    default_config::matching_mode_t::Class match_mode;
};

//! A global settings cache used across the application by
//! calling FAST_SETTING(name).
CREATE_STRUCT(Settings,
  (uint32_t, smooth_window),
  (float, cm_per_pixel),
  (uint32_t, frame_rate),
  (float, track_max_reassign_time),
  (float, speed_extrapolation),
  (bool, calculate_posture),
  (float, track_max_speed),
  (bool, debug),
  (BlobSizeRange, blob_size_ranges),
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
  (uint32_t, midline_resolution),
  (uint64_t, midline_samples),
  (float, meta_mass_mg),
  (inames_t, individual_names),
  (float, midline_stiff_percentage),
  (float, matching_probability_threshold),
  (uint16_t, posture_direction_smoothing),
  (file::Path, tags_path),
  (std::vector<Vec2>, grid_points),
  (std::vector<std::vector<Vec2>>, recognition_shapes),
  (float, grid_points_scaling),
  (std::vector<std::vector<Vec2>>, track_ignore),
  (std::vector<std::vector<Vec2>>, track_include),
  (bool, huge_timestamp_ends_segment),
  (double, huge_timestamp_seconds),
  (mapproved_t, manually_approved),
  (float, track_speed_decay),
  (bool, midline_invert),
  (bool, track_time_probability_enabled),
  (float, posture_head_percentage),
  (bool, track_absolute_difference),
  (bool, track_background_subtraction),
  (float, blobs_per_thread),
  (std::string, individual_prefix),
  (uint64_t, video_length),
  (analrange_t, analysis_range),
  (float, visual_field_eye_offset),
  (float, visual_field_eye_separation),
  (uint8_t, visual_field_history_smoothing),
  (bool, track_end_segment_for_speed),
  (default_config::matching_mode_t::Class, match_mode),
  (bool, track_do_history_split),
  (uint8_t, posture_closing_steps),
  (uint8_t, posture_closing_size),
  (float, individual_image_scale),
  (bool, analysis_paused),
  (float, track_trusted_probability),
  (float, recognition_segment_add_factor),
  (bool, output_interpolate_positions),
  (bool, track_consistent_categories),
  (std::vector<std::string>, categories_ordered),
  (std::vector<std::string>, track_only_categories),
  (std::vector<std::string>, track_only_labels),
  (float, track_label_confidence_threshold),
  (float, track_segment_max_length),
  (Size2, individual_image_size),
  (uint32_t, categories_min_sample_images)
)

//! Shorthand for defining slow settings cache entries:
#define DEF_SLOW_SETTINGS(X) inline static Settings:: X##_t X

//! Parameters that are only saved once per frame,
//! but have faster access than the settings cache.
//! Slower update, but faster access.
struct slow {
    DEF_SLOW_SETTINGS(frame_rate);
    DEF_SLOW_SETTINGS(track_max_speed);
    DEF_SLOW_SETTINGS(cm_per_pixel);
    DEF_SLOW_SETTINGS(analysis_range);
    DEF_SLOW_SETTINGS(track_threshold);
    DEF_SLOW_SETTINGS(track_max_reassign_time);
    DEF_SLOW_SETTINGS(calculate_posture);
    DEF_SLOW_SETTINGS(track_absolute_difference);
    DEF_SLOW_SETTINGS(track_background_subtraction);
    
    DEF_SLOW_SETTINGS(track_trusted_probability);
    DEF_SLOW_SETTINGS(huge_timestamp_ends_segment);
    DEF_SLOW_SETTINGS(huge_timestamp_seconds);
    DEF_SLOW_SETTINGS(track_end_segment_for_speed);
    DEF_SLOW_SETTINGS(track_segment_max_length);
};

#undef DEF_SLOW_SETTINGS

//! Fast updates, but slower access:
#define FAST_SETTING(NAME) (track::Settings::copy<track::Settings:: NAME>())
//! Slow updated, but faster access:
#define SLOW_SETTING(NAME) (track::slow:: NAME)

void initialize_slows();

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

}
