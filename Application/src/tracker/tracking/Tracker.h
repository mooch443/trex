#pragma once

#include <types.h>
#include <misc/PVBlob.h>
#include "Individual.h"
#include <pv.h>
#include <misc/ThreadPool.h>
#include <tracking/StaticBackground.h>
#include <tracking/FOI.h>
#include <tracking/Border.h>
#include <misc/Timer.h>
#include <misc/BlobSizeRange.h>
#include <misc/idx_t.h>
#include <misc/create_struct.h>

namespace Output {
    class TrackingResults;
}

namespace mem { struct TrackerMemoryStats; }

namespace track {
    class Posture;
    class Recognition;
    class TrainingData;
    class FOI;
    struct fdx_t;
    struct SplitData;
    
    struct IndividualStatus {
        const PhysicalProperties* prev;
        const PhysicalProperties* current;
        
        IndividualStatus() : prev(nullptr), current(nullptr) {}
    };

using mmatches_t = std::map<long_t, std::map<Idx_t, int64_t>>;
using msplits_t = std::map<long_t, std::set<int64_t>>;
using inames_t = std::map<uint32_t, std::string>;
using mapproved_t = std::map<long_t,long_t>;
using analrange_t = std::pair<long_t,long_t>;

CREATE_STRUCT(Settings,
  (uint32_t, smooth_window),
  (float, cm_per_pixel),
  (int, frame_rate),
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
  (size_t, posture_direction_smoothing),
  (file::Path, tags_path),
  (std::set<Idx_t>, manual_identities),
  (std::vector<Vec2>, grid_points),
  (std::vector<std::vector<Vec2>>, recognition_shapes),
  (float, grid_points_scaling),
  (std::vector<std::vector<Vec2>>, track_ignore),
  (std::vector<std::vector<Vec2>>, track_include),
  (bool, huge_timestamp_ends_segment),
  (double, huge_timestamp_seconds),
  (mapproved_t, manually_approved),
  (size_t, pixel_grid_cells),
  (float, track_speed_decay),
  (bool, midline_invert),
  (bool, track_time_probability_enabled),
  (float, posture_head_percentage),
  (bool, enable_absolute_difference),
  (float, blobs_per_thread),
  (std::string, individual_prefix),
  (size_t, video_length),
  (analrange_t, analysis_range),
  (bool, recognition_enable),
  (float, visual_field_eye_offset),
  (float, visual_field_eye_separation),
  (bool, track_end_segment_for_speed),
  (default_config::matching_mode_t::Class, match_mode),
  (bool, track_do_history_split),
  (uint8_t, posture_closing_steps),
  (uint8_t, posture_closing_size),
  (float, recognition_image_scale),
  (bool, analysis_paused),
  (float, track_trusted_probability),
  (float, recognition_segment_add_factor)
)

    class Tracker {
    public:
        static Tracker* instance();
        using set_of_individuals_t = std::unordered_set<Individual*>;
        
    protected:
        friend class Output::TrackingResults;
        friend struct mem::TrackerMemoryStats;
        
        GETTER_NCONST(GenericThreadPool, thread_pool)
        GenericThreadPool recognition_pool;
        
        GETTER_NCONST(Border, border)
        
        std::vector<FrameProperties> _added_frames;
    public:
        const std::vector<FrameProperties>& frames() const { return _added_frames; }
    protected:
        std::shared_ptr<Image> _average;
        GETTER_SETTER(cv::Mat, mask)
        
        GETTER(std::atomic<float>, midline_errors_frame)
        uint32_t _current_midline_errors, _overall_midline_errors;
        
        //! All the individuals that have been detected and are being maintained
        std::unordered_map<Idx_t, Individual*> _individuals;
        friend class Individual;
        
        set_of_individuals_t _active_individuals;
        std::unordered_map<long_t, set_of_individuals_t> _active_individuals_frame;
        
        std::recursive_timed_mutex _lock;
        
        std::atomic_long _startFrame, _endFrame;
        uint64_t _max_individuals;
        //GETTER_PTR(LuminanceGrid*, grid)
        GETTER_PTR(StaticBackground*, background)
        Recognition* _recognition;
        
        std::unordered_map<Idx_t, Individual::segment_map::const_iterator> _individual_add_iterator_map;
        std::unordered_map<Idx_t, size_t> _segment_map_known_capacity;
        std::vector<IndividualStatus> _warn_individual_status;
        
    public:
        struct LockGuard {
            std::lock_guard<std::recursive_timed_mutex> *lock;
            std::string _purpose;
            Timer _timer;
            bool _set_name;
            bool locked() const { return lock != NULL; }
            
            ~LockGuard();
            LockGuard(std::string purpose, uint32_t timeout_ms = 0);
        };
        
        static std::string thread_name_holding();
        
    #define FAST_SETTINGS(NAME) track::Settings::copy<track::Settings:: NAME>()
        
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
        
        std::mutex _statistics_mutex;
        std::map<long_t, Statistics> _statistics;
        long_t _approximative_enabled_in_frame;
        
        GETTER(std::deque<Range<long_t>>, consecutive)
        std::set<Idx_t, std::function<bool(Idx_t,Idx_t)>> _inactive_individuals;
        
    public:
        Tracker();
        ~Tracker();
        
        void print_memory();
        
        /**
         * Adds a frame to the known frames.
         */
        void add(PPFrame& frame);
    private:
        void add(long_t frameIndex, PPFrame& frame);
        
    public:
        //! Removes all frames after given index
        void _remove_frames(long_t frameIndex);
        
        void set_average(const Image::Ptr& average) {
            //average.copyTo(_average);
            _average = average;
            /*if(_grid)
                delete _grid;
            
            if(SETTING(correct_luminance))
                _grid = new LuminanceGrid(_average);
            else
                _grid = NULL;*/
            
            _background = new StaticBackground(_average, nullptr);
        }
        static const Image& average() { if(!instance()->_average) U_EXCEPTION("Pointer to average image is nullptr."); return *instance()->_average; }
        
        //! Returns true if a frame with the given index exists.
        static const FrameProperties* properties(long_t frameIndex, const CacheHints* cache = nullptr);
        static double time_delta(long_t frame_1, long_t frame_2, const CacheHints* cache = nullptr) {
            auto props_1 = properties(frame_1, cache);
            auto props_2 = properties(frame_2, cache);
            return props_1 && props_2 ? abs(props_1->time - props_2->time) : (abs(frame_1 - frame_2) / double(FAST_SETTINGS(frame_rate)));
        }
        static const FrameProperties* add_next_frame(const FrameProperties&);
        static void clear_properties();
        
        static Frame_t start_frame() { return Frame_t(instance()->_startFrame.load()); }
        static Frame_t end_frame() { return Frame_t(instance()->_endFrame.load()); }
        static size_t number_frames() { return instance()->_added_frames.size(); }
        static bool blob_matches_shapes(const pv::BlobPtr&, const std::vector<std::vector<Vec2>>&);
        
        // filters a given frames blobs for size and splits them if necessary
        static void preprocess_frame(PPFrame &frame, const std::unordered_set<Individual*>& active_individuals, GenericThreadPool* pool, std::ostream* = NULL, bool do_history_split = true);
        
        friend class VisualField;
        static const std::unordered_map<Idx_t, Individual*>& individuals() {
            LockGuard guard("individuals()");
            return instance()->_individuals;
        }
        static const std::unordered_set<Individual*>& active_individuals() {
            LockGuard guard("active_individuals()");
            return instance()->_active_individuals;
        }
        static const std::unordered_set<Individual*>& active_individuals(long_t frame) {
            //LockGuard guard;
            
            if(instance()->_active_individuals_frame.count(frame))
                return instance()->_active_individuals_frame.at(frame);
            
            U_EXCEPTION("Frame out of bounds.");
        }
        static uint32_t overall_midline_errors() { return instance()->_overall_midline_errors; }
        static Rangel analysis_range() {
            const auto [start, end] = FAST_SETTINGS(analysis_range);
            const long_t video_length = (long_t)FAST_SETTINGS(video_length)-1;
            return Rangel(max(0, start), max(end > -1 ? min(video_length, end) : video_length, max(0, start)));
        }
        
        void update_history_log();
        
        long_t update_with_manual_matches(const std::map<long_t, std::map<Idx_t, int64_t>>& manual_matches);
        void check_segments_identities(bool auto_correct, std::function<void(float)> callback, const std::function<void(const std::string&, const std::function<void()>&, const std::string&)>& add_to_queue = [](auto,auto,auto){}, long_t after_frame = -1);
        void clear_segments_identities();
        void prepare_shutdown();
        void wait();
        
        static pv::BlobPtr find_blob_noisy(std::map<uint32_t, pv::BlobPtr>& blob_to_id, int64_t bid, int64_t pid, const Bounds& bounds, long_t frame);
        
        //static bool generate_training_images(pv::File&, std::map<long_t, std::set<long_t>> individuals_per_frame, TrainingData&, const std::function<void(float)>& = [](float){}, const TrainingData* source = nullptr);
        //static bool generate_training_images(pv::File&, const std::set<long_t>& frames, TrainingData&, const std::function<void(float)>& = [](float){});
        //static bool generate_training_images(pv::File&, const Rangel& range, TrainingData&, const std::function<void(float)>& = [](float){});
        
        static Recognition* recognition();
        static std::vector<Rangel> global_segment_order();
        static void global_segment_order_changed();
        static void auto_calculate_parameters(pv::File& video, bool quiet = false);
        static void emergency_finish();
        static void delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range);
        
        enum class AnalysisState {
            PAUSED,
            UNPAUSED
        };
        static void analysis_state(AnalysisState);
        
    protected:
        friend class track::Posture;
        static void increase_midline_errors() {
            ++instance()->_current_midline_errors;
            ++instance()->_overall_midline_errors;
        }
        
        void update_consecutive(const set_of_individuals_t& active, long_t frameIndex, bool update_dataset = false);
        void update_warnings(long_t frameIndex, double time, long_t number_fish, long_t n_found, long_t n_prev, const FrameProperties *props, const FrameProperties *prev_props, const set_of_individuals_t& active_individuals, std::unordered_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
        
    private:
        static void filter_blobs(PPFrame& frame, GenericThreadPool *pool);
        static std::map<uint32_t, pv::BlobPtr> fill_proximity_grid(cmn::grid::ProximityGrid&, const std::vector<pv::BlobPtr>& blobs);
        void history_split(PPFrame& frame, const std::unordered_set<Individual*>& active_individuals, std::ostream* out = NULL, GenericThreadPool* pool = NULL);
        
        struct split_expectation {
            size_t number;
            bool allow_less_than;
            
            split_expectation(size_t number = 0, bool allow_less_than = false)
                : number(number), allow_less_than(allow_less_than)
            { }
            
            operator MetaObject () const {
                return MetaObject("{"+std::to_string(number)+","+(allow_less_than ? "true" : "false")+"}", "split_expectation");
            }
            static std::string class_name() {
                return "split_expectation";
            }
        };
        
        //static void changed_setting(const sprite::Map&, const std::string& key, const sprite::PropertyType& value);
        size_t found_individuals_frame(size_t frameIndex) const;
        void generate_pairdistances(long_t frameIndex);
        void check_save_tags(long_t frameIndex, const std::unordered_map<uint32_t, Individual*>&, const std::vector<tags::blob_pixel>&, const std::vector<tags::blob_pixel>&, const file::Path&);
        
        Individual* create_individual(Idx_t ID, set_of_individuals_t& active_individuals);
        
        struct PrefilterBlobs {
            std::vector<pv::BlobPtr> filtered;
            std::vector<pv::BlobPtr> filtered_out;
            std::vector<pv::BlobPtr> big_blobs;
            //std::vector<pv::BlobPtr> additional;
            
            long_t frame_index;
            BlobSizeRange fish_size;
            const Background* background;
            int threshold;
            
            PrefilterBlobs(long_t index, int threshold, const BlobSizeRange& fish_size, const Background& background)
            : frame_index(index), fish_size(fish_size), background(&background), threshold(threshold)
            {
                
            }
        };
        
        std::vector<pv::BlobPtr> split_big(std::vector<pv::BlobPtr>& filtered_out, const std::vector<std::shared_ptr<pv::Blob>>& big_blobs, const std::map<pv::BlobPtr, split_expectation> &expect, bool discard_small = false, std::ostream *out = NULL, GenericThreadPool* pool = nullptr);
        
        static void prefilter(std::shared_ptr<PrefilterBlobs>, std::vector<pv::BlobPtr>::const_iterator it, std::vector<pv::BlobPtr>::const_iterator end);
        
        void update_iterator_maps(long_t frame, const set_of_individuals_t& active_individuals, std::unordered_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
    };
}

STRUCT_META_EXTENSIONS(track::Settings)
