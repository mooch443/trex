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
#include <tracker/misc/default_config.h>

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
        const MotionRecord* prev;
        const MotionRecord* current;
        
        IndividualStatus() : prev(nullptr), current(nullptr) {}
    };

using mmatches_t = std::map<Frame_t, std::map<Idx_t, pv::bid>>;
using msplits_t = std::map<Frame_t, std::set<pv::bid>>;
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
  (uint16_t, posture_direction_smoothing),
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
  (float, track_speed_decay),
  (bool, midline_invert),
  (bool, track_time_probability_enabled),
  (float, posture_head_percentage),
  (bool, enable_absolute_difference),
  (float, blobs_per_thread),
  (std::string, individual_prefix),
  (uint64_t, video_length),
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
  (float, recognition_segment_add_factor),
  (bool, output_interpolate_positions),
  (bool, track_consistent_categories),
  (std::vector<std::string>, categories_ordered),
  (std::vector<std::string>, track_only_categories),
  (float, track_segment_max_length),
  (Size2, recognition_image_size),
  (uint32_t, categories_min_sample_images)
)

    class Tracker {
    public:
        static Tracker* instance();
        //using set_of_individuals_t = ska::bytell_hash_set<Individual*>;
        using set_of_individuals_t = std::unordered_set<Individual*>;

        std::map<Idx_t, pv::bid> automatically_assigned(Frame_t frame);
        
    protected:
        friend class Output::TrackingResults;
        friend struct mem::TrackerMemoryStats;
        
        GETTER_NCONST(GenericThreadPool, thread_pool)
        GenericThreadPool recognition_pool;
        
        GETTER_NCONST(Border, border)
        
        std::vector<std::unique_ptr<FrameProperties>> _added_frames;
    public:
        const std::vector<std::unique_ptr<FrameProperties>>& frames() const { return _added_frames; }
    protected:
        Image::Ptr _average;
        GETTER_SETTER(cv::Mat, mask)
        
        GETTER(std::atomic<float>, midline_errors_frame)
        uint32_t _current_midline_errors, _overall_midline_errors;
        
        //! All the individuals that have been detected and are being maintained
        ska::bytell_hash_map<Idx_t, Individual*> _individuals;
        friend class Individual;
        
    public:
        struct Clique {
            UnorderedVectorSet<pv::bid> bids;  // index of blob, not blob id
            UnorderedVectorSet<Idx_t> fishs; // index of fish
        };
        ska::bytell_hash_map<Frame_t, std::vector<Clique>> _cliques;
        
        set_of_individuals_t _active_individuals;
        //using active_individuals_t = ska::bytell_hash_map<Frame_t, set_of_individuals_t>;
        using active_individuals_t = std::unordered_map<Frame_t, set_of_individuals_t>;
        active_individuals_t _active_individuals_frame;
        
        std::atomic<Frame_t> _startFrame{ Frame_t() };
        std::atomic<Frame_t> _endFrame{ Frame_t() };
        uint64_t _max_individuals;
        //GETTER_PTR(LuminanceGrid*, grid)
        GETTER_PTR(StaticBackground*, background)
        Recognition* _recognition;
        
        ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator> _individual_add_iterator_map;
        ska::bytell_hash_map<Idx_t, size_t> _segment_map_known_capacity;
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
        std::map<Frame_t, Statistics> _statistics;
        Frame_t _approximative_enabled_in_frame;
        
        GETTER(std::deque<Range<Frame_t>>, consecutive)
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
        void add(Frame_t frameIndex, PPFrame& frame);
        
    public:
        //! Removes all frames after given index
        void _remove_frames(Frame_t frameIndex);
        
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
        static const Image& average() { if(!instance()->_average) throw U_EXCEPTION("Pointer to average image is nullptr."); return *instance()->_average; }
        
        
        static decltype(_added_frames)::const_iterator properties_iterator(Frame_t frameIndex);
        static const FrameProperties* properties(Frame_t frameIndex, const CacheHints* cache = nullptr);
        static double time_delta(Frame_t frame_1, Frame_t frame_2, const CacheHints* cache = nullptr) {
            auto props_1 = properties(frame_1, cache);
            auto props_2 = properties(frame_2, cache);
            return props_1 && props_2 ? abs(props_1->time - props_2->time) : (abs((frame_1 - frame_2).get()) / double(FAST_SETTINGS(frame_rate)));
        }
        static const FrameProperties* add_next_frame(const FrameProperties&);
        static void clear_properties();
        
        static Frame_t start_frame() { return instance()->_startFrame.load(); }
        static Frame_t end_frame() { return instance()->_endFrame.load(); }
        static size_t number_frames() { return instance()->_added_frames.size(); }
        static bool blob_matches_shapes(const pv::BlobPtr&, const std::vector<std::vector<Vec2>>&);
        
        // filters a given frames blobs for size and splits them if necessary
        static void preprocess_frame(PPFrame &frame, const Tracker::set_of_individuals_t& active_individuals, GenericThreadPool* pool, std::ostream* = NULL, bool do_history_split = true);
        
        friend class VisualField;
        static const ska::bytell_hash_map<Idx_t, Individual*>& individuals() {
            //LockGuard guard("individuals()");
            return instance()->_individuals;
        }
        static const set_of_individuals_t& active_individuals() {
            //LockGuard guard("active_individuals()");
            return instance()->_active_individuals;
        }
        static const set_of_individuals_t& active_individuals(Frame_t frame) {
            //LockGuard guard;
            
            if(instance()->_active_individuals_frame.count(frame))
                return instance()->_active_individuals_frame.at(frame);
            
            throw U_EXCEPTION("Frame out of bounds.");
        }
        static uint32_t overall_midline_errors() { return instance()->_overall_midline_errors; }
        static Range<Frame_t> analysis_range() {
            const auto [start, end] = FAST_SETTINGS(analysis_range);
            const long_t video_length = narrow_cast<long_t>(FAST_SETTINGS(video_length))-1;
            return Range<Frame_t>(Frame_t(max(0, start)), Frame_t(max(end > -1 ? min(video_length, end) : video_length, max(0, start))));
        }
        
        void update_history_log();
        
        Frame_t update_with_manual_matches(const Settings::manual_matches_t& manual_matches);
        void check_segments_identities(bool auto_correct, std::function<void(float)> callback, const std::function<void(const std::string&, const std::function<void()>&, const std::string&)>& add_to_queue = [](auto,auto,auto){}, Frame_t after_frame = {});
        void clear_segments_identities();
        void prepare_shutdown();
        void wait();
        
        static pv::BlobPtr find_blob_noisy(const PPFrame& frame, pv::bid bid, pv::bid pid, const Bounds& bounds);
        
        //static bool generate_training_images(pv::File&, std::map<long_t, std::set<long_t>> individuals_per_frame, TrainingData&, const std::function<void(float)>& = [](float){}, const TrainingData* source = nullptr);
        //static bool generate_training_images(pv::File&, const std::set<long_t>& frames, TrainingData&, const std::function<void(float)>& = [](float){});
        //static bool generate_training_images(pv::File&, const Rangel& range, TrainingData&, const std::function<void(float)>& = [](float){});
        
        static Recognition* recognition();
        static std::vector<Range<Frame_t>> global_segment_order();
        static void global_segment_order_changed();
        static void auto_calculate_parameters(pv::File& video, bool quiet = false);
        static void emergency_finish();
        static void delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range);
        
        static Match::PairedProbabilities calculate_paired_probabilities
                (const PPFrame& frame,
                 const Tracker::set_of_individuals_t& active_individuals,
                 const ska::bytell_hash_map<Individual*, bool>& fish_assigned,
                 const ska::bytell_hash_map<pv::Blob*, bool>& blob_assigned,
                 //std::unordered_map<pv::Blob*, pv::BlobPtr>& ptr2ptr,
                 GenericThreadPool* pool);
        static std::vector<Clique> generate_cliques(const Match::PairedProbabilities& paired);
        
        enum class AnalysisState {
            PAUSED = 0,
            UNPAUSED
        };
        static void analysis_state(AnalysisState);
        
    protected:
        friend class track::Posture;
        static void increase_midline_errors() {
            ++instance()->_current_midline_errors;
            ++instance()->_overall_midline_errors;
        }
        
        void update_consecutive(const set_of_individuals_t& active, Frame_t frameIndex, bool update_dataset = false);
        void update_warnings(Frame_t frameIndex, double time, long_t number_fish, long_t n_found, long_t n_prev, const FrameProperties *props, const FrameProperties *prev_props, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
        
    private:
        static void filter_blobs(PPFrame& frame, GenericThreadPool *pool);
        void history_split(PPFrame& frame, const Tracker::set_of_individuals_t& active_individuals, std::ostream* out = NULL, GenericThreadPool* pool = NULL);
        
        struct split_expectation {
            size_t number;
            bool allow_less_than;
            
            split_expectation(size_t number = 0, bool allow_less_than = false)
                : number(number), allow_less_than(allow_less_than)
            { }
            
            std::string toStr() const {
                return "{"+std::to_string(number)+","+(allow_less_than ? "true" : "false")+"}";
            }
            static std::string class_name() {
                return "split_expectation";
            }
        };
        
        //static void changed_setting(const sprite::Map&, const std::string& key, const sprite::PropertyType& value);
        size_t found_individuals_frame(Frame_t frameIndex) const;
        void generate_pairdistances(Frame_t frameIndex);
        void check_save_tags(Frame_t frameIndex, const ska::bytell_hash_map<pv::bid, Individual*>&, const std::vector<tags::blob_pixel>&, const std::vector<tags::blob_pixel>&, const file::Path&);
        
        Individual* create_individual(Idx_t ID, set_of_individuals_t& active_individuals);
        
        struct PrefilterBlobs {
            std::vector<pv::BlobPtr> filtered;
            std::vector<pv::BlobPtr> filtered_out;
            std::vector<pv::BlobPtr> big_blobs;
            //std::vector<pv::BlobPtr> additional;
            
            Frame_t frame_index;
            BlobSizeRange fish_size;
            const Background* background;
            int threshold;
            
            size_t overall_pixels = 0;
            size_t samples = 0;
            
            PrefilterBlobs(Frame_t index, int threshold, const BlobSizeRange& fish_size, const Background& background)
            : frame_index(index), fish_size(fish_size), background(&background), threshold(threshold)
            {
                
            }
            
            void commit(const pv::BlobPtr& b) {
                overall_pixels += b->num_pixels();
                ++samples;
                filtered.push_back(b);
            }
            
            void commit(const std::vector<pv::BlobPtr>& v) {
                for(auto &b:v)
                    overall_pixels += b->num_pixels();
                samples += v.size();
                filtered.insert(filtered.end(), v.begin(), v.end());
            }
            
            void filter_out(const pv::BlobPtr& b) {
                overall_pixels += b->num_pixels();
                ++samples;
                filtered_out.push_back(b);
            }
            
            void filter_out(const std::vector<pv::BlobPtr>& v) {
                for(auto &b:v)
                    overall_pixels += b->num_pixels();
                samples += v.size();
                filtered_out.insert(filtered_out.end(), v.begin(), v.end());
            }
        };
        
        struct BlobReceiver {
            const enum PPFrameType {
                noise,
                regular,
                none
            } _type = none;
            
            std::vector<pv::BlobPtr>* _base = nullptr;
            PPFrame* _frame = nullptr;
            Tracker::PrefilterBlobs *_prefilter = nullptr;
            
            BlobReceiver(Tracker::PrefilterBlobs& prefilter, PPFrameType type)
                : _type(type), _prefilter(&prefilter)
            { }
            
            BlobReceiver(PPFrame& frame, PPFrameType type)
                : _type(type), _frame(&frame)
            { }
            
            BlobReceiver(std::vector<pv::BlobPtr>& base)
                : _base(&base)
            { }
            
            void operator()(std::vector<pv::BlobPtr>&& v) const {
                if(_base) {
                    _base->insert(_base->end(), std::make_move_iterator(v.begin()), std::make_move_iterator(v.end()));
                } else if(_prefilter) {
                    switch(_type) {
                        case noise:
                            _prefilter->filter_out(std::move(v));
                            break;
                        case regular:
                            _prefilter->commit(std::move(v));
                            break;
                        case none:
                            break;
                    }
                } else {
                    switch(_type) {
                        case noise:
                            _frame->add_noise(std::move(v));
                            break;
                        case regular:
                            _frame->add_regular(std::move(v));
                            break;
                        case none:
                            break;
                    }
                }
            }
            
            void operator()(const pv::BlobPtr& b) const {
                if(_base) {
                    _base->insert(_base->end(), b);
                } else if(_prefilter) {
                    switch(_type) {
                        case noise:
                            _prefilter->filter_out(b);
                            break;
                        case regular:
                            _prefilter->commit(b);
                            break;
                        case none:
                            break;
                    }
                } else {
                    switch(_type) {
                        case noise:
                            _frame->add_noise(b);
                            break;
                        case regular:
                            _frame->add_regular(b);
                            break;
                        case none:
                            break;
                    }
                }
            }
        };
        
        std::vector<pv::BlobPtr> split_big(const BlobReceiver&, const std::vector<std::shared_ptr<pv::Blob>>& big_blobs, const robin_hood::unordered_map<pv::Blob*, split_expectation> &expect, bool discard_small = false, std::ostream *out = NULL, GenericThreadPool* pool = nullptr);
        
        static void prefilter(const std::shared_ptr<PrefilterBlobs>&, std::vector<pv::BlobPtr>::const_iterator it, std::vector<pv::BlobPtr>::const_iterator end);
        
        void update_iterator_maps(Frame_t frame, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
    };
}

STRUCT_META_EXTENSIONS(track::Settings)
