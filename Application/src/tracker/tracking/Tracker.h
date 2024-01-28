#pragma once

#include <commons.pc.h>
#include <misc/PVBlob.h>
#include "Individual.h"
#include <pv.h>
#include <misc/ThreadPool.h>
#include <processing/Background.h>
#include <misc/Border.h>
#include <misc/Timer.h>
#include <misc/BlobSizeRange.h>
#include <misc/idx_t.h>
#include <misc/create_struct.h>
#include <tracker/misc/default_config.h>
#include <misc/TrackingSettings.h>
#include <tracking/BlobReceiver.h>
#include <tracking/LockGuard.h>

namespace Output {
    class TrackingResults;
}

namespace mem { struct TrackerMemoryStats; }

namespace track {

class Posture;
class TrainingData;
class FOI;
struct SplitData;

struct IndividualStatus {
    const MotionRecord* prev;
    const MotionRecord* current;
    
    IndividualStatus() : prev(nullptr), current(nullptr) {}
};



class Tracker {
public:
    static Tracker* instance();

protected:
    friend class Output::TrackingResults;
    friend struct mem::TrackerMemoryStats;
    
    GenericThreadPool _thread_pool;
public:
    static GenericThreadPool& thread_pool() { return instance()->_thread_pool; }
    
protected:
    GETTER_NCONST(Border, border);
    
protected:
    mutable std::shared_mutex _vi_mutex;
    ska::bytell_hash_map<Frame_t, ska::bytell_hash_map<pv::bid, std::vector<float>>> _vi_predictions;
public:
    void transform_vi_predictions(auto&& fn) const {
        std::shared_lock g(_vi_mutex);
        for(auto const& [frame, pred] : _vi_predictions) {
            if constexpr(Predicate<decltype(fn), Frame_t, decltype(pred)>) {
                if(not fn(frame, pred))
                    break;
            } else {
                fn(frame, pred);
            }
        }
    }
    /*auto vi_predictions() const {
        return _vi_predictions;
    }*/
    size_t number_vi_predictions() const {
        std::shared_lock g(_vi_mutex);
        return _vi_predictions.size();
    }
    bool has_vi_predictions() const {
        std::shared_lock g(_vi_mutex);
        return !_vi_predictions.empty();
    }
    
    void set_vi_data(const decltype(_vi_predictions)& predictions);
    void predicted(Frame_t, pv::bid, std::span<float>);
    const std::vector<float>& get_prediction(Frame_t, pv::bid) const;
    const std::vector<float>* find_prediction(Frame_t, pv::bid) const;
    
protected:
    std::vector<FrameProperties::Ptr> _added_frames;
public:
    const std::vector<FrameProperties::Ptr>& frames() const { return _added_frames; }
protected:
    CallbackCollection _callback;
    Image::Ptr _average;
    GETTER_SETTER(cv::Mat, mask);
    
    //! All the individuals that have been detected and are being maintained
    friend class Individual;
    
public:
    ska::bytell_hash_map<Frame_t, std::vector<Clique>> _cliques;
    
    //set_of_individuals_t _active_individuals;
    //active_individuals_map_t _active_individuals_frame;
    
    std::atomic<Frame_t> _startFrame{ Frame_t() };
    std::atomic<Frame_t> _endFrame{ Frame_t() };
    
    std::atomic<uint64_t> _max_individuals;
    
public:
    static uint64_t max_individuals() { return instance()->_max_individuals.load(); }
    
protected:
    Background* _background{nullptr};
public:
    static Background* background() { return instance()->_background; }
    
    ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator> _individual_add_iterator_map;
    ska::bytell_hash_map<Idx_t, size_t> _segment_map_known_capacity;
    std::vector<IndividualStatus> _warn_individual_status;
    
public:
    using stats_map_t = std::map<Frame_t, Statistics>;
    GETTER(stats_map_t, statistics);
    
    struct SecondsPerFrame {
        double _seconds_per_frame, _frames_sampled;
        void add(double seconds, double num_individuals) {
            if(num_individuals > 0) {
                _seconds_per_frame += seconds / num_individuals;
                _frames_sampled++;
            }
        }
    };
    
private:
    inline static SecondsPerFrame _time_samples;
    GenericThreadPool recognition_pool;
    
public:
    static double average_seconds_per_individual();
    
    GETTER(std::deque<Range<Frame_t>>, consecutive);
    //std::set<Idx_t, std::function<bool(Idx_t,Idx_t)>> _inactive_individuals;
    
public:
    Tracker(Image::Ptr&& average, float meta_real_width);
    Tracker(Image::Ptr&& average, const pv::File& file);
    ~Tracker();
    
    /**
     * Adds a frame to the known frames.
     */
    void add(PPFrame& frame);
    
    //! Removes all frames after given index
    void _remove_frames(Frame_t frameIndex);
    
private:
    void add(Frame_t frameIndex, PPFrame& frame);
    
    CallbackManagerImpl<void> _delete_frame_callbacks;
    static inline std::atomic<bool> _segment_order_changed{false};
    
public:
    /**
     * Registers a new callback function and returns its unique ID.
     * The callback will be invoked when certain events or changes occur.
     *
     * @param callback The callback function to register.
     * @return A unique identifier for the registered callback.
     */
    std::size_t register_delete_callback(const std::function<void()>& callback) {
        return _delete_frame_callbacks.registerCallback(callback);
    }

    /**
     * Unregisters (removes) a previously registered callback using its unique ID.
     *
     * @param id The unique identifier of the callback to unregister.
     */
    void unregister_delete_callback(std::size_t id) {
        _delete_frame_callbacks.unregisterCallback(id);
    }
    
    void set_average(Image::Ptr&& average) {
        _average = std::move(average);
        _background = new Background(Image::Make(*_average), nullptr);
    }
    static const Image& average(cmn::source_location loc = cmn::source_location::current()) {
        if(!instance()->_average)
            throw _U_EXCEPTION(loc, "Pointer to average image is nullptr.");
        return *instance()->_average;
    }
    
    
    static decltype(_added_frames)::const_iterator properties_iterator(Frame_t frameIndex);
    static const FrameProperties* properties(Frame_t frameIndex, const CacheHints* cache = nullptr);
    static double time_delta(Frame_t frame_1, Frame_t frame_2, const CacheHints* cache = nullptr);
    static const FrameProperties* add_next_frame(const FrameProperties&);
    static void clear_properties();
    
    //! returns an ordered set of Idx_t for all individuals that exist
    static const std::set<Idx_t> identities();
    
    //! returns true only if track_max_individuals > 0
    static bool has_identities();
    
    static Frame_t start_frame() { return instance()->_startFrame.load(); }
    static Frame_t end_frame() { return instance()->_endFrame.load(); }
    static size_t number_frames() { return instance()->_added_frames.size(); }
    
    // filters a given frames blobs for size and splits them if necessary
    static void preprocess_frame(pv::Frame&&, PPFrame &frame, GenericThreadPool* pool, PPFrame::NeedGrid, const Size2& resolution, bool do_history_split = true);
    
    friend class VisualField;
    
    static const set_of_individuals_t& active_individuals(Frame_t frame);
    
    static const FrameRange& analysis_range();
    
    void update_history_log();
    
    Frame_t update_with_manual_matches(const Settings::manual_matches_t& manual_matches);

    static std::vector<Range<Frame_t>> global_segment_order();
    static void global_segment_order_changed();
    static std::vector<Range<Frame_t>> unsafe_global_segment_order();

    void check_segments_identities(bool auto_correct, IdentitySource, std::function<void(float)> callback, const std::function<void(const std::string&, const std::function<void()>&, const std::string&)>& add_to_queue = [](auto,auto,auto){}, Frame_t after_frame = {});
    void clear_segments_identities();

    void prepare_shutdown();
    
    static pv::BlobPtr find_blob_noisy(const PPFrame& frame, pv::bid bid, pv::bid pid, const Bounds& bounds);
    
    
    static void auto_calculate_parameters(pv::File& video, bool quiet = false);
    static void emergency_finish();
    
    //static std::vector<Clique> generate_cliques(const Match::PairedProbabilities& paired);
    
    enum class AnalysisState {
        PAUSED = 0,
        UNPAUSED
    };
    static void analysis_state(AnalysisState);
    
protected:
    friend class track::Posture;
    
    void update_consecutive(const set_of_individuals_t& active, Frame_t frameIndex, bool update_dataset = false);
    void update_warnings(Frame_t frameIndex, double time, long_t number_fish, long_t n_found, long_t n_prev, const FrameProperties *props, const FrameProperties *prev_props, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
    
private:
    static void filter_blobs(PPFrame& frame, GenericThreadPool *pool);
    
    //static void changed_setting(const sprite::Map&, const std::string& key, const sprite::PropertyType& value);
    size_t found_individuals_frame(Frame_t frameIndex) const;
    void generate_pairdistances(Frame_t frameIndex);
    void check_save_tags(Frame_t frameIndex, const ska::bytell_hash_map<pv::bid, Individual*>&, const std::vector<tags::blob_pixel>&, const std::vector<tags::blob_pixel>&, const file::Path&);
    
    friend struct TrackingHelper;
    //static Individual* create_individual(Idx_t ID, set_of_individuals_t& active_individuals);
    
public:
    static std::vector<pv::BlobPtr> split_big(const BlobReceiver&, const std::vector<pv::BlobPtr>& big_blobs, const robin_hood::unordered_map<pv::bid, split_expectation> &expect, bool discard_small = false, std::ostream *out = NULL, GenericThreadPool* pool = nullptr);
    
private:
    static void prefilter(
        PrefilterBlobs&,
          std::move_iterator<std::vector<pv::BlobPtr>::iterator> it,
          std::move_iterator<std::vector<pv::BlobPtr>::iterator> end);
    
    void update_iterator_maps(Frame_t frame, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
    void collect_matching_cliques(TrackingHelper& s, GenericThreadPool& thread_pool);
    static Match::PairedProbabilities calculate_paired_probabilities
     (
        const TrackingHelper& s,
        GenericThreadPool* pool
      );
    
public:
    void print_memory();
};
}

STRUCT_META_EXTENSIONS(track::Settings)
