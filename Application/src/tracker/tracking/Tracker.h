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
#include <tracking/TrackingSettings.h>
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
    GETTER_NCONST(Border, border)
    
protected:
    ska::bytell_hash_map<Frame_t, ska::bytell_hash_map<pv::bid, std::vector<float>>> _vi_predictions;
public:
    const auto& vi_predictions() const {
        return _vi_predictions;
    }
    bool has_vi_predictions() const {
        return !_vi_predictions.empty();
    }
    
    void set_vi_data(const decltype(_vi_predictions)& predictions);
    void predicted(Frame_t, pv::bid, std::vector<float>&&);
    const std::vector<float>& get_prediction(Frame_t, pv::bid) const;
    const std::vector<float>* find_prediction(Frame_t, pv::bid) const;
    static std::map<Idx_t, float> prediction2map(const std::vector<float>& pred);
    
protected:
    std::vector<FrameProperties::Ptr> _added_frames;
public:
    const std::vector<FrameProperties::Ptr>& frames() const { return _added_frames; }
protected:
    Image::Ptr _average;
    GETTER_SETTER(cv::Mat, mask)
    
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
    active_individuals_map_t _active_individuals_frame;
    
    std::atomic<Frame_t> _startFrame{ Frame_t() };
    std::atomic<Frame_t> _endFrame{ Frame_t() };
    
    std::atomic<uint64_t> _max_individuals;
    
public:
    static uint64_t max_individuals() { return instance()->_max_individuals.load(); }
    
protected:
    StaticBackground* _background{nullptr};
public:
    static StaticBackground* background() { return instance()->_background; }
    
    ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator> _individual_add_iterator_map;
    ska::bytell_hash_map<Idx_t, size_t> _segment_map_known_capacity;
    std::vector<IndividualStatus> _warn_individual_status;
    
public:
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
    
    
    std::map<Frame_t, Statistics> _statistics;
    
    struct SecondsPerFrame {
        double _seconds_per_frame, _frames_sampled;
        void add(double seconds, double num_individuals) {
            _seconds_per_frame += seconds / num_individuals;
            _frames_sampled++;
        }
    };
    
private:
    inline static SecondsPerFrame _time_samples;
    
public:
    static double average_seconds_per_individual();
    
    GETTER(std::deque<Range<Frame_t>>, consecutive)
    std::set<Idx_t, std::function<bool(Idx_t,Idx_t)>> _inactive_individuals;
    
public:
    Tracker();
    ~Tracker();
    
    /**
     * Adds a frame to the known frames.
     */
    void add(PPFrame& frame);
    
    //! Removes all frames after given index
    void _remove_frames(Frame_t frameIndex);
    
private:
    void add(Frame_t frameIndex, PPFrame& frame);
    
public:
    void set_average(const Image::Ptr& average) {
        _average = average;
        _background = new StaticBackground(_average, nullptr);
    }
    static const Image& average(cmn::source_location loc = cmn::source_location::current()) {
        if(!instance()->_average)
            throw U_EXCEPTION<FormatterType::UNIX, const char*>("Pointer to average image is nullptr.", loc);
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
    static bool blob_matches_shapes(const pv::BlobPtr&, const std::vector<std::vector<Vec2>>&);
    
    // filters a given frames blobs for size and splits them if necessary
    static void preprocess_frame(PPFrame &frame, const set_of_individuals_t& active_individuals, GenericThreadPool* pool, bool do_history_split = true);
    
    friend class VisualField;
    static const ska::bytell_hash_map<Idx_t, Individual*>& individuals();
    static const set_of_individuals_t& active_individuals();
    static const set_of_individuals_t& active_individuals(Frame_t frame);
    
    static const Range<Frame_t>& analysis_range();
    
    void update_history_log();
    
    Frame_t update_with_manual_matches(const Settings::manual_matches_t& manual_matches);

    

    static std::vector<Range<Frame_t>> global_segment_order();
    static void global_segment_order_changed();

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
    static Individual* create_individual(Idx_t ID, set_of_individuals_t& active_individuals);
    
public:
    static std::vector<pv::BlobPtr> split_big(const BlobReceiver&, const std::vector<pv::BlobPtr>& big_blobs, const robin_hood::unordered_map<pv::bid, split_expectation> &expect, bool discard_small = false, std::ostream *out = NULL, GenericThreadPool* pool = nullptr);
    
private:
    static void prefilter(const std::shared_ptr<PrefilterBlobs>&, std::vector<pv::BlobPtr>::const_iterator it, std::vector<pv::BlobPtr>::const_iterator end);
    
    void update_iterator_maps(Frame_t frame, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators);
    
public:
    void print_memory();
};
}

STRUCT_META_EXTENSIONS(track::Settings)
