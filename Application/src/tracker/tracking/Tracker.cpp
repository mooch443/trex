#include "Tracker.h"
#include <misc/GlobalSettings.h>
#include <tracking/SplitBlob.h>
#include <misc/Timer.h>
#include "PairingGraph.h"
#include <tracking/OutputLibrary.h>
#include <tracking/DetectTag.h>
#include <misc/cnpy_wrapper.h>
#include <processing/CPULabeling.h>
#include <misc/ReverseAdapter.h>
#include <misc/default_settings.h>
#include <misc/pretty.h>
#include <tracking/DatasetQuality.h>
#include <misc/PixelTree.h>
#include <misc/CircularGraph.h>
#include <tracking/MemoryStats.h>
#include <tracking/Categorize.h>
#include <tracking/VisualField.h>
#include <file/DataLocation.h>
#include <misc/FOI.h>

#include <tracking/TrackingHelper.h>
#include <tracking/AutomaticMatches.h>
#include <tracking/HistorySplit.h>
#include <tracking/IndividualManager.h>
#include <misc/SettingsInitializer.h>

#if !COMMONS_NO_PYTHON
#include <misc/PythonWrapper.h>
#include <tracking/RecTask.h>
#include <tracking/Accumulation.h>
namespace py = Python;
#endif

#ifndef NDEBUG
//#define PAIRING_PRINT_STATS
#endif
//#define TREX_DEBUG_IDENTITIES

namespace track {

std::mutex _statistics_mutex;

FrameRange _analysis_range;

void initialize_slows() {
#define DEF_CALLBACK(X) Settings::set_callback(Settings:: X , [](auto&, auto& value) { SLOW_SETTING( X ) = value.template value<Settings:: X##_t >(); })
    
    static std::once_flag flag;
    std::call_once(flag, [](){
        Settings::init();
        
        DEF_CALLBACK(frame_rate);
        DEF_CALLBACK(track_max_speed);
        DEF_CALLBACK(track_consistent_categories);
        DEF_CALLBACK(cm_per_pixel);
        //DEF_CALLBACK(analysis_range);
        DEF_CALLBACK(track_threshold);
        DEF_CALLBACK(track_max_reassign_time);
        DEF_CALLBACK(calculate_posture);
        DEF_CALLBACK(track_absolute_difference);
        DEF_CALLBACK(track_background_subtraction);
        
        DEF_CALLBACK(track_trusted_probability);
        DEF_CALLBACK(huge_timestamp_ends_segment);
        DEF_CALLBACK(huge_timestamp_seconds);
        DEF_CALLBACK(track_end_segment_for_speed);
        DEF_CALLBACK(track_segment_max_length);
        
        static const auto update_range = [](){
            const auto video_length = narrow_cast<long_t>(SETTING(video_length).value<Settings::video_length_t>())-1;
            //auto analysis_range = FAST_SETTING(analysis_range);
            const auto analysis_range = SETTING(analysis_range).value<Settings::analysis_range_t>();
            const auto& [start, end] = analysis_range;
            
            _analysis_range = FrameRange(Range<Frame_t>{
                Frame_t((uint32_t)max(0, start)),
                Frame_t((uint32_t)max(end > -1
                            ? min(video_length, end)
                            : video_length, max(0, start)))
            });
            
            assert(_analysis_range.start().valid()
                   && _analysis_range.end().valid());
        };
        
        Settings::set_callback(Settings::analysis_range, [](auto&, auto& value) {
            SLOW_SETTING(analysis_range) = value.template value<Settings::analysis_range_t>();
            update_range();
        });
        
        Settings::set_callback(Settings::video_length, [](auto&, auto&) {
            update_range();
        });
        
        for(auto &n : Settings :: names())
            Settings::variable_changed(sprite::Map::Signal::NONE, cmn::GlobalSettings::map(), n, cmn::GlobalSettings::get(n).get());
    });
}

const set_of_individuals_t& Tracker::active_individuals(Frame_t frame) {
    //assert(LockGuard::owns_read());
    auto result = IndividualManager::active_individuals(frame);
    if(result)
        return *result.value();
    
    throw U_EXCEPTION("active_individuals(",frame,") reporting: ", result.error());
}

const FrameRange& Tracker::analysis_range() {
    return _analysis_range;
}

Tracker* _instance = NULL;
std::vector<Range<Frame_t>> _global_segment_order;

std::mutex _identities_mutex;
std::set<Idx_t> _fixed_identities;

Tracker* Tracker::instance() {
    return _instance;
}
    
void Tracker::predicted(Frame_t frame, pv::bid bdx, std::span<float> ps) {
    std::unique_lock g(_vi_mutex);
    auto &ff = _vi_predictions[frame];
#ifndef NDEBUG
    if(ff.count(bdx)) {
        FormatWarning("bdx ", bdx, " already in predictions (forgot to clear?).");
    }
#endif
    ff[bdx] = std::vector<float>{ps.data(), ps.data() + ps.size()};
}

const std::vector<float>& Tracker::get_prediction(Frame_t frame, pv::bid bdx) const {
    auto ptr = find_prediction(frame, bdx);
    if(!ptr)
        throw U_EXCEPTION("Assumed that prediction in ", frame, " for ", bdx, " exists. It does not.");
    return *ptr;
}

const std::vector<float>* Tracker::find_prediction(Frame_t frame, pv::bid bdx) const
{
    if(not frame.valid() || not bdx.valid())
        return nullptr;
    
    std::shared_lock g(_vi_mutex);
    auto it = _vi_predictions.find(frame);
    if(it == _vi_predictions.end())
        return nullptr;
    
    auto kit = it->second.find(bdx);
    if(kit == it->second.end())
        return nullptr;
    
    return &kit->second;
}

auto& properties_cache() {
    static CacheHints _properties_cache;
    return _properties_cache;
}


auto& properties_mutex() {
    static std::shared_mutex _properties_mutex;
    return _properties_mutex;
}

double Tracker::time_delta(Frame_t frame_1, Frame_t frame_2, const CacheHints* cache) {
    auto props_1 = properties(frame_1, cache);
    auto props_2 = properties(frame_2, cache);
    return props_1 && props_2 ? abs(props_1->time - props_2->time) : (abs((frame_1 - frame_2).get()) / double(FAST_SETTING(frame_rate)));
}

const FrameProperties* Tracker::properties(Frame_t frameIndex, const CacheHints* hints) {
    if(!frameIndex.valid())
        return nullptr;
    
    if(hints) {
        //! check if its just meant to disable it
        if(hints != (const CacheHints*)0x1) {
            auto ptr = hints->properties(frameIndex);
            if(ptr)
                return ptr;
        }
        
    } else {
        std::shared_lock guard(properties_mutex());
        auto ptr = properties_cache().properties(frameIndex);
        if(ptr)
            return ptr;
    }
    
    auto &frames = instance()->frames();
    auto it = properties_iterator(frameIndex);
    if(it == frames.end())
        return nullptr;
    return (*it).get();
}

decltype(Tracker::_added_frames)::const_iterator Tracker::properties_iterator(Frame_t frameIndex) {
    auto &frames = instance()->frames();
    
    auto it = std::upper_bound(frames.begin(), frames.end(), frameIndex, [](Frame_t frame, const auto& prop) -> bool {
        return frame < prop->frame;
    });
    
    if((it == frames.end() && !frames.empty()) || (it != frames.begin())) {
        --it;
        
        if((*it)->frame == frameIndex) {
            return it;
        }
    }
    
    return frames.end();
}
        
    void Tracker::print_memory() {
        LockGuard guard(ro_t{}, "print_memory");
        mem::TrackerMemoryStats stats;
        stats.print();
    }
    
void Tracker::analysis_state(AnalysisState pause) {
    if(!instance())
        throw U_EXCEPTION("No tracker instance can be used to pause.");

    std::packaged_task<void(bool)> task([](bool value) {
        SETTING(analysis_paused) = value;
    });
    
    std::thread tmp(std::move(task), pause == AnalysisState::PAUSED);
    tmp.detach();
}

Tracker::Tracker(Image::Ptr&& average, const pv::File& video)
    : Tracker(std::move(average), settings::infer_meta_real_width_from(video))
{ }

Tracker::Tracker(Image::Ptr&& average, float meta_real_width)
      : _thread_pool(max(1u, cmn::hardware_concurrency()), "Tracker::thread_pool"),
        _max_individuals(0),
        _background(NULL),
        _border{ nullptr },
        recognition_pool(max(1u, 5u), "RecognitionPool")
        /*_inactive_individuals([this](Idx_t A, Idx_t B){
            auto it = _individuals.find(A);
            assert(it != _individuals.end());
            const Individual* a = it->second;
            
            it = _individuals.find(B);
            assert(it != _individuals.end());
            const Individual* b = it->second;
    
            return a->end_frame() > b->end_frame() || (a->end_frame() == b->end_frame() && A > B);
        })*/
{
    Identity::Reset(); // reset Identities if the tracker is created
    initialize_slows();
    
    _instance = this;
    set_average(std::move(average));
    
    if(not GlobalSettings::has("meta_real_width")
       || SETTING(meta_real_width).value<float>() == 0)
    {
        SETTING(meta_real_width) = meta_real_width;
    }
    
    // setting cm_per_pixel after average has been generated (and offsets have been set)
    if(!GlobalSettings::has("cm_per_pixel")
       || SETTING(cm_per_pixel).value<float>() == 0)
    {
        SETTING(cm_per_pixel) = settings::infer_cm_per_pixel();
    }
    
    if(GlobalSettings::is_runtime_quiet())
    {
        print("Initialized with ", _thread_pool.num_threads()," threads.");
    }
    
    Settings::set_callback(Settings::outline_resample, [](auto&, auto&value){
        static_assert(std::is_same<Settings::outline_resample_t, float>::value, "outline_resample assumed to be float.");
        auto v = value.template value<float>();
        if(v <= 0) {
            print("outline_resample defaulting to 1.0 instead of ",v);
            SETTING(outline_resample) = 1.f;
        }
    });
    Settings::set_callback(Settings::manually_approved, [](auto&, auto&){
        DatasetQuality::update();
    });
    
    auto track_list_update = [](std::string_view key, auto&value)
    {
        auto update = [key = std::string(key), tmp = value.template value<Settings::track_ignore_t>()]() mutable
        {
            bool changed = false;
            for(auto &vec : tmp) {
                if(vec.size() > 2) {
                    auto ptr = poly_convex_hull(&vec);
                    if(ptr) {
                        if(vec != *ptr) {
                            vec = *ptr;
                            changed = true;
                        }
                    }
                }
            }
            
            if(changed) {
                GlobalSettings::get(key) = tmp;
            }
        };
        
        update();
    };
    Settings::set_callback(Settings::track_ignore, track_list_update);
    Settings::set_callback(Settings::track_include, track_list_update);
    Settings::set_callback(Settings::frame_rate, [](auto&, auto&){
        std::unique_lock guard(properties_mutex());
        properties_cache().clear(); //! TODO: need to refill as well
    });
    Settings::set_callback(Settings::posture_direction_smoothing, [](auto&key, auto&value) {
        static_assert(std::is_same<Settings::posture_direction_smoothing_t, uint16_t>::value, "posture_direction_smoothing assumed to be uint16_t.");
        size_t v = value.template value<uint16_t>();
        
        if(v != FAST_SETTING(posture_direction_smoothing))
        {
            auto worker = [key](){
                {
                    LockGuard guard(w_t{}, "Updating midlines in changed_setting("+std::string(key)+")");
                    IndividualManager::transform_parallel(Tracker::instance()->thread_pool(), [](auto fdx, auto fish)
                    {
                        print("\t", fdx);
                        fish->clear_post_processing();
                        fish->update_midlines(nullptr);
                    });
                }
                DatasetQuality::update();
            };
            
            /*if(GUI::instance()) {
                GUI::work().add_queue("updating midlines / head positions...", worker);
            } else*/
                worker();
        }
    });
    
    if (not _callback) {
        auto callback = [](std::string_view name){
            //LockGuard guard(ro_t{}, "changed_settings");
            Settings :: variable_changed(sprite::Map::Signal::NONE, GlobalSettings::map(), name, cmn::GlobalSettings::map().at(name).get());
        };
        _callback = cmn::GlobalSettings::map().register_callbacks(Settings::names(), callback);
        GlobalSettings::map().register_shutdown_callback([this](auto) {
            _callback.reset();
        });
        
        for(auto n : Settings::names())
            callback(n);
    }
}

Tracker::~Tracker() {
    assert(_instance);
    //Settings::clear_callbacks();
    
#if !COMMONS_NO_PYTHON
    Accumulation::on_terminate();
    RecTask::deinit();
#endif
    Individual::shutdown();
    
    _thread_pool.force_stop();
    
    const bool quiet = GlobalSettings::is_runtime_quiet();
    if(!quiet)
        print("Waiting for recognition...");
    recognition_pool.force_stop();
    if(!quiet)
        print("Done waiting.");
    
    _instance = NULL;
    
    {
        std::unique_lock g(_identities_mutex);
        _fixed_identities.clear();
    }
    _global_segment_order.clear();
    //_analysis_range = {};
    
    IndividualManager::clear();
    emergency_finish();

    if (_callback)
        cmn::GlobalSettings::map().unregister_callbacks(std::move(_callback));
}

void Tracker::emergency_finish() {
    PPFrame::CloseLogs();
}

void Tracker::prepare_shutdown() {
    _thread_pool.force_stop();
    recognition_pool.force_stop();
    Match::PairingGraph::prepare_shutdown();
#if !COMMONS_NO_PYTHON
    Accumulation::on_terminate();
    try {
        py::deinit().get();
    } catch(...) {
        FormatWarning("Exception during py::deinit().");
    }
#endif
    IndividualManager::clear();
}

Frame_t Tracker::update_with_manual_matches(const Settings::manual_matches_t& manual_matches) {
    LockGuard guard(ro_t{}, "update_with_manual_matches");
    
    static std::atomic_bool first_run(true);
    static Settings::manual_matches_t compare = manual_matches;
    if(first_run) {
        first_run = false;
        //auto str = Meta::toStr(compare);
        //SETTING(manual_matches) = manual_matches;
        return {};
    }
    
    //auto str0 = Meta::toStr(compare), str1 = Meta::toStr(manual_matches);
    auto copy = manual_matches; // another copy
    auto next = copy;
    
    // iterate over old to find frames that are not in the current version
    Frame_t first_change;
    
    auto itn = compare.begin(), ito = copy.begin();
    for (; itn != compare.end() && ito != copy.end(); ++itn, ++ito) {
        if(itn->first != ito->first || itn->second != ito->second) {
            first_change = Frame_t(min(itn->first, ito->first));
            break;
        }
    }
    
    // if one of the iterators reached the end, but the other one didnt
    if((itn == compare.end()) ^ (ito == copy.end())) {
        if(itn == compare.end())
            first_change = Frame_t(ito->first);
        else
            first_change = Frame_t(itn->first);
    }
    
    if(first_change.valid() && (not Tracker::end_frame().valid()
                                || first_change <= Tracker::end_frame()))
    {
        Tracker::analysis_state(Tracker::AnalysisState::UNPAUSED);
    }
    
    //SETTING(manual_matches) = next;
    //FAST_SETTING(manual_matches) = next;
    //auto str = Meta::toStr(FAST_SETTING(manual_matches));
    compare = next;
    
    return first_change;
}

bool operator<(Frame_t frame, const FrameProperties& props) {
    return frame < props.frame;
}

//! Assumes a sorted array.
template<typename T, typename Q>
inline bool contains_sorted(const Q& v, T obj) {
    auto it = std::lower_bound(v.begin(), v.end(), obj, [](const auto& v, T number) -> bool {
        return *v < number;
    });
    
    if(it != v.end()) {
        auto end = std::upper_bound(it, v.end(), obj, [](T number, const auto& v) -> bool {
            return number < *v;
        });
        
        if(end == v.end() || !(*(*end) < obj)) {
            return true;
        }
    }
    
    return false;
}

void Tracker::add(PPFrame &frame) {
    static Timing timing("Tracker::add(PPFrame)", 100);
    TakeTiming take(timing);
    
    LockGuard guard(w_t{}, "Tracker::add(PPFrame)");
    
    assert(frame.index().valid());
    initialize_slows();
    
    if (contains_sorted(_added_frames, frame.index())) {
        print("Frame ",frame.index()," already in tracker.");
        return;
    }
    
    if(frame.timestamp > uint64_t(INT64_MAX)) {
        print("frame timestamp is bigger than INT64_MAX! (",frame.timestamp," time)");
    }
    
    auto props = properties(frame.index() - 1_f);
    if(props && frame.timestamp < props->org_timestamp.get()) {
        FormatError("Cannot add frame with timestamp smaller than previous timestamp. Frames have to be in order. Skipping.");
        return;
    }
    
    if(start_frame().valid() && frame.index() < end_frame() + 1_f)
        throw UtilsException("Cannot add intermediate frames out of order.");
    
    add(frame.index(), frame);
}

double Tracker::average_seconds_per_individual() {
    std::lock_guard<std::mutex> lguard(_statistics_mutex);
    if(_time_samples._frames_sampled == 0)
        return 0;
    return _time_samples._seconds_per_frame / _time_samples._frames_sampled;
}

class PairProbability {
private:
    GETTER_PTR(Individual*, idx);
    GETTER_PTR(pv::bid, bdx);
    GETTER(Match::prob_t, p);
    
public:
    PairProbability() = default;
    PairProbability(Individual* idx, pv::bid bdx, Match::prob_t p)
        : _idx(idx), _bdx(bdx), _p(p)
    {}
    
    bool operator<(const PairProbability& other) const {
        return std::make_tuple(_p, _idx->identity().ID(), _bdx) < std::make_tuple(other._p, other._idx->identity().ID(), other._bdx);
    }
    bool operator>(const PairProbability& other) const {
        return std::make_tuple(_p, _idx->identity().ID(), _bdx) > std::make_tuple(other._p, other._idx->identity().ID(), other._bdx);
    }
    bool operator<=(const PairProbability& other) const {
        return std::make_tuple(_p, _idx->identity().ID(), _bdx) <= std::make_tuple(other._p, other._idx->identity().ID(), other._bdx);
    }
    bool operator>=(const PairProbability& other) const {
        return std::make_tuple(_p, _idx->identity().ID(), _bdx) >= std::make_tuple(other._p, other._idx->identity().ID(), other._bdx);
    }
    bool operator==(const PairProbability& other) const {
        return std::make_tuple(_p, _idx->identity().ID(), _bdx) == std::make_tuple(other._p, other._idx->identity().ID(), other._bdx);
    }
};

void Tracker::update_history_log() {
    LockGuard guard(ro_t{}, "update_history_log");
    PPFrame::UpdateLogs();
}

void Tracker::preprocess_frame(pv::Frame&& frame, PPFrame& pp, GenericThreadPool* pool, PPFrame::NeedGrid need, const Size2& resolution, bool do_history_split)
{
    double time = double(frame.timestamp()) / double(1000*1000);
    
    //! Free old memory
    pp.clear();
    
    pp.time = time;
    assert(frame.index().valid());
    pp.set_index(frame.index());
    pp.set_source_index(frame.source_index());
    pp.timestamp = frame.timestamp();
    pp.set_loading_time(frame.loading_time());
    pp.init_from_blobs(std::move(frame).steal_blobs());
    if(SETTING(image_invert)) {
        pp.transform_all([](pv::Blob& blob){
            auto &pixels = blob.pixels();
            for(auto &p : *pixels)
                p = 255 - p;
        });
    }
    
    frame.clear();
    
    filter_blobs(pp, pool);
    pp.fill_proximity_grid(resolution);
    
    if(do_history_split)
        HistorySplit{pp, need, pool};
    
    //! discarding frame...
    //frame.clear();
}

void Tracker::prefilter(
        PrefilterBlobs& result,
        std::move_iterator<std::vector<pv::BlobPtr>::iterator> it,
        std::move_iterator<std::vector<pv::BlobPtr>::iterator> end)
{
    static Timing timing("prefilter", 100);
    TakeTiming take(timing);
    
    const float cm_sqr = SQR(SLOW_SETTING(cm_per_pixel));
    
    const auto track_include = FAST_SETTING(track_include);
    const auto track_ignore = FAST_SETTING(track_ignore);
    const auto manual_ignore_bdx = FAST_SETTING(manual_ignore_bdx);
    
    std::vector<pv::BlobPtr> ptrs;
    auto track_only_categories = FAST_SETTING(track_only_categories);
    auto track_label_confidence_threshold = FAST_SETTING(track_label_confidence_threshold);
    const auto detect_classes = GlobalSettings::has("detect_classes")
                        ? SETTING(detect_classes).value<std::vector<std::string>>()
                        : std::vector<std::string>{};
    [[maybe_unused]] auto get_class_name = [&detect_classes](size_t i) -> std::string {
        if(i < detect_classes.size())
            return detect_classes[i];
        return Meta::toStr(i);
    };
    auto get_class_id = [&detect_classes](const std::string& name)
        -> std::optional<uint8_t>
    {
        for(size_t i=0, N=detect_classes.size(); i<N; ++i) {
            if(detect_classes[i] == name)
                return narrow_cast<uint8_t>(i);
        }
        return std::nullopt;
    };
    
    const auto track_only_classes = FAST_SETTING(track_only_classes);
    std::unordered_set<uint8_t> only_classes;
    for(auto &name : track_only_classes) {
        auto id = get_class_id(name);
        if(id)
            only_classes.insert(id.value());
    }
    
    auto is_class_ignored = [&only_classes](uint8_t clid)
        -> bool
    {
        if(only_classes.contains(clid))
            return true;
        return false;
    };
    
    const auto tags_dont_track = SETTING(tags_dont_track).value<bool>();
    const auto track_only_segmentations = SETTING(track_only_segmentations).value<bool>();
    
    auto check_precise_not_ignored = [&track_ignore, &track_include, &result](pv::BlobPtr&& b){
        if (!track_ignore.empty()) {
            if (PrefilterBlobs::blob_matches_shapes(*b, track_ignore)) {
                result.filter_out(std::move(b), FilterReason::InsideIgnore);
                return false;
            }
        }

        if (!track_include.empty()) {
            if (!PrefilterBlobs::blob_matches_shapes(*b, track_include)) {
                result.filter_out(std::move(b), FilterReason::OutsideInclude);
                return false;
            }
        }
        
        return true;
    };
    
    std::optional<const decltype(manual_ignore_bdx)::mapped_type*> manual_ignore_bdx_c;
    if(auto it = manual_ignore_bdx.find(result.frame_index);
       it != manual_ignore_bdx.end())
    {
        manual_ignore_bdx_c = &it->second;
    }
    
    auto check_blob = [&track_only_segmentations, &tags_dont_track, &check_precise_not_ignored, &track_include, &result, &cm_sqr, &manual_ignore_bdx_c](pv::BlobPtr&& b, bool precise_check_boundaries)
    {
        // TODO: magic numbers
        if(b->pixels()->size() * cm_sqr > result.fish_size.max_range().end * 100)
            b->force_set_recount(result.threshold);
        else
            b->recount(result.threshold, *result.background);
        
        if(manual_ignore_bdx_c.has_value()) {
            if(manual_ignore_bdx_c.value()->contains(b->blob_id())
               || (b->parent_id().valid() && manual_ignore_bdx_c.value()->contains(b->parent_id())))
            {
                result.filter_out(std::move(b), FilterReason::Unknown);
                return false;
            }
        }
        
        if(b->is_tag() && tags_dont_track) {
            result.filter_out(std::move(b), FilterReason::DontTrackTags);
            return false;
        }
        
        if(not b->is_instance_segmentation() and track_only_segmentations) {
            result.filter_out(std::move(b), FilterReason::OnlySegmentations);
            return false;
        }
        
        //! use imprecise boundary check
        if(not precise_check_boundaries) {
            if (!track_include.empty()) {
                if (not PrefilterBlobs::rect_overlaps_shapes(b->bounds(), track_include)) {
                    result.filter_out(std::move(b), FilterReason::OutsideInclude);
                    return false;
                }
            }
            return true;
        }
        
        return check_precise_not_ignored(std::move(b));
    };

    for(; it != end; ++it) {
        auto &&own = *it;
        if(!own) {
            ++it;
            continue;
        }
        
        //! check if this blob is of valid size.
        //! if it is NOT of valid size, it will
        //! be moved to one of the other arrays.
        if(!check_blob(std::move(own), false))
            continue;
        
        // it has NOT been moved, continue here...
        float recount = own->recount(-1);
        ptrs.clear();
        
        //! If the size is appropriately big, try to split the blob using the minimum of threshold and
        //  posture_threshold. Using the minimum ensures that the thresholds dont depend on each other
        //  as the threshold used here will reduce the number of available pixels for posture analysis
        //  or tracking respectively (pixels below used threshold will be removed).
        if(result.fish_size.close_to_minimum_of_one(recount, 0.5)
           && result.threshold > 0
           )
        {
            auto pblobs = pixel::threshold_blob(result.cache, own.get(), result.threshold, result.background);
            
            // only use blobs that split at least into 2 new blobs
            for(auto &&add : pblobs) {
                add->set_split(false, own); // set_split even if the blob has just been thresholded normally?
                if(!check_blob(std::move(add), true))
                    continue;

                ptrs.push_back(std::move(add));
            }
        }
        
        // if we havent found any blobs, add the unthresholded
        // blob instead:
        if (ptrs.empty()) {
            if(check_precise_not_ignored(std::move(own)))
                ptrs.push_back(std::move(own));
            else
                continue;
        }
        
        //! actually add the blob(s) to the filtered/filtered_out arrays
        for(auto&& ptr : ptrs) {
            recount = ptr->recount(-1);

            if(result.fish_size.in_range_of_one(recount)) {
                if(FAST_SETTING(track_threshold_2) > 0) {
                    auto second_count = ptr->recount(FAST_SETTING(track_threshold_2), *result.background);
                    
                    ptr->force_set_recount(result.threshold, recount / cm_sqr);
                    
                    if(!(FAST_SETTING(threshold_ratio_range) * recount).contains(second_count)) {
                        result.filter_out(std::move(ptr), FilterReason::SecondThreshold);
                        continue;
                    }
                }
                
#if !COMMONS_NO_PYTHON
                if(!track_only_categories.empty()) {
                    auto ldx = Categorize::DataStore::_ranged_label_unsafe(Frame_t(result.frame_index), ptr->blob_id());
                    if(ldx == -1 || !contains(track_only_categories, Categorize::DataStore::label(ldx)->name)) {
                        result.filter_out(std::move(ptr), FilterReason::Category);
                        continue;
                    }
                }
#endif
                //! TODO: translate track_only_classes to IDs and check those...
                if(not track_only_classes.empty()) {
                    if(ptr->prediction().valid()) {
                        auto clid = ptr->prediction().clid;
                        if(is_class_ignored(clid)) {
                            result.filter_out(std::move(ptr), FilterReason::Label);
                            continue;
                        }
                    } else {
                        result.filter_out(std::move(ptr), FilterReason::Label);
                        continue;
                    }
                }
                
                if(ptr->prediction().valid()) {
                    if(float(ptr->prediction().p) / 255.f < track_label_confidence_threshold) {
                        //! TODO: use own filter reason
                        result.filter_out(std::move(ptr), FilterReason::LabelConfidenceThreshold);
                        continue;
                    }
                }
                
                //! only after all the checks passed, do we commit the blob
                /// to the "filtered" array:
                result.commit(std::move(ptr));
                
            } else if(recount < result.fish_size.max_range().start) {
                result.filter_out(std::move(ptr), FilterReason::OutsideRange);
            } else
                result.big_blob(std::move(ptr));
        }
        
#ifndef NDEBUG
        //! all pointers should have been moved, check if thats true
        for(auto &b : ptrs) {
            assert(b == nullptr);
        }
#endif
    }
    
    for(auto &blob : result.filtered())
        blob->calculate_moments();
    
    if (not Tracker::start_frame().valid()
        || result.frame_index == Tracker::start_frame())
    {
#if !COMMONS_NO_PYTHON
        std::vector<pv::BlobPtr> noises;
#endif
        PrefilterBlobs::split_big(
              std::move(result.big_blobs),
              BlobReceiver(result, BlobReceiver::noise, nullptr, FilterReason::SplitFailed),
              BlobReceiver(result, BlobReceiver::regular,
#if !COMMONS_NO_PYTHON
                  [&](auto& blob) {
                      if(track_only_categories.empty())
                          return false;
                      
                      auto ldx = Categorize::DataStore::_ranged_label_unsafe(Frame_t(result.frame_index), blob->blob_id());
                      if (ldx == -1 || !contains(track_only_categories, Categorize::DataStore::label(ldx)->name))
                      {
                          noises.push_back(std::move(blob));
                          return true;
                      }
                      
                      return false;
                  }),
#endif
              {});
        
        result.big_blobs.clear();
#if !COMMONS_NO_PYTHON
        result.filter_out(std::move(noises), FilterReason::SplitFailed);
#endif
    }
}

void Tracker::filter_blobs(PPFrame& frame, GenericThreadPool *pool) {
    static Timing timing("filter_blobs", 100);
    TakeTiming take(timing);
    
    const BlobSizeRange fish_size = FAST_SETTING(blob_size_ranges);
    const uint32_t num_blobs = (uint32_t)frame.N_blobs();
    const int threshold = FAST_SETTING(track_threshold);
    
    size_t available_threads = 1 + (pool ? pool->num_threads() : 0);
    size_t maximal_threads = num_blobs;
    size_t needed_threads = min(maximal_threads / (size_t)FAST_SETTING(blobs_per_thread), available_threads);
    std::shared_lock guard(Categorize::DataStore::range_mutex());
    
    if (maximal_threads > 1
        && needed_threads > 1
        && available_threads > 1
        && pool)
    {
        std::vector<PrefilterBlobs> prefilters;
        std::mutex lock;
        
        prefilters.reserve(needed_threads + 1);
        for(size_t i=0; i<needed_threads + 1; ++i) {
            prefilters.emplace_back(frame.index(), threshold, fish_size, *Tracker::background());
        }
        
        PrefilterBlobs global(frame.index(), threshold, fish_size, *Tracker::background());
        
        distribute_indexes([&](auto, auto start, auto end, auto j){
            //auto result = std::make_shared<PrefilterBlobs>(frame.index(), threshold, fish_size, *Tracker::instance()->_background);
            {
#ifndef NDEBUG
                std::lock_guard guard(lock);
                assert(prefilters.size() > j);
#endif
                //prefilters.at(j) = result;
            }
            
            prefilter(prefilters.at(j), start, end);
            
            std::unique_lock guard(lock);
            std::move(prefilters.at(j)).to(global);
            
        }, *pool, std::make_move_iterator(frame.unsafe_access_all_blobs().begin()), std::make_move_iterator(frame.unsafe_access_all_blobs().end()), (uint32_t)needed_threads);
        
        frame.clear_blobs();
        std::move(global).to(frame);
        
    } else {
        PrefilterBlobs pref(frame.index(), threshold, fish_size, *Tracker::instance()->_background);
        prefilter(pref, std::make_move_iterator(frame.unsafe_access_all_blobs().begin()), std::make_move_iterator(frame.unsafe_access_all_blobs().end()));
        
        frame.clear_blobs();
        std::move(pref).to(frame);
    }
}

const FrameProperties* Tracker::add_next_frame(const FrameProperties & props) {
    auto &frames = instance()->frames();
    auto capacity = frames.capacity();
    instance()->_added_frames.emplace_back(FrameProperties::Make(props));
    
    if(frames.capacity() != capacity) {
        std::unique_lock guard(properties_mutex());
        properties_cache().clear();
        
        auto it = frames.rbegin();
        while(it != frames.rend() && !properties_cache().full())
        {
            properties_cache().push((*it)->frame, (*it).get());
            ++it;
        }
        assert((frames.empty() && !end_frame().valid()) || (end_frame().valid() && (*frames.rbegin())->frame == end_frame()));
        
    } else {
        std::unique_lock guard(properties_mutex());
        properties_cache().push(props.frame, frames.back().get());
    }
    
    return frames.back().get();
}

bool Tracker::has_identities() {
    return FAST_SETTING(track_max_individuals) > 0;
}

const std::set<Idx_t> Tracker::identities() {
    if(!has_identities())
        return {};
    
    //std::set<Idx_t> set;
    //std::mutex mutex;
    
    std::unique_lock guard(_identities_mutex);
    if(_fixed_identities.empty())
    {
        //LockGuard guard("Tracker::identities");
        //for(auto &[id, fish] : Tracker::individuals())
        //    set.insert(id);
        
        //if(set.empty()) {
            for(Idx_t i = Idx_t(0); i < Idx_t(FAST_SETTING(track_max_individuals)); i = i + Idx_t(1)) {
                _fixed_identities.insert(i);
            }
        //}
    }
    
    return _fixed_identities;
}

void Tracker::clear_properties() {
    std::unique_lock guard(properties_mutex());
    properties_cache().clear();
}

Match::PairedProbabilities Tracker::calculate_paired_probabilities
 (
    const TrackingHelper& s,
    GenericThreadPool* pool
 )
{
    // now that the blobs array has been cleared of all the blobs for fixed matches,
    // get pairings for all the others:
    //static std::unordered_map<Individual*, Match::prob_t> max_probs;
    //std::unordered_map<const pv::Blob*, std::unordered_map<Individual*, Match::prob_t>> paired_blobs;
    //max_probs.clear();
    Match::PairedProbabilities paired_blobs;
    std::mutex paired_mutex;
    //std::condition_variable is_paired_free;
    auto frameIndex = s.frame.index();
    
    using namespace default_config;
    
    {
        using namespace Match;
        
        static Timing probs("Tracker::paired", 30);
        TakeTiming take(probs);
        
        // see how many are missing
        std::vector<Individual*> unassigned_individuals;
        unassigned_individuals.reserve(s._manager.current().size());
        
        s._manager.transform_active([&](Individual* fish) {
            if(not s._manager.fish_assigned(fish->identity().ID()))
                unassigned_individuals.push_back(fish);
        });
        
        // Create Individuals for unassigned blobs
        std::vector<int> blob_labels(s.frame.N_blobs());
        std::vector<pv::bid> bdxes(blob_labels.size());
        std::vector<pv::BlobWeakPtr> ptrs(blob_labels.size());
        std::vector<size_t> unassigned_blobs;
        unassigned_blobs.reserve(blob_labels.size());
        //std::vector<std::tuple<const pv::BlobPtr*, int>> unassigned_blobs;
        //unassigned_blobs.reserve(frame.blobs().size());
        
#if !COMMONS_NO_PYTHON
        const bool enable_labels = FAST_SETTING(track_consistent_categories) || !FAST_SETTING(track_only_categories).empty();
        if(enable_labels) {
            s.frame.transform_blobs([&](size_t i, pv::Blob& blob) {
                auto bdx = blob.blob_id();
                bdxes[i] = bdx;
                
                if(!s._manager.blob_assigned(bdx)) {
                    auto label = Categorize::DataStore::ranged_label(Frame_t(frameIndex), bdx);
                    blob_labels[i] = label ? label->id : -1;
                    ptrs[i] = &blob;
                    unassigned_blobs.push_back(i);
                } else {
                    blob_labels[i] = -1;
                    ptrs[i] = nullptr;
                }
            });
            
        } else {
#endif
            s.frame.transform_blobs([&](size_t i, pv::Blob& blob) {
                auto bdx = blob.blob_id();
                if(!s._manager.blob_assigned(bdx)) {
                    unassigned_blobs.push_back(i);
                    ptrs[i] = &blob;
                } else
                    ptrs[i] = nullptr;
                bdxes[i] = bdx;
            });
            
#if !COMMONS_NO_PYTHON
        }
#endif

        std::vector<pv::bid> ff;
        for(auto idx : unassigned_blobs)
            ff.push_back(bdxes[idx]);
        PPFrame::Log("unassigned_blobs = ", ff);
        
        const auto N_blobs = unassigned_blobs.size();
        const auto N_fish  = unassigned_individuals.size();
        const auto matching_probability_threshold = FAST_SETTING(matching_probability_threshold);
        IndividualCache empty{.valid=false, .individual_empty=true};
        
        auto work = [&](auto, auto start, auto end, auto){
            size_t pid = 0;
            std::vector< PairedProbabilities::ordered_assign_map_t > _probs(std::distance(start, end));
            
            for (auto it = start; it != end; ++it, ++pid) {
                auto fish = *it;
                if(fish->empty())
                    continue;
                
                auto cache = s.frame.cached(fish->identity().ID());
                if(!cache || fish->empty()) {
                    cache = &empty;
                } else {
                    assert(cache != nullptr);
                    //assert(cache->_idx == fish->identity().ID());
                }
                
                auto &probs = _probs[pid];
                
                for (size_t i = 0; i < N_blobs; ++i) {
                    auto &bix = unassigned_blobs[i];
                    auto ptr = ptrs[bix];//s.frame.bdx_to_ptr(bdxes[bix]);
                    //auto &own = s.frame.unsafe_access_all_blobs()[bix];
                    //if(!own.regular)
                    //    continue;
                    //auto ptr = s.frame.bdx_to_ptr(blob);
                    //assert(own.blob != nullptr);
                    auto p = fish->probability(blob_labels[bix], *cache, frameIndex, *ptr);
                    //PPFrame::Log(bdxes[bix], " + ", fish->identity().ID(), " => ", p, " (t = ", cache->time_probability,")");
                    if (p > matching_probability_threshold)
                        probs[bdxes[bix]] = p;
                }
            }
            
            pid = 0;
            std::unique_lock guard(paired_mutex);
            for (auto it = start; it != end; ++it, ++pid)
                paired_blobs.add((*it)->identity().ID(), std::move(_probs[pid]));
        };
        
        IndividualManager::Protect{};
#if defined(TREX_THREADING_STATS)
        {
            static std::mutex mStats;
            struct Bin {
                size_t N;
                double average_t, average_s;
                double samples_t, samples_s;
            };
            
            static std::vector<Bin> bins;
            size_t max_elements = 2000u;
            size_t step_size = 100u;
            size_t step = max_elements / step_size;
            
            {
                std::unique_lock guard(mStats);
                if(bins.empty()) {
                    bins.resize(step);
                    
                    for(size_t i=0; i<bins.size(); ++i) {
                        bins[i] = {
                            .N = step_size * i,
                            .average_t = 0, .average_s = 0,
                            .samples_t = 0, .samples_s = 0
                        };
                    }
                }
            }
            
            Timer timer;
            //if(pool && N_fish > 100)
            distribute_indexes(work, *pool, unassigned_individuals.begin(), unassigned_individuals.end());
            {
                auto s = timer.elapsed();
                
                std::unique_lock guard(mStats);
                for(auto &b : bins) {
                    if(b.N + step > N_fish) {
                        b.average_t += s;
                        ++b.samples_t;
                        break;
                    }
                }
            }
            
            paired_blobs = Match::PairedProbabilities{};
            
            timer.reset();
            work(0, unassigned_individuals.begin(), unassigned_individuals.end(), N_fish);
            
            {
                auto s = timer.elapsed();
                
                std::unique_lock guard(mStats);
                for(auto &b : bins) {
                    if(b.N + step > N_fish) {
                        b.average_s += s;
                        ++b.samples_s;
                        break;
                    }
                }
            }
            
            if(frame.index().get() % 1000 == 0) {
                std::unique_lock guard(mStats);
                cv::Mat mat = cv::Mat::zeros(255, 1000, CV_8UC3);
                using namespace gui;
                float max_y = 0;
                for(auto &b : bins) {
                    max_y = max(max_y, float(b.average_t / b.samples_t));
                    max_y = max(max_y, float(b.average_s / b.samples_s));
                }
                
                {
                    Vec2 prev{0, 0};
                    for(auto &b : bins) {
                        Vec2 pos{ float(b.N) / float(max_elements) * float(mat.cols), sqrt(float(b.average_t / b.samples_t) / max_y) * float(mat.rows) };
                        cv::line(mat, prev, pos, White);
                        prev = pos;
                    }
                }
                
                {
                    Vec2 prev{0, 0};
                    for(auto &b : bins) {
                        Vec2 pos{ float(b.N) / float(max_elements) * float(mat.cols), sqrt(float(b.average_s / b.samples_s) / max_y) * float(mat.rows) };
                        cv::line(mat, prev, pos, Red);
                        prev = pos;
                    }
                }
                
                for(auto &b : bins) {
                    if(b.N % (2 * step_size) == 0)
                        cv::putText(mat,Meta::toStr(b.N), Vec2(float(b.N) / float(max_elements) * float(mat.cols), float(mat.rows) - 10), cv::FONT_HERSHEY_PLAIN, 0.5, White);
                    cv::line(mat, Vec2(float(b.N) / float(max_elements) * float(mat.cols), float(mat.rows) - 11), Vec2(float(b.N) / float(max_elements) * float(mat.cols), 0), Gray);
                }
                
                cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
                tf::imshow("stats", mat);
                
                std::vector<float> values;
                for(auto &b : bins) {
                    values.push_back(b.N);
                    values.push_back(b.average_t / b.samples_t);
                    values.push_back(b.average_s / b.samples_s);
                }
                
                try {
                    auto path = file::DataLocation::parse("output", (std::string)SETTING(filename).value<file::Path>().filename()+"_threading_stats.npz").str();
                    npz_save(path, "values", values.data(), std::vector<size_t>{bins.size(), 3});
                    print("Saved threading stats at ", path,".");
                } catch(...) {
                    FormatWarning("Error saving threading stats.");
                }
            }
        }
#else
        if(pool && N_fish > 100)
            distribute_indexes(work, *pool, unassigned_individuals.begin(), unassigned_individuals.end());
        else
            work(0, unassigned_individuals.begin(), unassigned_individuals.end(), 0);
#endif
    }
    
    return paired_blobs;
}

//! Collect groups of individuals that have to be matched together.
//!
//! A simplified version of this algorithm can be described as follows:
//! if individuals are farther away from each other than `track_max_speed x dt`,
//! then they are not part of the same matching problem and thus can be handled
//! independently of each other.
//!
//! The only real difference here is that instead of distance, the probability
//! is used (from `calculate_paired_probabilities`), which is the value actually
//! used by the matching function.
void Tracker::collect_matching_cliques(TrackingHelper& s, GenericThreadPool& thread_pool) {
    using namespace default_config;
    
    struct IndexClique {
        UnorderedVectorSet<Match::blob_index_t> bids;
        UnorderedVectorSet<Match::fish_index_t> fids;

        void clear() {
            bids.clear();
            fids.clear();
        }

    } clique;

    UnorderedVectorSet<Match::fish_index_t> all_individuals; // collect all relevant individuals
    std::vector<IndexClique> cliques; // collect all cliques

    const auto p_threshold = FAST_SETTING(matching_probability_threshold);
    auto N_cliques = cliques.size();
    std::vector<bool> to_merge(N_cliques);
    
    for(auto &[row, idx] : s.paired.row_indexes()) {
        if(s.paired.degree(idx) > 1) {
            auto edges = s.paired.edges_for_row(idx);
            clique.clear();
            
            size_t matches = 0;
            
            //! collect all cliques that contain this individual
            //! or any of the blobs associated with this individual
            for (size_t i = 0; i<N_cliques; ++i) {
                auto ct = cliques.data() + i;
                if(contains(ct->fids, idx)) {
                    to_merge[i] = true;
                    ++matches;
                } else if(std::any_of(edges.begin(), edges.end(), [&](const Match::PairedProbabilities::Edge& e){
                    return e.p < p_threshold || ct->bids.contains(e.cdx);
                })) {
                    to_merge[i] = true;
                    ++matches;
                } else
                    to_merge[i] = false;
            }
            
            //! search edges to see if any of them
            //! have to be considered (>threshold)
            for(auto &col : edges) {
                if(col.p >= p_threshold) {
                    clique.bids.insert(col.cdx);
                }
            }
            
            //! we have found blobs that are associated with this individual
            //! so we need to consider it
            if(!clique.bids.empty()) {
                IndexClique* first = nullptr;
                clique.fids.insert(idx);
                
                if(matches > 0) {
                    auto it = cliques.begin();
                    
                    for(size_t i=0; i<N_cliques; ++i) {
                        if(!to_merge[i]) {
                            ++it;
                            continue;
                        }
                        
                        //! this is the first to-be-merged element we have found
                        //! which we will use from now on to add ids to
                        if(!first) {
                            first = &(*it);
                            
                            first->fids.insert(clique.fids.begin(), clique.fids.end());
                            first->bids.insert(clique.bids.begin(), clique.bids.end());
                            
                            ++it;
                            continue;
                        }
                        
                        //! not the first element -> add to "first" clique
                        //! and erase it.
                        {
                            // merge into current clique
                            const auto &c = *it;
                            first->fids.insert(c.fids.begin(), c.fids.end());
                            first->bids.insert(c.bids.begin(), c.bids.end());
                        }
                        
                        it = cliques.erase(it);
                    }
                }
                
                // this individual is connected with any blobs,
                // so it can be active
                all_individuals.insert(idx);
                
                // no cliques have been merged, but we still want a clique
                // that only contains our individual and the associated blobs
                // (the clique is disconnected from everything else)
                if(!first)
                    cliques.emplace_back(std::move(clique));
                
                // adapt array sizes
                N_cliques = cliques.size();
                to_merge.resize(N_cliques);
            }
        }
    }
    
    if(cliques.empty()) {
        // if there are absolutely no cliques found, no need to perform any crazy
        // matching at all:
        s.match_mode = matching_mode_t::approximate;
    } else {
        //! there are cliques, so we need to check each clique and match all
        //! individuals in it with objects in the frame
        
        // try to extend cliques as far as possible (and merge)
        IndexClique indexes;
        for(size_t index = 0; index < cliques.size(); ++index) {
            indexes.bids = cliques[index].bids;
            //UnorderedVectorSet<Match::fish_index_t> added_individuals;
            //UnorderedVectorSet<Match::blob_index_t> added_blobs(cliques[index].bids);
            
            do {
                indexes.fids.clear();
                
                for(auto cdx : indexes.bids) {
#ifndef NDEBUG
                    auto blob = s.paired.col(cdx);
#endif
                    auto &bedges = s.paired.edges_for_col(cdx);
                    
#ifndef NDEBUG
                    if(!s.frame.has_bdx(blob)) {
                        print("Frame ", s.frame.index(),": Cannot find blob ",blob," in map.");
                        continue;
                    }
#endif
                    
#ifdef TREX_DEBUG_MATCHING
                    auto str = Meta::toStr(bedges);
                    print("\t\tExploring blob ", cdx," (aka ", (*blob)->blob_id(),") with edges ",str);
#endif
                    for(auto fdi : bedges) {
                        if(   !contains(cliques[index].fids, fdi)
                           && !contains(indexes.fids, fdi))
                        {
                            indexes.fids.insert(fdi);
                            
                            for(size_t j=0; j<cliques.size(); ++j) {
                                if(j == index)
                                    continue;
                                
                                if(   contains(cliques[j].bids, cdx)
                                   || contains(cliques[j].fids, fdi))
                                {
#ifdef TREX_DEBUG_MATCHING
                                    // merge cliques
                                    auto str0 = Meta::toStr(cliques[index].fishs);
                                    auto str1 = Meta::toStr(cliques[j].fishs);
                                    
                                    auto str2 = Meta::toStr(cliques[index].bids);
                                    auto str3 = Meta::toStr(cliques[j].bids);
#endif
                                    
                                    indexes.fids.insert(cliques[j].fids.begin(), cliques[j].fids.end());
                                    cliques[index].bids.insert(cliques[j].bids.begin(), cliques[j].bids.end());
                                    
#ifdef TREX_DEBUG_MATCHING
                                    auto afterf = Meta::toStr(added_individuals);
                                    auto afterb = Meta::toStr(cliques[index].bids);
#endif
                                    
                                    if(j < index) {
                                        --index;
                                    }
                                    cliques.erase(cliques.begin() + j);
                                    
#ifdef TREX_DEBUG_MATCHING
                                    Debug("Frame %d: Merging cliques fishs:%S+%d vs. %S and blobs:%S+%u vs. %S -> added_fishs:%S and blobs:%S.",
                                          frameIndex,
                                          &str0, fdi, &str1,
                                          &str2, cdx, &str3,
                                          &afterf, &afterb);
#endif
                                    break;
                                }
                            }
                        }
                    }
                }
                
                indexes.bids.clear();
                for(auto i : indexes.fids) {
                    auto edges = s.paired.edges_for_row(i);
#ifdef TREX_DEBUG_MATCHING
                    print("\t\tExploring row ", i," (aka fish", s.paired.row(i)->identity().ID(),") with edges=",edges);
#endif
                    for(auto &e : edges) {
                        if(!contains(cliques[index].bids, e.cdx))
                            indexes.bids.insert(e.cdx);
                    }
                }
                
#ifdef TREX_DEBUG_MATCHING
                if(!added_individuals.empty()) {
                    auto str = Meta::toStr(added_individuals);
                    print("Adding ", str," to clique ",index);
                }
#endif
                cliques[index].fids.insert(indexes.fids.begin(), indexes.fids.end());
                
            } while(!indexes.fids.empty());
        }

        std::mutex thread_mutex;

        //! this function does the actual matching (threaded)
        //! for each clique in the frame:
        auto work_cliques = [frameIndex = s.frame.index(), &s, &thread_mutex] (auto, const decltype(cliques)::const_iterator& start, const decltype(cliques)::const_iterator& end, auto)
        {
            using namespace Match;
            
            for(auto it = start; it != end; ++it) {
                Match::PairedProbabilities paired;
                auto &clique = *it;
                
                {
                    for (auto& [fish, fdi] : s.paired.row_indexes()) {
                        if (!clique.fids.contains(fdi))
                            continue;
                        
                        if (s._manager.fish_assigned(fish))
                            continue;
                        
                        auto edges = s.paired.edges_for_row(fdi);
                        
                        PairedProbabilities::ordered_assign_map_t probs;
                        for (auto& e : edges) {
                            auto blob = s.paired.col(e.cdx);
                            if (!s._manager.blob_assigned(blob))
                                probs[blob] = e.p;
                        }
                        
                        if (!probs.empty())
                            paired.add(fish, probs);
                    }
                }
                
                PairingGraph graph(*s.props, frameIndex, std::move(paired));
                
                try {
                    // we use hungarian matching here by default, although
                    // the tree-based method is basically equal:
                    auto& optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
                    
                    std::unique_lock g(thread_mutex);
                    //const auto& paired = optimal.pairings;
                    //auto blobs = s.frame.extract_from_blobs_unsafe(paired);
                    
                    s._manager.assign<false>(AssignInfo{
                        .frame = &s.frame,
                        .f_prop = s.props,
                        .f_prev_prop = s.prev_props,
                        .match_mode = default_config::matching_mode_t::hungarian
                    }, std::move(optimal.pairings), [](pv::bid, Idx_t, Individual*)
                    {},
                    [frameIndex](pv::bid bdx, Idx_t fdx, Individual*, const char* error)
                    {
                        FormatExcept("Cannot assign ", fdx, " to ", bdx, " in frame ", frameIndex, " reporting: ", error);
                    });
                }
                catch (...) {
                    FormatExcept("Failed to generate optimal solution (frame ", frameIndex,").");
                }
            }
        };
        
        const auto frameIndex = s.frame.index();
        distribute_indexes(work_cliques, thread_pool, cliques.begin(), cliques.end());
        
        //! update cliques in the global array:
        Clique translated;
        Tracker::instance()->_cliques[frameIndex].clear();

        for (auto& clique : cliques) {
            translated.bids.clear();
            translated.fishs.clear();

            for (auto bdi : clique.bids)
                translated.bids.insert(s.paired.col(bdi));
            for (auto fdi : clique.fids)
                translated.fishs.insert(s.paired.row(fdi));

            Tracker::instance()->_cliques[frameIndex].emplace_back(std::move(translated));
        }

#ifdef TREX_DEBUG_MATCHING
        size_t index = 0;
        for(auto &clique : cliques) {
            std::set<uint32_t> fishs, blobs;
            for(auto fdi : clique.fishs)
                fishs.insert(s.paired.row(fdi)->identity().ID());
            for(auto bdi : clique.bids)
                blobs.insert((*s.paired.col(bdi))->blob_id());
            
            auto str = Meta::toStr(fishs);
            auto str1 = Meta::toStr(blobs);
            print("Frame ",frameIndex,": Clique ",index,", Matching fishs ",str.c_str()," and blobs ",str1.c_str()," together.");
            ++index;
            
            for(auto &cdx : clique.bids) {
                print("\tBlob ", (*s.paired.col(cdx))->blob_id()," edges:");
                for(auto &e : s.paired.edges_for_col(cdx)) {
                    print("\t\tFish", s.paired.row(e)->identity().raw_name().c_str());
                }
            }
        }
#endif
        
        Match::PairedProbabilities paired;
        for(auto &[fish, idx] : s.paired.row_indexes()) {
            if(!s._manager.fish_assigned(fish)) {
                auto edges = s.paired.edges_for_row(idx);
                
                Match::PairedProbabilities::ordered_assign_map_t probs;
                for(auto &e : edges) {
                    auto blob = s.paired.col(e.cdx);
                    if(s._manager.blob_assigned(blob))
                        continue;
#ifndef NDEBUG
                    if(!s.frame.has_bdx(blob)) {
                        print("Frame ", frameIndex,": Cannot find blob ",blob," in map.");
                        continue;
                    }
#endif
                    probs[blob] = e.p;
                }
                
                if(!probs.empty())
                    paired.add(fish, probs);
            }
        }
        
        PPFrame::Log(s.frame.index(), " s.paired = ", s.paired, " and paired = ", paired);
        s.paired = std::move(paired);
        s.match_mode = matching_mode_t::approximate;
    }
}

/**
 * Adding a frame that has been preprocessed previously in a different thread.
 */
void Tracker::add(Frame_t frameIndex, PPFrame& frame) {
    static Timer overall_timer;
    overall_timer.reset();
    
    if (!start_frame().valid() || start_frame() > frameIndex) {
        _startFrame = frameIndex;
    }
    
    if (!end_frame().valid() || end_frame() < frameIndex) {
        if(end_frame().valid() && end_frame() < start_frame())
          FormatError("end frame is ", end_frame()," < ",start_frame());
        _endFrame = frameIndex;
    }
    
    //! Perform blob splitting based on recent history.
    //! E.g.: We know there should be two individuals where we only
    //! find one object -> split the object in order to try to split
    //! the potentially overlapping individuals apart.
    HistorySplit{frame, PPFrame::NeedGrid::NoNeed, &_thread_pool};
    
    //! Initialize helper structure that encapsulates the substeps
    //! of the Tracker::add method:
    TrackingHelper s(frame, _added_frames);
    const auto number_fish = FAST_SETTING(track_max_individuals);
    
    // now that the blobs array has been cleared of all the blobs for fixed matches,
    // get pairings for all the others:
    //std::unordered_map<pv::Blob*, pv::BlobPtr> ptr2ptr;
    s.paired = calculate_paired_probabilities( s, &_thread_pool);
    PPFrame::Log(s.frame.index(), " calculated paired probabilities: ", s.paired);
    
#ifdef TREX_DEBUG_MATCHING
    {
        Match::PairingGraph graph(frameIndex, s.paired);
        
        try {
            auto &optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
            pairs = optimal.pairings;
            
        } catch(...) {
            FormatExcept("Failed to generate optimal solution (frame ", frameIndex,").");
        }
    }
#endif
    
    //! if we're in automatic matching mode, we need to collect cliques first.
    //! a clique is a number of close-by individuals that are potentially conflicting
    //! (either directly or indirectly) when it comes to matching them to objects.
    //! since they may compete for the same objects, the matching problem needs to be
    //! solved fully (e.g. hungarian algorithm) for each clique.
    if(s.match_mode == default_config::matching_mode_t::automatic) {
        collect_matching_cliques(s, _thread_pool);
    }
    
    //! now actually apply the matching
    //! the below method basically calls PairingGraph.find_optimal_pairing
    s.apply_matching();
    
    // Create Individuals for unassigned blobs
    std::vector<pv::bid> unassigned_blobs;
    frame.transform_blob_ids([&](pv::bid bdx) {
        if(!s._manager.blob_assigned(bdx))
            unassigned_blobs.emplace_back(bdx);
    });
    
    if(not number_fish) {
        // the number of individuals is unlimited (==0)
        // fallback to creating new individuals if the blobs cant be matched
        // to existing ones
        s._manager.assign_to_inactive(AssignInfo{
            .frame = &s.frame,
            .f_prop = s.props,
            .f_prev_prop = s.prev_props,
            .match_mode = default_config::matching_mode_t::none
        }, unassigned_blobs, [](pv::bid, Individual*) {
            // nothing
            //print("Assigned inactive ", bdx, " to ", fish);
        }, [frameIndex](pv::bid bdx, Individual* fish, const char* error) {
            if(!fish)
                throw U_EXCEPTION(frameIndex, ": Cannot create individual for blob ", bdx, ". Reason: ", error);
            else {
                FormatWarning(frameIndex, ": Cannot assign individual ", fish," with blob ", bdx," because ", error);
            }
        });
        
    } else if(s._manager.assigned_count() < number_fish
              && not unassigned_blobs.empty())
    {
        //  + the count of individuals is fixed (which means that no new individuals can
        //    be created after max)
        //  + the number of individuals is limited
        //  + there are unassigned individuals
        //    (the number of currently active individuals is smaller than the limit)
        //  + there are blobs left to be assigned
      
        // now find all individuals that have left the "active individuals" group already
        // and re-assign them if needed
        {
            // if an individual needs to be reassigned, choose the blobs that are the closest
            // to the estimated position.
            using namespace Match;
            
            //std::multiset<PairProbability> new_table;
            std::vector<std::tuple<Match::prob_t, Idx_t, pv::bid>> new_table;
            std::map<Idx_t, std::map<Match::Blob_t, Match::prob_t>> new_pairings;
            std::map<Individual*, Match::prob_t> max_probs;
            const Match::prob_t p_threshold = FAST_SETTING(matching_probability_threshold);
            
            static PairedProbabilities pairs;
            pairs.clear();

            if (pairs.probabilities().capacity() < IndividualManager::num_individuals() * unassigned_blobs.size()) {
#ifndef NDEBUG
                print("Reserving ", IndividualManager::num_individuals() * unassigned_blobs.size(), " pairs...");
#endif
                pairs.reserve(IndividualManager::num_individuals() * unassigned_blobs.size());
            }

            auto previous = Tracker::properties(frameIndex - 1_f);
            
            IndividualManager::transform_inactive([&](auto fish)
            {
                Match::PairedProbabilities::ordered_assign_map_t for_this;
                if(fish->empty()) {
                    for (auto& blob : unassigned_blobs) {
                        //assert(not s._manager.blob_assigned(blob));
                        //new_table.emplace_back(p_threshold, fish->identity().ID(), blob);
                        //new_pairings[fish->identity().ID()][blob] = p_threshold;
                        for_this[blob] = p_threshold;
                    }
                    
                } else {
                    auto pos_fish = fish->cache_for_frame(previous, frameIndex, frame.time);
                    if(not pos_fish) {
                        throw U_EXCEPTION("Cannot calculate cache_for_frame for ", fish->identity(), " in ", frameIndex, " because: ", pos_fish.error());
                    }
                    
                    for (auto& blob : unassigned_blobs) {
                        auto ptr = s.frame.bdx_to_ptr(blob);
                        assert(ptr != nullptr);
                        auto pos_blob = ptr->center();
                        Match::prob_t dist = sqdistance(pos_fish.value().last_seen_px, pos_blob);
                        Match::prob_t p =
                        pos_fish.value().local_tdelta == 0 ?
                            std::numeric_limits<float>::infinity() :
                              (dist > 0 ?
                                    Match::prob_t(1.0) / dist / pos_fish.value().local_tdelta
                                 :  Match::prob_t(1.0) / pos_fish.value().local_tdelta);
#ifndef NDEBUG
                        if (std::isinf(p)) {
                            throw U_EXCEPTION("Probability is invalid for ", fish->identity(), " in ", frameIndex, " (", pos_fish.value().last_seen_px, " -> ", pos_blob, ", tdelta=", pos_fish.value().local_tdelta,") => ", p);
                        }
#endif
                        for_this[blob] = p_threshold + p * (1 - p_threshold);
                        
                        //new_table.emplace_back(p, fish->identity().ID(), blob);
                        //new_pairings[fish->identity().ID()][blob] = p;
                        //new_table.insert(PairProbability(fish, blob, p));
                    }
                }
                
                //print(fish->identity(), " -> ", for_this);
                pairs.add(fish->identity().ID(), std::move(for_this));
            });
            
            //print("Ended up with ", pairs.probabilities().size(), "pairs.");
            PPFrame::Log("Ended up with ", pairs.probabilities().size(), "pairs.");
            PPFrame::Log(pairs);
            PairingGraph g(*s.props, s.frame.index(), std::move(pairs));
            auto& optimal = g.get_optimal_pairing(false, (s.match_mode == default_config::matching_mode_t::automatic) ? default_config::matching_mode_t::approximate : s.match_mode);
            PPFrame::Log("Got optimal.pairings = ", prettify_array(Meta::toStr(optimal.pairings)));
            //std::sort(new_table.begin(), new_table.end());
            
            s._manager.assign(AssignInfo{
                .frame = &s.frame,
                .f_prop = s.props,
                .f_prev_prop = s.prev_props,
                .match_mode = default_config::matching_mode_t::none
                
            }, std::move(optimal.pairings), [](pv::bid bdx, Idx_t fdx, Individual*) {
                //print("Test for ", bdx, " -> ", fdx);
                /*if(not new_pairings.contains(fdx)) {
                    print("\t not accepting ", bdx, " -> ", fdx);
                    return false;
                }*/
                    
                /*{
                    auto it = new_pairings.find(fdx);
                    assert(it != new_pairings.end());
                    new_pairings.erase(it);
                }*/
                
                PPFrame::Log("\t *accepting ", bdx, " -> ", fdx);
                
            }, [frameIndex](pv::bid bdx, Idx_t fdx, Individual*, const char* error) {
                throw U_EXCEPTION(frameIndex, ": Cannot assign individual ", fdx," to blob ", bdx, ". Reason: ", error);
            });

            pairs = std::move(g.paired());
        }
    }
    
#ifndef NDEBUG
    if(not number_fish) {
        static std::set<Idx_t> lost_ids;
        IndividualManager::transform_all([&](auto fdx, auto fish){
            if(   not s._manager.is_active(fish)
               && not s._manager.is_inactive(fish))
            {
                if(lost_ids.find(fdx) != lost_ids.end())
                    return;
                lost_ids.insert(fdx);
                auto basic = fish->empty() ? nullptr : fish->find_frame(frameIndex);
                
                if(basic && basic->frame == frameIndex) {
                    FormatWarning("Fish ", fdx," not in any of the arrays, but has frame ",frameIndex,".");
                } else
                    FormatWarning("Fish ", fdx," is gone (",basic ? basic->frame : Frame_t{},")");
            } else if(lost_ids.find(fdx) != lost_ids.end()) {
                lost_ids.erase(fdx);
                FormatWarning("Fish ", fdx," found again in frame ",frameIndex,".");
            }
        });
    }
#endif
    
    //! See if we are supposed to save tags (`tags_path` not empty),
    //! and if so then enqueue check_save_tags on a thread pool.
    std::future<void> tags_saver;
    std::promise<void> promise;
    if(s.save_tags()) {
        tags_saver = promise.get_future();
        
        try {
            _thread_pool.enqueue([&]() {
                std::shared_lock guard(s.blob_fish_mutex);
                this->check_save_tags(frameIndex, s.blob_fish_map, s.tagged_fish, s.noise, FAST_SETTING(tags_path));
                promise.set_value();
            });
        } catch(const UtilsException& e) {
            FormatExcept("Exception when starting check_save_tags threads on _thread_pool: ", e.what());
        }
    }
    
    Timer posture_timer;
    
    const auto combined_posture_seconds = s.process_postures();
    const auto posture_seconds = posture_timer.elapsed();
    
    Output::Library::frame_changed(frameIndex);
    
    if(number_fish && s._manager.assigned_count() >= number_fish) {
        update_consecutive(s._manager.current(), frameIndex, true);
    }
    
    _max_individuals = cmn::max(_max_individuals.load(), s._manager.assigned_count());
    _added_frames.back()->active_individuals = narrow_cast<long_t>(s._manager.assigned_count());
    
    uint32_t n = 0;
    uint32_t prev = 0;
    if(!has_identities()) {
        for(auto fish : s._manager.current()) {
            assert((fish->end_frame().valid() && fish->end_frame() == frameIndex) == (fish->has(frameIndex)));
            
            if(fish->end_frame().valid()
               && fish->end_frame() == frameIndex)
                ++n;
            if(fish->has(frameIndex - 1_f))
                ++prev;
        }
        
    } else {
        IndividualManager::transform_ids_with_error(Tracker::identities(), [&](auto, auto fish)
        {
            assert((fish->end_frame().valid() && fish->end_frame() == frameIndex) == (fish->has(frameIndex)));
            if(fish->end_frame().valid()
               && fish->end_frame() == frameIndex)
                ++n;
            if(fish->has(frameIndex - 1_f))
                ++prev;
            
        }, [](auto){});
    }
    
    update_warnings(frameIndex, s.frame.time, number_fish, n, prev, s.props, s.prev_props, s._manager.current(), _individual_add_iterator_map);
    
#if !COMMONS_NO_PYTHON
    //! Iterate through qrcodes in this frame and try to assign them
    //! to the optimal individuals. These qrcodes can currently only
    //! come from tgrabs directly.
    if (not frame.tags().empty()) // <-- only added from tgrabs
    {
        //! calculate probabilities of assigning qrcodes / tags
        //! to individuals based on position, similarly to how
        //! blobs are assigned to individuals, too.
        Match::PairedProbabilities paired;
        const Match::prob_t p_threshold = FAST_SETTING(matching_probability_threshold);
        std::unordered_map<pv::bid, pv::BlobPtr> owner;
        for(auto && blob : frame.tags()) {
            owner[blob->blob_id()] = std::move(blob);
        }
        frame.tags().clear(); // is invalidated now, clear it
        
        s._manager.transform_active([&](Individual* fish){
            Match::PairedProbabilities::ordered_assign_map_t probs;
            
            auto cache = frame.cached(fish->identity().ID());
            if (!cache)
                return;

            for (const auto &[bdx, blob] : owner) {
                auto p = fish->probability(-1, *cache, frameIndex, *blob);
                if (p >= p_threshold)
                    probs[bdx] = p;
            }

            if(!probs.empty())
                paired.add(fish->identity().ID(), probs);
        });

        //! use calculated probabilities to find optimal assignments
        //! then add the qrcode to the matched individual
        Match::PairingGraph graph(*s.props, frameIndex, std::move(paired));
        try {
            auto& optimal = graph.get_optimal_pairing(false, default_config::matching_mode_t::hungarian);
            for (auto& [bdx, fish] : optimal.pairings) {
                IndividualManager::transform_if_exists(fish, [&, bdx=bdx](Individual * fish) {
                    if (!fish->add_qrcode(frameIndex, std::move(owner.at(bdx))))
                    {
                        //FormatWarning("Fish ", fish->identity(), " rejected tag at ", (*blob)->bounds());
                    }
                });
            }
        }
        catch (...) {
            FormatExcept("Exception during tags to individuals matching.");
        }
    }
#endif
    
    //! wait for potentially enqueued tasks to save tags
    if(s.save_tags())
        tags_saver.wait();
    
    auto adding = (float)overall_timer.elapsed();
    auto loading = (float)frame.loading_time();
    
    std::lock_guard<std::mutex> guard(_statistics_mutex);
    auto& entry = _statistics[frameIndex];
    entry.number_fish = s._manager.assigned_count();
    entry.posture_seconds = posture_seconds;
    entry.combined_posture_seconds = combined_posture_seconds;
    
    entry.adding_seconds = adding;
    entry.loading_seconds = loading;
    entry.match_improvements_made = frame._split_objects;
    entry.match_leafs_visited = frame._split_pixels;
    
    _time_samples.add(adding, entry.number_fish);
}

void Tracker::update_iterator_maps(Frame_t frame, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators)
{
    for(auto fish : active_individuals) {
        auto fit = individual_iterators.find(fish->identity().ID());
        
        //! check if iterator is valid (in case vector size changed and it got invalidated)
        if(_segment_map_known_capacity[fish->identity().ID()] != ((const Individual*)fish)->frame_segments().capacity()) {
            // all iterators are invalid
            if(fit != individual_iterators.end()) {
                individual_iterators.erase(fit);
                fit = individual_iterators.end();
            }
            _segment_map_known_capacity[fish->identity().ID()] = ((const Individual*)fish)->frame_segments().capacity();
        }
        
        const auto end = ((const Individual*)fish)->frame_segments().end();
        
        if(fit == individual_iterators.end()) {
            fit = individual_iterators.insert({
                fish->identity().ID(),
                fish->iterator_for(frame)
            }).first;
        }
        
        assert(fit != individual_iterators.end());
        while(fit->second != end && (*fit->second)->end() < frame)
            ++fit->second;
        
        if(fit->second == end) {
            individual_iterators.erase(fit);
            continue;
        }
    }
}
            
    void Tracker::update_warnings(Frame_t frameIndex, double time, long_t /*number_fish*/, long_t n_found, long_t n_prev, const FrameProperties *props, const FrameProperties *prev_props, const set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators) {
        std::map<std::string, std::set<FOI::fdx_t>> merge;
        
        if(n_found < n_prev-1) {
            FOI::add(FOI(frameIndex, "lost >=2 fish"));
        }
        
        //if(!prev_props) prev_props = properties(frameIndex - 1);
        if(prev_props && time - prev_props->time >= FAST_SETTING(huge_timestamp_seconds)) {
            FOI::add(FOI(frameIndex, "huge time jump"));
            for(auto fish : active_individuals)
                merge["correcting"].insert(FOI::fdx_t(fish->identity().ID()));
        }
        
        std::set<FOI::fdx_t> found_matches;
        for(auto fish : active_individuals) {
            if(fish->is_manual_match(frameIndex))
                found_matches.insert(FOI::fdx_t(fish->identity().ID()));
        }
        
        if(!found_matches.empty()) {
            FOI::add(FOI(frameIndex, found_matches, "manual match"));
            merge["correcting"].insert(found_matches.begin(), found_matches.end());
        }
        
        update_iterator_maps(frameIndex - 1_f, active_individuals, individual_iterators);
        for(auto &fish : active_individuals) {
            if(_warn_individual_status.size() <= (size_t)fish->identity().ID().get()) {
                _warn_individual_status.resize(fish->identity().ID().get() + 1);
            }
            
            auto &property = _warn_individual_status[fish->identity().ID().get()];
            
            auto fit = individual_iterators.find(fish->identity().ID());
            if(fit == individual_iterators.end()) {
                property.prev = property.current = nullptr;
                continue;
            }
            
            auto &it = fit->second;
            if(it != fish->frame_segments().end() && (*it)->contains(frameIndex - 1_f)) {
                // prev
                auto idx = (*it)->basic_stuff(frameIndex - 1_f);
                property.prev = idx != -1 ? &fish->basic_stuff()[uint32_t(idx)]->centroid : nullptr;
                
                // current
                idx = (*it)->basic_stuff(frameIndex);
                property.current = idx != -1 ? &fish->basic_stuff()[uint32_t(idx)]->centroid : nullptr;
                
            } else
                property.prev = property.current = nullptr;
        }
        
#ifndef NDEBUG
        for(auto &fish : active_individuals) {
            if(_warn_individual_status.size() <= fish->identity().ID().get()) {
                assert(!fish->has(frameIndex - 1_f));
                continue;
            }
            
            auto &property = _warn_individual_status.at(fish->identity().ID().get());
            if(property.prev == nullptr) {
                assert(!fish->has(frameIndex - 1_f));
            } else {
                assert((property.prev != nullptr) == fish->has(frameIndex - 1_f));
                if(property.prev != nullptr) {
                    if(property.current == nullptr) {
                        assert(fish->segment_for(frameIndex - 1_f) != fish->segment_for(frameIndex));
                    } else
                        if(fish->segment_for(frameIndex - 1_f) != fish->segment_for(frameIndex)) {
                            auto A = fish->segment_for(frameIndex-1_f);
                            auto B = fish->segment_for(frameIndex);
                            FormatWarning(frameIndex - 1_f, " != ", frameIndex, ": ", A, " vs. ", B);
                            assert(false);
                        }
                } else
                    assert(property.current == nullptr);
            }
        }
#endif
        
        if(prev_props && props) {
            std::set<FOI::fdx_t> weird_distance, weird_angle, segment_end;
            std::set<FOI::fdx_t> fdx;
            
            for(auto fish : active_individuals) {
                auto properties = _warn_individual_status.size() > (size_t)fish->identity().ID().get() ? &_warn_individual_status[fish->identity().ID().get()] : nullptr;
                
                if(properties && properties->current) {
                    if(properties->current->speed<Units::CM_AND_SECONDS>() >= Individual::weird_distance()) {
                        weird_distance.insert(FOI::fdx_t{fish->identity().ID()});
                    }
                }
                
                if(properties && properties->prev && properties->current) {
                    // only if both current and prev are set, do we have
                    // both frameIndex-1 and frameIndex present in the same segment:
                    assert(fish->has(frameIndex - 1_f) && fish->has(frameIndex));
                    if(cmn::abs(angle_difference(properties->prev->angle(), properties->current->angle())) >= M_PI * 0.8)
                    {
                        weird_angle.insert(FOI::fdx_t{fish->identity().ID()});
                    }
                    
                } else if(properties && properties->prev) {
                    segment_end.insert(FOI::fdx_t{fish->identity().ID()});
                    
                    if(!fish->has(frameIndex)) {
                        assert(fish->has(frameIndex - 1_f) && !fish->has(frameIndex));
                        fdx.insert(FOI::fdx_t{fish->identity().ID()});
                    }
                    
                } else if(!properties)
                    print("No properties for fish ",fish->identity().ID());
            }
            
#ifndef NDEBUG
            IndividualManager::transform_ids(segment_end, [frameIndex](auto, auto fish)
            {
                assert(fish->segment_for(frameIndex) != fish->segment_for(frameIndex - 1_f));
            });
            
            IndividualManager::transform_ids(fdx, [frameIndex](auto, auto fish)
            {
                assert(not fish->has(frameIndex));
                assert(frameIndex != start_frame() && fish->has(frameIndex - 1_f));
            });
#endif
            
            if(!fdx.empty()) {
                FOI::add(FOI(frameIndex, fdx, "lost >=1 fish"));
                merge["correcting"].insert(fdx.begin(), fdx.end());
            }
            
            if(!weird_distance.empty()) {
                FOI::add(FOI(frameIndex, weird_distance, "weird distance"));
                merge["correcting"].insert(weird_distance.begin(), weird_distance.end());
                
                if(!found_matches.empty()) {
                    std::set<FOI::fdx_t> combined;
                    for(auto id : found_matches)
                        if(weird_distance.find(id) != weird_distance.end())
                            combined.insert(id);
                    FOI::add(FOI(frameIndex, combined, "weird distance + mm"));
                }
            }
            if(!weird_angle.empty())
                FOI::add(FOI(frameIndex, weird_angle, "weird angle"));
            if(!segment_end.empty())
                FOI::add(FOI(frameIndex - 1_f, segment_end, "segment end"));
        }
        
        /*if(n_found < n_prev || frameIndex == start_frame()) {
            std::set<FOI::fdx_t> fdx;
            static std::atomic<size_t> finds = 0, misses = 0;
            
            update_iterator_maps(frameIndex, active_individuals, individual_iterators);
            for(auto & fish : active_individuals) {
                auto fit = individual_iterators.find(fish->identity().ID());
                if(fit == individual_iterators.end() || fit->second == fish->frame_segments().end() || !(*fit->second)->contains(frameIndex))
                {
                    fdx.insert(FOI::fdx_t(fish->identity().ID()));
                }
            }
            
#ifndef NDEBUG
            {
                std::set<FOI::fdx_t> fdx1;
                auto it = _active_individuals_frame.find(frameIndex);
                if(it != _active_individuals_frame.end()) {
                    for(auto fish : it->second) {
                        if(!fish->has(frameIndex))
                            fdx1.insert(FOI::fdx_t(fish->identity().ID()));
                    }
                }
                if(fdx1 != fdx) {
                    auto str0 = Meta::toStr(fdx);
                    auto str1 = Meta::toStr(fdx1);
                    throw U_EXCEPTION("",str0," != ",str1,"");
                }
            }
#endif
            
            if(!fdx.empty()) {
                FOI::add(FOI(frameIndex, fdx, "lost >=1 fish"));
                merge["correcting"].insert(fdx.begin(), fdx.end());
            }
        }
        */
        for(auto && [key, value] : merge)
            FOI::add(FOI(frameIndex, value, key));
    }

    void Tracker::update_consecutive(const set_of_individuals_t &active, Frame_t frameIndex, bool update_dataset) {
        bool all_good = FAST_SETTING(track_max_individuals) == (uint32_t)active.size();
        
        //auto manual_identities = FAST_SETTING(manual_identities);
        for(auto fish : active) {
            //if(manual_identities.empty() || manual_identities.count(fish->identity().ID()))
            {
                if(!fish->has(frameIndex) /*|| fish->centroid_weighted(frameIndex)->speed() >= SLOW_SETTING(track_max_speed) * 0.25*/) {
                    all_good = false;
                    break;
                }
            }
        }
        
        if(all_good) {
            if(!_consecutive.empty() && _consecutive.back().end == frameIndex - 1_f) {
                _consecutive.back().end = frameIndex;
                if(frameIndex == analysis_range().end()) {
                    DatasetQuality::update();
                }
            } else {
                if(!_consecutive.empty()) {
                    FOI::add(FOI(_consecutive.back(), "global segment"));
                }
                
                _consecutive.push_back(Range<Frame_t>(frameIndex, frameIndex));
                if(update_dataset) {
                    DatasetQuality::update();
                }
            }
        }
    }

    void Tracker::generate_pairdistances(Frame_t) {
        /*std::vector<Individual*> frame_individuals;
        for (auto fish : _active_individuals) {
            if (fish->centroid(frameIndex)) {
                frame_individuals.push_back(fish);
            }
        }
        
        std::multiset<PairDistance> distances;
        //distances.reserve(frame_individuals.size()*ceil(frame_individuals.size()*0.5));
        
        PairDistance d(NULL, NULL, 0);
        for (long_t i=0; i<long_t(frame_individuals.size()); i++) {
            for (long_t j=0; j<ceil(frame_individuals.size()*0.5); j++) {
                long_t idx1 = i - 1 - j;
                if(idx1 < 0)
                    idx1 += frame_individuals.size();
                
                d.set_fish0(frame_individuals.at(i));
                d.set_fish1(frame_individuals.at(idx1));
                d.set_d(euclidean_distance(frame_individuals.at(i)->centroid(frameIndex)->pos<Units::PX_AND_SECONDS>(), frame_individuals.at(idx1)->centroid(frameIndex)->pos<Units::PX_AND_SECONDS>()));
                
                distances.insert(d);
            }
        }
*/
        throw U_EXCEPTION("Pair distances need to implement the new properties.");
    }
    
    void Tracker::_remove_frames(Frame_t frameIndex) {
        assert(frameIndex.valid());
        
#if !COMMONS_NO_PYTHON
        Categorize::DataStore::reanalysed_from(Frame_t(frameIndex));
#endif
        
        LockGuard guard(w_t{}, "_remove_frames("+Meta::toStr(frameIndex)+")");
        _thread_pool.wait();
        
        _individual_add_iterator_map.clear();
        _segment_map_known_capacity.clear();
        
        if(TrackingHelper::_approximative_enabled_in_frame.valid()
           && TrackingHelper::_approximative_enabled_in_frame >= frameIndex)
        {
            TrackingHelper::_approximative_enabled_in_frame.invalidate();
        }
        
        print("Removing frames after and including ", frameIndex);
        
        if (not start_frame().valid() || end_frame() < frameIndex || start_frame() > frameIndex)
            return;
        
        PPFrame::CloseLogs();
        update_history_log();
        
        if(!_consecutive.empty()) {
            while(!_consecutive.empty()) {
                if(_consecutive.back().start < frameIndex)
                    break;
                
                _consecutive.erase(--_consecutive.end());
            }
            print("Last remaining ", _consecutive.size());
            if(!_consecutive.empty()) {
                if(_consecutive.back().end >= frameIndex)
                    _consecutive.back().end = frameIndex - 1_f;
                print(_consecutive.back().start,"-",_consecutive.back().end);
            }
        }
        
        DatasetQuality::remove_frames(frameIndex);
        IndividualManager::remove_frames(frameIndex
#ifndef NDEBUG
            , [this](Individual* fish){
                assert (_individual_add_iterator_map.find(fish->identity().ID()) == _individual_add_iterator_map.end() );
            }
#endif
        );
        
        {
            //! update the cache for frame properties
            std::unique_lock guard(properties_mutex());
            while(!_added_frames.empty()) {
                if((*(--_added_frames.end()))->frame < frameIndex)
                    break;
                _added_frames.erase(--_added_frames.end());
            }
            
            properties_cache().clear();
            
            auto it = _added_frames.rbegin();
            while(it != _added_frames.rend() && !properties_cache().full())
            {
                properties_cache().push((*it)->frame, (*it).get());
                ++it;
            }
        }
        
        for (auto it=_statistics.begin(); it != _statistics.end();) {
            if(it->first < frameIndex)
                ++it;
            else
                it = _statistics.erase(it);
        }
        
        _endFrame = frameIndex - 1_f;
        while (end_frame().valid()
               && not properties(end_frame()))
        {
            if (end_frame() < start_frame()) {
                _endFrame = _startFrame = Frame_t();
                break;
            }
            
            _endFrame = end_frame() - 1_f;
        }
        
        if(not end_frame().valid()
           or end_frame() < analysis_range().start())
        {
            _endFrame = _startFrame = Frame_t();
        }
        
        assert((_added_frames.empty() && !end_frame().valid()) || (end_frame().valid() && (*_added_frames.rbegin())->frame == end_frame()));
        
        FilterCache::clear();
        //! TODO: MISSING remove_frames
        //_recognition->remove_frames(frameIndex);
        DatasetQuality::update();
        
        VisualField::remove_frames_after(frameIndex);
        FOI::remove_frames(frameIndex);
        
        global_segment_order_changed();
        
        print("After removing frames: ", gui::CacheObject::memory());
        print("posture: ", Midline::saved_midlines());
        print("all blobs: ", pv::Blob::all_blobs());
        print("Range: ", start_frame(),"-",end_frame());
    }

    size_t Tracker::found_individuals_frame(Frame_t frameIndex) const {
        if(!properties(frameIndex))
            return 0;
        
        auto &a = active_individuals(frameIndex);
        size_t n = 0;
        for (auto i : a) {
            n += i->has(frameIndex) ? 1 : 0;
        }
        
        return n;
    }

    void Tracker::global_segment_order_changed() {
        _segment_order_changed = true;
    }

    std::vector<Range<Frame_t>> Tracker::unsafe_global_segment_order() {
        LockGuard guard(ro_t{}, "Tracker::max_range()");
        return _global_segment_order;
    }
    
    std::vector<Range<Frame_t>> Tracker::global_segment_order() {
        LockGuard guard(ro_t{}, "Tracker::max_range()");
        if(_segment_order_changed) {
            LockGuard guard(w_t{}, "Tracker::max_range()::write");
            std::set<Range<Frame_t>> manuals;
            auto manually_approved = FAST_SETTING(manually_approved);
            for(auto && [from, to] : manually_approved)
                manuals.insert(Range<Frame_t>(Frame_t(from), Frame_t(to)));
            
            std::set<Range<Frame_t>, std::function<bool(Range<Frame_t>, Range<Frame_t>)>> ordered([&manuals](Range<Frame_t> A, Range<Frame_t> B) -> bool {
                if(manuals.find(A) != manuals.end() && manuals.find(B) == manuals.end())
                    return true;
                if(manuals.find(B) != manuals.end() && manuals.find(A) == manuals.end())
                    return false;
                return ((DatasetQuality::has(A)
                         ? DatasetQuality::quality(A)
                         : DatasetQuality::Quality())
                        > (DatasetQuality::has(B)
                           ? DatasetQuality::quality(B)
                           : DatasetQuality::Quality()));
            });
            
            if(!manually_approved.empty()) {
                auto str = Meta::toStr(manually_approved);
                for(auto && [from, to] : manually_approved) {
                    ordered.insert(Range<Frame_t>(Frame_t(from), Frame_t(to)));
                }
            }
            
            std::set<Range<Frame_t>> consecutive;
            for(auto &range : instance()->consecutive())
                consecutive.insert(range);
            
            for(auto& range : instance()->consecutive()) {
                bool found = false;
                for(auto& existing : ordered) {
                    if(existing.overlaps(range)) {
                        found = true;
                        break;
                    }
                }
                
                if(!found)
                    ordered.insert(range);
            }
            
            _global_segment_order = std::vector<Range<Frame_t>>(ordered.begin(), ordered.end());
            _segment_order_changed = false;
        }
        
        return _global_segment_order;
    }
    
    struct IndividualImages {
        std::vector<Frame_t> frames;
        std::vector<Image::SPtr> images;
    };
    
    struct SplitData {
        std::map<long_t, IndividualImages> training;
        std::map<long_t, IndividualImages> validation;
        std::map<long_t, Rangel> ranges;
        TrainingData::MidlineFilters filters;
        
        GETTER(default_config::individual_image_normalization_t::Class, normalized);
        
    public:
        SplitData();
        void add_frame(Frame_t frame, long_t id, Image::SPtr image);
    };
    
    SplitData::SplitData() : _normalized(SETTING(recognition_normalize_direction).value<default_config::individual_image_normalization_t::Class>()) {
        
    }
    
    void SplitData::add_frame(Frame_t frame, long_t id, Image::SPtr image) {
        assert(image);
        
        if(training.size() <= validation.size() * 1.25) {
            training[id].frames.push_back(frame);
            training[id].images.push_back(image);
        } else {
            validation[id].frames.push_back(frame);
            validation[id].images.push_back(image);
        }
    }
    
    
    template<typename... Args>
    void log(FILE* f, const Args&... args) {
        std::string output = format<FormatterType::NONE>(args...);
        output += "\n";
        
        if(f)
            fwrite(output.c_str(), sizeof(char), output.length(), f);
    }
    
    void Tracker::clear_segments_identities() {
        LockGuard guard(w_t{}, "clear_segments_identities");
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(Frame_t(0), fid);
        
        IndividualManager::transform_all([](auto, auto fish){
            fish->clear_recognition();
        });
        
        AutoAssign::clear_automatic_ranges();
    }

namespace v {
using fdx_t = Idx_t;
using range_t = FrameRange;
using namespace Match;

struct VirtualFish {
    std::set<range_t> segments;
    std::map<range_t, Match::prob_t> probs;
    std::map<range_t, size_t> samples;
    std::map<Range<Frame_t>, Idx_t> track_ids;
};

template<typename F>
void process_vi
  (Frame_t after_frame,
   const Individual::small_segment_map& segments,
   const Idx_t fdx,
   const Individual* fish,
   const std::map<fdx_t, std::map<Range<Frame_t>, fdx_t>>& assigned_ranges,
   std::map<fdx_t, VirtualFish>& virtual_fish,
   F&& apply)
{
    //! find identities for
    for (auto& [start, segment] : segments) {
        auto previous = segments.end(),
            next = segments.end();

        const auto current = segments.find(start);
        if (after_frame.valid() && segment.range.end < after_frame)
            continue;
        
        if (current != segments.end()) {
            auto it = current;
            if ((++it) != segments.end())
                next = it;

            it = current;
            if (it != segments.begin())
                previous = (--it);
        }

        if (assigned_ranges.count(fdx) && assigned_ranges.at(fdx).count(segment.range)) {
            continue; // already assigned this frame segment to someone...
        }

        if (next != segments.end() && next->second.start().valid()) {
        }
        else
            continue;

        Idx_t prev_id, next_id;
        MotionRecord* prev_pos = nullptr, * next_pos = nullptr;

        auto it = assigned_ranges.find(fdx);
        if (it != assigned_ranges.end()) {
            decltype(it->second.begin()) rit;
            const Frame_t max_frames{ SLOW_SETTING(frame_rate) * 15 };

            // skip some frame segments to find the next assigned id
            do {
                // dont assign anything after one second
                if (next->second.start() >= current->second.end() + max_frames)
                    break;

                rit = it->second.find(next->second.range);
                if (rit != it->second.end()) {
                    next_id = rit->second;

                    if (virtual_fish.count(next_id) && virtual_fish.at(next_id).track_ids.count(rit->first)) {
                        auto org_id = virtual_fish.at(next_id).track_ids.at(rit->first);
                        IndividualManager::transform_if_exists(org_id, [&](auto fish){
                            auto blob = fish->centroid_weighted(next->second.start());
                            if (blob)
                                next_pos = blob;
                        });
                    }
                    break;
                }

            } while ((++next) != segments.end());

            // skip some frame segments to find the previous assigned id
            while (previous != segments.end()) {
                // dont assign anything after one second
                if (previous->second.end() + max_frames < current->second.start())
                    break;

                rit = it->second.find(previous->second.range);
                if (rit != it->second.end()) {
                    prev_id = rit->second;

                    if (virtual_fish.count(prev_id) && virtual_fish.at(prev_id).track_ids.count(rit->first)) {
                        auto org_id = virtual_fish.at(prev_id).track_ids.at(rit->first);
                        IndividualManager::transform_if_exists(org_id, [&](auto fish){
                            auto pos = fish->centroid_weighted(previous->second.end());
                            if (pos) {
                                prev_pos = pos;
                            }
                        });
                    }
                    break;
                }

                if (previous != segments.begin())
                    --previous;
                else
                    break;
            }
        }

        if (next_id.valid() && prev_id.valid() && next_id == prev_id && prev_pos && next_pos) {

        }
        else
            continue;

        apply(fdx, fish, segment, prev_id, next_id, prev_pos, next_pos);
    }
}

template<typename F>
void process_qr(Frame_t after_frame,
                const Idx_t fdx,
                const Individual* fish,
                const std::map<fdx_t, std::map<Range<Frame_t>, fdx_t>>& assigned_ranges,
                std::map<fdx_t, VirtualFish>& virtual_fish,
                F&& apply)
{
    for (auto& [start, ids] : fish->qrcodes()) {
        const auto current = fish->find_segment_with_start(start);
        if (current == fish->frame_segments().end()) {
            FormatWarning("Cannot find frame segment ", start, " with ", ids, " for ", fish->identity());
            continue;
        }

        auto segment = *(*current);
        auto previous = fish->frame_segments().end(),
            next = fish->frame_segments().end();

        if (after_frame.valid() && segment.range.end < after_frame)
            continue;
        //if(start == 741 && fish->identity().ID() == 1)

        if (current != fish->frame_segments().end()) {
            auto it = current;
            if ((++it) != fish->frame_segments().end())
                next = it;

            it = current;
            if (it != fish->frame_segments().begin())
                previous = (--it);
        }

        if (assigned_ranges.count(fdx) && assigned_ranges.at(fdx).count(segment.range)) {
            continue; // already assigned this frame segment to someone...
        }

        if (next != fish->frame_segments().end() && /*previous.start() != -1 &&*/ (*next)->start().valid()) {
        }
        else
            continue;

        Idx_t prev_id, next_id;
        MotionRecord* prev_pos = nullptr, * next_pos = nullptr;

        auto it = assigned_ranges.find(fdx);
        if (it != assigned_ranges.end()) {
            decltype(it->second.begin()) rit;
            const Frame_t max_frames{ SLOW_SETTING(frame_rate) * 15 };

            // skip some frame segments to find the next assigned id
            do {
                // dont assign anything after one second
                if ((*next)->start() >= (*current)->end() + max_frames)
                    break;

                rit = it->second.find((*next)->range);
                if (rit != it->second.end()) {
                    next_id = rit->second;

                    if (virtual_fish.count(next_id) && virtual_fish.at(next_id).track_ids.count(rit->first)) {
                        auto org_id = virtual_fish.at(next_id).track_ids.at(rit->first);
                        IndividualManager::transform_if_exists(org_id, [&](auto fish){
                            auto blob = fish->centroid_weighted((*next)->start());
                            if (blob)
                                next_pos = blob;
                        });
                    }
                    break;
                }

            } while ((++next) != fish->frame_segments().end());

            // skip some frame segments to find the previous assigned id
            while (previous != fish->frame_segments().end()) {
                // dont assign anything after one second
                if ((*previous)->end() + max_frames < (*current)->start())
                    break;

                rit = it->second.find((*previous)->range);
                if (rit != it->second.end()) {
                    prev_id = rit->second;

                    if (virtual_fish.count(prev_id) && virtual_fish.at(prev_id).track_ids.count(rit->first)) {
                        auto org_id = virtual_fish.at(prev_id).track_ids.at(rit->first);
                        IndividualManager::transform_if_exists(org_id, [&](auto fish){
                            auto pos = fish->centroid_weighted((*previous)->end());
                            if (pos) {
                                prev_pos = pos;
                            }
                        });
                    }
                    break;
                }

                if (previous != fish->frame_segments().begin())
                    --previous;
                else
                    break;
            }
        }

        if (next_id.valid() && prev_id.valid() && next_id == prev_id && prev_pos && next_pos) {

        }
        else
            continue;

        apply(fdx, fish, segment, prev_id, next_id, prev_pos, next_pos);
    }
}

void apply(
#ifdef TREX_DEBUG_IDENTITIES
 auto f,
#endif
 auto& still_unassigned,
 auto& manual_splits,
 auto &assigned_ranges,
 auto& tmp_assigned_ranges,
 Idx_t fdx,
 const Individual* fish,
 const FrameRange& segment,
 Idx_t prev_id,
 Idx_t next_id,
 MotionRecord* prev_pos,
 MotionRecord* next_pos)
{
    Vec2 pos_start(FLT_MAX), pos_end(FLT_MAX);
    auto blob_start = fish->centroid_weighted(segment.start());
    auto blob_end = fish->centroid_weighted(segment.end());
    if (blob_start)
        pos_start = blob_start->pos<Units::CM_AND_SECONDS>();
    if (blob_end)
        pos_end = blob_end->pos<Units::CM_AND_SECONDS>();

    if (blob_start && blob_end) {
        auto dprev = euclidean_distance(prev_pos->pos<Units::CM_AND_SECONDS>(), pos_start)
            / abs(blob_start->time() - prev_pos->time());
        auto dnext = euclidean_distance(next_pos->pos<Units::CM_AND_SECONDS>(), pos_end)
            / abs(next_pos->time() - blob_end->time());
        Idx_t chosen_id;

        if (dnext < dprev) {
            if (dprev < FAST_SETTING(track_max_speed) * 0.1)
                chosen_id = next_id;
        }
        else if (dnext < FAST_SETTING(track_max_speed) * 0.1)
            chosen_id = prev_id;

        if (chosen_id.valid()) {
#ifdef TREX_DEBUG_IDENTITIES
            if (segment.start() == 0_f) {
                log(f, "Fish ", fdx, ": chosen_id ", chosen_id, ", assigning ", segment, " (", dprev, " / ", dnext, ")...");
            }
#endif

            bool remove = false;
#ifndef NDEBUG
            std::set<Range<Frame_t>> remove_from;
#endif
            std::vector<pv::bid> blob_ids;
            
            for (Frame_t frame = segment.start(); frame <= segment.end(); ++frame) {
                auto blob = fish->compressed_blob(frame);

                if (blob) {
                    //automatically_assigned_blobs[frame][blob->blob_id()] = fdx;
                    blob_ids.push_back(blob->blob_id());
                    //if(blob->split() && blob->parent_id().valid())
                    //    manual_splits[frame].insert(blob->parent_id());
                }
                else
                    blob_ids.push_back(pv::bid::invalid);

                auto it = std::find(tmp_assigned_ranges.begin(), tmp_assigned_ranges.end(), chosen_id);
                if (it != tmp_assigned_ranges.end()) {
                    for (auto&& [range, blobs] : it->ranges)
                    {
                        if (range != segment.range && range.contains(frame)) {
#ifndef NDEBUG
                            remove_from.insert(range);
#endif
                            remove = true;
                            break;
                        }
                    }
                }
            }

            if (remove) {
#ifndef NDEBUG
                FormatWarning("[ignore] While assigning ", segment.range.start, "-", segment.range.end, " to ", chosen_id, " -> same fish already assigned in ranges ", remove_from);
#else
                FormatWarning("[ignore] While assigning ", segment.range.start, "-", segment.range.end, " to ", chosen_id, " -> same fish already assigned in another range.");
#endif
            }
            else {
                assert((int64_t)blob_ids.size() == (segment.range.end - segment.range.start + 1_f).get());
                AutoAssign::add_assigned_range(tmp_assigned_ranges, chosen_id, segment.range, std::move(blob_ids));

                auto blob = fish->blob(segment.start());
                if (blob && blob->split() && blob->parent_id().valid())
                    manual_splits[segment.start()].insert(blob->parent_id());

                assigned_ranges[fdx][segment.range] = chosen_id;
            }

            return;
        }
    }

    still_unassigned[fdx].insert(segment.range);
}
}

void Tracker::set_vi_data(const decltype(_vi_predictions)& predictions) {
    std::unique_lock g(_vi_mutex);
    _vi_predictions = std::move(predictions);
}
    
    void Tracker::check_segments_identities(bool auto_correct, IdentitySource source, std::function<void(float)> callback, const std::function<void(const std::string&, const std::function<void()>&, const std::string&)>& add_to_queue, Frame_t after_frame) {
        
        print("Waiting for lock...");
        LockGuard guard(w_t{}, "check_segments_identities");
        print("Updating automatic ranges starting from ", !after_frame.valid() ? Frame_t(0) : after_frame);
        
        if (source == IdentitySource::QRCodes)
            print("Using physical tag information.");
        else
            print("Using machine learning data.");

        std::atomic<size_t> count{0u};
        
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(after_frame.valid() ? Frame_t(0) : after_frame, fid);
        
#ifdef TREX_DEBUG_IDENTITIES
        auto f = fopen(file::DataLocation::parse("output", "identities.log").c_str(), "wb");
#endif
        float N = float(IndividualManager::num_individuals());
        IndividualManager::transform_parallel(recognition_pool, [&](auto, auto fish) {
            fish->clear_recognition();
            fish->calculate_average_recognition();
            
            callback(count / N * 0.5f);
            ++count;
        });
        
        using namespace v;
        Settings::manual_matches_t automatic_matches;
        std::map<fdx_t, VirtualFish> virtual_fish;
        
        // wrong fish -> set of unassigned ranges
        std::map<fdx_t, std::set<Range<Frame_t>>> unassigned_ranges;
        std::map<fdx_t, std::map<Range<Frame_t>, fdx_t>> assigned_ranges;
        
        std::vector<AutoAssign::RangesForID> tmp_assigned_ranges;
        //automatically_assigned_ranges.clear();
        
        /*static const auto compare = [](const std::pair<long_t, float>& pair0, const std::pair<long_t, float>& pair1){
            return pair0.second < pair1.second;
        };*/
        
        static const auto compare_greatest = [](const std::pair<Idx_t, Match::prob_t>& pair0, const std::pair<Idx_t, Match::prob_t>& pair1)
        {
            return pair0.second > pair1.second;
        };
        
        static const auto sigmoid = [](Match::prob_t x) {
            return 1.0/(1.0 + exp((0.5-x)*2.0*M_PI));
        };
        
        const size_t n_lower_bound =  source == IdentitySource::QRCodes ? 2  : max(5, SLOW_SETTING(frame_rate) * 0.1f);

        auto collect_virtual_fish = [n_lower_bound, &virtual_fish, &assigned_ranges
#ifdef TREX_DEBUG_IDENTITIES
            ,f
#endif
        ]
            (Idx_t fdx, Individual* fish, const FrameRange& range, size_t N, const std::map<Idx_t, float>& average)
        {
            if (N >= n_lower_bound || (range.start() == fish->start_frame() && N > 0)) {
            }
            else
                return false;

#ifdef TREX_DEBUG_IDENTITIES
            log(f, "fish ", fdx, ": segment ", range, " has ", N, " samples");
#endif
            //print("fish ",fdx,": segment ",segment.start(),"-",segment.end()," has ",n," samples");

            std::set<std::pair<Idx_t, Match::prob_t>, decltype(compare_greatest)> sorted(compare_greatest);
            sorted.insert(average.begin(), average.end());

            // check if the values for this segment are too close, this probably
            // means that we shouldnt correct here.
            if (sorted.size() >= 2) {
                Match::prob_t ratio = sorted.begin()->second / ((++sorted.begin())->second);
                if (ratio > 1)
                    ratio = 1 / ratio;

                if (ratio >= 0.6) {
#ifdef TREX_DEBUG_IDENTITIES
                    log(f, "\ttwo largest probs ", sorted.begin()->second, " and ", (++sorted.begin())->second, " are too close (ratio ", ratio, ")");
#endif
                    return false;
                }
            }

            //auto it = std::max_element(average.begin(), average.end(), compare);
            auto it = sorted.begin();

            // see if there is already something found for given segment that
            // overlaps with this segment
            auto fit = virtual_fish.find(it->first);
            if (fit != virtual_fish.end()) {
                // fish exists
                auto& A = range;

                std::set<range_t> matches;
                auto rit = fit->second.segments.begin();
                for (; rit != fit->second.segments.end(); ++rit) {
                    auto& B = *rit;
                    //if(B.overlaps(A))
                    if (A.end() > B.start() && A.start() < B.end())
                        //if((B.start() >= A.start() && A.end() >= B.start())
                        //   || (A.start() >= B.start() && B.end() >= A.start()))
                    {
                        matches.insert(B);
                    }
                }

                if (!matches.empty()) {
                    // if there are multiple matches, we can already assume that this
                    // is a much longer segment (because it overlaps multiple smaller segments
                    // because it starts earlier, cause thats the execution order)
                    auto rit = matches.begin();
#ifdef TREX_DEBUG_IDENTITIES
                    log(f, "\t", fdx, " (as ", it->first, ") Found range(s) ", *rit, " for search range ", range, " p:", fit->second.probs.at(*rit), " n:", fit->second.samples.at(*rit), " (self:", it->second, ",n:", N, ")");
#endif

                    Match::prob_t n_me = N;//segment.end() - segment.start();
                    Match::prob_t n_he = fit->second.samples.at(*rit);//rit->end() - rit->start();
                    const Match::prob_t N = n_me + n_he;

                    n_me /= N;
                    n_he /= N;

                    Match::prob_t sum_me = sigmoid(it->second) * sigmoid(n_me);
                    Match::prob_t sum_he = sigmoid(fit->second.probs.at(*rit)) * sigmoid(n_he);

#ifdef TREX_DEBUG_IDENTITIES
                    log(f, "\tself:", range.length(), " ", it->second, " other:", rit->length(), " ", fit->second.probs.at(*rit), " => ", sum_me, " / ", sum_he);
#endif

                    if (sum_me > sum_he) {
#ifdef TREX_DEBUG_IDENTITIES
                        log(f, "\t* Replacing");
#endif

                        for (auto rit = matches.begin(); rit != matches.end(); ++rit) {
                            fit->second.probs.erase(*rit);
                            fit->second.track_ids.erase(rit->range);
                            fit->second.segments.erase(*rit);
                            fit->second.samples.erase(*rit);
                        }

                    }
                    else
                        return false;
                }
            }

#ifdef TREX_DEBUG_IDENTITIES
            log(f, "\tassigning ", it->first, " to ", fdx, " with p ", it->second, " for ", range);
#endif
            virtual_fish[it->first].segments.insert(range);
            virtual_fish[it->first].probs[range] = it->second;
            virtual_fish[it->first].samples[range] = N;
            virtual_fish[it->first].track_ids[range.range] = fdx;

            assigned_ranges[fdx][range.range] = it->first;

            return true;
        };
        
        // iterate through segments, find matches for segments.
        // try to find the longest segments and assign them to virtual fish
        if(source == IdentitySource::QRCodes) {
            //! When using QRCodes...
            IndividualManager::transform_all([&](auto fdx, auto fish) {
                for (auto& [start, assign] : fish->qrcodes()) {
                    auto segment = fish->segment_for(start);
                    if (segment && segment->contains(start)) {
                        collect_virtual_fish(fdx, fish, *segment, assign.samples, {
                            { Idx_t(assign.best_id), assign.p }
                        });
                    }
                }
            });
            
        } else {
            //! When using other ML information (visual identification)...
            assert(source == IdentitySource::VisualIdent);
            
            IndividualManager::transform_all([&](auto fdx, auto fish) {
                //! TODO: MISSING recalculate recognition for all segments
                //fish->clear_recognition();

                for (auto&& [start, segment] : fish->recognition_segments()) {
                    if (after_frame.valid() && segment.end() < after_frame)
                        continue;

                    auto& [n, average] = fish->processed_recognition(start);
                    collect_virtual_fish(fdx, fish, segment, n, average);
                }
            });
        }
        
        Settings::manual_splits_t manual_splits;
        
#ifdef TREX_DEBUG_IDENTITIES
        log(f, "Found segments:");
#endif
        for(auto && [fdx, fish] : virtual_fish) {
#ifdef TREX_DEBUG_IDENTITIES
            log(f, "\t", fdx,":");
#endif
            // manual_match for first segment
            if(!fish.segments.empty()) {
                auto segment = *fish.segments.begin();
                
                if(!fish.probs.count(segment))
                    throw U_EXCEPTION("Cannot find ",segment.start(),"-",segment.end()," in fish.probs");
                if(!fish.track_ids.count(segment.range))
                    throw U_EXCEPTION("Cannot find ",segment.start(),"-",segment.end()," in track_ids");
                
                auto tid = fish.track_ids.at(segment.range);
                IndividualManager::transform_if_exists(tid, [&, fdx=fdx](auto track){
                    if(segment.first_usable.valid() && segment.first_usable != segment.start()) {
                        auto blob = track->compressed_blob(segment.first_usable);
                        if(blob)
                            automatic_matches[segment.first_usable][fdx] = blob->blob_id();
                        else
                            FormatWarning("Have first_usable (=", segment.first_usable,"), but blob is null (fish ",fdx,")");
                    }
                    
                    auto blob = track->compressed_blob(segment.start());
                    if(blob) {
                        automatic_matches[segment.start()][fdx] = blob->blob_id();
                        if(blob->split() && blob->parent_id.valid())
                            manual_splits[segment.start()].insert(blob->parent_id);
                    }
                    
                }).or_else([tid](auto message){
                    FormatWarning("Cannot find individual with ID ", tid, ": ", message);
                });
            }
            
            for(auto segment : fish.segments) {
                if(after_frame.valid() && segment.range.end < after_frame)
                    continue;
                
                if(!fish.probs.count(segment))
                    throw U_EXCEPTION("Cannot find ",segment.start(),"-",segment.end()," in fish.probs");
                if(!fish.track_ids.count(segment.range))
                    throw U_EXCEPTION("Cannot find ",segment.start(),"-",segment.end()," in track_ids");
#ifdef TREX_DEBUG_IDENTITIES
                log(f, "\t\t",segment,": ",fish.probs.at(segment)," (from ", fish.track_ids.at(segment.range),")");
#endif
                auto tid = fish.track_ids.at(segment.range);
                IndividualManager::transform_if_exists(tid, [&, fdx=fdx](auto track) {
                    assert(track->compressed_blob(segment.start()));
                    
                    //automatic_matches[segment.start()][fdx] = track->blob(segment.start())->blob_id();
                    if(!assigned_ranges.count(track->identity().ID()) || !assigned_ranges.at(track->identity().ID()).count(segment.range))
                        assigned_ranges[track->identity().ID()][segment.range] = fdx;
                    
                    auto blob = track->compressed_blob(segment.start());
                    if(blob && blob->split() && blob->parent_id.valid())
                        manual_splits[segment.start()].insert(blob->parent_id);
                    
                    std::vector<pv::bid> blob_ids;
                    for(Frame_t frame=segment.start(); frame<=segment.end(); ++frame) {
                        blob = track->compressed_blob(frame);
                        if(blob) {
                            //automatically_assigned_blobs[frame][blob->blob_id()] = fdx;
                            blob_ids.push_back(blob->blob_id());
                            //if(blob->split() && blob->parent_id().valid())
                            //    manual_splits[frame].insert(blob->parent_id());
                        } else
                            blob_ids.push_back(pv::bid::invalid);
                        
                        std::set<Range<Frame_t>> remove_from;
                        auto it = std::find(tmp_assigned_ranges.begin(), tmp_assigned_ranges.end(), fdx);
                        if(it != tmp_assigned_ranges.end()) {
                            for(auto & assign : it->ranges) {
                                if(assign.range != segment.range && assign.range.contains(frame)) {
                                    remove_from.insert(assign.range);
                                }
                            }
                        }
                        
                        if(!remove_from.empty()) {
                            for(auto range : remove_from) {
                                // find individual fdx,
                                // then delete range range:
                                auto it = std::find(tmp_assigned_ranges.begin(), tmp_assigned_ranges.end(), fdx);
                                if(it != tmp_assigned_ranges.end()) {
                                    for(auto dit = it->ranges.begin(); dit != it->ranges.end(); ) {
                                        if(dit->range == range) {
                                            it->ranges.erase(dit);
                                            break;
                                        }
                                    }
                                }
                            }
                            
                            FormatWarning("While assigning ",frame,",",blob ? (int64_t)blob->blob_id() : -1," to ",fdx," -> same fish already assigned in ranges ",remove_from);
                        }
                    }
                    
                    assert(Frame_t(blob_ids.size()) == segment.range.end - segment.range.start + 1_f);
                    AutoAssign::add_assigned_range(tmp_assigned_ranges, fdx, segment.range, std::move(blob_ids));
                    
                }).or_else([tid](auto message) {
                    FormatWarning("Cannot find individual with ID ", tid, ": ", message);
                });
            }
        }
#ifdef TREX_DEBUG_IDENTITIES
        log(f, "----");
#endif
        decltype(unassigned_ranges) still_unassigned;

        const auto _apply = [&](
           Idx_t fdx,
           const Individual* fish,
           const FrameRange& segment,
           Idx_t prev_id,
           Idx_t next_id,
           MotionRecord* prev_pos,
           MotionRecord* next_pos)
        {
           apply(
#ifdef TREX_DEBUG_IDENTITIES
                 f,
#endif
                 still_unassigned,
                 manual_splits,
                 assigned_ranges,
                 tmp_assigned_ranges,
                 fdx,
                 fish,
                 segment,
                 prev_id,
                 next_id,
                 prev_pos,
                 next_pos
           );
        };
        
        if(source == IdentitySource::QRCodes) {
            IndividualManager::transform_all([&](auto fdx, auto fish){
                process_qr(after_frame, fdx, fish, assigned_ranges, virtual_fish, _apply);
            });
            
        } else {
            assert(source == IdentitySource::VisualIdent);
            IndividualManager::transform_all([&](auto fdx, auto fish){
                process_vi(after_frame, fish->recognition_segments(), fdx, fish, assigned_ranges, virtual_fish, _apply);
            });
        }
        
        //auto str = prettify_array(Meta::toStr(still_unassigned));
        print("auto_assign is ", auto_correct ? 1 : 0);
        if(auto_correct) {
            add_to_queue("", [after_frame, automatic_matches, manual_splits, tmp_assigned_ranges = std::move(tmp_assigned_ranges)]() mutable {
                print("Assigning to queue from frame ", after_frame);
                
                //std::lock_guard<decltype(GUI::instance()->gui().lock())> guard(GUI::instance()->gui().lock());
                
                {
                    LockGuard guard(w_t{}, "check_segments_identities::auto_correct");
                    Tracker::instance()->_remove_frames(!after_frame.valid() ? Tracker::analysis_range().start() : after_frame);
                    IndividualManager::transform_all([](auto, auto fish){
                        fish->clear_recognition();
                    });
                    
                    print("automatically_assigned_ranges ", tmp_assigned_ranges.size());
                    AutoAssign::set_automatic_ranges(std::move(tmp_assigned_ranges));
                    
                    if(!after_frame.valid()) {
                        Settings::set<Settings::Variables::manual_matches>(Settings::manual_matches_t{automatic_matches});
                        Settings::set<Settings::Variables::manual_splits>(Settings::manual_splits_t{manual_splits});
                    }
                }
                
                if(!after_frame.valid())
                    SETTING(manual_matches) = automatic_matches;
                if(!after_frame.valid())
                    SETTING(manual_splits) = manual_splits;
                
                Tracker::analysis_state(Tracker::AnalysisState::UNPAUSED);
            }, "");
        }
        
#ifdef TREX_DEBUG_IDENTITIES
        log(f, "Done.");
        if(f)
            fclose(f);
#endif
    }

pv::BlobPtr Tracker::find_blob_noisy(const PPFrame& pp, pv::bid bid, pv::bid, const Bounds&)
{
    auto blob = pp.bdx_to_ptr(bid);
    if(!blob) {
        return nullptr;
        
        /*if(pid.valid()) {
            blob = pp.bdx_to_ptr(pid);
            if(blob) {
                auto blobs = pixel::threshold_blob(blob, FAST_SETTING(track_threshold), Tracker::instance()->background());
                
                for(auto && sub : blobs) {
                    if(sub->blob_id() == bid) {
                        //print("Found perfect match for ", bid," in blob ",b->blob_id());//blob_to_id[bid] = sub;
                        //sub->calculate_moments();
                        return std::move(sub);
                        //break;
                    }
                }
            }
        }*/
        
        /*if(!blob_to_id.count(bid)) {
            //std::set<std::tuple<Match::PairingGraph::prob_t, long_t, Vec2>> sorted;
            //for(auto && [id, ptr] : blob_to_id) {
            //    sorted.insert({euclidean_distance(ptr->center(), bounds.pos() + bounds.size() * 0.5), id, ptr->center()});
            //}
            //auto str = Meta::toStr(sorted);
            
            //Error("Cannot find blob %d (%.0f,%.0f) in frame %d with threshold=%d. (%S)", bid, bounds.x,bounds.y, frame, FAST_SETTING(track_threshold), &str);
            return nullptr;
        }*/
        
        return nullptr;
    }
    
    return pv::Blob::Make(*blob);
}

    void Tracker::check_save_tags(Frame_t frameIndex, const ska::bytell_hash_map<pv::bid, Individual*>& blob_fish_map, const std::vector<tags::blob_pixel> &tagged_fish, const std::vector<tags::blob_pixel> &noise, const file::Path &) {
        static Timing tag_timing("tags", 0.1);
        TakeTiming take(tag_timing);
        
        auto result = tags::prettify_blobs(tagged_fish, noise, {}, *_average);
        for (auto &r : result) {
            auto && [var, bid, ptr, f] = tags::is_good_image(r);
            if(ptr) {
                auto it = blob_fish_map.find(r.bdx);
                if(it != blob_fish_map.end())
                    it->second->add_tag_image(tags::Tag{var, r.bdx, std::move(ptr), frameIndex} /* ptr? */);
                else
                    FormatWarning("Blob ", r.bdx," in frame ",frameIndex," contains a tag, but is not associated with an individual.");
            }
        }
        
        auto active = IndividualManager::active_individuals(frameIndex - 1_f);
        if(active) {
            for(auto fish : *active.value()) {
                if(fish->start_frame() < frameIndex && fish->has(frameIndex - 1_f) && !fish->has(frameIndex))
                {
                    // - none
                }
            }
        }
    }
    
    void Tracker::auto_calculate_parameters(pv::File& video, bool quiet) {
        if(video.length() > 1000_f && (SETTING(auto_minmax_size) || SETTING(auto_number_individuals))) {
            gpuMat average;
            video.average().copyTo(average);
            if(average.cols == video.size().width && average.rows == video.size().height)
                video.processImage(average, average);
            
            Image local_average(average.rows, average.cols, 1);
            average.copyTo(local_average.get());
            
            if(!quiet)
                print("Determining blob size in ", video.length()," frames...");
            
            Median<float> blob_size;
            pv::Frame frame;
            std::multiset<float> values;
            const uint32_t number_fish = SETTING(track_max_individuals).value<uint32_t>();
            
            std::vector<std::multiset<float>> blobs;
            Median<float> median;
            
            auto step = Frame_t((video.length().get() - video.length().get()%500) / 500);
            for (Frame_t i=0_f; i<video.length(); i+=step) {
                video.read_frame(frame, i);
                
                std::multiset<float> frame_values;
                
                for (size_t k=0; k<frame.n(); k++) {
                    auto pair = imageFromLines(*frame.mask().at(k), NULL, NULL, NULL, frame.pixels().at(k).get(), SETTING(track_threshold).value<int>(), &local_average);
                    
                    float value = pair.second * SQR(SETTING(cm_per_pixel).value<float>());
                    if(value > 0) {
                        frame_values.insert(value);
                    }
                }
                
                blobs.push_back(frame_values);
                
                if(!frame_values.empty()) {
                    auto result = percentile(frame_values, {0.75, 0.90});
                    /*if(*frame_values.rbegin() > 10) {
                        auto str = Meta::toStr(frame_values);
                        auto str0 = Meta::toStr(result);
                        print(i / step,": ",*frame_values.rbegin()," ",str0.c_str());
                    }*/
                    
                    values.insert(result.begin(), result.end());
                    median.addNumber(*result.begin());
                    median.addNumber(*result.rbegin());
                }
                //values.insert(percentile(frame_values, 0.75));
                //values.insert(percentile(frame_values, 0.90));
            }
            
            /*float middle = 0;
            for(auto &v : values)
                middle += v;
            if(!values.empty())
                middle /= float(values.size());*/
            
            auto ranges = percentile(values, {0.25, 0.75});
            /*middle = median.getValue();
            middle = (ranges[1] - ranges[0]) * 0.5 + ranges[0];*/
            
            if(SETTING(auto_minmax_size))
                SETTING(blob_size_ranges) = BlobSizeRange({Rangef(ranges[0] * 0.25, ranges[1] * 1.75)});
            
            auto blob_range = SETTING(blob_size_ranges).value<BlobSizeRange>();
            
            std::multiset<size_t> number_individuals;
            
            for(auto && vector : blobs) {
                size_t number = 0;
                std::map<size_t, Median<float>> min_ratios;
                std::map<size_t, float> sizes;
                for(auto v : vector) {
                    if(blob_range.in_range_of_one(v))
                        ++number;
                }
                
                number_individuals.insert(number);
                //number_median.addNumber(number);
            }
            
            uint64_t median_number = percentile(number_individuals, 0.95);
            //median_number = number_median.getValue();
            
            //if(!quiet)
            
            if(median_number != number_fish) {
                if(!quiet)
                    FormatWarning("The set (", number_fish,") number of individuals differs from the detected number of individuals / frame (",median_number,").");
                
                if(SETTING(auto_number_individuals).value<bool>()) {
                    if(!quiet)
                        print("Setting number of individuals as ", median_number,".");
                    SETTING(track_max_individuals) = uint32_t(median_number);
                }
            }
        }
    }
}
