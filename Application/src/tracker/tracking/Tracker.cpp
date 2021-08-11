#include "Tracker.h"
#include <misc/GlobalSettings.h>
#include <tracking/SplitBlob.h>
#include <misc/Timer.h>
#include <iomanip>
#include <random>
#include "PairingGraph.h"
#include <misc/OutputLibrary.h>
#include <tracking/DetectTag.h>
#include <misc/cnpy_wrapper.h>
#include <processing/CPULabeling.h>
#include <misc/ReverseAdapter.h>
#include <processing/CPULabeling.h>
#include <misc/ProximityGrid.h>
#include <tracking/Recognition.h>
#include <misc/default_settings.h>
#include <python/GPURecognition.h>
#include <misc/pretty.h>
#include <tracking/DatasetQuality.h>
#include <gui/gui.h>
#include <misc/PixelTree.h>
#include <misc/CircularGraph.h>
#include <misc/MemoryStats.h>
#include <gui/WorkProgress.h>
#include <tracking/Categorize.h>

#ifndef NDEBUG
//#define PAIRING_PRINT_STATS
#endif

namespace track {
    auto *tracker_lock = new std::recursive_timed_mutex;

    std::shared_ptr<std::ofstream> history_log;
    std::mutex log_mutex;
    inline void Log(std::ostream* out, const char* cmd, ...) {
        if(!out)
            return;
        
        va_list args;
        va_start(args, cmd);
        
        std::string str;
        DEBUG::ParseFormatString(str, cmd, args);
        if(dynamic_cast<std::ofstream*>(out)) {
            str = settings::htmlify(str) + "</br>";
        }
        
        std::lock_guard<std::mutex> guard(log_mutex);
        *out << str << std::endl;
        
        va_end(args);
    }
    
    Tracker* _instance = NULL;
    std::vector<Rangel> _global_segment_order;

    Tracker* Tracker::instance() {
        return _instance;
    }
    
    inline void analyse_posture_pack(long_t frameIndex, const std::vector<std::tuple<Individual*, std::shared_ptr<Individual::BasicStuff>>>& p) {
        Timer t;
        double collected = 0;
        for(auto && [f, b] : p) {
            t.reset();
            f->save_posture(b, frameIndex);
            collected += t.elapsed();
        }
        
        std::lock_guard<std::mutex> guard(Tracker::instance()->_statistics_mutex);
        Tracker::instance()->_statistics[frameIndex].combined_posture_seconds += narrow_cast<float>(collected);
    }
    
    //std::map<long_t, std::map<uint32_t, long_t>> automatically_assigned_blobs;
    std::map<Idx_t, std::map<Rangel, std::vector<int64_t>>> automatically_assigned_ranges;
    
    std::map<Idx_t, int64_t> Tracker::automatically_assigned(long_t frame) {
        //LockGuard guard;
        std::map<Idx_t, int64_t> blob_for_fish;
        
        for(auto && [fdx, bff] : automatically_assigned_ranges) {
            blob_for_fish[fdx] = -1;
            
            for(auto && [range, blob_ids] : bff) {
                if(range.contains(frame)) {
                    assert(frame >= range.start && range.end >= frame);
                    blob_for_fish[fdx] = blob_ids.at(sign_cast<size_t>(frame - range.start));
                    break;
                }
            }
        }
        
        return blob_for_fish;
    }

static std::string _last_thread = "<none>", _last_purpose = "";
static Timer _thread_holding_lock_timer;
static std::map<std::string, Timer> _last_printed_purpose;
static std::thread::id _last_thread_id;

Tracker::LockGuard::~LockGuard() {
    if(lock) {
        if(_set_name) {
            if(_timer.elapsed() >= 0.1) {
                auto name = get_thread_name();
                if(_last_printed_purpose.find(_purpose) == _last_printed_purpose.end() || _last_printed_purpose[_purpose].elapsed() >= 10) {
                    auto str = Meta::toStr(DurationUS{uint64_t(_timer.elapsed() * 1000 * 1000)});
                    Debug("thread '%S' held the lock for %S with purpose '%S'", &name, &str, &_purpose);
                    _last_printed_purpose[_purpose].reset();
                }
            }
            
            _last_purpose = "";
            _last_thread = "<none>";
            _thread_holding_lock_timer.reset();
            _last_thread_id = std::thread::id();
        }
        delete lock;
    }
}

Tracker::LockGuard::LockGuard(std::string purpose, uint32_t timeout_ms) : _purpose(purpose), _set_name(false)
{
    assert(Tracker::instance());
    assert(!purpose.empty());
    lock = NULL;
    
    if(timeout_ms) {
        auto duration = std::chrono::milliseconds(timeout_ms);
        if(!tracker_lock->try_lock_for(duration)) {
            // did not get the lock... :(
            return;
        }
        
    } else {
        auto duration = std::chrono::milliseconds(10);
        Timer timer, print_timer;
        while(true) {
            if(tracker_lock->try_lock_for(duration)) {
                // acquired the lock :)
                break;
                
            } else if(timer.elapsed() > 10 && print_timer.elapsed() > 10) {
                auto name = _last_thread;
                auto myname = get_thread_name();
                Warning("(%S) Possible dead-lock with '%S' ('%S') thread holding the lock for %.2fs (waiting for %.2fs, current purpose is '%S')", &myname, &name, &_last_purpose, _thread_holding_lock_timer.elapsed(), timer.elapsed(), &_purpose);
                print_timer.reset();
            }
        }
    }
    
    lock = new std::lock_guard<std::recursive_timed_mutex>(*tracker_lock, std::adopt_lock);
    
    auto my_id = std::this_thread::get_id();
    if(my_id != _last_thread_id) {
        _last_thread = get_thread_name();
        _last_purpose = _purpose;
        _thread_holding_lock_timer.reset();
        _timer.reset();
        _last_thread_id = my_id;
        _set_name = true;
    }
}

static CacheHints _properties_cache;
static std::shared_mutex _properties_mutex;

const FrameProperties* Tracker::properties(long_t frameIndex, const CacheHints* hints) {
    if(frameIndex < 0)
        return nullptr;
    
    if(hints) {
        //! check if its just meant to disable it
        if(hints != (const CacheHints*)0x1) {
            auto ptr = hints->properties(frameIndex);
            if(ptr)
                return ptr;
        }
        
    } else {
        std::shared_lock guard(_properties_mutex);
        auto ptr = _properties_cache.properties(frameIndex);
        if(ptr)
            return ptr;
    }
    
    auto &frames = instance()->frames();
    auto it = properties_iterator(frameIndex);
    return it != frames.end() ? &(*it) : nullptr;
}

decltype(Tracker::_added_frames)::const_iterator Tracker::properties_iterator(long_t frameIndex) {
    auto &frames = instance()->frames();
    
    auto it = std::upper_bound(frames.begin(), frames.end(), frameIndex, [](long_t frame, const FrameProperties& prop) -> bool {
        return frame < prop.frame;
    });
    
    if((it == frames.end() && !frames.empty()) || (it != frames.begin())) {
        --it;
        
        if(it->frame == frameIndex) {
            return it;
        }
    }
    
    return frames.end();
}

    std::string Tracker::thread_name_holding() {
        return _last_thread;
    }
        
    void Tracker::print_memory() {
        LockGuard guard("print_memory");
        mem::TrackerMemoryStats stats;
        stats.print();
    }

    void Tracker::delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range) {
        auto it = automatically_assigned_ranges.find(fish_id);
        if(it == automatically_assigned_ranges.end()) {
            Except("Cannot find fish %d in automatic assignments");
            return;
        }
        
        std::set<Rangel> ranges_to_remove;
        for(auto && [range, blob_ids] : it->second) {
            if(frame_range.overlaps(range)) {
                ranges_to_remove.insert(range);
            }
        }
        for(auto range : ranges_to_remove)
            it->second.erase(range);
    }

    bool callback_registered = false;
    
    Recognition* Tracker::recognition() {
        if(!_instance)
            U_EXCEPTION("There is no valid instance if Tracker available (Tracker::recognition).");
        
        return _instance->_recognition;
    }

void Tracker::analysis_state(AnalysisState pause) {
    if(!instance())
        U_EXCEPTION("No tracker instance can be used to pause.");
    instance()->recognition_pool.enqueue([](bool value){
        SETTING(analysis_paused) = value;
    }, pause == AnalysisState::PAUSED);
}

    Tracker::Tracker()
          : _thread_pool(max(1u, cmn::hardware_concurrency())),
            recognition_pool(max(1u, cmn::hardware_concurrency())),
            _midline_errors_frame(0), _overall_midline_errors(0),
            _startFrame(-1), _endFrame(-1), _max_individuals(0),
            _background(NULL), _recognition(NULL),
            _approximative_enabled_in_frame(std::numeric_limits<long_t>::lowest()),
            _inactive_individuals([this](Idx_t A, Idx_t B){
                auto it = _individuals.find(A);
                assert(it != _individuals.end());
                const Individual* a = it->second;
                
                it = _individuals.find(B);
                assert(it != _individuals.end());
                const Individual* b = it->second;
        
                return a->end_frame() > b->end_frame() || (a->end_frame() == b->end_frame() && A > B);
            })
    {
        _instance = this;
        if(!SETTING(quiet))
            Debug("Initialized with %ld threads.", _thread_pool.num_threads());
        
        Settings::set_callback(Settings::outline_resample, [](auto&, auto&value){
            static_assert(std::is_same<Settings::outline_resample_t, float>::value, "outline_resample assumed to be float.");
            auto v = value.template value<float>();
            if(v <= 0) {
                Warning("outline_resample defaulting to 1.0 instead of %f", v);
                SETTING(outline_resample) = 1.f;
            }
        });
        Settings::set_callback(Settings::manually_approved, [](auto&, auto&){
            if(recognition() && recognition()->dataset_quality()) {
                recognition()->update_dataset_quality();
            }
        });
        
        auto track_list_update = [](auto&key, auto&value)
        {
            auto update = [key = key, tmp = value.template value<Settings::track_ignore_t>()]() mutable
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
                
                if(changed && GUI::instance()) {
                    std::lock_guard<std::recursive_mutex> guard(GUI::instance()->gui().lock());
                    GlobalSettings::get(key) = tmp;
                } else if(changed) {
                    GlobalSettings::get(key) = tmp;
                }
            };
            
            if(GUI::instance()) {
                GUI::work().add_queue("", update);
            } else
                update();
        };
        Settings::set_callback(Settings::track_ignore, track_list_update);
        Settings::set_callback(Settings::track_include, track_list_update);
        Settings::set_callback(Settings::frame_rate, [this](auto&, auto&){
            std::unique_lock guard(_properties_mutex);
            _properties_cache.clear(); //! TODO: need to refill as well
        });
        Settings::set_callback(Settings::posture_direction_smoothing, [](auto&key, auto&value) {
            static_assert(std::is_same<Settings::posture_direction_smoothing_t, size_t>::value, "posture_direction_smoothing assumed to be size_t.");
            size_t v = value.template value<size_t>();
            
            if(v != FAST_SETTINGS(posture_direction_smoothing))
            {
                auto worker = [key](){
                    LockGuard guard("Updating midlines in changed_setting("+key+")");
                    
                    for (auto && [id, fish] : Tracker::individuals()) {
                        Tracker::instance()->_thread_pool.enqueue([](long_t id, Individual *fish){
                            Debug("\t%d", id);
                            fish->clear_post_processing();
                            fish->update_midlines(nullptr);
                        }, id, fish);
                    }
                    
                    Tracker::instance()->_thread_pool.wait();
                    if(Tracker::recognition() && Tracker::recognition()->dataset_quality()) {
                        Tracker::recognition()->dataset_quality()->remove_frames(start_frame());
                        Tracker::recognition()->update_dataset_quality();
                    }
                };
                
                if(GUI::instance()) {
                    GUI::work().add_queue("updating midlines / head positions...", worker);
                } else
                    worker();
            }
        });
        
        if (!callback_registered) {
            auto ptr = "Tracker::Settings";
            auto variable_changed = [ptr](sprite::Map::Signal signal, auto&map, auto&key, auto&value)
            {
                if(signal == sprite::Map::Signal::EXIT) {
                    map.unregister_callback(ptr);
                    return;
                }
                
                if(contains(Settings::names(), key)) {
                    Tracker::LockGuard guard("changed_settings");
                    Settings :: variable_changed(signal, map, key, value);
                }
            };
            cmn::GlobalSettings::map().register_callback(ptr, variable_changed);
            for(auto &n : Settings :: names())
                variable_changed(sprite::Map::Signal::NONE, cmn::GlobalSettings::map(), n, cmn::GlobalSettings::get(n).get());
            
        }
        
        _recognition = new Recognition();
    }
    Tracker::~Tracker() {
        assert(_instance);
        Settings::clear_callbacks();
        
        _thread_pool.force_stop();
        if(!SETTING(quiet))
            Debug("Waiting for recognition...");
        recognition_pool.force_stop();
        if(!SETTING(quiet))
            Debug("Done waiting.");
        
        _instance = NULL;
        
        auto individuals = _individuals;
        for (auto& fish_ptr : individuals)
            delete fish_ptr.second;
        
        emergency_finish();
    }
    
    void Tracker::emergency_finish() {
        std::lock_guard<std::mutex> guard(log_mutex);
        if(history_log != nullptr && history_log->is_open()) {
            Debug("Closing history log.");
            *history_log << "</body></html>";
            history_log->flush();
            history_log->close();
        }
        history_log = nullptr;
    }

    void Tracker::prepare_shutdown() {
        _thread_pool.force_stop();
        recognition_pool.force_stop();
        _recognition->prepare_shutdown();
        Match::PairingGraph::prepare_shutdown();
        PythonIntegration::quit();
    }

    long_t Tracker::update_with_manual_matches(const Settings::manual_matches_t& manual_matches) {
        LockGuard guard("update_with_manual_matches");
        
        static std::atomic_bool first_run(true);
        static Settings::manual_matches_t compare = manual_matches;
        if(first_run) {
            first_run = false;
            //auto str = Meta::toStr(compare);
            //SETTING(manual_matches) = manual_matches;
            //Debug("Manual matches have been updated %S.", &str);
            return -1;
            
        } else {
            auto str0 = Meta::toStr(compare), str1 = Meta::toStr(manual_matches);
            //Debug("Manual matches have been updated %S -> %S.", &str0, &str1);
            auto copy = manual_matches; // another copy
            auto next = copy;
            
            // iterate over old to find frames that are not in the current version
            long_t first_change = -1;
            
            auto itn = compare.begin(), ito = copy.begin();
            for (; itn != compare.end() && ito != copy.end(); ++itn, ++ito) {
                if(itn->first != ito->first || itn->second != ito->second) {
                    first_change = min(itn->first, ito->first);
                    break;
                }
            }
            
            // if one of the iterators reached the end, but the other one didnt
            if((itn == compare.end()) ^ (ito == copy.end())) {
                if(itn == compare.end())
                    first_change = ito->first;
                else
                    first_change = itn->first;
            }
            
            //Debug("First changed frame is %d", first_change);
            if(first_change != -1 && first_change <= Tracker::end_frame()) {
                //bool analysis_paused = SETTING(analysis_paused);
                GUI::reanalyse_from(first_change, true);
                //if(!analysis_paused)
                Tracker::analysis_state(Tracker::AnalysisState::UNPAUSED);
            }
            
            //SETTING(manual_matches) = next;
            //FAST_SETTINGS(manual_matches) = next;
            //auto str = Meta::toStr(FAST_SETTINGS(manual_matches));
            //Debug("Updating fast settings with %S", &str);
            compare = next;
            
            return first_change;
        }
    }

bool operator<(long_t frame, const FrameProperties& props) {
    return frame < props.frame;
}

    //! Assumes a sorted array.
    template<typename T, typename Q>
    inline bool contains_sorted(const Q& v, T obj) {
        auto it = std::lower_bound(v.begin(), v.end(), obj, [](const auto& v, T number) -> bool {
            return v < number;
        });
        
        if(it != v.end()) {
            auto end = std::upper_bound(it, v.end(), obj, [](T number, const auto& v) -> bool {
                return number < v;
            });
            
            if(end == v.end() || !((*end) < obj)) {
                return true;
            }
        }
        
        return false;
    }

    void Tracker::add(PPFrame &frame) {
        static Timing timing("Tracker::add(PPFrame)", 10);
        TakeTiming take(timing);
        
        Timer overall_timer;
        LockGuard guard("Tracker::add(PPFrame)");
        
        assert(frame.index() != -1);
        
        if (contains_sorted(_added_frames, frame.index())) {
            Warning("Frame %d already in tracker.", frame.index());
            return;
        }
        
        if(frame.frame().timestamp() > uint64_t(INT64_MAX)) {
            Warning("frame timestamp is bigger than INT64_MAX! (%f time)", time);
        }
        
        auto props = properties(frame.index() - 1);
        if(props && frame.frame().timestamp() < props->org_timestamp) {
            Error("Cannot add frame with timestamp smaller than previous timestamp. Frames have to be in order. Skipping.");
            return;
        }
        
        if(_startFrame != -1 && frame.index() < _endFrame+1)
            throw new UtilsException("Cannot add intermediate frames out of order.");
        
        history_split(frame, _active_individuals, history_log != nullptr && history_log->is_open() ? history_log.get() : nullptr, &_thread_pool);
        add(frame.index(), frame);
        
        //! Update recognition if enabled and end of video reached
        if(Recognition::recognition_enabled()) {
            const long_t video_length = Tracker::analysis_range().end;
            if(frame.index() >= video_length)
                Recognition::notify();
        }
        
        std::lock_guard<std::mutex> lguard(_statistics_mutex);
        _statistics[frame.index()].adding_seconds = (float)overall_timer.elapsed();
        _statistics[frame.index()].loading_seconds = (float)frame.frame().loading_time();
    }

    class PairProbability {
    private:
        GETTER_PTR(Individual*, idx)
        GETTER_PTR(pv::BlobPtr, bdx)
        GETTER(Match::prob_t, p)
        
    public:
        PairProbability() {}
        PairProbability(Individual* idx, pv::BlobPtr bdx, Match::prob_t p)
            : _idx(idx), _bdx(bdx), _p(p)
        {}
        
        bool operator<(const PairProbability& other) const {
            return std::make_tuple(_p, _idx->identity().ID(), _bdx->blob_id()) < std::make_tuple(other._p, other._idx->identity().ID(), other._bdx->blob_id());
        }
        bool operator>(const PairProbability& other) const {
            return std::make_tuple(_p, _idx->identity().ID(), _bdx->blob_id()) > std::make_tuple(other._p, other._idx->identity().ID(), other._bdx->blob_id());
        }
        bool operator<=(const PairProbability& other) const {
            return std::make_tuple(_p, _idx->identity().ID(), _bdx->blob_id()) <= std::make_tuple(other._p, other._idx->identity().ID(), other._bdx->blob_id());
        }
        bool operator>=(const PairProbability& other) const {
            return std::make_tuple(_p, _idx->identity().ID(), _bdx->blob_id()) >= std::make_tuple(other._p, other._idx->identity().ID(), other._bdx->blob_id());
        }
        bool operator==(const PairProbability& other) const {
            return std::make_tuple(_p, _idx->identity().ID(), _bdx->blob_id()) == std::make_tuple(other._p, other._idx->identity().ID(), other._bdx->blob_id());
        }
    };

    void Tracker::update_history_log() {
        Tracker::LockGuard guard("update_history_log");
        if(history_log == nullptr && !SETTING(history_matching_log).value<file::Path>().empty()) {
            history_log = std::make_shared<std::ofstream>();
            
            auto path = SETTING(history_matching_log).value<file::Path>();
            if(!path.empty()) {
                path = pv::DataLocation::parse("output", path);
                DebugCallback("Opening history_log at '%S'...", &path.str());
                
                history_log->open(path.str(), std::ios_base::out | std::ios_base::binary);
                if(history_log->is_open()) {
                    auto &ss = *history_log;
                    ss << "<html><head>";
                    ss << "<style>";
                    ss << "map{ \
                    display: table; \
                    width: 100%; \
                    } \
                    row { \
                    display: table-row; \
                    } \
                    row.header { \
                    background-color: #EEE; \
                    } \
                    key, value, doc { \
                    border: 1px solid #999999; \
                    display: table-cell; \
                    padding: 3px 10px; \
                    } \
                    row.readonly { color: gray; background-color: rgb(242, 242, 242); } \
                    doc { overflow-wrap: break-word; }\
                    value { overflow-wrap: break-word;max-width: 300px; }\
                    row.header { \
                    background-color: #EEE; \
                    font-weight: bold; \
                    } \
                    row.footer { \
                    background-color: #EEE; \
                    display: table-footer-group; \
                    font-weight: bold; \
                    } \
                    string { display:inline; color: red; font-style: italic; }    \
                    ref { display:inline; font-weight:bold; } ref:hover { color: gray; } \
                    number { display:inline; color: green; } \
                    keyword { display:inline; color: purple; } \
                    .body { \
                    display: table-row-group; \
                    }";
                    
                    ss <<"</style>";
                    ss <<"</head><body>";
                }
            }
        }
    }
    
    void Tracker::preprocess_frame(PPFrame& frame, const Tracker::set_of_individuals_t& active_individuals, GenericThreadPool* pool, std::ostream* out, bool do_history_split)
    {
        double time = frame.frame().timestamp() / double(1000*1000);
        
        //! Free old memory
        frame.clear();
        
        frame.time = time;
        frame.timestamp = frame.frame().timestamp();
        frame.set_index(frame.frame().index());
        //assert(frame.index() == frame.frame().index());
        frame.init_from_blobs(std::move(frame.frame().get_blobs()));
        
        filter_blobs(frame, pool);
        frame.fill_proximity_grid();
        
        if(do_history_split) {
            //Tracker::LockGuard guard("preprocess_frame");
            Tracker::instance()->history_split(frame, active_individuals, out, pool);
        }
    }
            
    bool Tracker::blob_matches_shapes(const pv::BlobPtr & b, const std::vector<std::vector<Vec2> > & shapes) {
        for(auto &rect : shapes) {
            if(rect.size() == 2) {
                // its a boundary
                if(Bounds(rect[0], rect[1] - rect[0]).contains(b->center()))
                {
                    return true;
                }
                
            } else if(rect.size() > 2) {
                // its a polygon
                if(pnpoly(rect, b->center())) {
                    return true;
                }
            }
#ifndef NDEBUG
            else {
                static bool warned = false;
                if(!warned) {
                    auto str = Meta::toStr(rect);
                    Warning("Array of numbers %S is not a polygon (or rectangle).", &str);
                    warned = true;
                }
            }
#endif
        }
        
        return false;
    }
    
    void Tracker::prefilter(const std::shared_ptr<Tracker::PrefilterBlobs>& result, std::vector<pv::BlobPtr>::const_iterator it, std::vector<pv::BlobPtr>::const_iterator end)
    {
        static Timing timing("prefilter", 10);
        TakeTiming take(timing);
        
        const float cm_sqr = SQR(FAST_SETTINGS(cm_per_pixel));
        
        auto &big_blobs = result->big_blobs;
        auto &filtered  = result->filtered;
        auto &filtered_out = result->filtered_out;

        const auto track_include = FAST_SETTINGS(track_include);
        const auto track_ignore = FAST_SETTINGS(track_ignore);
        
        std::vector<pv::BlobPtr> ptrs;
        auto only_allowed = FAST_SETTINGS(track_only_categories);
        
        auto check_blob = [&track_ignore, &track_include, &result, &cm_sqr](pv::BlobPtr& b){
            if(b->pixels()->size() * cm_sqr > result->fish_size.max_range().end * 100)
                b->force_set_recount(result->threshold);
            else
                b->recount(result->threshold, *result->background);
            
            if (!track_ignore.empty()) {
                if (blob_matches_shapes(b, track_ignore)) {
                    result->filter_out(b);
                    return false;
                }
            }

            if (!track_include.empty()) {
                if (!blob_matches_shapes(b, track_include)) {
                    result->filter_out(b);
                    return false;
                }
            }
            
            return true;
        };
        
        for(; it != end; ++it) {
            ptrs.clear();
            
            auto b = *it;
            
            if(!check_blob(b))
                continue;
            
            float recount = b->recount(-1);
            
            // TODO: magic numbers
            //! If the size is appropriately big, try to split the blob using the minimum of threshold and
            //  posture_threshold. Using the minimum ensures that the thresholds dont depend on each other
            //  as the threshold used here will reduce the number of available pixels for posture analysis
            //  or tracking respectively (pixels below used threshold will be removed).
            if(result->fish_size.close_to_minimum_of_one(recount, 0.5)) {
                Timer timer;
                auto pblobs = pixel::threshold_blob(b, result->threshold, result->background);
                
                // only use blobs that split at least into 2 new blobs
                for(auto &add : pblobs) {
                    add->set_split(false, b); // set_split even if the blob has just been thresholded normally?
                    if(!check_blob(add))
                        continue;

                    ptrs.push_back(add);
                }

                if (ptrs.empty()) {
                    ptrs.push_back(b);
                }
                
            } else {
                ptrs.push_back(b);
            }
            
            //! actually add the blob(s) to the filtered/filtered_out arrays
            for(auto& ptr : ptrs) {
                //if(!result->fish_size.close_to_maximum_of_one( ptr->pixels()->size() * cm_sqr, 100))
                //    ptr->force_set_recount(result->threshold);
                //recount = ptr->recount(result->threshold, *result->background);
                recount = ptr->recount(-1);

                if(result->fish_size.in_range_of_one(recount)) {
                    if(FAST_SETTINGS(track_threshold_2) > 0) {
                        auto second_count = ptr->recount(FAST_SETTINGS(track_threshold_2), *result->background);
                        
                        ptr->force_set_recount(result->threshold, recount / cm_sqr);
                        
                        if(!(FAST_SETTINGS(threshold_ratio_range) * recount).contains(second_count)) {
                            result->filter_out(ptr);
                            continue;
                        }
                    }
                    
                    if(!only_allowed.empty()) {
                        auto ldx = Categorize::DataStore::_ranged_label_unsafe(Frame_t(result->frame_index), ptr->blob_id());
                        if(ldx == -1 || !contains(only_allowed, Categorize::DataStore::label(ldx)->name)) {
                            result->filter_out(ptr);
                            continue;
                        }
                    }
                    
                    //! only after all the checks passed, do we commit the blob
                    /// to the "filtered" array:
                    result->commit(ptr);
                    
                } else if(recount < result->fish_size.max_range().start) {
                    result->filter_out(ptr);
                } else
                    big_blobs.push_back(ptr);
            }
        }
        
        /*if(!big_blobs.empty()) {
            Debug("Frame %d: %d big blobs", result->frame_index, big_blobs.size());
        }*/
        
        for(auto &blob : filtered)
            blob->calculate_moments();
        
        if (result->frame_index == Tracker::start_frame() || Tracker::start_frame() == -1)
            big_blobs = Tracker::instance()->split_big(
                    BlobReceiver(*result, BlobReceiver::noise),
                    big_blobs,
                    {});

        if (!only_allowed.empty()) {
            for (auto it = big_blobs.begin(); it != big_blobs.end(); ) {
                auto ldx = Categorize::DataStore::_ranged_label_unsafe(Frame_t(result->frame_index), (*it)->blob_id());
                if (ldx == -1 || !contains(only_allowed, Categorize::DataStore::label(ldx)->name)) {
                    result->filter_out(*it);
                    it = big_blobs.erase(it);
                    continue;
                }
                ++it;
            }
        }

        if (!big_blobs.empty()) {
            result->commit(big_blobs);
            big_blobs.clear();
        }
    }

    void Tracker::filter_blobs(PPFrame& frame, GenericThreadPool *pool) {
        static Timing timing("filter_blobs", 20);
        TakeTiming take(timing);
        
        const BlobSizeRange fish_size = FAST_SETTINGS(blob_size_ranges);
        const uint32_t num_blobs = (uint32_t)frame.blobs().size();
        const int threshold = FAST_SETTINGS(track_threshold);
        
        //static const unsigned concurrentThreadsSupported = cmn::hardware_concurrency();
        
        //static Timing initial_filter("initial_filter", 1);
        //initial_filter.start_measure();
        
        size_t available_threads = 1 + (pool ? pool->num_threads() : 0);
        size_t maximal_threads = frame.blobs().size();
        size_t needed_threads = min(maximal_threads / (size_t)FAST_SETTINGS(blobs_per_thread), available_threads);
        std::shared_lock guard(Categorize::DataStore::range_mutex());
        
        if (maximal_threads > 1 && needed_threads > 1 && available_threads > 1 && pool) {
            size_t used_threads = min(needed_threads, available_threads);
            size_t last = num_blobs % used_threads;
            size_t per_thread = (num_blobs - last) / used_threads;
            
            std::vector<std::shared_ptr<PrefilterBlobs>> prefilters;
            prefilters.resize(used_threads);
            
            auto start = frame.blobs().begin();
            auto end = start + per_thread;
            
            for(size_t i=0; i<used_threads - 1; ++i) {
                assert(end < frame.blobs().end());
                
                prefilters.at(i) = std::make_shared<PrefilterBlobs>(frame.index(), threshold, fish_size, *Tracker::instance()->_background);
                pool->enqueue(prefilter, prefilters[i], start, end);
                
                start = end;
                end = end + per_thread;
            }
            
            prefilters.back() = std::make_shared<PrefilterBlobs>(frame.index(), threshold, fish_size, *Tracker::instance()->_background);
            prefilter(prefilters.back(), start, frame.blobs().end());
            pool->wait();
            
            frame.clear_blobs();
            
            for(auto& filter : prefilters) {
                if(!filter)
                    continue;
                frame.add_blobs(std::move(filter->filtered), std::move(filter->filtered_out), filter->overall_pixels, filter->samples);
            }

        } else {
            auto pref = std::make_shared<PrefilterBlobs>(frame.index(), threshold, fish_size, *Tracker::instance()->_background);
            prefilter(pref, frame.blobs().begin(), frame.blobs().end());
            
            frame.clear_blobs();
            frame.add_blobs(std::move(pref->filtered),
                            std::move(pref->filtered_out),
                            pref->overall_pixels, pref->samples);
        }
        
        //initial_filter.conclude_measure();
    }

    std::vector<pv::BlobPtr> Tracker::split_big(
        const BlobReceiver& filter_out,
        const std::vector<pv::BlobPtr> &big_blobs,
        const std::map<pv::BlobPtr, split_expectation> &expect,
        bool discard_small,
        std::ostream* out,
        GenericThreadPool* pool)
    {
        std::vector<pv::BlobPtr> result;
        const int threshold = FAST_SETTINGS(track_threshold);
        const BlobSizeRange fish_size = FAST_SETTINGS(blob_size_ranges);
        const float cm_sq = SQR(FAST_SETTINGS(cm_per_pixel));
        const auto track_ignore = FAST_SETTINGS(track_ignore);
        const auto track_include = FAST_SETTINGS(track_include);
        
        std::mutex thread_mutex;
        
        auto check_blob = [&track_ignore, &track_include](const pv::BlobPtr& b) {
            if (!track_ignore.empty()) {
                if (blob_matches_shapes(b, track_ignore))
                    return false;
            }

            if (!track_include.empty()) {
                if (!blob_matches_shapes(b, track_include))
                    return false;
            }
            
            return true;
        };
        
        auto work = [&](auto, auto start, auto end, auto)
        {
            std::vector<pv::BlobPtr> noise, regular;
            
            for(auto it = start; it != end; ++it) {
                auto &b = *it;
                
                if(!fish_size.close_to_maximum_of_one(b->pixels()->size() * cm_sq, 1000))
                {
                    noise.push_back(b);
                    continue;
                }
                
                split_expectation ex(2, false);
                if(!expect.empty() && expect.count(b))
                    ex = expect.at(b);
                
                auto rec = b->recount(threshold, *_background);
                if(!fish_size.close_to_maximum_of_one(rec, 10 * ex.number)) {
                    noise.push_back(b);
                    continue;
                }
                
                SplitBlob s(*_background, b);
                std::vector<pv::BlobPtr> copy;
                auto ret = s.split(ex.number);
                
                for(auto &ptr : ret) {
                    if(b->blob_id() != ptr->blob_id())
                        ptr->set_split(true, b);
                }
                
                if(ex.allow_less_than && ret.empty()) {
                    if((!discard_small || fish_size.close_to_minimum_of_one(rec, 0.25))) {
                        result.push_back(b);
                    } else {
                        noise.push_back(b);
                    }
                    
                } else {
                    std::vector<pv::BlobPtr> for_this_blob;
                    std::set<std::tuple<float, pv::BlobPtr>, std::greater<>> found;
                    for(auto &ptr : ret) {
                        float recount = ptr->recount(0, *_background);
                        found.insert({recount, ptr});
                    }
                    
                    size_t counter = 0;
                    for(auto & [r, ptr] : found) {
                        ptr->add_offset(b->bounds().pos());
                        ptr->set_split(true, b);

                        ptr->calculate_moments();

                        if(!check_blob(ptr)) {
                            noise.push_back(ptr);
                            continue;
                        }
                        
                        if(fish_size.in_range_of_one(r, 0.35, 1) && (!discard_small || counter < ex.number)) {
                            for_this_blob.push_back(ptr);
                            ++counter;
                        } else {
                            noise.push_back(ptr);
                        }
                    }
                    
                    if(ret.empty()) {
                        noise.push_back(b);
                    } /*else if(for_this_blob.size() < ex.number) {
                        Log(out, "Not allowing less than %d, but only found %d blobs", ex.number, for_this_blob.size());
                        receiver(std::move(for_this_blob));
                        //filtered_out.insert(filtered_out.end(), for_this_blob.begin(), for_this_blob.end());
                        
                        if(out)
                            added.clear();
                    }*/ else
                        regular.insert(regular.end(),
                                       std::make_move_iterator(for_this_blob.begin()),
                                       std::make_move_iterator(for_this_blob.end()));
                }
            }
            
            std::unique_lock guard(thread_mutex);
            result.insert(result.end(),
                          std::make_move_iterator(regular.begin()),
                          std::make_move_iterator(regular.end()));
            filter_out(std::move(noise));
        };
        
        if(big_blobs.size() >= 2 && pool) {
            distribute_vector(work, *pool, big_blobs.begin(), big_blobs.end());
        } else
            work(0, big_blobs.begin(), big_blobs.end(), 0);
        
        return result;
    }

    void Tracker::history_split(PPFrame &frame, const std::unordered_set<Individual *> &active_individuals, std::ostream* out, GenericThreadPool* pool) {
        static Timing timing("history_split", 20);
        TakeTiming take(timing);

        float tdelta;
        
        auto resolution = _average->bounds().size();
        //ProximityGrid proximity(resolution);
        {
            Tracker::LockGuard guard("history_split#1");
            auto props = properties(frame.index() - 1);
            tdelta = props ? (frame.time - props->time) : 0;
        }
        const float max_d = FAST_SETTINGS(track_max_speed) * tdelta / FAST_SETTINGS(cm_per_pixel) * 0.5;

        Log(out, "");
        Log(out, "------------------------");
        Log(out, "HISTORY MATCHING for frame %d: (%f)", frame.index(), max_d);
        
        if(out) {
            auto str = Meta::toStr(active_individuals);
            Log(out, "frame %d active: %S", frame.index(), &str);
        }
        
        using namespace Match;
        std::map<long_t, std::set<uint32_t>> fish_mappings;
        std::map<uint32_t, std::set<long_t>> blob_mappings;
        std::map<long_t, std::map<uint32_t, Match::prob_t>> paired;

        const auto frame_limit = FAST_SETTINGS(frame_rate) * FAST_SETTINGS(track_max_reassign_time);
        
        {
            //static Timing just_splitting("caching", 0.1);
            //TakeTiming take_(just_splitting);
            
            size_t num_threads = pool ? hardware_concurrency() : 0;
            //num_threads = 1;
            std::mutex thread_mutex;
            auto space_limit = Individual::weird_distance() * 0.5;
            std::condition_variable variable;

            size_t count = 0;
            std::mutex mutex;

            auto fn = [&](const Tracker::set_of_individuals_t& active_individuals,
                          size_t start,
                          size_t N)
            {
                auto it = active_individuals.begin();
                std::advance(it, start);
                
                for(auto i = start; i < start + N; ++i, ++it) {
                    auto fish = *it;
                    auto &cache = frame.individual_cache()[i];
					
                    Vec2 last_pos(-1,-1);
                    auto last_frame = -1;
                    long_t last_L = -1;
                    float time_limit;

                    // IndividualCache is in the same position as the indexes here
                    //auto& obj = frame.cached_individuals.at(fish->identity().ID());
                    cache = fish->cache_for_frame(frame.index(), frame.time);
                    time_limit = cache.previous_frame - frame_limit;
                        
                    size_t counter = 0;
                    auto sit = fish->iterator_for(cache.previous_frame);
                    if (sit != fish->frame_segments().end() && (*sit)->contains(cache.previous_frame)) {
                        for (; sit != fish->frame_segments().end() && min((*sit)->end(), cache.previous_frame) >= time_limit && counter < frame_limit; ++counter)
                        {
                            auto pos = fish->basic_stuff().at((*sit)->basic_stuff((*sit)->end()))->centroid->pos(Units::DEFAULT);

                            if ((*sit)->length() > FAST_SETTINGS(frame_rate) * FAST_SETTINGS(track_max_reassign_time) * 0.25)
                            {
                                //! segment is long enough, we can stop. but only actually use it if its not too far away:
                                if (last_pos.x == -1 || euclidean_distance(pos, last_pos) < space_limit) {
                                    last_frame = min((*sit)->end(), cache.previous_frame);
                                    last_L = last_frame - (*sit)->start();
                                }
                                break;
                            }

                            last_pos = fish->basic_stuff().at((*sit)->basic_stuff((*sit)->start()))->centroid->pos(Units::DEFAULT);

                            if (sit != fish->frame_segments().begin())
                                --sit;
                            else
                                break;
                        }
                    }
                    
                    if(last_frame < time_limit) {
                        Log(out, "\tNot processing fish %d because its last respected frame is %d, best segment length is %d and were in frame %d.", fish->identity().ID(), last_frame, last_L, frame.index());
                        continue;
                    }
                    
                    auto set = frame.blob_grid().query(cache.estimated_px, max_d);
                    
                    std::string str = "";
                    if(out)
                        str = Meta::toStr(set);
                    
                    if(!set.empty()) {
                        auto fdx = fish->identity().ID();
                        
                        std::unique_lock guard(thread_mutex);
                        auto &map = fish_mappings[fdx];
                        auto &pair_map = paired[fdx];
                        
                        for(auto && [d, bdx] : set) {
                            if(!frame.find_bdx(bdx)) {
                                continue;
                            }
                            
                            map.insert(bdx);
                            blob_mappings[bdx].insert(fdx);
                            pair_map[bdx] = d;
                        }
                    }
                    
                    Log(out, "\tFish %d (%f,%f) proximity: %S", fish->identity().ID(), cache.estimated_px.x, cache.estimated_px.y, &str);
                }

                std::unique_lock lock(mutex);
                ++count;
                variable.notify_one();
            };
            
            //pool = nullptr;
            frame.individual_cache().clear();
            frame.individual_cache().resize(active_individuals.size());
            
            if(num_threads < 2 || !pool || active_individuals.size() < num_threads) {
                Tracker::LockGuard guard("history_split#2");
                fn(active_individuals, 0, active_individuals.size());
                
            } else if(active_individuals.size()) {
                size_t last = active_individuals.size() % num_threads;
                size_t per_thread = (active_individuals.size() - last) / num_threads;
                size_t i = 0;

                Tracker::LockGuard guard("history_split#2");
                for (; (i<=num_threads && last) || (!last && i<num_threads); ++i) {
                    size_t n = per_thread;
                    if(i == num_threads)
                        n = last;
                    
                    pool->enqueue(fn,
                                  active_individuals,
                                  i * per_thread, n);
                }
                
                std::unique_lock lock(mutex);
                while (count < i)
                    variable.wait(lock);
            }
        }
        
        std::set<long_t> already_walked;
        std::vector<pv::BlobPtr> big_blobs;
        std::map<pv::BlobPtr, split_expectation> expect;
        
        auto manual_splits = FAST_SETTINGS(manual_splits);
        auto manual_splits_frame = manual_splits.empty() || manual_splits.count(frame.index()) == 0 ? decltype(manual_splits)::mapped_type() : manual_splits.at(frame.index());
        std::string manualstr = out ? Meta::toStr(manual_splits) : "";
        Log(out, "manual_splits = %S", &manualstr);
        
        if(!manual_splits_frame.empty()) {
            for(auto bdx : manual_splits_frame) {
                auto it = blob_mappings.find(bdx);
                if(it == blob_mappings.end()) {
                    blob_mappings[bdx] = {-1 };
                    //Debug("%d: Inserting 2 additional matches for %d", frame.index(), bdx);
                } else{
                    it->second.insert(-1);
                    //Debug("%d: Inserting additional match for %d", frame.index(), bdx);
                }
                
                Log(out, "\t\tManually splitting %d", bdx);
                auto ptr = frame.erase_anywhere(bdx);
                if(ptr) {
                    big_blobs.push_back(ptr);
                    
                    expect[ptr].number = 2;
                    expect[ptr].allow_less_than = false;
                    
                    already_walked.insert(bdx);
                }
            }
            
        } else
            Log(out, "\t\tNo manual splits for frame %d", frame.index());
        
        if(out) {
            std::string str = Meta::toStr(fish_mappings);
            Log(out, "fish_mappings %S", &str);
            str = Meta::toStr(blob_mappings);
            Log(out, "blob_mappings %S", &str);
            str = Meta::toStr(paired);
            Log(out, "Paired %S", &str);
        }
        
        if(!FAST_SETTINGS(track_do_history_split)) {
            frame.finalize();
            return;
        }
        
        for(auto && [bdx, set] : blob_mappings) {
            if(already_walked.count(bdx)) {
                Log(out, "\tblob %d already walked", bdx);
                continue;
            }
            Log(out, "\tblob %d has %d fish mapped to it", bdx, set.size());
            
            if(set.size() <= 1)
                continue;
            Log(out, "\tFinding clique of this blob:");
            
            std::set<uint32_t> clique;
            std::set<uint32_t> others;
            std::queue<long_t> q;
            q.push(bdx);
            
            while(!q.empty()) {
                auto current = q.front();
                q.pop();
                
                for(auto fdx: blob_mappings.at(current)) {
                    // ignore manually forced splits
                    if(fdx < 0)
                        continue;
                    
                    for(auto b : fish_mappings.at(fdx)) {
                        if(!others.count(b)) {
                            q.push(b);
                            others.insert(b);
                            already_walked.insert(b);
                        }
                    }
                    
                    clique.insert(fdx);
                }
            }
            
            assert(bdx > 0);
            frame.blob_cliques[(uint32_t)bdx] = clique;
            frame.fish_cliques[(uint32_t)bdx] = others;
            
            if(out) {
                auto str = Meta::toStr(clique);
                auto str1 = Meta::toStr(others);
                Log(out, "\t\t%S %S", &str, &str1);
            }
            
            if(clique.size() > others.size()) {
                using namespace Match;
                std::map<long_t, std::pair<long_t, Match::prob_t>> assign_blob; // blob: individual
                std::map<long_t, std::set<std::tuple<Match::prob_t, long_t>>> all_combinations;
                std::map<long_t, std::set<std::tuple<Match::prob_t, long_t>>> complete_combinations;
                
                if(out) {
                    Log(out, "\t\tMismatch between blobs and number of fish assigned to them.");
                    if(clique.size() > others.size() + 1)
                        Log(out, "\t\tSizes: %d != %d", clique.size(), others.size());
                }
                
                bool allow_less_than = false;
                /*for(auto fdx : clique) {
                    if(_individuals.at(fdx)->recently_manually_matched(frame.index)) {
                        allow_less_than = true;
                        break;
                    }
                }*/
                
                auto check_combinations = [&assign_blob, out](long_t c, std::set<std::tuple<Match::prob_t, long_t>>& combinations, std::queue<long_t>& q) -> bool
                {
                    if(!combinations.empty()) {
                        auto b = std::get<1>(*combinations.begin());
                        
                        if(assign_blob.count(b) == 0) {
                            // great! this blob has not been assigned at all (yet)
                            // so just assign it to this fish
                            assign_blob[b] = {c, std::get<0>(*combinations.begin())};
                            Log(out, "\t\t%d(%d): %f", b, c, std::get<0>(*combinations.begin()));
                            return true;
                            
                        } else if(assign_blob[b].first != c) {
                            // this blob has been assigned to a different fish!
                            // check for validity (which one is closer)
                            if(assign_blob[b].second <= std::get<0>(*combinations.begin())) {
                                Log(out, "\t\tBlob %d is already assigned to %d (%d)...", b, assign_blob[b], c);
                            } else {
                                auto oid = assign_blob[b].first;
                                if(out) {
                                    Log(out, "\t\tBlob %d is already assigned to %d, but fish %d is closer (need to check combinations of fish %d again)", b, assign_blob[b], c, oid);
                                    Log(out, "\t\t%d(%d): %f", b, c, std::get<0>(*combinations.begin()));
                                }
                                assign_blob[b] = {c, std::get<0>(*combinations.begin())};
                                
                                q.push(oid);
                                return true;
                            }
                        }
                        
                        combinations.erase(combinations.begin());
                    }
                    
                    return false;
                };
                
                // 1. assign best matches (remove if better one is found)
                // 2. assign 2. best matches... until nothing is free
                std::queue<long_t> checks;
                for(auto c : clique) {
                    std::set<std::tuple<Match::prob_t, long_t>> combinations;
                    for(auto && [bdx, d] : paired.at(c)) {
                        combinations.insert({d, bdx});
                    }
                    
                    complete_combinations[c] = combinations;
                    all_combinations[c] = combinations;
                    
                    checks.push(c);
                }
                
                while(!checks.empty()) {
                    auto c = checks.front();
                    checks.pop();
                    
                    auto &combinations = all_combinations.at(c);
                    
                    if(!combinations.empty() && !check_combinations(c, combinations, checks))
                        checks.push(c);
                }
                
                size_t counter = 0;
                for(auto && [fdx, set] : all_combinations) {
                    if(out) {
                        auto str = Meta::toStr(set);
                        Log(out, "Combinations %d: %S", fdx, &str);
                    }
                    
                    if(set.empty()) {
                        ++counter;
                        Log(out, "No more alternatives for %d", fdx);
                        
                        if(!complete_combinations.at(fdx).empty()) {
                            //float max_s = 0;
                            long_t max_id = -1;
                            /*for(auto && [d, bdx] : complete_combinations.at(fdx)) {
                                if(bdx_to_ptr.at(bdx)->recount(-1) > max_s) {
                                    max_s = bdx_to_ptr.at(bdx)->recount(-1);
                                    max_id = bdx;
                                }
                            }*/
                            
                            //auto copy = complete_combinations.at(fdx);
                            if(out) {
                                for(auto && [d, bdx] : complete_combinations.at(fdx)) {
                                    Log(out, "\t%d: %f", bdx, d);
                                }
                            }
                            
                            max_id = std::get<1>(*complete_combinations.at(fdx).begin());
                            
                            if(max_id > 0) {
                                frame.split_blobs.insert((uint32_t)max_id);
                                auto ptr = frame.erase_regular(max_id);
                                
                                if(ptr) {
                                    Log(out, "Splitting blob %d", max_id);
                                    
                                    for(auto && [ind, blobs] : paired) {
                                        auto it = blobs.find(max_id);
                                        if(it != blobs.end()) {
                                            blobs.erase(it);
                                            //Debug("Frame %d: Erasing blob %u from paired (ind=%d)", frame.index(), max_id, ind);
                                        }
                                    }
                                    
                                    ++expect[ptr].number;
                                    big_blobs.push_back(ptr);
                                }
                                else if((ptr = frame.find_bdx(max_id))) {
                                    if(expect.count(ptr)) {
                                        Log(out, "Increasing expect number for blob %d.", max_id);
                                        ++expect[ptr].number;
                                    }
                                    
                                    Log(out, "Would split blob %d, but its part of additional.", max_id);
                                }
                                
                                if(allow_less_than)
                                    expect[ptr].allow_less_than = allow_less_than;
                            }
                        }
                    }
                }
                
                if(out) {
                    auto str = Meta::toStr(expect);
                    Log(out, "expect: %S", &str);
                    if(counter > 1) {
                        Log(out, "Lost %d fish (%S)", counter, &str);
                    }
                }
            }
        }
        
        for(auto && [blob, e] : expect)
            ++e.number;
        
        if(!manual_splits_frame.empty()) {
            for(auto bdx : manual_splits_frame) {
                auto ptr = frame.find_bdx(bdx);
                if(ptr) {
                    expect[ptr].allow_less_than = false;
                    expect[ptr].number = 2;
                }
            }
        }
        
        //static Timing tim("history_split_split_big", 0.1);
        //TakeTiming tak(tim);
        
        auto big_filtered = split_big(BlobReceiver(frame, BlobReceiver::noise), big_blobs, expect, true, out, pool);
        
        if(!big_filtered.empty())
            frame.add_regular(std::move(big_filtered));
        
        for(size_t i=0; i<frame.blobs().size(); ) {
            if(!FAST_SETTINGS(blob_size_ranges).in_range_of_one(frame.blobs()[i]->recount(-1))) {
                frame.move_to_noise(i);
            } else
                ++i;
        }
        
        frame.finalize();
    }
    
    Individual* Tracker::create_individual(Idx_t ID, Tracker::set_of_individuals_t& active_individuals) {
        if(_individuals.find(ID) != _individuals.end())
            U_EXCEPTION("Cannot assign identity (%d) twice.", ID);
        
        Individual *fish = new Individual();
        fish->identity().set_ID(ID);
        
        _individuals[fish->identity().ID()] = fish;
        active_individuals.insert(fish);
        
        if(ID >= Identity::running_id()) {
            Identity::set_running_id(ID + 1);
        }
        
        return fish;
    }

const FrameProperties* Tracker::add_next_frame(const FrameProperties & props) {
    auto &frames = instance()->frames();
    auto capacity = frames.capacity();
    instance()->_added_frames.push_back(props);
    
    if(frames.capacity() != capacity) {
        std::unique_lock guard(_properties_mutex);
        _properties_cache.clear();
        
        long_t frame = end_frame();
        auto it = frames.rbegin();
        while(it != frames.rend() && !_properties_cache.full())
        {
            _properties_cache.push_front(it->frame, &(*it));
            ++it;
        }
        assert((frames.empty() && end_frame() == -1) || (end_frame() != -1 && frames.rbegin()->frame == end_frame()));
        
    } else {
        std::unique_lock guard(_properties_mutex);
        _properties_cache.push(props.frame, &frames.back());
    }
    
    return &frames.back();
}

void Tracker::clear_properties() {
    std::unique_lock guard(_properties_mutex);
    _properties_cache.clear();
}

Match::PairedProbabilities Tracker::calculate_paired_probabilities
 (
    const PPFrame& frame,
    const Tracker::set_of_individuals_t& active_individuals,
    const std::unordered_map<Individual*, bool>& fish_assigned,
    const std::unordered_map<pv::Blob*, bool>& blob_assigned,
    //std::unordered_map<pv::Blob*, pv::BlobPtr>& ptr2ptr,
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
    auto frameIndex = frame.index();
    
    using namespace default_config;
    
    {
        using namespace Match;
        static const unsigned concurrentThreadsSupported = cmn::hardware_concurrency();
        
        static Timing probs("Tracker::paired", 30);
        TakeTiming take(probs);
        
        // see how many are missing
        std::vector<Individual*> unassigned_individuals;
        unassigned_individuals.reserve(active_individuals.size());
        
        for(auto &p : active_individuals) {
            if(!fish_assigned.at(p))
                unassigned_individuals.push_back(p);
        }
        
        // Create Individuals for unassigned blobs
        std::vector<std::tuple<const pv::BlobPtr*, int>> unassigned_blobs;
        unassigned_blobs.reserve(frame.blobs().size());
        
        const bool enable_labels = FAST_SETTINGS(track_consistent_categories) || !FAST_SETTINGS(track_only_categories).empty();
        for(size_t i=0; i<frame.blobs().size(); ++i) {//auto &p : frame.blobs) {
            if(!blob_assigned.at(frame.blobs()[i].get())) {
                auto bdx = frame.blobs()[i]->blob_id();
                auto label = enable_labels
                    ? Categorize::DataStore::ranged_label(Frame_t(frameIndex), bdx)
                    : nullptr;
                auto ptr = &frame.bdx_to_ptr(bdx);
                unassigned_blobs.push_back(std::make_tuple(ptr, label ? label->id : -1));
                //ptr2ptr[p.get()] = p;
            }
        }
        
        size_t last = unassigned_individuals.size() % concurrentThreadsSupported;
        size_t per_thread = (unassigned_individuals.size() - last) / concurrentThreadsSupported;
        
        size_t num_threads = max(1, min((float)concurrentThreadsSupported, floorf(unassigned_individuals.size() / SETTING(individuals_per_thread).value<float>())));
        if(num_threads > 1) {
            last = unassigned_individuals.size() % num_threads;
            per_thread = (unassigned_individuals.size() - last) / num_threads;
        } else
            per_thread = 0;
        
        size_t processed = 0;
        std::condition_variable variable;
        
        const auto work = [&](size_t from, size_t to)
        {
            //std::unordered_set<pv::BlobPtr> blobs_used;
            //std::unordered_set<Individual*> individuals_used;
            const auto matching_probability_threshold = FAST_SETTINGS(matching_probability_threshold);

            for(size_t i=from; i<to; i++) {
                auto fish = unassigned_individuals[i];
                //Match::prob_t max_p = 0;
                std::unordered_map<Match::Blob_t, Match::prob_t> probs;
                
                auto cache = frame.cached(fish->identity().ID());
                if(!cache) {
                    Except("Fish %d not found in cache.", fish->identity().ID());
                    continue;
                }

                for (auto &[blob, label]: unassigned_blobs) {
/*#ifndef NDEBUG
                    if(!frame.find_bdx(bdx)) {
                        auto it = std::find_if(frame.original_blobs().begin(), frame.original_blobs().end(), [bdx=bdx](const pv::BlobPtr& b) {
                            if(b->blob_id() == bdx) {
                                return true;
                            }
                            return false;
                        });
                        U_EXCEPTION("Frame %d: Blob %u not found (in original=>%d).", frameIndex, bdx, it != frame.original_blobs().end());
                    }
#endif*/
                    auto p = fish->probability(label, *cache, frameIndex, *blob).p;//blob->center(), blob->num_pixels()).p;

                    // discard elements with probabilities that are too low
                    if (p <= matching_probability_threshold)
                        continue;

                    //Debug("%d: %d -> %d: %f", frameIndex, fish->identity().ID(), blob->blob_id(), p);
                    
                    probs[blob] = p;
                    //max_p = max(max_p, p);

                    //blobs_used.insert(blob);
                    //individuals_used.insert(fish);
                }

                if(!probs.empty()) {
                    std::lock_guard<std::mutex> guard(paired_mutex);
                    paired_blobs.add(fish, probs);
                }
                //local_paired[fish] = probs;
                //local_max_probs[fish] = max_p;
            }
            
            std::lock_guard<std::mutex> guard(paired_mutex);
            ++processed;
            variable.notify_one();

            /*std::lock_guard<std::mutex> lock(guard);
            relevant_individuals.insert(individuals_used.begin(), individuals_used.end());
            relevant_blobs.insert(blobs_used.begin(), blobs_used.end());*/
        };

        if(num_threads <= 1 && pool) {
            work(0, unassigned_individuals.size());
        } else {
            for(size_t i=0; i<num_threads; i++) {
                pool->enqueue(work, i*per_thread, (i+1)*per_thread + (i == num_threads-1 ? last : 0));
            }
            
            std::unique_lock g(paired_mutex);
            while(processed < num_threads)
                variable.wait(g);
        }
    }
    
    return paired_blobs;
}

    void Tracker::add(long_t frameIndex, PPFrame& frame) {
        static const unsigned concurrentThreadsSupported = cmn::hardware_concurrency();
        double time = frame.frame().timestamp() / double(1000*1000);
        
        if(_endFrame < _startFrame) {
            Error("end frame is %d < %d", _endFrame.load(), _startFrame.load());
            _endFrame = _startFrame.load();
        }
        
        if (_startFrame > frameIndex || _startFrame == -1) {
            _startFrame = frameIndex;
        }
        
        if (_endFrame < frameIndex || _endFrame == -1) {
            _endFrame = frameIndex;
        }
        
        auto props = add_next_frame(FrameProperties(frame.index(), time, frame.frame().timestamp()));
        const FrameProperties* prev_props = nullptr;
        {
            auto it = --_added_frames.end();
            if(it != _added_frames.begin()) {
                --it;
                if(it->frame == frame.index() - 1)
                    prev_props = &(*it);
            }
        }
        
        static Timing timing("Tracker::add(frameIndex,PPFrame)", 10);
        TakeTiming assign(timing);
        
        // transfer the blobs and calculate the necessary properties
        // (also filter for blob size)
        //std::vector<Blob*> blobs;
        const float track_max_reassign_time = FAST_SETTINGS(track_max_reassign_time);
        const bool do_posture = FAST_SETTINGS(calculate_posture);
        const file::Path tags_path = FAST_SETTINGS(tags_path);
        const bool save_tags = !tags_path.empty();
        
        _current_midline_errors = 0;
        
        // ------------------------------------
        // filter and calculate blob properties
        // ------------------------------------
        std::queue<std::tuple<Individual*, std::shared_ptr<Individual::BasicStuff>>> need_postures;
        
        std::unordered_map<pv::Blob*, bool> blob_assigned;
        std::unordered_map<Individual*, bool> fish_assigned;
        
        const uint32_t number_fish = (uint32_t)FAST_SETTINGS(track_max_individuals);
        const BlobSizeRange minmax = FAST_SETTINGS(blob_size_ranges);
        
        size_t assigned_count = 0;
        
        std::vector<tags::blob_pixel> tagged_fish, noise;
        std::unordered_map<uint32_t, Individual*> blob_fish_map;
        
        //auto blob_to_pixels = filter_blobs(frame);
        auto assign_blob_individual = [&tagged_fish, &blob_fish_map, &fish_assigned, &blob_assigned, &assigned_count, &do_posture, &need_postures, save_tags]
            (size_t frameIndex, const PPFrame& frame, Individual* fish, const pv::BlobPtr& blob, default_config::matching_mode_t::Class match_mode)
        {
            // transfer ownership of blob to individual
            // delete the copied objects from the original array.
            // otherwise they would be deleted after the RawProcessing
            // object gets deleted (ownership of blobs is gonna be
            // transferred to Individuals)
            /*auto it = std::find(frame.blobs.begin(), frame.blobs.end(), blob);
            if(it != frame.blobs.end())
                frame.blobs.erase(it);
            else if((it = std::find(frame.filtered_out.begin(), frame.filtered_out.end(), blob)) != frame.filtered_out.end()) {
                frame.filtered_out.erase(it);
            }
#ifndef NDEBUG
            else
                U_EXCEPTION("Cannot find blob in frame.");
#endif*/
#ifndef NDEBUG
            if(!contains(frame.blobs(), blob)
               && !contains(frame.noise(), blob))
            {
                Except("Cannot find blob %u in frame %d.", blob->blob_id(), frameIndex);
            }
#endif
            
            //auto &pixels = *blob_to_pixels.at(blob);
            assert(blob->properties_ready());
            if(!blob->moments().ready) {
                blob->calculate_moments();
            }
            auto basic = fish->add(frameIndex, frame, blob, -1, match_mode);
            if(!basic) {
                Except("Was not able to assign individual %d with blob %u", fish->identity().ID(), blob->blob_id());
                return;
            }
            
            fish_assigned[fish] = true;
            blob_assigned[blob.get()] = true;
            
            if(save_tags) {
                if(!blob->split()){
                    blob_fish_map[blob->blob_id()] = fish;
                    if(blob->parent_id() != -1)
                        blob_fish_map[blob->parent_id()] = fish;
                    
                    //pv::BlobPtr copy = std::make_shared<pv::Blob>((Blob*)blob.get(), std::make_shared<std::vector<uchar>>(*blob->pixels()));
                    tagged_fish.push_back(std::make_shared<pv::Blob>(blob->lines(), blob->pixels()));
                }
            }
            
            if (do_posture)
                need_postures.push({fish, basic});
            else //if(!Recognition::recognition_enabled())
                basic->pixels = nullptr;
            
            assigned_count++;
        };
        
        if(save_tags) {
            for(auto &blob : frame.noise()) {
                if(blob->recount(-1) <= minmax.max_range().start) {
                    pv::BlobPtr copy = std::make_shared<pv::Blob>((Blob*)blob.get(), std::make_shared<std::vector<uchar>>(*blob->pixels()));
                    noise.emplace_back(std::move(copy));
                }
            }
        }
        
        //blobs = frame.blobs;
        for(auto &blob: frame.blobs())
            blob_assigned[blob.get()] = false;
        
        // collect all the currently active individuals
        Tracker::set_of_individuals_t active_individuals;
        
        //std::unordered_map<Individual*, std::unordered_map<pv::Blob*, Match::prob_t>> paired;
        //std::unordered_map<uint32_t, pv::BlobPtr> id_to_blob;
        
        //! TODO: Can probably reuse frame.blob_grid here, but need to add noise() as well
        static grid::ProximityGrid blob_grid(_average->bounds().size());
        blob_grid.clear();
        
        const auto manual_identities = FAST_SETTINGS(manual_identities);
        if(!number_fish && !manual_identities.empty()) {
            SETTING(manual_identities) = Settings::manual_identities_t();
        }
        
        for(auto &b : frame.blobs()) {
            //id_to_blob[b->blob_id()] = b;
            blob_grid.insert(b->bounds().x + b->bounds().width * 0.5f, b->bounds().y + b->bounds().height * 0.5f, b->blob_id());
        }
        for(auto &b : frame.noise()) {
            //id_to_blob[b->blob_id()] = b;
            blob_grid.insert(b->bounds().x + b->bounds().width * 0.5f, b->bounds().y + b->bounds().height * 0.5f, b->blob_id());
        }
        
        // see if there are manually fixed matches for this frame
        Settings::manual_matches_t::mapped_type current_fixed_matches;
        {
            auto manual_matches = Settings::get<Settings::manual_matches>();
            auto it = manual_matches->find(frameIndex);
            if (it != manual_matches->end())
                current_fixed_matches = it->second;
        }
        
        // prepare active_individuals array and assign fixed matches for which
        // the individuals already exist
        std::unordered_map<uint32_t, std::set<Idx_t>> cannot_find;
        std::unordered_map<uint32_t, std::set<Idx_t>> double_find;
        std::unordered_map<uint32_t, Idx_t> actually_assign;
        
        for(auto && [fdx, bdx] : current_fixed_matches) {
            auto it = _individuals.find(fdx);
            if(it != _individuals.end()) { //&& size_t(fm.second) < blobs.size()) {
                auto fish = it->second;
                
                if(bdx < 0) {
                    // dont assign this fish! (bdx == -1)
                    
                    continue;
                }
                
                auto blob = frame.find_bdx((uint32_t)bdx);
                if(!blob) {
                    //Error("Blob number %d out of range in frame %d", fm.second, frameIndex);
                    cannot_find[(uint32_t)bdx].insert(fdx);
                    continue;
                }
                
                if(actually_assign.count((uint32_t)bdx) > 0) {
                    Error("(fixed matches) Trying to assign blob %d twice in frame %d (fish %d and %d).", (uint32_t)bdx, frameIndex, fdx, actually_assign.at((uint32_t)bdx));
                    double_find[(uint32_t)bdx].insert(fdx);
                    
                } else if(blob_assigned[blob.get()]) {
                    Error("(fixed matches, blob_assigned) Trying to assign blob %d twice in frame %d (fish %d).", bdx, frameIndex, fdx);
                    // TODO: remove assignment from the other fish as well and add it to cannot_find
                    double_find[(uint32_t)bdx].insert(fdx);
                    
                    /*for(auto fish : active_individuals) {
                        auto blob = fish->blob(frameIndex);
                        if(blob && blob->blob_id() == fm.second) {
                            double_find[fm.second].insert(fish->identity().ID());
                        }
                    }*/
                    
                } else if(fish_assigned[fish]) {
                    Error("Trying to assign fish %d twice in frame %d.", fish->identity().ID(), frameIndex);
                } else {
                    actually_assign[blob->blob_id()] = fdx;
                    //active_individuals.push_back(fish);
                    //fish->add_manual_match(frameIndex);
                    //assign_blob_individual(frameIndex, frame, fish, blob);
                    //Debug("Manually assigning %d -> %d", fish->identity().ID(), blob->blob_id());
                }
                
            } else {
                if(frameIndex != _startFrame)
                    Warning("Individual number %d out of range in frame %d. Creating new one.", fdx, frameIndex);
                
                auto blob = frame.find_bdx((uint32_t)bdx);
                if(!blob) {
                    //Warning("Cannot find blob %d in frame %d. Fallback to normal assignment behavior.", it->second, frameIndex);
                    cannot_find[(uint32_t)bdx].insert(fdx);
                    continue;
                }
                
                if(actually_assign.count((uint32_t)bdx) > 0) {
                    Error("(fixed matches) Trying to assign blob %d twice in frame %d (fish %d and %d).", bdx, frameIndex, fdx, actually_assign.at((uint32_t)bdx));
                    double_find[(uint32_t)bdx].insert(fdx);
                } else
                    actually_assign[(uint32_t)bdx] = fdx;
                
                //auto fish = create_individual(fm.first, active_individuals);
               // fish->add_manual_match(frameIndex);
                //assign_blob_individual(frameIndex, frame, fish, blob_it->second);
            }
        }
        
        for(auto && [bdx, fdxs] : double_find) {
            if(actually_assign.count(bdx) > 0) {
                fdxs.insert(actually_assign.at(bdx));
                actually_assign.erase(bdx);
            }
            
            cannot_find[bdx].insert(fdxs.begin(), fdxs.end());
        }
        
        for(auto && [bdx, fdx] : actually_assign) {
            auto &blob = frame.bdx_to_ptr(bdx);
            Individual *fish = NULL;
            
            auto it = _individuals.find(fdx);
            if(it == _individuals.end()) {
                fish = create_individual(fdx, active_individuals);
            } else {
                fish = it->second;
                active_individuals.insert(fish);
            }
            
            fish->add_manual_match(frameIndex);
            assign_blob_individual(frameIndex, frame, fish, blob, default_config::matching_mode_t::benchmark);
            //frame.erase_anywhere(blob);
        }
        
        if(!cannot_find.empty()) {
            struct Blaze {
                PPFrame *_frame;
                Blaze(PPFrame& frame) : _frame(&frame) {
                    _frame->_finalized = false;
                }
                
                ~Blaze() {
                    _frame->finalize();
                }
            } blaze(frame);
            
            std::unordered_map<uint32_t, std::vector<std::tuple<Idx_t, Vec2, uint32_t>>> assign_blobs;
            const auto max_speed_px = FAST_SETTINGS(track_max_speed) / FAST_SETTINGS(cm_per_pixel);
            
            for(auto && [bdx, fdxs] : cannot_find) {
                assert(bdx >= 0);
                auto pos = pv::Blob::position_from_id(bdx);
                //Debug("Trying to find blob for %d (-> fish %d) at %f,%f", bdx, fdx, pos.x, pos.y);
                auto list = blob_grid.query(pos, max_speed_px);
                //auto str = Meta::toStr(list);
                //Debug("\t%S", &str);
                
                if(!list.empty()) {
                    // blob ids will not be < 0, as they have been inserted into the
                    // grid before directly from the file. so we can assume (uint32_t)
                    for(auto fdx: fdxs)
                        assign_blobs[(uint32_t)std::get<1>(*list.begin())].push_back({fdx, pos, (uint32_t)bdx});
                }
            }
            
            //auto str = prettify_array(Meta::toStr(assign_blobs));
            //Debug("replacing blobids / potentially splitting:\n%S", &str);
            
            std::map<Idx_t, uint32_t> actual_assignments;
            
            for(auto && [bdx, clique] : assign_blobs) {
                //if(clique.size() > 1)
                {
                    // have to split blob...
                    auto blob = frame.bdx_to_ptr(bdx);
                    
                    //std::vector<pv::BlobPtr> additional;
                    std::map<pv::BlobPtr, split_expectation> expect;
                    expect[blob] = split_expectation(clique.size() == 1 ? 2 : clique.size(), false);
                    
                    auto big_filtered = split_big(BlobReceiver(frame, BlobReceiver::noise),
                                                  {blob}, expect);
                    if(!big_filtered.empty()) {
                        /*std::map<Individual*, std::map<pv::BlobPtr, float>> distances;
                        std::map<Individual*, float> max_probs;
                        
                        for(auto b : big_filtered) {
                            for(auto && [fdx, pos] : clique) {
                                Individual *fish = nullptr;
                                
                                auto it = _individuals.find(fdx);
                                if(it == _individuals.end()) {
                                    fish = create_individual(fdx, blob, active_individuals);
                                } else
                                    fish = it->second;
                                
                                float d = 1 / sqdistance(pos, b->bounds().pos() + b->bounds().size() * 0.5);
                                
                                distances[fish][b] = d;
                                if(max_probs[fish] < d)
                                    max_probs[fish] = d;
                            }
                        }
                        
                        Match::PairingGraph graph(frameIndex, distances, max_probs);
                        for(auto && [fdx, pos] : clique)
                            graph.add(_individuals.at(fdx));
                        for(auto b : big_filtered)
                            graph.add(b);
                        
                        auto & result = graph.get_optimal_pairing();
                        for(auto && [fish, blob] : result->path) {
                            auto fdx = fish->identity().ID();
                            if(blob)
                                actual_assignments[fdx] = blob->blob_id();
                        }
                        
                        frame.blobs.insert(frame.blobs.end(), big_filtered.begin(), big_filtered.end());*/
                        
                        size_t found_perfect = 0;
                        for(auto && [fdx, pos, original_bdx] : clique) {
                            for(auto b : big_filtered) {
                                if(b->blob_id() == original_bdx) {
#ifndef NDEBUG
                                    Debug("frame %d: Found perfect match for individual %d, bdx %d after splitting %d", frame.index(), fdx, b->blob_id(), b->parent_id());
#endif
                                    actual_assignments[fdx] = original_bdx;
                                    //frame.blobs.insert(frame.blobs.end(), b);
                                    ++found_perfect;
                                    
                                    break;
                                }
                            }
                        }
                        
                        if(found_perfect) {
                            frame.add_regular(std::move(big_filtered));
                            // remove the blob thats to be split from all arrays
                            frame.erase_anywhere(blob);
                        }
                        
                        if(found_perfect == clique.size()) {
#ifndef NDEBUG
                            Debug("frame %d: All missing manual matches perfectly matched.", frame.index());
#endif
                        } else {
                            auto str = Meta::toStr(clique);
                            Error("frame %d: Missing some matches (%d/%d) for blob %d (identities %S).", frame.index(), found_perfect, clique.size(), bdx, &str);
                        }
                    }
                    
                } //else
                    //actual_assignments[std::get<0>(*clique.begin())] = bdx;
            }
            
            if(!actual_assignments.empty()) {
                auto str = prettify_array(Meta::toStr(actual_assignments));
                Debug("frame %d: actually assigning:\n%S", frame.index(), &str);
            }
            
            std::set<FOI::fdx_t> identities;
            
            for(auto && [fdx, bdx] : actual_assignments) {
                auto blob = frame.bdx_to_ptr(bdx);
                
                Individual *fish = nullptr;
                auto it = _individuals.find(fdx);
                
                // individual doesnt exist yet. create it
                if(it == _individuals.end()) {
                    U_EXCEPTION("Should have created it already."); //fish = create_individual(fdx, blob, active_individuals);
                } else
                    fish = it->second;
                
                if(blob_assigned[blob.get()]) {
                    Error("Trying to assign blob %d twice.", bdx);
                } else if(fish_assigned[fish]) {
                    Error("Trying to assign fish %d twice.", fdx);
                } else {
                    fish->add_manual_match(frameIndex);
                    assign_blob_individual(frameIndex, frame, fish, blob, default_config::matching_mode_t::benchmark);
                    
                    frame.erase_anywhere(blob);
                    active_individuals.insert(fish);
                    
                    identities.insert(FOI::fdx_t(fdx));
                    //Debug("Manually assigning %d -> %d", fish->identity().ID(), blob->blob_id());
                }
            }
            
            FOI::add(FOI(frameIndex, identities, "failed matches"));
        }
        
        if(frameIndex == _startFrame && !manual_identities.empty()) {
            // create correct identities
            //assert(_individuals.empty());
            
            Idx_t max_id(Identity::running_id());
            
            for (auto m : manual_identities) {
                if(_individuals.find(m) == _individuals.end()) {
                    Individual *fish = new Individual((uint32_t)m);
                    //fish->identity().set_ID(m);
                    assert(fish->identity().ID() == m);
                    max_id = Idx_t(max((uint32_t)max_id, (uint32_t)m));
                    
                    _individuals[m] = fish;
                    //active_individuals.push_back(fish);
                }
            }
            
            if(max_id.valid()) {
                Identity::set_running_id(max_id + 1);
            }
        }
        
        auto automatic_assignments = automatically_assigned(frameIndex);
        for(auto && [fdx, bdx] : automatic_assignments) {
            if(bdx < 0)
                continue; // dont assign this fish
            
            Individual *fish = nullptr;
            if(_individuals.find(fdx) != _individuals.end())
                fish = _individuals.at(fdx);
            
            pv::BlobPtr blob = frame.find_bdx((uint32_t)bdx);
            if(fish && blob && !fish_assigned[fish] && !blob_assigned[blob.get()]) {
                assign_blob_individual(frameIndex, frame, fish, blob, default_config::matching_mode_t::benchmark);
                //frame.erase_anywhere(blob);
                fish->add_automatic_match(frameIndex);
                active_individuals.insert(fish);
                
            } else {
#ifndef NDEBUG
                Error("frame %d: Automatic assignment cannot be executed with fdx %d(%s) and bdx %ld(%s)", frameIndex, fdx, fish ? (fish_assigned[fish] ? "assigned" : "unassigned") : "no fish", bdx, blob ? (blob_assigned[blob.get()] ? "assigned" : "unassigned") : "no blob");
#endif
            }
        }
        
        for(auto fish: _active_individuals) {
            // jump over already assigned individuals
            if(!fish_assigned[fish]) {
                if(fish->empty()) {
                    //fish_assigned[fish] = false;
                    //active_individuals.push_back(fish);
                } else {
                    auto found_idx = fish->find_frame(frameIndex)->frame;
                    float tdelta = cmn::abs(frame.time - properties(found_idx)->time);
                    if (tdelta < track_max_reassign_time)
                        active_individuals.insert(fish);
                }
            }
        }
        // now that the blobs array has been cleared of all the blobs for fixed matches,
        // get pairings for all the others:
        //std::unordered_map<pv::Blob*, pv::BlobPtr> ptr2ptr;
        auto paired_blobs = calculate_paired_probabilities(frame,
                                                           active_individuals,
                                                           fish_assigned,
                                                           blob_assigned,
                                                           //ptr2ptr,
                                                           &_thread_pool);
        
        if(!manual_identities.empty() && manual_identities.size() < paired_blobs.n_rows()) {
            using namespace Match;
            
            for (auto r : paired_blobs.rows()) {
                if(r->identity().manual()) {
                    // this is an important fish, check
                    auto idx = paired_blobs.index(r);
                    
                    if(paired_blobs.degree(idx) == 1) {
                        auto edges = paired_blobs.edges_for_row(idx);
                        
                        // only one possibility!
                        auto blob = edges.front();
                        Individual *other = NULL;
                        size_t count = 0;
                        
                        for (auto f : paired_blobs.rows()) {
                            if(f == r)
                                continue;
                            
                            auto e = paired_blobs.edges_for_row(paired_blobs.index(f));
                            if(contains(e, blob)) {
                                // also contains the blob
                                count++;
                                
                                if(manual_identities.find(f->identity().ID()) == manual_identities.end()) {
                                    // the other fish is not important
                                    if(e.size() > 1)
                                        continue; // check if only one possibility
                                    other = f;
                                }
                            }
                        }
                        
                        // only one other identity is sharing this blob, or no other identity
                        if(!other || count > 1)
                            continue;
                        
                        // found another fish, and its the only possibility
                        //Debug("Prioritizing %d over %d in frame %d for blob %d.", r->identity().ID(), other->identity().ID(), frameIndex, blob.blob->blob_id());
                        if(paired_blobs.has(other))
                            paired_blobs.erase(other);
                    }
                }
            }
        }
        
        
        using namespace default_config;
        const long_t approximation_delay_time = max(1, FAST_SETTINGS(frame_rate) * 0.25);
        bool frame_uses_approximate = (_approximative_enabled_in_frame >= 0 && frameIndex - _approximative_enabled_in_frame < approximation_delay_time);
        
        auto match_mode = frame_uses_approximate
                ? default_config::matching_mode_t::hungarian
                : FAST_SETTINGS(match_mode);
//#define TREX_DEBUG_MATCHING
#ifdef TREX_DEBUG_MATCHING
        std::vector<std::pair<Individual*, Match::Blob_t>> pairs;
        
        {
            Match::PairingGraph graph(frameIndex, paired_blobs);
            
            try {
                auto &optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
                pairs = optimal.pairings;
                
            } catch(...) {
                Except("Failed to generate optimal solution (frame %d).", frameIndex);
            }
        }
#endif
        
        if(match_mode == default_config::matching_mode_t::automatic) {
            std::unordered_set<uint32_t> all_individuals;
            std::vector<std::set<uint32_t>> blob_cliques;
            std::vector<Clique> cliques;
            
            /*for(auto &&[bdx, c] : frame.blob_cliques) {
                if(!contains(blob_cliques, c)) {
                    auto &fishies = frame.fish_cliques.at(bdx);
                    blob_cliques.push_back(c);
                    
                    Clique clique;
                    for(auto i : c) {
                        if(contains(all_individuals, i)) {
                            for(auto &sub : cliques) {
                                if(contains(sub.fishs, i)) {
                                    // merge cliques
                                    auto str0 = Meta::toStr(c);
                                    auto str1 = Meta::toStr(sub.fishs);
                                    Debug("Frame %d: Should merge cliques %S and %S.", frameIndex, &str0, &str1);
                                    break;
                                }
                            }
                        } else
                            all_individuals.insert(i);
                    }
                    
                    clique.fishs.insert(c.begin(), c.end());
                    clique.bids.insert(fishies.begin(), fishies.end());
                    cliques.push_back(clique);
                }
            }*/
            
            Clique clique;
            
            for(auto &row : paired_blobs.rows()) {
                auto idx = paired_blobs.index(row);
                if(paired_blobs.degree(idx) > 1) {
                    auto edges = paired_blobs.edges_for_row(idx);
                    clique.bids.clear();
                    clique.fishs.clear();
                    
                    for(auto &col : edges) {
                        if(!frame.find_bdx((*paired_blobs.col(col.cdx))->blob_id())) {
                            Debug("Frame %d: Cannot find blob %u in map.", frameIndex, (*paired_blobs.col(col.cdx))->blob_id());
                            continue;
                        }
                        
                        if(col.p >= FAST_SETTINGS(matching_probability_threshold)) {
                            clique.bids.insert(col.cdx);
                            
                            for (auto it = cliques.begin(); it != cliques.end();) {
                                if(contains(it->fishs, idx) || contains(it->bids, col.cdx)) {
                                    clique.fishs.insert(it->fishs.begin(), it->fishs.end());
                                    clique.bids.insert(it->bids.begin(), it->bids.end());
                                    
                                    it = cliques.erase(it);
                                    
                                } else
                                    ++it;
                            }
                        }
                    }
                    
                    if(!clique.bids.empty()) {
                        all_individuals.insert(idx);
                        clique.fishs.insert(idx);
                        cliques.emplace_back(std::move(clique));
                    }
                }
            }
            
            if(cliques.empty()) {
                match_mode = matching_mode_t::approximate;
            } else {
                // try to extend cliques as far as possible (and merge)
                for(size_t index = 0; index < cliques.size(); ++index) {
                    std::unordered_set<uint32_t> added_individuals, added_blobs = cliques[index].bids;
                    do {
                        added_individuals.clear();
                        
                        for(auto cdx : added_blobs) {
                            auto blob = paired_blobs.col(cdx);
                            auto bedges = paired_blobs.edges_for_col(cdx);
                            
                            if(!frame.find_bdx((*blob)->blob_id())) {
                                Debug("Frame %d: Cannot find blob %u in map.", frameIndex, (*blob)->blob_id());
                                continue;
                            }
                            
#ifdef TREX_DEBUG_MATCHING
                            auto str = Meta::toStr(bedges);
                            Debug("\t\tExploring blob %u (aka %u) with edges %S", cdx, (*blob)->blob_id(), &str);
#endif
                            for(auto fdi : bedges) {
                                if(!contains(cliques[index].fishs, fdi) && !contains(added_individuals, fdi)) {
                                    added_individuals.insert(fdi);
                                    
                                    for(size_t j=0; j<cliques.size(); ++j) {
                                        if(j == index)
                                            continue;
                                        
                                        if(contains(cliques[j].bids, cdx) || contains(cliques[j].fishs, fdi))
                                        {
#ifdef TREX_DEBUG_MATCHING
                                            // merge cliques
                                            auto str0 = Meta::toStr(cliques[index].fishs);
                                            auto str1 = Meta::toStr(cliques[j].fishs);
                                            
                                            auto str2 = Meta::toStr(cliques[index].bids);
                                            auto str3 = Meta::toStr(cliques[j].bids);
#endif
                                            
                                            added_individuals.insert(cliques[j].fishs.begin(), cliques[j].fishs.end());
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
                        
                        added_blobs.clear();
                        for(auto i : added_individuals) {
                            auto edges = paired_blobs.edges_for_row(i);
#ifdef TREX_DEBUG_MATCHING
                            auto estr = Meta::toStr(edges);
                            Debug("\t\tExploring row %d (aka fish%d) with edges=%S", i, paired_blobs.row(i)->identity().ID(), &estr);
#endif
                            for(auto &e : edges) {
                                if(!contains(cliques[index].bids, e.cdx))
                                    added_blobs.insert(e.cdx);
                            }
                        }
                        
#ifdef TREX_DEBUG_MATCHING
                        if(!added_individuals.empty()) {
                            auto str = Meta::toStr(added_individuals);
                            Debug("Adding %S to clique %lu", &str, index);
                        }
#endif
                        cliques[index].fishs.insert(added_individuals.begin(), added_individuals.end());
                        
                    } while(!added_individuals.empty());
                }
                
                Clique translated;
                _cliques[frameIndex].clear();
                
                for(auto &clique : cliques) {
                    translated.bids.clear();
                    translated.fishs.clear();
                    
                    for(auto bdi : clique.bids)
                        translated.fishs.emplace((*paired_blobs.col(bdi))->blob_id());
                    for(auto fdi : clique.fishs)
                        translated.fishs.emplace(paired_blobs.row(fdi)->identity().ID());
                    
                    _cliques[frameIndex].emplace_back(std::move(translated));
                }
                
                std::mutex thread_mutex;
                std::condition_variable _variable;
                size_t executed = 0;
                
                size_t index = 0;
                for(auto &clique : cliques) {
#ifdef TREX_DEBUG_MATCHING
                    std::set<uint32_t> fishs, blobs;
                    for(auto fdi : clique.fishs)
                        fishs.insert(paired_blobs.row(fdi)->identity().ID());
                    for(auto bdi : clique.bids)
                        blobs.insert((*paired_blobs.col(bdi))->blob_id());
                    
                    auto str = Meta::toStr(fishs);
                    auto str1 = Meta::toStr(blobs);
                    Debug("Frame %d: Clique %lu, Matching fishs %S and blobs %S together.", frameIndex,index, &str, &str1);
                    ++index;
                    
                    for(auto &cdx : clique.bids) {
                        Debug("\tBlob %u edges:", (*paired_blobs.col(cdx))->blob_id());
                        for(auto &e : paired_blobs.edges_for_col(cdx)) {
                            Debug("\t\tFish%s", paired_blobs.row(e)->identity().raw_name().c_str());
                        }
                    }
#endif
                    
                    _thread_pool.enqueue([&paired_blobs, &blob_assigned, &fish_assigned, &frame, &assign_blob_individual, &thread_mutex, &active_individuals, &executed, &_variable
#ifdef TREX_DEBUG_MATCHING
                                          , &pairs
#endif
                                          ]
                                         (const Clique& clique, long_t frameIndex)
                    {
                        using namespace Match;
                        Match::PairedProbabilities paired;
                        for(auto fish : paired_blobs.rows()) {
                            auto fdi = paired_blobs.index(fish);
                            if(!contains(clique.fishs, fdi)
                               || (fish_assigned.count(fish) && fish_assigned.at(fish)))
                                continue;
                            
                            auto edges = paired_blobs.edges_for_row(fdi);
                            
                            std::unordered_map<Match::Blob_t, prob_t> probs;
                            for(auto &e : edges) {
                                auto blob = paired_blobs.col(e.cdx);
                                if(!blob_assigned.count(blob->get()) || !blob_assigned.at(blob->get()))
                                    probs[blob] = e.p;
                            }
                            
                            if(!probs.empty())
                                paired.add(fish, probs);
                        }
                        
                        PairingGraph graph(frameIndex, paired);
                        
                        try {
                            auto &optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
                            for (auto &p: optimal.pairings) {
    #ifdef TREX_DEBUG_MATCHING
                                for(auto &[i, b] : pairs) {
                                    if(i == p.first) {
                                        if(b != p.second) {
                                            Warning("Frame %d: Assigning individual %d to %u instead of %u", frameIndex, i->identity().ID(), p.second ? (*p.second)->blob_id() : 0,  b ? (*b)->blob_id() : 0);
                                        }
                                        break;
                                    }
                                }
    #endif
                                std::unique_lock g(thread_mutex);
                                assign_blob_individual(frameIndex, frame, p.first, *p.second, matching_mode_t::hungarian);
                                active_individuals.insert(p.first);
                            }
                            
                        } catch(...) {
                            Except("Failed to generate optimal solution (frame %d).", frameIndex);
                        }
                        
                        std::unique_lock g(thread_mutex);
                        ++executed;
                        _variable.notify_one();
                        
                    }, clique, frameIndex);
                }
                
                {
                    std::unique_lock g(thread_mutex);
                    while(executed < cliques.size()) {
                        _variable.wait(g);
                    }
                }
                
                Match::PairedProbabilities paired;
                auto in_map = paired_blobs.rows();
                for(auto fish : in_map) {
                    if(fish_assigned.find(fish) == fish_assigned.end() || !fish_assigned.at(fish)) {
                        auto edges = paired_blobs.edges_for_row(paired_blobs.index(fish));
                        
                        std::unordered_map<Match::Blob_t, Match::prob_t> probs;
                        for(auto &e : edges) {
                            auto blob = paired_blobs.col(e.cdx);
                            if(!frame.find_bdx((*blob)->blob_id())) {
                                Debug("Frame %d: Cannot find blob %u in map.", frameIndex, (*blob)->blob_id());
                                continue;
                            }
                            auto it = blob_assigned.find(blob->get());
                            if(it == blob_assigned.end() || !it->second) {
                                probs[blob] = e.p;
                            }
                        }
                        
                        if(!probs.empty())
                            paired.add(fish, probs);
                    }
                }
                
                paired_blobs = std::move(paired);
                match_mode = matching_mode_t::approximate;
            }
        }
        
        //Debug("Frame %d: %s", frameIndex, match_mode.name());
        
        {
            // calculate optimal permutation of blob assignments
            static Timing perm_timing("PairingGraph", 30);
            TakeTiming take(perm_timing);
            
            using namespace Match;
            PairingGraph graph(frameIndex, paired_blobs);
            
            size_t nedges = 0;
            size_t max_edges_per_fish = 0, max_edges_per_blob = 0;
            double mean_edges_per_blob = 0, mean_edges_per_fish = 0;
            size_t fish_with_one_edge = 0, fish_with_more_edges = 0;
            double average_probability = 0;
            
            std::map<long_t, size_t> edges_per_blob;
            /*for(auto && [fish, edges] : graph.edges()) {
                //double sum = 1 / double(edges.size()+1);
                //average_probability += sum;
                
                if(edges.size() > max_edges_per_fish) {
                    max_edges_per_fish = edges.size();
                }
                
                for(auto &edge : edges) {
                    ++edges_per_blob[edge.blob_index];
                }
                
                if(edges.size() == 1) {
                    ++fish_with_one_edge;
                } else
                    ++fish_with_more_edges;
                
                if(edges.empty())
                    U_EXCEPTION("FU");
                
                nedges += edges.size();
                mean_edges_per_fish += edges.size();
            }
            mean_edges_per_fish /= double(graph.edges().size());
            mean_edges_per_fish ++;
            //average_probability /= double(graph.edges().size());
            
            size_t blobs_with_one_edge = 0, blobs_with_more_edges = 0;
            for(auto && [blob, edges] : edges_per_blob) {
                if(edges > max_edges_per_blob)
                    max_edges_per_blob = edges;
                average_probability += 1 / double(edges);
                
                if(edges == 1)
                    ++blobs_with_one_edge;
                else
                    ++blobs_with_more_edges;
                
                mean_edges_per_blob += edges;
            }
            mean_edges_per_blob /= double(edges_per_blob.size());
            
            average_probability /= double(edges_per_blob.size());
            
            size_t one_to_ones = 0;
            for(auto && [fish, edges] : graph.edges()) {
                if(edges.size() > 1)
                    continue;
                
                if(edges.front().blob_index != -1 && edges_per_blob[edges.front().blob_index] == 1)
                {
                    ++one_to_ones;
                }
            }*/
            
#if defined(PAIRING_PRINT_STATS)
            double one_edge_probability = double(fish_with_one_edge) / double(fish_with_one_edge + fish_with_more_edges);
            double blob_one_edge = double(blobs_with_one_edge) / double(blobs_with_one_edge + blobs_with_more_edges);
            double one_to_one = double(one_to_ones) / double(graph.edges().size());
            
            //graph.print_summary();
            
            auto print_statistics = [&](const PairingGraph::Result& optimal, bool force = false){
                std::lock_guard<std::mutex> guard(_statistics_mutex);
                
                static double average_improvements = 0, samples = 0, average_leafs = 0, average_objects = 0, average_bad_probabilities = 0;
                static Timer timer;
                
                size_t bad_probs = 0;
                for(auto && [fish, mp] : max_probs) {
                    if(mp <= 0.5)
                        ++bad_probs;
                }
                
                average_bad_probabilities += bad_probs;
                average_improvements += optimal.improvements_made;
                average_leafs += optimal.leafs_visited;
                average_objects += optimal.objects_looked_at;
                ++samples;
                
                if(size_t(samples) % 50 == 0 || force) {
                    Debug("frame %d: %d of %d / %d objects. %.2f improvements on average, %.2f leafs visited on average, %.0f objects on average (%f mean edges per fish and %f mean edges per blob). On average we encounter %.2f bad probabilities below 0.5 (currently %d).", frameIndex, optimal.improvements_made, optimal.leafs_visited, optimal.objects_looked_at, average_improvements / samples, average_leafs / samples, average_objects / samples, mean_edges_per_fish, mean_edges_per_blob, average_bad_probabilities / samples, bad_probs);
                    Debug("g fish_has_one_edge * mean_edges_per_fish = %f * %f = %f", one_edge_probability, mean_edges_per_fish, one_edge_probability * (mean_edges_per_fish));
                    Debug("g fish_has_one_edge * mean_edges_per_blob = %f * %f = %f", one_edge_probability, mean_edges_per_blob, one_edge_probability * (mean_edges_per_blob));
                    Debug("g blob_has_one_edge * mean_edges_per_fish = %f * %f = %f", blob_one_edge, mean_edges_per_fish, blob_one_edge * mean_edges_per_fish);
                    Debug("g blob_has_one_edge * mean_edges_per_blob = %f * %f = %f", blob_one_edge, mean_edges_per_blob, blob_one_edge * mean_edges_per_blob);
                    Debug("g mean_edges_per_fish / mean_edges_per_blob = %f", mean_edges_per_fish / mean_edges_per_blob);
                    Debug("g one_to_one = %f, one_to_one * mean_edges_per_fish = %f / blob: %f /// %f, %f", one_to_one, one_to_one * mean_edges_per_fish, one_to_one * mean_edges_per_blob, average_probability, average_probability * mean_edges_per_fish);
                    Debug("g --");
                    timer.reset();
                }
            };
            
            if(average_probability * mean_edges_per_fish <= 1) {
                Warning("(%d) Warning: %f", frameIndex, average_probability * mean_edges_per_fish);
            }
    #endif
            
            try {
                //if(match_mode == default_config::matching_mode_t::accurate)
                //    U_EXCEPTION("Test %d", frameIndex);
                auto &optimal = graph.get_optimal_pairing(false, match_mode);
                
                if(!frame_uses_approximate) {
                    std::lock_guard<std::mutex> guard(_statistics_mutex);
                    _statistics[frameIndex].match_number_blob = paired_blobs.n_cols();
                    _statistics[frameIndex].match_number_fish = paired_blobs.n_rows();
                    _statistics[frameIndex].match_number_edges = nedges;
                    _statistics[frameIndex].match_stack_objects = optimal.objects_looked_at;
                    _statistics[frameIndex].match_max_edges_per_blob = max_edges_per_blob;
                    _statistics[frameIndex].match_max_edges_per_fish = max_edges_per_fish;
                    _statistics[frameIndex].match_mean_edges_per_blob = mean_edges_per_blob;
                    _statistics[frameIndex].match_mean_edges_per_fish = mean_edges_per_fish;
                    _statistics[frameIndex].match_improvements_made = optimal.improvements_made;
                    _statistics[frameIndex].match_leafs_visited = optimal.leafs_visited;
                    _statistics[frameIndex].method_used = (int)match_mode.value();
                }
                
    #if defined(PAIRING_PRINT_STATS)
                print_statistics(optimal);
    #endif
                
                for (auto &p: optimal.pairings) {
#ifdef TREX_DEBUG_MATCHING
                    for(auto &[i, b] : pairs) {
                        if(i == p.first) {
                            if(b != p.second) {
                                Warning("Frame %d: Assigning individual %d to %u instead of %u", frameIndex, i->identity().ID(), p.second ? (*p.second)->blob_id() : 0,  b ? (*b)->blob_id() : 0);
                            }
                            break;
                        }
                    }
#endif
                    
                    assign_blob_individual(frameIndex, frame, p.first, *p.second, match_mode);
                    active_individuals.insert(p.first);
                }
                
            } catch (const UtilsException& e) {
    #if !defined(NDEBUG) && defined(PAIRING_PRINT_STATS)
                if(graph.optimal_pairing())
                    print_statistics(*graph.optimal_pairing(), true);
                else
                    Warning("No optimal pairing object.");
                
                graph.print_summary();
    #endif
                            
#if defined(PAIRING_PRINT_STATS)
                // matching did not work
                Warning("Falling back to approximative matching in frame %d. (p=%f,%f, %f, %f)", frameIndex, one_edge_probability, mean_edges_per_fish, one_edge_probability * (mean_edges_per_fish), one_edge_probability * mean_edges_per_blob);
                Warning("frame %d: (%f mean edges per fish and %f mean edges per blob).", frameIndex, mean_edges_per_fish, mean_edges_per_blob);
                
                Debug("gw Probabilities: fish_has_one_edge=%f blob_has_one_edge=%f", one_edge_probability, blob_one_edge);
                Debug("gw fish_has_one_edge * mean_edges_per_fish = %f * %f = %f", one_edge_probability, mean_edges_per_fish, one_edge_probability * (mean_edges_per_fish));
                Debug("gw fish_has_one_edge * mean_edges_per_blob = %f * %f = %f", one_edge_probability, mean_edges_per_blob, one_edge_probability * (mean_edges_per_blob));
                Debug("gw blob_has_one_edge * mean_edges_per_fish = %f * %f = %f", blob_one_edge, mean_edges_per_fish, blob_one_edge * mean_edges_per_fish);
                Debug("gw blob_has_one_edge * mean_edges_per_blob = %f * %f = %f", blob_one_edge, mean_edges_per_blob, blob_one_edge * mean_edges_per_blob);
                Debug("gw one_to_one = %f, one_to_one * mean_edges_per_fish = %f / blob: %f /// %f, %f", one_to_one, one_to_one * mean_edges_per_fish, one_to_one * mean_edges_per_blob, average_probability, average_probability * mean_edges_per_fish);
                Debug("gw mean_edges_per_fish / mean_edges_per_blob = %f", mean_edges_per_fish / mean_edges_per_blob);
                Debug("gw ---");
#endif
                
                auto &optimal = graph.get_optimal_pairing(false, default_config::matching_mode_t::hungarian);
                for (auto &p: optimal.pairings) {
                    assign_blob_individual(frameIndex, frame, p.first, *p.second, default_config::matching_mode_t::hungarian);
                    active_individuals.insert(p.first);
                }
                
                _approximative_enabled_in_frame = frameIndex;
                
                FOI::add(FOI(Rangel(frameIndex, frameIndex + approximation_delay_time - 1), "apprx matching"));
            }
        }
        
        static Timing rest("rest", 30);
        TakeTiming take(rest);
        // see how many are missing
        std::vector<Individual*> unassigned_individuals;
        for(auto &p : fish_assigned) {
            if(!p.second) {
                unassigned_individuals.push_back(p.first);
            }
        }
        
        // Create Individuals for unassigned blobs
        std::vector<pv::BlobPtr> unassigned_blobs;
        for(auto &p: frame.blobs()) {
            if(!blob_assigned[p.get()])
                unassigned_blobs.emplace_back(p);
        }
        
        if(!number_fish /*|| (number_fish && number_individuals < number_fish)*/) {
            // the number of individuals is limited
            // fallback to creating new individuals if the blobs cant be matched
            // to existing ones
            if(frameIndex > 1) {
                static std::random_device rng;
                static std::mt19937 urng(rng());
                std::shuffle(unassigned_blobs.begin(), unassigned_blobs.end(), urng);
            }
            
            for(auto fish :_active_individuals)
                if(active_individuals.find(fish) == active_individuals.end())
                    _inactive_individuals.insert(fish->identity().ID());
            
            for (auto &blob: unassigned_blobs) {
                // we measure the number of currently assigned fish based on whether a maximum number has been set. if there is a maximum, then we only look at the currently active individuals and extend that array with new individuals if necessary.
                const size_t number_individuals = number_fish ? _individuals.size() : active_individuals.size();
                if(number_fish && number_individuals >= number_fish) {
                    static bool warned = false;
                    if(!warned) {
                        Warning("Running out of assignable fish (track_max_individuals %d/%d)", active_individuals.size(), number_fish);
                        warned = true;
                    }
                    break;
                }
                
                if(number_fish)
                    Warning("Frame %d: Creating new individual (%d) for blob %d.", frameIndex, Identity::running_id(), blob->blob_id());
                
                Individual *fish = nullptr;
                if(!_inactive_individuals.empty()) {
                    fish = _individuals.at(*_inactive_individuals.begin());
                    _inactive_individuals.erase(_inactive_individuals.begin());
                } else {
                    fish = new Individual;
                    if(_individuals.find(fish->identity().ID()) != _individuals.end()) {
                        U_EXCEPTION("Cannot assign identity (%d) twice.", fish->identity().ID());
                        //assert(_individuals[fish->identity().ID()] != fish);
                        //mark_to_delete.insert(_individuals[fish->identity().ID()]);
                    }
                    //Debug("Creating new identity %d", fish->identity().ID());
                    _individuals[fish->identity().ID()] = fish;
                }
                assign_blob_individual(frameIndex, frame, fish, blob, default_config::matching_mode_t::benchmark);
                active_individuals.insert(fish);
            }
        }
        
        if(number_fish && active_individuals.size() < _individuals.size()) {
            //  + the count of individuals is fixed (which means that no new individuals can
            //    be created after max)
            //  + the number of individuals is limited
            //  + there are unassigned individuals
            //    (the number of currently active individuals is smaller than the limit)
            
            /*if(_individuals.size() < number_fish) {
                for(auto id : manual_identities) {
                    if(!_individuals.count(id)) {
                        Individual *fish = new Individual();
                        fish->identity().set_ID(id);
                        _individuals[id] = fish;
                        
                        Debug("Creating new individual %d", id);
                    }
                }
            }*/
            
            if(!unassigned_blobs.empty()) {
                // there are blobs left to be assigned
                
                // now find all individuals that have left the "active individuals" group already
                // and re-assign them if needed
                //if(_individuals.size() != active_individuals.size())
                
                // yep, theres at least one who is not active anymore. we may reassign them.
                std::vector<Individual*> lost_individuals;
                
                for (auto &pair: _individuals) {
                    auto fish = pair.second;
                    if(fish->empty())
                        lost_individuals.push_back(fish);
                    else {
                        auto found = fish->find_frame(frameIndex)->frame;
                        
                        if (found != frameIndex) {
                            // this fish is not active in frameIndex
                            auto props = properties(found);
                            
                            // dont reassign them directly!
                            // this would lead to a lot of jumping around. mostly these problems
                            // solve themselves after a few frames.
                            if(frame.time - props->time >= track_max_reassign_time)
                                lost_individuals.push_back(fish);
                        }
                    }
                }
                
                if(!lost_individuals.empty()) {
                    // if an individual needs to be reassigned, chose the blobs that are the closest
                    // to the estimated position.
                    using namespace Match;
                    
                    std::multiset<PairProbability> new_table;
                    std::map<Individual*, std::map<pv::BlobPtr, Match::prob_t>> new_pairings;
                    std::map<Individual*, Match::prob_t> max_probs;
                    const Match::prob_t p_threshold = FAST_SETTINGS(matching_probability_threshold);
                    
                    for (auto& fish : lost_individuals) {
                        if(fish->empty()) {
                            for (auto& blob : unassigned_blobs) {
                                new_table.insert(PairProbability(fish, blob, p_threshold));
                                new_pairings[fish][blob] = p_threshold;
                            }
                            
                        } else {
                            auto pos_fish = fish->cache_for_frame(frameIndex, frame.time);
                            
                            for (auto& blob : unassigned_blobs) {
                                auto pos_blob = blob->center();
                                
                                Match::prob_t p = p_threshold + Match::prob_t(1.0) / sqdistance(pos_fish.last_seen_px, pos_blob) / pos_fish.tdelta;
                                
                                new_pairings[fish][blob] = p;
                                new_table.insert(PairProbability(fish, blob, p));
                            }
                        }
                    }
                    
                    /*{
                        // calculate optimal permutation of blob assignments
                        static Timing perm_timing("PairingGraph(lost)", 1);
                        TakeTiming take(perm_timing);
                     
                        using namespace Match;
                        PairingGraph graph(frameIndex, new_pairings, max_probs);
                     
                        for (auto r : lost_individuals)
                            graph.add(r);
                        for (auto r : unassigned_blobs)
                            graph.add(r);
                        //graph.print_summary();
                        auto &optimal = graph.get_optimal_pairing();
                     
                        for (auto &p: optimal.pairings) {
                            if(p.second && !fish_assigned[p.first] && !blob_assigned[p.second.get()]) {
                                assert(new_pairings.at(p.first).at(p.second) > FAST_SETTINGS(matching_probability_threshold));
                                assign_blob_individual(frameIndex, frame, p.first, p.second);
                     
                                auto it = std::find(lost_individuals.begin(), lost_individuals.end(), p.first);
                                assert(it != lost_individuals.end());
                                lost_individuals.erase(it);
                                if(!contains(active_individuals, p.first))
                                    active_individuals.push_back(p.first);
                     
                                Debug("Assigning individual because its the most likely (fixed_count, %d-%d in frame %d, p:%f).", p.first->identity().ID(), p.second->blob_id(), frameIndex, new_pairings.at(p.first).at(p.second));
                            }
                        }
                    }*/
                    
                    for(auto it = new_table.rbegin(); it != new_table.rend(); ++it) {
                        auto &r = *it;
                    //for (auto &r : new_table) {
                        if(!blob_assigned.at(r.bdx().get()) && contains(lost_individuals, r.idx())) {
                            blob_assigned.at(r.bdx().get()) = true;
                            
                            auto it = std::find(lost_individuals.begin(), lost_individuals.end(), r.idx());
                            assert(it != lost_individuals.end());
                            lost_individuals.erase(it);
                            
                            Individual *fish = r.idx();
                            
                            //Debug("Best match for blob %d is %d in %d (%f)", r.bdx()->blob_id(), fish->identity().ID(), frameIndex, r.p());
                            
                            assign_blob_individual(frameIndex, frame, fish, r.bdx(), default_config::matching_mode_t::benchmark);
                            active_individuals.insert(fish);
                            
                            //Debug("Assigning individual because its the most likely (fixed_count, %d-%d in frame %d, p:%f).", r.idx()->identity().ID(), r.bdx()->blob_id(), frameIndex, r.p());
                        }
                    }
                }
            }
        }
        
        _active_individuals = active_individuals;
        
#ifndef NDEBUG
        if(!number_fish) {
            static std::set<Idx_t> lost_ids;
            for(auto && [fdx, fish] : _individuals) {
                if(active_individuals.find(fish) == active_individuals.end() && _inactive_individuals.find(fdx) == _inactive_individuals.end()) {
                    if(lost_ids.find(fdx) != lost_ids.end())
                        continue;
                    lost_ids.insert(fdx);
                    auto basic = fish->empty() ? nullptr : fish->find_frame(frameIndex);
                    
                    if(basic && basic->frame == frameIndex) {
                        Warning("Fish %d not in any of the arrays, but has frame %d.", fdx, frameIndex);
                    } else
                        Warning("Fish %d is gone (%d)", fdx, basic ? basic->frame : -1);
                } else if(lost_ids.find(fdx) != lost_ids.end()) {
                    lost_ids.erase(fdx);
                    Warning("Fish %d found again in frame %d.", fdx, frameIndex);
                }
            }
        }
#endif
        
        if(save_tags) {
           _thread_pool.enqueue([&, bmf = blob_fish_map](){
                this->check_save_tags(frameIndex, bmf, tagged_fish, noise, tags_path);
            });
        }
        
        std::vector<std::vector<std::tuple<Individual*, std::shared_ptr<Individual::BasicStuff>>>> vector;
        Timer posture_timer;
        
        {
            static Timing timing("Tracker::need_postures", 30);
            TakeTiming take(timing);
            
            if(do_posture && !need_postures.empty()) {
                size_t num_threads = max(1, min((float)concurrentThreadsSupported, need_postures.size() / SETTING(postures_per_thread).value<float>()));
            
                size_t last = need_postures.size() % num_threads;
                size_t per_thread = (need_postures.size() - last) / num_threads;
                
                //if(frameIndex % 100 == 0)
                //    Debug("Posture in %d threads (%d per thread)", num_threads, per_thread);
                
                vector.reserve(num_threads+1);
                
                for(size_t i=0; i<=num_threads; ++i) {
                    const size_t elements = i == num_threads ? last : per_thread;
                    if(!elements)
                        break;
                    
                    decltype(vector)::value_type v;
                    v.reserve(elements);
                    
                    while(!need_postures.empty() && v.size() < elements) {
                        v.push_back(need_postures.front());
                        need_postures.pop();
                    }
                    
                    vector.push_back(v);
                    
                    if(i) {
                        _thread_pool.enqueue(analyse_posture_pack, frameIndex, vector.back());
                    }
                }
                
                if(!vector.empty())
                    analyse_posture_pack(frameIndex, vector.front());
                assert(need_postures.empty());
            }
            
            //if(assigned_count < 5)
            //    generate_pairdistances(frameIndex);
            
            _thread_pool.wait();
        }
        
        /*for(auto && [fish, assigned] : fish_assigned) {
            if(assigned) {
                long_t prev_frame = frameIndex - 1;
                if(!fish->empty()) {
                    if(frameIndex > fish->start_frame()) {
                        auto previous = fish->find_frame(frameIndex - 1);
                        prev_frame = previous.first;
                    }
                }
                
                fish->push_to_segments(frameIndex, prev_frame);
            }
        }*/
        
        if(do_posture) {
            _midline_errors_frame = (_midline_errors_frame + _current_midline_errors) * 0.5f;
        }
        auto posture_seconds = posture_timer.elapsed();
        
        Output::Library::frame_changed(frameIndex);
        
        if(number_fish && assigned_count >= number_fish) {
            update_consecutive(_active_individuals, frameIndex, true);
        }
        
        _max_individuals = cmn::max(_max_individuals, assigned_count);
        _active_individuals_frame[frameIndex] = _active_individuals;
        _added_frames.back().active_individuals = assigned_count;
        
        uint32_t n = 0;
        uint32_t prev = 0;
        if(manual_identities.empty()) {
            for(auto fish : _active_individuals) {
                assert((fish->end_frame() == frameIndex) == (fish->has(frameIndex)));
                
                if(fish->end_frame() == frameIndex)
                    ++n;
                if(fish->has(frameIndex-1))
                    ++prev;
            }
            
        } else {
            for(auto id : manual_identities) {
                auto it = _individuals.find(id);
                if(it != _individuals.end()) {
                    auto& fish = it->second;
                    assert((fish->end_frame() == frameIndex) == (fish->has(frameIndex)));
                    
                    if(fish->end_frame() == frameIndex)
                        ++n;
                    if(fish->has(frameIndex-1))
                        ++prev;
                }
            }
        }
        
        update_warnings(frameIndex, frame.time, number_fish, n, prev, props, prev_props, _active_individuals, _individual_add_iterator_map);
        
        std::lock_guard<std::mutex> guard(_statistics_mutex);
        _statistics[frameIndex].number_fish = assigned_count;
        _statistics[frameIndex].posture_seconds = posture_seconds;
    }

void Tracker::update_iterator_maps(long_t frame, const Tracker::set_of_individuals_t& active_individuals, std::unordered_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators)
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
            
    void Tracker::update_warnings(long_t frameIndex, double time, long_t /*number_fish*/, long_t n_found, long_t n_prev, const FrameProperties *props, const FrameProperties *prev_props, const Tracker::set_of_individuals_t& active_individuals, std::unordered_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators) {
        std::map<std::string, std::set<FOI::fdx_t>> merge;
        
        if(n_found < n_prev-1) {
            FOI::add(FOI(frameIndex, "lost >=2 fish"));
        }
        
        //if(!prev_props) prev_props = properties(frameIndex - 1);
        if(prev_props && time - prev_props->time >= FAST_SETTINGS(huge_timestamp_seconds)) {
            FOI::add(FOI(frameIndex, "huge time jump"));
            for(auto id : FAST_SETTINGS(manual_identities))
                merge["correcting"].insert(FOI::fdx_t(id));
        }
        
        std::set<FOI::fdx_t> found_matches;
        for(auto fish : _active_individuals_frame.at(frameIndex)) {
            if(fish->is_manual_match(frameIndex))
                found_matches.insert(FOI::fdx_t(fish->identity().ID()));
        }
        
        if(!found_matches.empty()) {
            FOI::add(FOI(frameIndex, found_matches, "manual match"));
            merge["correcting"].insert(found_matches.begin(), found_matches.end());
        }
        
        update_iterator_maps(frameIndex - 1, active_individuals, individual_iterators);
        for(auto &fish : active_individuals) {
            if(_warn_individual_status.size() <= (size_t)fish->identity().ID()) {
                _warn_individual_status.resize(fish->identity().ID() + 1);
            }
            
            auto &property = _warn_individual_status[fish->identity().ID()];
            
            auto fit = individual_iterators.find(fish->identity().ID());
            if(fit == individual_iterators.end()) {
                property.prev = property.current = nullptr;
                continue;
            }
            
            auto &it = fit->second;
            if(it != fish->frame_segments().end() && (*it)->contains(frameIndex - 1)) {
                // prev
                auto idx = (*it)->basic_stuff(frameIndex - 1);
                property.prev = idx != -1 ? fish->basic_stuff()[uint32_t(idx)]->centroid : nullptr;
                
                // current
                idx = (*it)->basic_stuff(frameIndex);
                property.current = idx != -1 ? fish->basic_stuff()[uint32_t(idx)]->centroid : nullptr;
                
            } else
                property.prev = property.current = nullptr;
        }
        
#ifndef NDEBUG
        for(auto &fish : active_individuals) {
            if(_warn_individual_status.size() <= fish->identity().ID()) {
                assert(!fish->has(frameIndex-1));
                continue;
            }
            
            auto &property = _warn_individual_status.at(fish->identity().ID());
            if(property.prev == nullptr) {
                assert(!fish->has(frameIndex-1));
            } else {
                assert((property.prev != nullptr) == fish->has(frameIndex-1));
                if(property.prev != nullptr) {
                    if(property.current == nullptr) {
                        assert(fish->segment_for(frameIndex-1) != fish->segment_for(frameIndex));
                    } else
                        assert(fish->segment_for(frameIndex-1) == fish->segment_for(frameIndex));
                } else
                    assert(property.current == nullptr);
            }
        }
#endif
        
        if(prev_props && props) {
            std::set<FOI::fdx_t> weird_distance, weird_angle, segment_end;
            std::set<FOI::fdx_t> fdx;
            
            for(auto fish : active_individuals) {
                auto properties = _warn_individual_status.size() > (size_t)fish->identity().ID() ? &_warn_individual_status[fish->identity().ID()] : nullptr;
                
                if(properties && properties->current) {
                    if(properties->current->speed(Units::CM_AND_SECONDS) >= Individual::weird_distance()) {
                        weird_distance.insert(FOI::fdx_t{fish->identity().ID()});
                    }
                }
                
                if(properties && properties->prev && properties->current) {
                    // only if both current and prev are set, do we have
                    // both frameIndex-1 and frameIndex present in the same segment:
                    assert(fish->has(frameIndex-1) && fish->has(frameIndex));
                    if(cmn::abs(angle_difference(properties->prev->angle(), properties->current->angle())) >= M_PI * 0.8)
                    {
                        weird_angle.insert(FOI::fdx_t{fish->identity().ID()});
                    }
                    
                } else if(properties && properties->prev) {
                    segment_end.insert(FOI::fdx_t{fish->identity().ID()});
                    
                    if(!fish->has(frameIndex)) {
                        assert(fish->has(frameIndex-1) && !fish->has(frameIndex));
                        fdx.insert(FOI::fdx_t{fish->identity().ID()});
                    }
                    
                } else if(!properties)
                    Warning("No properties for fish %d", fish->identity().ID());
            }
            
#ifndef NDEBUG
            for(auto id : segment_end) {
                assert(individuals().at(Idx_t(id.id))->segment_for(frameIndex) != individuals().at(Idx_t(id.id))->segment_for(frameIndex-1));
            }
            for(auto id : fdx) {
                assert(!individuals().at(Idx_t(id.id))->has(frameIndex));
                assert(frameIndex != start_frame() && _individuals.at(Idx_t(id.id))->has(frameIndex-1));
            }
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
                FOI::add(FOI(frameIndex-1, segment_end, "segment end"));
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
                    U_EXCEPTION("%S != %S", &str0, &str1);
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

    void Tracker::update_consecutive(const Tracker::set_of_individuals_t &active, long_t frameIndex, bool update_dataset) {
        bool all_good = FAST_SETTINGS(track_max_individuals) == (uint32_t)active.size();
        //bool manual = false;
        /*Rangel manual_approval;
        
        for(auto && [from, to] : FAST_SETTINGS(manually_approved)) {
            if(Rangel(from, to+1).contains(frameIndex)) {
                manual = true;
                manual_approval = Rangel(from,to);
                break;
            }
        }
        
        if(!manual) {*/
        auto manual_identities = FAST_SETTINGS(manual_identities);
            for(auto fish : active) {
                if(manual_identities.empty() || manual_identities.count(fish->identity().ID())) {
                    if(!fish->has(frameIndex) /*|| fish->centroid_weighted(frameIndex)->speed() >= FAST_SETTINGS(track_max_speed) * 0.25*/) {
                        all_good = false;
                        break;
                    }
                }
            }
        /*} else if(manual_approval.end == frameIndex) {
            Warning("Letting frame %d-%d slip because its manually approved.", manual_approval.start, manual_approval.end);
        }*/
        
        if(all_good) {
            if(!_consecutive.empty() && _consecutive.back().end == frameIndex-1) {
                _consecutive.back().end = frameIndex;
                if(frameIndex == analysis_range().end)
                    _recognition->update_dataset_quality();
            } else {
                if(!_consecutive.empty()) {
                    FOI::add(FOI(_consecutive.back(), "global segment"));
                }
                
                _consecutive.push_back(Range<long_t>(frameIndex, frameIndex));
                if(update_dataset)
                    _recognition->update_dataset_quality();
            }
        }
    }

    void Tracker::generate_pairdistances(long_t frameIndex) {
        std::vector<Individual*> frame_individuals;
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
                d.set_d(euclidean_distance(frame_individuals.at(i)->centroid(frameIndex)->pos(Units::PX_AND_SECONDS), frame_individuals.at(idx1)->centroid(frameIndex)->pos(Units::PX_AND_SECONDS)));
                
                distances.insert(d);
            }
        }

        U_EXCEPTION("Pair distances need to implement the new properties.");
        //std::copy(distances.begin(), distances.end(), std::back_inserter(properties(frameIndex)->_pair_distances));
    }
    
    void Tracker::_remove_frames(long_t frameIndex) {
        Categorize::DataStore::reanalysed_from(Frame_t(frameIndex));
        
        LockGuard guard("_remove_frames("+Meta::toStr(frameIndex)+")");
        recognition_pool.wait();
        _thread_pool.wait();
        
        _individual_add_iterator_map.clear();
        _segment_map_known_capacity.clear();
        
        if(_approximative_enabled_in_frame >= frameIndex)
            _approximative_enabled_in_frame = -1;
        
        Debug("Removing frames after and including %ld", frameIndex);
        
        if (_endFrame < frameIndex || _startFrame > frameIndex)
            return;
        
        if(history_log && history_log->is_open()) {
            DebugCallback("Restarting history log from %d", frameIndex);
            history_log = nullptr;
            update_history_log();
        }
        
        if(!_consecutive.empty()) {
            while(!_consecutive.empty()) {
                if(_consecutive.back().start < frameIndex)
                    break;
                
                _consecutive.erase(--_consecutive.end());
            }
            Debug("Last remaining %d", _consecutive.size());
            if(!_consecutive.empty()) {
                if(_consecutive.back().end >= frameIndex)
                    _consecutive.back().end = frameIndex-1;
                Debug("%d-%d", _consecutive.back().start, _consecutive.back().end);
            }
        }
        
        auto manual_identities = FAST_SETTINGS(manual_identities);
        std::vector<Idx_t> to_delete;
        std::vector<Individual*> ptrs;
        for(auto && [fdx, fish] : _individuals) {
            fish->remove_frame(frameIndex);
            
            if(FAST_SETTINGS(track_max_individuals) == 0 || manual_identities.find(fdx) == manual_identities.end()) {
                if(fish->empty()) {
                    to_delete.push_back(fdx);
                    ptrs.push_back(fish);
                }
            }
        }
        
        for(auto fdx : to_delete)
            _individuals.erase(fdx);
        
        for(auto it = _active_individuals_frame.begin(); it != _active_individuals_frame.end();) {
            if(it->first >= frameIndex)
                it = _active_individuals_frame.erase(it);
            else
                ++it;
        }
        
        while(!_added_frames.empty()) {
            if((--_added_frames.end())->frame < frameIndex)
                break;
            _added_frames.erase(--_added_frames.end());
        }
        
        for (auto it=_statistics.begin(); it != _statistics.end();) {
            if(it->first < frameIndex)
                ++it;
            else
                it = _statistics.erase(it);
        }
        
        _endFrame = frameIndex-1;
        while (!properties(_endFrame)) {
            if (_endFrame < _startFrame) {
                _endFrame = _startFrame = -1;
                break;
            }
            
            _endFrame--;
        }
        
        if(_endFrame != -1 && _endFrame < analysis_range().start)
            _endFrame = _startFrame = -1;
        
        if(properties(_endFrame))
            _active_individuals = _active_individuals_frame.at(_endFrame);
        else
            _active_individuals = {};
        
        _inactive_individuals.clear();
        //! assuming that most of the active / inactive individuals will stay the same, this should actually be more efficient
        for(auto&& [id, fish] : _individuals) {
            if(_active_individuals.find(fish) == _active_individuals.end())
                _inactive_individuals.insert(id);
        }
        
        for (auto ptr : ptrs) {
            assert (_individual_add_iterator_map.find(ptr->identity().ID()) == _individual_add_iterator_map.end() );
            delete ptr;
        }
        
        if(_individuals.empty())
            Identity::set_running_id(0);
        
        if(_recognition) {
            _recognition->clear_filter_cache();
            
            _recognition->remove_frames(frameIndex);
            
            if(_recognition->dataset_quality()) {
                _recognition->dataset_quality()->remove_frames(frameIndex);
                _recognition->update_dataset_quality();
            }
        }
        
        {
            //! update the cache for frame properties
            std::unique_lock guard(_properties_mutex);
            _properties_cache.clear();
            
            long_t frame = end_frame();
            auto it = _added_frames.rbegin();
            while(it != _added_frames.rend() && !_properties_cache.full())
            {
                _properties_cache.push_front(it->frame, &(*it));
                ++it;
            }
            assert((_added_frames.empty() && end_frame() == -1) || (end_frame() != -1 && _added_frames.rbegin()->frame == end_frame()));
        }
        
        FOI::remove_frames(frameIndex);
        global_segment_order_changed();
        
        auto str = Meta::toStr(_inactive_individuals);
        Debug("Inactive individuals: %S", &str);
        str = Meta::toStr(_active_individuals);
        Debug("Active individuals: %S", &str);
        
        Debug("After removing frames: %d", gui::CacheObject::memory());
        Debug("posture: %d", Midline::saved_midlines());
        Debug("physical props: %d", PhysicalProperties::saved_midlines());
        Debug("all blobs: %d", Blob::all_blobs());
        Debug("Range: %d-%d", _startFrame.load(), _endFrame.load());
    }

    size_t Tracker::found_individuals_frame(size_t frameIndex) const {
        if(!properties(frameIndex))
            return 0;
        
        auto &a = active_individuals(frameIndex);
        size_t n = 0;
        for (auto i : a) {
            n += i->has(frameIndex) ? 1 : 0;
        }
        
        return n;
    }

    void Tracker::wait() {
        recognition_pool.wait();
    }

    void Tracker::global_segment_order_changed() {
        LockGuard guard("Tracker::global_segment_order_changed");
        _global_segment_order.clear();
    }
    
    std::vector<Rangel> Tracker::global_segment_order() {
        LockGuard guard("Tracker::max_range()");
        if(_global_segment_order.empty()) {
            std::set<Rangel> manuals;
            auto manually_approved = FAST_SETTINGS(manually_approved);
            for(auto && [from, to] : manually_approved)
                manuals.insert(Rangel(from, to));
            
            std::set<Rangel, std::function<bool(Rangel, Rangel)>> ordered([&manuals](Rangel A, Rangel B) -> bool {
                if(manuals.find(A) != manuals.end() && manuals.find(B) == manuals.end())
                    return true;
                if(manuals.find(B) != manuals.end() && manuals.find(A) == manuals.end())
                    return false;
                return (recognition() && recognition()->dataset_quality() ? ((recognition()->dataset_quality()->has(A) ? recognition()->dataset_quality()->quality(A) : DatasetQuality::Quality()) > (recognition()->dataset_quality()->has(B) ? recognition()->dataset_quality()->quality(B) : DatasetQuality::Quality())) : (A.length() > B.length()));
            });
            
            if(!manually_approved.empty()) {
                auto str = Meta::toStr(manually_approved);
                //Debug("Inserting %S", &str);
                for(auto && [from, to] : manually_approved) {
                    ordered.insert(Rangel(from, to));
                }
            }
            
            std::set<Rangel> consecutive;
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
            
            _global_segment_order = std::vector<Rangel>(ordered.begin(), ordered.end());
        }
        
        return _global_segment_order;
    }
    
    struct IndividualImages {
        std::vector<long_t> frames;
        std::vector<Image::Ptr> images;
    };
    
    struct SplitData {
        std::map<long_t, IndividualImages> training;
        std::map<long_t, IndividualImages> validation;
        std::map<long_t, Rangel> ranges;
        TrainingData::MidlineFilters filters;
        
        GETTER(default_config::recognition_normalization_t::Class, normalized)
        
    public:
        SplitData();
        void add_frame(long_t frame, long_t id, Image::Ptr image);
    };
    
    SplitData::SplitData() : _normalized(SETTING(recognition_normalize_direction).value<default_config::recognition_normalization_t::Class>()) {
        
    }
    
    void SplitData::add_frame(long_t frame, long_t id, Image::Ptr image) {
        assert(image);
        
        if(training.size() <= validation.size() * 1.25) {
            training[id].frames.push_back(frame);
            training[id].images.push_back(image);
        } else {
            validation[id].frames.push_back(frame);
            validation[id].images.push_back(image);
        }
    }
    
    

    void log(FILE* f, const char* cmd, ...) {
        std::string output;
        
        va_list args;
        va_start(args, cmd);
        
        DEBUG::ParseFormatString(output, cmd, args);
        va_end(args);
        
        output += "\n";
        
        if(f)
            fwrite(output.c_str(), sizeof(char), output.length(), f);
    }
    
    void Tracker::clear_segments_identities() {
        LockGuard guard("clear_segments_identities");
        
        recognition_pool.wait();
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(0, fid);
        
        for(auto && [fdx, fish] : _individuals) {
            fish->clear_recognition();
        }
        
        automatically_assigned_ranges.clear();
    }
    
    void Tracker::check_segments_identities(bool auto_correct, std::function<void(float)> callback, const std::function<void(const std::string&, const std::function<void()>&, const std::string&)>& add_to_queue, long_t after_frame) {
        
        Debug("Waiting for gui...");
        if(GUI::instance()) {
            std::lock_guard<decltype(GUI::instance()->gui().lock())> guard(GUI::instance()->gui().lock());
            GUI::work().set_item("updating with automatic ranges");
        }
        
        Debug("Waiting for lock...");
        LockGuard guard("check_segments_identities");
        Debug("Updating automatic ranges starting from %d", after_frame == -1 ? 0 : after_frame);
        
        const auto manual_identities = FAST_SETTINGS(manual_identities);
        size_t count=0;
        
        recognition_pool.wait();
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(after_frame != -1 ? 0 : after_frame, fid);
        
#ifdef TREX_DEBUG_IDENTITIES
        auto f = fopen(pv::DataLocation::parse("output", "identities.log").c_str(), "wb");
#endif
        distribute_vector([this, &count, &callback, &manual_identities](auto i, auto it, auto nex, auto step){
            auto & [fdx, fish] = *it;
            
            if(manual_identities.empty() || manual_identities.find(fdx) != manual_identities.end()) {
                fish->clear_recognition();
                fish->calculate_average_recognition();
                
                callback(count / float(_individuals.size()) * 0.5f);
                ++count;
            }
            
        }, recognition_pool, _individuals.begin(), _individuals.end());
        
        using fdx_t = Idx_t;
        using range_t = FrameRange;
        using namespace Match;
        
        struct VirtualFish {
            std::set<range_t> segments;
            std::map<range_t, Match::prob_t> probs;
            std::map<range_t, size_t> samples;
            std::map<Rangel, Idx_t> track_ids;
        };
        
        std::map<long_t, std::map<Idx_t, int64_t>> automatic_matches;
        std::map<fdx_t, VirtualFish> virtual_fish;
        
        // wrong fish -> set of unassigned ranges
        std::map<fdx_t, std::set<Rangel>> unassigned_ranges;
        std::map<fdx_t, std::map<Rangel, fdx_t>> assigned_ranges;
        
        decltype(automatically_assigned_ranges) tmp_assigned_ranges;
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
        
        const size_t n_lower_bound = max(5, FAST_SETTINGS(frame_rate) * 0.1f);
        
        // iterate through segments, find matches for segments.
        // try to find the longest segments and assign them to virtual fish
        for(auto && [fdx, fish] : _individuals) {
            if(manual_identities.empty() || manual_identities.find(fdx) != manual_identities.end()) {
                // recalculate recognition for all segments
                //fish->clear_recognition();
                
                for(auto && [start, segment] : fish->recognition_segments()) {
                    auto && [n, average] = fish->processed_recognition(start);
                    
                    if(after_frame != -1 && segment.range.end < after_frame)
                        continue;
                    
                    if(n >= n_lower_bound || (segment.start() == fish->start_frame() && n > 0)) {
#ifdef TREX_DEBUG_IDENTITIES
                        log(f, "fish %d: segment %d-%d has %d samples", fdx, segment.start(), segment.end(), n);
#endif
                        Debug("fish %d: segment %d-%d has %d samples", fdx, segment.start(), segment.end(), n);
                        
                        std::set<std::pair<Idx_t, Match::prob_t>, decltype(compare_greatest)> sorted(compare_greatest);
                        sorted.insert(average.begin(), average.end());
                        
                        // check if the values for this segment are too close, this probably
                        // means that we shouldnt correct here.
                        if(sorted.size() >= 2) {
                            Match::prob_t ratio = sorted.begin()->second / ((++sorted.begin())->second);
                            if(ratio > 1)
                                ratio = 1 / ratio;
                            
                            if(ratio >= 0.6) {
                                //Debug("Fish %d (%d-%d)", fdx, segment.start(), segment.end());
                                //Debug("\ttwo largest probs %f and %f are too close (ratio %f)", sorted.begin()->second, (++sorted.begin())->second, ratio);
#ifdef TREX_DEBUG_IDENTITIES
                                log(f, "\ttwo largest probs %f and %f are too close (ratio %f)", sorted.begin()->second, (++sorted.begin())->second, ratio);
#endif
                                continue;
                            }
                        }
                        
                        //auto it = std::max_element(average.begin(), average.end(), compare);
                        auto it = sorted.begin();
                        
                        // see if there is already something found for given segment that
                        // overlaps with this segment
                        auto fit = virtual_fish.find(it->first);
                        if(fit != virtual_fish.end()) {
                            // fish exists
                            auto &A = segment;
                            
                            std::set<range_t> matches;
                            auto rit = fit->second.segments.begin();
                            for(; rit != fit->second.segments.end(); ++rit) {
                                auto &B = *rit;
                                //if(B.overlaps(A))
                                if(A.end() > B.start() && A.start() < B.end())
                                //if((B.start() >= A.start() && A.end() >= B.start())
                                //   || (A.start() >= B.start() && B.end() >= A.start()))
                                {
                                    matches.insert(B);
                                }
                            }
                            
                            if(!matches.empty()) {
                                // if there are multiple matches, we can already assume that this
                                // is a much longer segment (because it overlaps multiple smaller segments
                                // because it starts earlier, cause thats the execution order)
                                auto rit = matches.begin();
#ifdef TREX_DEBUG_IDENTITIES
                                log(f, "\t%d (as %d) Found range(s) %d-%d for search range %d-%d p:%f n:%d (self:%f,n:%d)", fdx, it->first, rit->start(), rit->end(), segment.start(), segment.end(), fit->second.probs.at(*rit), fit->second.samples.at(*rit), it->second, n);
#endif
                                
                                Match::prob_t n_me = n;//segment.end() - segment.start();
                                Match::prob_t n_he = fit->second.samples.at(*rit);//rit->end() - rit->start();
                                const Match::prob_t N = n_me + n_he;
                                
                                n_me /= N;
                                n_he /= N;
                                
                                Match::prob_t sum_me = sigmoid(it->second) * sigmoid(n_me);
                                Match::prob_t sum_he = sigmoid(fit->second.probs.at(*rit)) * sigmoid(n_he);
                                
#ifdef TREX_DEBUG_IDENTITIES
                                log(f, "\tself:%d %f other:%d %f => %f / %f", segment.length(), it->second, rit->length(), fit->second.probs.at(*rit), sum_me, sum_he);
#endif
                                
                                if(sum_me > sum_he) {
#ifdef TREX_DEBUG_IDENTITIES
                                    log(f, "\t* Replacing");
#endif
                                    
                                    for(auto rit = matches.begin(); rit != matches.end(); ++rit) {
                                        fit->second.probs.erase(*rit);
                                        fit->second.track_ids.erase(rit->range);
                                        fit->second.segments.erase(*rit);
                                        fit->second.samples.erase(*rit);
                                    }
                                    
                                } else
                                    continue;
                            }
                        }
                        
#ifdef TREX_DEBUG_IDENTITIES
                        log(f, "\tassigning %d to %d with p %f for %d-%d", it->first, fdx, it->second, segment.start(), segment.end());
#endif
                        virtual_fish[it->first].segments.insert(segment);
                        virtual_fish[it->first].probs[segment] = it->second;
                        virtual_fish[it->first].samples[segment] = n;
                        virtual_fish[it->first].track_ids[segment.range] = fdx;
                        
                        assigned_ranges[fdx][segment.range] = it->first;
                    }
                }
            }
        }
        
        Settings::manual_splits_t manual_splits;
        
#ifdef TREX_DEBUG_IDENTITIES
        log(f, "Found segments:");
#endif
        for(auto && [fdx, fish] : virtual_fish) {
#ifdef TREX_DEBUG_IDENTITIES
            log(f, "\t%d:", fdx);
#endif
            // manual_match for first segment
            if(!fish.segments.empty()) {
                auto segment = *fish.segments.begin();
                
                if(!fish.probs.count(segment))
                    U_EXCEPTION("Cannot find %d-%d in fish.probs", segment.start(), segment.end());
                if(!fish.track_ids.count(segment.range))
                    U_EXCEPTION("Cannot find %d-%d in track_ids", segment.start(), segment.end());
                
                auto track = _individuals.at(fish.track_ids.at(segment.range));
                
                if(segment.first_usable != -1 && segment.first_usable != segment.start()) {
                    auto blob = track->compressed_blob(segment.first_usable);
                    if(blob)
                        automatic_matches[segment.first_usable][fdx] = blob->blob_id();
                    else
                        Warning("Have first_usable (=%d), but blob is null (fish %d)", segment.first_usable, fdx);
                }
                
                auto blob = track->compressed_blob(segment.start());
                if(blob) {
                    automatic_matches[segment.start()][fdx] = blob->blob_id();
                    if(blob->split() && blob->parent_id != -1)
                        manual_splits[segment.start()].insert(blob->parent_id);
                }
            }
            
            for(auto segment : fish.segments) {
                if(after_frame != -1 && segment.range.end < after_frame)
                    continue;
                
                if(!fish.probs.count(segment))
                    U_EXCEPTION("Cannot find %d-%d in fish.probs", segment.start(), segment.end());
                if(!fish.track_ids.count(segment.range))
                    U_EXCEPTION("Cannot find %d-%d in track_ids", segment.start(), segment.end());
#ifdef TREX_DEBUG_IDENTITIES
                log(f, "\t\t%d-%d: %f (from %d)", segment.start(), segment.end(), fish.probs.at(segment), fish.track_ids.at(segment.range));
#endif
                auto track = _individuals.at(fish.track_ids.at(segment.range));
                assert(track->compressed_blob(segment.start()));
                
                //automatic_matches[segment.start()][fdx] = track->blob(segment.start())->blob_id();
                if(!assigned_ranges.count(track->identity().ID()) || !assigned_ranges.at(track->identity().ID()).count(segment.range))
                    assigned_ranges[track->identity().ID()][segment.range] = fdx;
                
                auto blob = track->compressed_blob(segment.start());
                if(blob && blob->split() && blob->parent_id != -1)
                    manual_splits[segment.start()].insert(blob->parent_id);
                
                std::vector<int64_t> blob_ids;
                for(long_t frame=segment.start(); frame<=segment.end(); ++frame) {
                    blob = track->compressed_blob(frame);
                    if(blob) {
                        //automatically_assigned_blobs[frame][blob->blob_id()] = fdx;
                        blob_ids.push_back(blob->blob_id());
                        //if(blob->split() && blob->parent_id() != -1)
                        //    manual_splits[frame].insert(blob->parent_id());
                    } else
                        blob_ids.push_back(-1);
                    
                    std::set<Rangel> remove_from;
                    for(auto && [range, blobs] : tmp_assigned_ranges[fdx]) {
                        if(range != segment.range && range.contains(frame)) {
                            //if(!blob || contains(blobs, blob->blob_id())) {
                                remove_from.insert(range);
                            //}
                            
                            //break;
                        }
                    }
                    
                    
                    /*for(auto && [b, f] : automatically_assigned_blobs[frame]) {
                        if(f == fdx && (!blob || (blob && b != blob->blob_id()))) {
                            remove_from.insert(b);
                        }
                    }*/
                    
                    if(!remove_from.empty()) {
                        for(auto range : remove_from)
                            tmp_assigned_ranges[fdx].erase(range);
                        
                        auto str = Meta::toStr(remove_from);
                        Warning("While assigning %d,%d to %d -> same fish already assigned in ranges %S", frame, blob ? (int64_t)blob->blob_id() : -1, fdx, &str);
                    }
                }
                
                assert((long_t)blob_ids.size() == segment.range.end - segment.range.start + 1);
                tmp_assigned_ranges[fdx][segment.range] = blob_ids;
            }
        }
#ifdef TREX_DEBUG_IDENTITIES
        log(f, "----");
#endif
        decltype(unassigned_ranges) still_unassigned;
        //auto manual_identities = FAST_SETTINGS(manual_identities);
        for(auto && [fdx, fish] : _individuals) {
            if(!manual_identities.count(fdx))
                continue;
            
            for(auto && [start, segment] : fish->recognition_segments()) {
                auto previous = fish->recognition_segments().end(),
                     next = fish->recognition_segments().end();
                
                const auto current = fish->recognition_segments().find(start);
                if(after_frame != -1 && segment.range.end < after_frame)
                    continue;
                //if(start == 741 && fish->identity().ID() == 1)
                //    Debug("Here");
                
                if(current != fish->recognition_segments().end()) {
                    auto it = current;
                    if((++it) != fish->recognition_segments().end())
                        next = it;
                    
                    it = current;
                    if(it != fish->recognition_segments().begin())
                        previous = (--it);
                }
                
                if(assigned_ranges.count(fdx) && assigned_ranges.at(fdx).count(segment.range)) {
                    continue; // already assigned this frame segment to someone...
                }
                
                if(next != fish->recognition_segments().end() && /*previous.start() != -1 &&*/ next->second.start() != -1) {
                    Idx_t prev_id, next_id;
                    PhysicalProperties *prev_pos = nullptr, *next_pos = nullptr;
                    int64_t prev_blob = -1;
                    
                    auto it = assigned_ranges.find(fdx);
                    if(it != assigned_ranges.end()) {
                        decltype(it->second.begin()) rit;
                        const long_t max_frames = FAST_SETTINGS(frame_rate)*15;
                        
                        // skip some frame segments to find the next assigned id
                        do {
                            // dont assign anything after one second
                            if(next->second.start() >= current->second.end() + max_frames)
                                break;
                            
                            rit = it->second.find(next->second.range);
                            if(rit != it->second.end()) {
                                next_id = rit->second;
                                
                                if(virtual_fish.count(next_id) && virtual_fish.at(next_id).track_ids.count(rit->first)) {
                                    auto org_id = virtual_fish.at(next_id).track_ids.at(rit->first);
                                    auto blob = _individuals.at(org_id)->centroid_weighted(next->second.start());
                                    if(blob)
                                        next_pos = blob;
                                }
                                break;
                            }
                            
                        } while((++next) != fish->recognition_segments().end());
                        
                        // skip some frame segments to find the previous assigned id
                        while(previous != fish->recognition_segments().end()) {
                            // dont assign anything after one second
                            if(previous->second.end() + max_frames < current->second.start())
                                break;
                            
                            rit = it->second.find(previous->second.range);
                            if(rit != it->second.end()) {
                                prev_id = rit->second;
                                
                                if(virtual_fish.count(prev_id) && virtual_fish.at(prev_id).track_ids.count(rit->first)) {
                                    auto org_id = virtual_fish.at(prev_id).track_ids.at(rit->first);
                                    auto pos = _individuals.at(org_id)->centroid_weighted(previous->second.end());
                                    if(pos) {
                                        prev_pos = pos;
                                        prev_blob = previous->second.end();
                                    }
                                }
                                break;
                            }
                            
                            if(previous != fish->recognition_segments().begin())
                                --previous;
                            else
                                break;
                        }
                    }
                    
                    if(next_id.valid() && prev_id.valid() && next_id == prev_id && prev_pos && next_pos) {
                        //Debug("Fish %d: virtual prev_id %d == virtual next_id %d, assigning...", fdx, prev_id, next_id);
                        Vec2 pos_start(FLT_MAX), pos_end(FLT_MAX);
                        auto blob_start = fish->centroid_weighted(segment.start());
                        auto blob_end = fish->centroid_weighted(segment.end());
                        if(blob_start)
                            pos_start = blob_start->pos(Units::CM_AND_SECONDS);
                        if(blob_end)
                            pos_end = blob_end->pos(Units::CM_AND_SECONDS);
                        
                        if(blob_start && blob_end) {
                            auto dprev = euclidean_distance(prev_pos->pos(Units::CM_AND_SECONDS), pos_start) / Tracker::time_delta(blob_start->frame(), prev_pos->frame());
                            auto dnext = euclidean_distance(next_pos->pos(Units::CM_AND_SECONDS), pos_end) / Tracker::time_delta(next_pos->frame(), blob_end->frame());
                            Idx_t chosen_id;
                            
                            if(dnext < dprev) {
                                if(dprev < FAST_SETTINGS(track_max_speed) * 0.1)
                                    chosen_id = next_id;
                            } else if(dnext < FAST_SETTINGS(track_max_speed) * 0.1)
                                chosen_id = prev_id;
                            
                            if(chosen_id.valid()) {
#ifdef TREX_DEBUG_IDENTITIES
                                if(segment.start() == 0) {
                                    log(f, "Fish %d: chosen_id %d, assigning %d-%d (%f / %f)...", fdx, chosen_id, segment.start(), segment.end(), dprev, dnext);
                                }
#endif
                                
                                if(prev_blob != -1 && prev_id.valid()) {
                                    // we found the previous blob/segment quickly:
                                    auto range = _individuals.at(prev_id)->get_segment_safe(prev_blob);
                                    if(!range.empty()) {
                                        long_t frame = range.end();
                                        while(frame >= range.start()) {
                                            auto blob = _individuals.at(prev_id)->compressed_blob(frame);
                                            if(blob->split()) {
                                                if(blob->parent_id != -1) {
                                                    //manual_splits[frame].insert(blob->parent_id());
                                                    //Debug("Inserting manual split %d : %d (%d)", frame, blob->parent_id(), blob->blob_id());
                                                }
                                            } else
                                                break;
                                            
                                            --frame;
                                        }
                                    }
                                }
                                
                                // find and remove duplicates
                                /*auto it = automatic_matches.find(segment.start());
                                if(it != automatic_matches.end()) {
                                    long_t to_erase = -1;
                                    for(auto && [fdx, bdx] : it->second) {
                                        if(fdx != chosen_id && bdx == fish->blob(segment.start())->blob_id()) {
                                            to_erase = fdx;
                                            break;
                                        }
                                    }
                                    
                                    if(to_erase != -1)
                                        it->second.erase(to_erase);
                                }*/
                                
                                std::set<Rangel> remove_from;
                                
                                std::vector<int64_t> blob_ids;
                                for(long_t frame=segment.start(); frame<=segment.end(); ++frame) {
                                    auto blob = fish->compressed_blob(frame);
                                    
                                    if(blob) {
                                        //automatically_assigned_blobs[frame][blob->blob_id()] = fdx;
                                        blob_ids.push_back(blob->blob_id());
                                        //if(blob->split() && blob->parent_id() != -1)
                                        //    manual_splits[frame].insert(blob->parent_id());
                                    } else
                                        blob_ids.push_back(-1);
                                    
                                    
                                    for(auto && [range, blobs] : tmp_assigned_ranges[chosen_id]) {
                                        if(range != segment.range && range.contains(frame)) {
                                            remove_from.insert(range);
                                            break;
                                        }
                                    }
                                }
                                
                                if(!remove_from.empty()) {
                                    //for(auto range : remove_from)
                                    //    automatically_assigned_ranges[chosen_id].erase(range);
                                    
                                    auto str = Meta::toStr(remove_from);
                                    Warning("[ignore] While assigning %d-%d to %d -> same fish already assigned in ranges %S", segment.range.start, segment.range.end, (uint32_t)chosen_id, &str);
                                } else {
                                    assert((int64_t)blob_ids.size() == segment.range.end - segment.range.start + 1);
                                    tmp_assigned_ranges[chosen_id][segment.range] = blob_ids;
                                    
                                    auto blob = fish->blob(segment.start());
                                    if(blob && blob->split() && blob->parent_id() != -1)
                                        manual_splits[segment.start()].insert(blob->parent_id());
                                    
                                    assigned_ranges[fdx][segment.range] = chosen_id;
                                }
                                
                                continue;
                            }
                        }
                    }
                }
                
                still_unassigned[fdx].insert(segment.range);
            }
        }
        
        //auto str = prettify_array(Meta::toStr(still_unassigned));
        //Debug("still unassigned: %S", &str);
        Debug("auto_assign is %d", auto_correct ? 1 : 0);
        if(auto_correct) {
            add_to_queue("", [after_frame, automatic_matches, manual_splits, tmp_assigned_ranges](){
                Debug("Assigning to queue from frame %d", after_frame);
                
                std::lock_guard<decltype(GUI::instance()->gui().lock())> guard(GUI::instance()->gui().lock());
                
                {
                    Tracker::LockGuard guard("check_segments_identities::auto_correct");
                    Tracker::instance()->_remove_frames(after_frame == -1 ? Tracker::analysis_range().start : after_frame);
                    for(auto && [fdx, fish] : instance()->individuals()) {
                        fish->clear_recognition();
                    }
                    
                    Debug("automatically_assigned_ranges %d", tmp_assigned_ranges.size());
                    automatically_assigned_ranges = tmp_assigned_ranges;
                }
                
                if(after_frame == -1)
                    SETTING(manual_matches) = automatic_matches;
                if(after_frame == -1)
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

pv::BlobPtr Tracker::find_blob_noisy(const PPFrame& pp, int64_t bid, int64_t pid, const Bounds& bounds)
{
    auto blob = pp.find_bdx(bid);
    if(!blob) {
        return nullptr;
        
        if(pid != -1) {
            blob = pp.find_bdx((uint32_t)pid);
            if(blob) {
                auto blobs = pixel::threshold_blob(blob, FAST_SETTINGS(track_threshold), Tracker::instance()->background());
                
                for(auto & sub : blobs) {
                    if(sub->blob_id() == bid) {
                        //Debug("Found perfect match for %d in blob %d", bid, b->blob_id());//blob_to_id[bid] = sub;
                        //sub->calculate_moments();
                        return sub;
                        //break;
                    }
                }
                
                /*if(!blob_to_id.count(bid)) {
                    return nullptr;*/
                    /*int first_found = -1, last_found = -1;
                    
                    for(int threshold = FAST_SETTINGS(track_threshold)+1; threshold<100; ++threshold) {
                        auto blobs = pixel::threshold_blob(b, threshold, Tracker::instance()->background());
                        
                        for(auto & sub : blobs) {
                            if((long_t)sub->blob_id() == bid) {
                                if(first_found == -1) {
                                    first_found = threshold;
                                    blob_to_id[bid] = sub;
                                }
                                last_found = threshold;
                                
                                break;
                            }
                        }
                    }
                    
                    if(first_found != -1) {
                        Debug("Found blob %d in parent %d within thresholds [%d - %d]", bid, pid, first_found, last_found);
                    } else {
                        //Warning("Cannot find blob %d in it, but can find the parent %d in frame %d (threshold=%d).", bid, pid, frame, FAST_SETTINGS(track_threshold));
                    //}*/
                //}
            }
        }
        
        /*if(!blob_to_id.count(bid)) {
            //std::set<std::tuple<Match::PairingGraph::prob_t, long_t, Vec2>> sorted;
            //for(auto && [id, ptr] : blob_to_id) {
            //    sorted.insert({euclidean_distance(ptr->center(), bounds.pos() + bounds.size() * 0.5), id, ptr->center()});
            //}
            //auto str = Meta::toStr(sorted);
            
            //Error("Cannot find blob %d (%.0f,%.0f) in frame %d with threshold=%d. (%S)", bid, bounds.x,bounds.y, frame, FAST_SETTINGS(track_threshold), &str);
            return nullptr;
        }*/
        
        return nullptr;
    }
    
    return blob;
}

    void Tracker::check_save_tags(long_t frameIndex, const std::unordered_map<uint32_t, Individual*>& blob_fish_map, const std::vector<tags::blob_pixel> &tagged_fish, const std::vector<tags::blob_pixel> &noise, const file::Path & tags_path) {
        static Timing tag_timing("tags", 0.1);
        TakeTiming take(tag_timing);
        
        auto result = tags::prettify_blobs(tagged_fish, noise, *_average);
        for (auto &r : result) {
            auto && [var, bid, ptr, f] = tags::is_good_image(r, *_average);
            if(ptr) {
                auto it = blob_fish_map.find(r.blob->blob_id());
                if(it != blob_fish_map.end())
                    it->second->add_tag_image(tags::Tag{var, r.blob->blob_id(), ptr, frameIndex} /* ptr? */);
                else
                    Warning("Blob %u in frame %d contains a tag, but is not associated with an individual.", r.blob->blob_id(), frameIndex);
            }
        }
        
        if(_active_individuals_frame.find(frameIndex-1) != _active_individuals_frame.end())
        {
            for(auto fish : _active_individuals_frame.at(frameIndex-1)) {
                if(fish->start_frame() < frameIndex && fish->has(frameIndex-1) && !fish->has(frameIndex))
                {
                    // - none
                }
            }
        }
    }
    
    void Tracker::auto_calculate_parameters(pv::File& video, bool quiet) {
        if(video.length() > 1000 && (SETTING(auto_minmax_size) || SETTING(auto_number_individuals))) {
            gpuMat average;
            video.average().copyTo(average);
            if(average.cols == video.size().width && average.rows == video.size().height)
                video.processImage(average, average);
            
            Image local_average(average.rows, average.cols, 1);
            average.copyTo(local_average.get());
            
            if(!quiet)
                Debug("Determining blob size in %d frames...", video.length());
            
            Median<float> blob_size;
            pv::Frame frame;
            std::multiset<float> values;
            const uint32_t number_fish = SETTING(track_max_individuals).value<uint32_t>();
            
            std::vector<std::multiset<float>> blobs;
            Median<float> median;
            
            auto step = (video.length() - video.length()%500) / 500;
            for (size_t i=0; i<video.length(); i+=step) {
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
                        Debug("%d: %f %S", i / step, *frame_values.rbegin(), &str0);
                    }*/
                    
                    values.insert(result.begin(), result.end());
                    median.addNumber(*result.begin());
                    median.addNumber(*result.rbegin());
                }
                //values.insert(percentile(frame_values, 0.75));
                //values.insert(percentile(frame_values, 0.90));
            }
            
            float middle = 0;
            for(auto &v : values)
                middle += v;
            if(!values.empty())
                middle /= float(values.size());
            
            auto ranges = percentile(values, {0.25, 0.75});
            middle = median.getValue();
            middle = (ranges[1] - ranges[0]) * 0.5 + ranges[0];
            
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
            //    Debug("Calculated blob_size_range as: %f-%f median %f %f-%f", values.empty() ? -1.f : *values.begin(), values.empty() ? *values.rbegin() : -1.f, blob_size.added() ? blob_size.getValue() : -1.f, ten * 0.75, it == values.end() ? -1.f : (*it * 1.25));
            
            if(median_number != number_fish) {
                if(!quiet)
                    Warning("The set (%d) number of individuals differs from the detected number of individuals / frame (%d).", number_fish, median_number);
                
                //auto str = Meta::toStr(number_individuals);
                //Warning("%S", &str);
                
                if(SETTING(auto_number_individuals).value<bool>()) {
                    if(!quiet)
                        Debug("Setting number of individuals as %d.", median_number);
                    SETTING(track_max_individuals) = uint32_t(median_number);
                }
            }
        }
    }
}
