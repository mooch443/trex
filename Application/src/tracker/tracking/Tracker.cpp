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
//#include <gui/gui.h>
#include <misc/PixelTree.h>
#include <misc/CircularGraph.h>
#include <misc/MemoryStats.h>
//#include <gui/WorkProgress.h>
#include <tracking/Categorize.h>

#ifndef NDEBUG
//#define PAIRING_PRINT_STATS
#endif

namespace track {
    auto *tracker_lock = new std::recursive_timed_mutex;

    std::shared_ptr<std::ofstream> history_log;
    std::mutex log_mutex;

    template<typename... Args>
    inline void Log(std::ostream* out, Args... args) {
        if(!out)
            return;
        
        std::string str = format<FormatterType::NONE>(args...);
        if(dynamic_cast<std::ofstream*>(out)) {
            str = settings::htmlify(str) + "</br>";
        }
        
        std::lock_guard<std::mutex> guard(log_mutex);
        *out << str << std::endl;
    }
    
    Tracker* _instance = NULL;
    std::vector<Range<Frame_t>> _global_segment_order;

    Tracker* Tracker::instance() {
        return _instance;
    }
    
    inline void analyse_posture_pack(Frame_t frameIndex, const std::vector<std::tuple<Individual*, const Individual::BasicStuff*>>& p) {
        Timer t;
        double collected = 0;
        for(auto && [f, b] : p) {
            t.reset();
            f->save_posture(*b, frameIndex);
            collected += t.elapsed();
        }
        
        std::lock_guard<std::mutex> guard(Tracker::instance()->_statistics_mutex);
        Tracker::instance()->_statistics[frameIndex].combined_posture_seconds += narrow_cast<float>(collected);
    }
    
    //std::map<long_t, std::map<uint32_t, long_t>> automatically_assigned_blobs;

    struct RangesForID {
        struct AutomaticRange {
            Range<Frame_t> range;
            std::vector<pv::bid> bids;
            
            bool operator==(const Range<Frame_t>& range) const {
                return this->range.start == range.start && this->range.end == range.end;
            }
        };
        
        Idx_t id;
        std::vector<AutomaticRange> ranges;
        
        bool operator==(const Idx_t& idx) const {
            return id == idx;
        }
    };
    std::vector<RangesForID> _automatically_assigned_ranges;

void add_assigned_range(std::vector<RangesForID>& assigned, Idx_t fdx, const Range<Frame_t>& range, std::vector<pv::bid>&& bids) {
    auto it = std::find(assigned.begin(), assigned.end(), fdx);
    if(it == assigned.end()) {
        assigned.push_back(RangesForID{ fdx, { RangesForID::AutomaticRange{range, std::move(bids)} } });
        
    } else {
        it->ranges.push_back(RangesForID::AutomaticRange{ range, std::move(bids) });
    }
}
    
    std::map<Idx_t, pv::bid> Tracker::automatically_assigned(Frame_t frame) {
        //LockGuard guard;
        std::map<Idx_t, pv::bid> blob_for_fish;
        
        for(auto && [fdx, bff] : _automatically_assigned_ranges) {
            blob_for_fish[fdx] = -1;
            
            for(auto & assign : bff) {
                if(assign.range.contains(frame)) {
                    assert(frame >= assign.range.start && assign.range.end >= frame);
                    blob_for_fish[fdx] = assign.bids.at(sign_cast<size_t>((frame - assign.range.start).get()));
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
                    print("thread ",name," held the lock for ",str.c_str()," with purpose ",_purpose.c_str());
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
                
            } else if(timer.elapsed() > 60 && print_timer.elapsed() > 120) {
                auto name = _last_thread;
                auto myname = get_thread_name();
                FormatWarning("(",myname.c_str(),") Possible dead-lock with ",name," (",_last_purpose,") thread holding the lock for ",dec<2>(_thread_holding_lock_timer.elapsed()),"s (waiting for ",timer.elapsed(),"s, current purpose is ",_purpose,")");
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
        std::shared_lock guard(_properties_mutex);
        auto ptr = _properties_cache.properties(frameIndex);
        if(ptr)
            return ptr;
    }
    
    auto &frames = instance()->frames();
    auto it = properties_iterator(frameIndex);
    return it != frames.end() ? (*it).get() : nullptr;
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

    std::string Tracker::thread_name_holding() {
        return _last_thread;
    }
        
    void Tracker::print_memory() {
        LockGuard guard("print_memory");
        mem::TrackerMemoryStats stats;
        stats.print();
    }

    void Tracker::delete_automatic_assignments(Idx_t fish_id, const FrameRange& frame_range) {
        LockGuard guard("delete_automatic_assignments");
        
        auto it = std::find(_automatically_assigned_ranges.begin(), _automatically_assigned_ranges.end(), fish_id);
        if(it == _automatically_assigned_ranges.end()) {
            FormatExcept("Cannot find fish ",fish_id," in automatic assignments");
            return;
        }
        
        std::set<Range<Frame_t>> ranges_to_remove;
        for(auto && [range, blob_ids] : it->ranges) {
            if(frame_range.overlaps(range)) {
                ranges_to_remove.insert(range);
            }
        }
        for(auto range : ranges_to_remove) {
            std::erase_if(it->ranges, [&](auto &assign){
                return assign.range == range;
            });
        }
    }

    bool callback_registered = false;
    
    Recognition* Tracker::recognition() {
        if(!_instance)
            throw U_EXCEPTION("There is no valid instance if Tracker available (Tracker::recognition).");
        
        return _instance->_recognition;
    }

void Tracker::analysis_state(AnalysisState pause) {
    if(!instance())
        throw U_EXCEPTION("No tracker instance can be used to pause.");
    static std::future<void> current_state;
    if(current_state.valid())
        current_state.get();
    
    current_state = std::async(std::launch::async, [](bool pause){
        SETTING(analysis_paused) = pause;
    }, pause == AnalysisState::PAUSED);
}

    Tracker::Tracker()
          : _thread_pool(max(1u, cmn::hardware_concurrency()), nullptr, "Tracker::thread_pool"),
            recognition_pool(max(1u, cmn::hardware_concurrency()), nullptr, "RecognitionPool"),
            _midline_errors_frame(0), _overall_midline_errors(0),
            _max_individuals(0),
            _background(NULL), _recognition(NULL),
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
            print("Initialized with ", _thread_pool.num_threads()," threads.");
        
        Settings::set_callback(Settings::outline_resample, [](auto&, auto&value){
            static_assert(std::is_same<Settings::outline_resample_t, float>::value, "outline_resample assumed to be float.");
            auto v = value.template value<float>();
            if(v <= 0) {
                print("outline_resample defaulting to 1.0 instead of ",v);
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
                
                if(changed) {
                    GlobalSettings::get(key) = tmp;
                }
            };
            
                update();
        };
        Settings::set_callback(Settings::track_ignore, track_list_update);
        Settings::set_callback(Settings::track_include, track_list_update);
        Settings::set_callback(Settings::frame_rate, [](auto&, auto&){
            std::unique_lock guard(_properties_mutex);
            _properties_cache.clear(); //! TODO: need to refill as well
        });
        Settings::set_callback(Settings::posture_direction_smoothing, [](auto&key, auto&value) {
            static_assert(std::is_same<Settings::posture_direction_smoothing_t, uint16_t>::value, "posture_direction_smoothing assumed to be uint16_t.");
            size_t v = value.template value<uint16_t>();
            
            if(v != FAST_SETTINGS(posture_direction_smoothing))
            {
                auto worker = [key](){
                    LockGuard guard("Updating midlines in changed_setting("+key+")");
                    
                    for (auto && [id, fish] : Tracker::individuals()) {
                        Tracker::instance()->_thread_pool.enqueue([](long_t id, Individual *fish){
                            print("\t", id);
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
                
                /*if(GUI::instance()) {
                    GUI::work().add_queue("updating midlines / head positions...", worker);
                } else*/
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
            print("Waiting for recognition...");
        recognition_pool.force_stop();
        if(!SETTING(quiet))
            print("Done waiting.");
        
        _instance = NULL;
        
        auto individuals = _individuals;
        for (auto& fish_ptr : individuals)
            delete fish_ptr.second;
        
        emergency_finish();
    }
    
    void Tracker::emergency_finish() {
        std::lock_guard<std::mutex> guard(log_mutex);
        if(history_log != nullptr && history_log->is_open()) {
            print("Closing history log.");
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

    Frame_t Tracker::update_with_manual_matches(const Settings::manual_matches_t& manual_matches) {
        LockGuard guard("update_with_manual_matches");
        
        static std::atomic_bool first_run(true);
        static Settings::manual_matches_t compare = manual_matches;
        if(first_run) {
            first_run = false;
            //auto str = Meta::toStr(compare);
            //SETTING(manual_matches) = manual_matches;
            return {};
            
        } else {
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
            
            if(first_change.valid() && first_change <= Tracker::end_frame()) {
                Tracker::analysis_state(Tracker::AnalysisState::UNPAUSED);
            }
            
            //SETTING(manual_matches) = next;
            //FAST_SETTINGS(manual_matches) = next;
            //auto str = Meta::toStr(FAST_SETTINGS(manual_matches));
            compare = next;
            
            return first_change;
        }
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
        static Timing timing("Tracker::add(PPFrame)", 10);
        TakeTiming take(timing);
        
        Timer overall_timer;
        LockGuard guard("Tracker::add(PPFrame)");
        
        assert(frame.index().valid());
        
        if (contains_sorted(_added_frames, frame.index())) {
            print("Frame ",frame.index()," already in tracker.");
            return;
        }
        
        if(frame.frame().timestamp() > uint64_t(INT64_MAX)) {
            print("frame timestamp is bigger than INT64_MAX! (",time," time)");
        }
        
        auto props = properties(frame.index() - 1_f);
        if(props && frame.frame().timestamp() < props->org_timestamp.get()) {
            FormatError("Cannot add frame with timestamp smaller than previous timestamp. Frames have to be in order. Skipping.");
            return;
        }
        
        if(start_frame().valid() && frame.index() < end_frame() + 1_f)
            throw UtilsException("Cannot add intermediate frames out of order.");
        
        history_split(frame, _active_individuals, history_log != nullptr && history_log->is_open() ? history_log.get() : nullptr, &_thread_pool);
        add(frame.index(), frame);
        
        //! Update recognition if enabled and end of video reached
        //if(Recognition::recognition_enabled()) 
        {
            const auto video_length = Tracker::analysis_range().end;
            if (frame.index() >= video_length) {
                if(Recognition::recognition_enabled())
                    Recognition::notify();
            }
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
                DebugCallback("Opening history_log at ", path, "...");
                //!TODO: CHECK IF THIS WORKS
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
        double time = double(frame.frame().timestamp()) / double(1000*1000);
        
        //! Free old memory
        frame.clear();
        
        frame.time = time;
        frame.timestamp = frame.frame().timestamp();
        frame.set_index(Frame_t(frame.frame().index()));
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
                    print("Array of numbers ",rect," is not a polygon (or rectangle).");
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

        const auto track_include = FAST_SETTINGS(track_include);
        const auto track_ignore = FAST_SETTINGS(track_ignore);
        
        std::vector<pv::BlobPtr> ptrs;
        auto only_allowed = FAST_SETTINGS(track_only_categories);
        
        auto check_blob = [&track_ignore, &track_include, &result, &cm_sqr](const pv::BlobPtr& b){
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
            
            auto &b = *it;
            
            if(!check_blob(b))
                continue;
            
            float recount = b->recount(-1);
            
            // TODO: magic numbers
            //! If the size is appropriately big, try to split the blob using the minimum of threshold and
            //  posture_threshold. Using the minimum ensures that the thresholds dont depend on each other
            //  as the threshold used here will reduce the number of available pixels for posture analysis
            //  or tracking respectively (pixels below used threshold will be removed).
            if(result->fish_size.close_to_minimum_of_one(recount, 0.5)) {
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
            print("Frame ", result->frame_index,": ",big_blobs.size()," big blobs");
        }*/
        
        for(auto &blob : filtered)
            blob->calculate_moments();
        
        if (result->frame_index == Tracker::start_frame() || !Tracker::start_frame().valid())
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
        const robin_hood::unordered_map<pv::Blob*, split_expectation> &expect,
        bool discard_small,
        std::ostream* out,
        GenericThreadPool* pool)
    {
        UNUSED(out);
        
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
                if(!expect.empty() && expect.count(b.get()))
                    ex = expect.at(b.get());
                
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
                    std::set<std::tuple<float, pv::bid, pv::BlobPtr>, std::greater<>> found;
                    for(auto &ptr : ret) {
                        float recount = ptr->recount(0, *_background);
                        found.insert({recount, ptr->blob_id(), ptr});
                    }
                    
                    size_t counter = 0;
                    for(auto & [r, id, ptr] : found) {
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

    void Tracker::history_split(PPFrame &frame, const Tracker::set_of_individuals_t &active_individuals, std::ostream* out, GenericThreadPool* pool) {
        static Timing timing("history_split", 20);
        TakeTiming take(timing);

        float tdelta;
        {
            Tracker::LockGuard guard("history_split#1");
            auto props = properties(frame.index() - 1_f);
            tdelta = props ? (frame.time - props->time) : 0;
        }
        const float max_d = FAST_SETTINGS(track_max_speed) * tdelta / FAST_SETTINGS(cm_per_pixel) * 0.5;

        Log(out, "");
        Log(out, "------------------------");
        Log(out, "HISTORY MATCHING for frame ", frame.index(), ": (", max_d, ")");
        
        if(out) {
            Log(out, "frame ", frame.index()," active: ", active_individuals);
        }
        
        using namespace Match;
        robin_hood::unordered_map<long_t, std::set<pv::bid>> fish_mappings;
        robin_hood::unordered_map<pv::bid, std::set<Idx_t>> blob_mappings;
        robin_hood::unordered_map<Idx_t, ska::bytell_hash_map<pv::bid, Match::prob_t>> paired;

        const auto frame_limit = FAST_SETTINGS(frame_rate) * FAST_SETTINGS(track_max_reassign_time);
        const auto N = active_individuals.size();

        {
            const size_t num_threads = pool ? min(hardware_concurrency(), N / 200u) : 0;
            auto space_limit = Individual::weird_distance() * 0.5;
            std::condition_variable variable;

            size_t count = 0;
            std::mutex mutex;
            CacheHints hints;
            if(frame.index().valid() && frame.index() > Tracker::start_frame())
                hints.push(frame.index()-1_f, properties(frame.index()-1_f));
            hints.push(frame.index(), properties(frame.index()));

            auto fn = [&](const Tracker::set_of_individuals_t& active_individuals,
                          size_t start,
                          size_t N)
            {
                struct FishAssignments {
                    Idx_t fdx;
                    std::vector<int64_t> blobs;
                    std::vector<float> distances;
                };
                struct BlobAssignments {
                    UnorderedVectorSet<Idx_t> idxs;
                };

                std::vector<FishAssignments> fish_assignments(N);
                ska::bytell_hash_map<int64_t, BlobAssignments> blob_assignments;

                auto it = active_individuals.begin();
                std::advance(it, start);
                
                for(auto i = start; i < start + N; ++i, ++it) {
                    auto fish = *it;
                    auto &cache = frame.individual_cache()[i];
					
                    Vec2 last_pos(-1,-1);
                    Frame_t last_frame;
                    long_t last_L = -1;
                    float time_limit;

                    // IndividualCache is in the same position as the indexes here
                    //auto& obj = frame.cached_individuals.at(fish->identity().ID());
                    cache = fish->cache_for_frame(frame.index(), frame.time, &hints);
                    time_limit = cache.previous_frame.get() - frame_limit;
                        
                    size_t counter = 0;
                    auto sit = fish->iterator_for(cache.previous_frame);
                    if (sit != fish->frame_segments().end() && (*sit)->contains(cache.previous_frame)) {
                        for (; sit != fish->frame_segments().end() && min((*sit)->end(), cache.previous_frame).get() >= time_limit && counter < frame_limit; ++counter)
                        {
                            auto pos = fish->basic_stuff().at((*sit)->basic_stuff((*sit)->end()))->centroid.pos<Units::DEFAULT>();

                            if ((*sit)->length() > FAST_SETTINGS(frame_rate) * FAST_SETTINGS(track_max_reassign_time) * 0.25)
                            {
                                //! segment is long enough, we can stop. but only actually use it if its not too far away:
                                if (last_pos.x == -1 || euclidean_distance(pos, last_pos) < space_limit) {
                                    last_frame = min((*sit)->end(), cache.previous_frame);
                                    last_L = (last_frame - (*sit)->start()).get();
                                }
                                break;
                            }

                            last_pos = fish->basic_stuff().at((*sit)->basic_stuff((*sit)->start()))->centroid.pos<Units::DEFAULT>();

                            if (sit != fish->frame_segments().begin())
                                --sit;
                            else
                                break;
                        }
                    }
                    
                    if(last_frame.get() < time_limit) {
                        Log(out, "\tNot processing fish ", fish->identity()," because its last respected frame is ", last_frame,", best segment length is ", last_L," and were in frame ", frame.index(),".");
                        continue;
                    }
                    
                    auto set = frame.blob_grid().query(cache.estimated_px, max_d);
                    
                    if(!set.empty()) {
                        auto fdx = fish->identity().ID();
                        
                        //std::unique_lock guard(thread_mutex);
                        //auto &map = fish_mappings[fdx];
                        //auto &pair_map = paired[fdx];
                        auto& map = fish_assignments[i - start];
                        map.fdx = fdx;

                        for(auto && [d, bdx] : set) {
                            if(!frame.find_bdx(bdx))
                                continue;
                            
                            map.blobs.push_back(bdx);
                            map.distances.push_back(d);
                            blob_assignments[bdx].idxs.insert(fdx);
                        }
                    }
                    
                    Log(out, "\tFish ", fish->identity()," (", cache.estimated_px.x, ",", cache.estimated_px.y, ") proximity: ", set);
                }

                std::unique_lock lock(mutex);
                for (auto& [fdx, blobs, distances] : fish_assignments) {
                    fish_mappings[fdx].insert(blobs.begin(), blobs.end());
                    auto N = blobs.size();
                    for(size_t i=0; i<N; ++i)
                        paired[fdx][pv::bid(blobs[i])] = distances[i];
                }
                for (auto& [bdx, assign] : blob_assignments) {
                    blob_mappings[bdx].insert(assign.idxs.begin(), assign.idxs.end());
                }

                ++count;
                variable.notify_one();
            };
            
            //pool = nullptr;
            frame.individual_cache().clear();
            frame.individual_cache().resize(N);
            
            if(num_threads < 2 || !pool || N < num_threads) {
                Tracker::LockGuard guard("history_split#2");
                fn(active_individuals, 0, N);
                
            } else if(N) {
                size_t last = N % num_threads;
                size_t per_thread = (N - last) / num_threads;
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
        
        UnorderedVectorSet<pv::bid> already_walked;
        std::vector<pv::BlobPtr> big_blobs;
        robin_hood::unordered_map<pv::Blob*, split_expectation> expect;
        
        auto manual_splits = FAST_SETTINGS(manual_splits);
        auto manual_splits_frame =
            (manual_splits.empty() || manual_splits.count(frame.index()) == 0)
                ? decltype(manual_splits)::mapped_type()
                : manual_splits.at(frame.index());
        
        Log(out, "manual_splits = ", manual_splits);
        
        if(!manual_splits_frame.empty()) {
            for(auto bdx : manual_splits_frame) {
                if(!bdx.valid())
                    continue;
                
                auto it = blob_mappings.find(bdx);
                if(it == blob_mappings.end()) {
                    blob_mappings[bdx] = { Idx_t() };
                } else{
                    it->second.insert(Idx_t());
                }
                
                Log(out, "\t\tManually splitting ", (uint32_t)bdx);
                auto ptr = frame.erase_anywhere(bdx);
                if(ptr) {
                    big_blobs.push_back(ptr);
                    
                    expect[ptr.get()].number = 2;
                    expect[ptr.get()].allow_less_than = false;
                    
                    already_walked.insert(bdx);
                }
            }
            
        } else
            Log(out, "\t\tNo manual splits for frame ", frame.index());
        
        if(out) {
            Log(out, "fish_mappings ", fish_mappings);
            Log(out, "blob_mappings ", blob_mappings);
            Log(out, "Paired ", paired);
        }
        
        if(!FAST_SETTINGS(track_do_history_split)) {
            frame.finalize();
            return;
        }
        
        for(auto && [bdx, set] : blob_mappings) {
            if(already_walked.contains(bdx)) {
                Log(out, "\tblob ", bdx," already walked");
                continue;
            }
            Log(out, "\tblob ", bdx," has ", set.size()," fish mapped to it");
            
            if(set.size() <= 1)
                continue;
            Log(out, "\tFinding clique of this blob:");
            
            UnorderedVectorSet<Idx_t> clique;
            UnorderedVectorSet<pv::bid> others;
            std::queue<pv::bid> q;
            q.push(bdx);
            
            while(!q.empty()) {
                auto current = q.front();
                q.pop();
                
                for(auto fdx: blob_mappings.at(current)) {
                    // ignore manually forced splits
                    if(fdx < 0)
                        continue;
                    
                    for(auto &b : fish_mappings.at(fdx)) {
                        if(!others.contains(b)) {
                            q.push(b);
                            others.insert(b);
                            already_walked.insert(b);
                        }
                    }
                    
                    clique.insert(fdx);
                }
            }
            
            assert(bdx.valid());
            frame.clique_for_blob[bdx] = clique;
            frame.clique_second_order[bdx] = others;
            
            if(out) {
                Log(out, "\t\t", clique, " ", others);
            }
            
            if(clique.size() <= others.size())
                continue;
            
            using namespace Match;
            robin_hood::unordered_map<pv::bid, std::pair<Idx_t, Match::prob_t>> assign_blob; // blob: individual
            robin_hood::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> all_probs_per_fish;
            robin_hood::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> probs_per_fish;
            
            if(out) {
                Log(out, "\t\tMismatch between blobs and number of fish assigned to them.");
                if(clique.size() > others.size() + 1)
                    Log(out, "\t\tSizes: ", clique.size()," != ",others.size());
            }
            
            bool allow_less_than = false;
            /*for(auto fdx : clique) {
                if(_individuals.at(fdx)->recently_manually_matched(frame.index)) {
                    allow_less_than = true;
                    break;
                }
            }*/
            
            auto check_combinations = [&assign_blob, out](Idx_t c, decltype(probs_per_fish)::mapped_type& combinations, std::queue<Idx_t>& q) -> bool
            {
                if(!combinations.empty()) {
                    auto b = std::get<1>(*combinations.begin());
                    
                    if(assign_blob.count(b) == 0) {
                        // great! this blob has not been assigned at all (yet)
                        // so just assign it to this fish
                        assign_blob[b] = {c, std::get<0>(*combinations.begin())};
                        Log(out, "\t\t",b,"(",c,"): ", std::get<0>(*combinations.begin()));
                        return true;
                        
                    } else if(assign_blob[b].first != c) {
                        // this blob has been assigned to a different fish!
                        // check for validity (which one is closer)
                        if(assign_blob[b].second <= std::get<0>(*combinations.begin())) {
                            Log(out, "\t\tBlob ", b," is already assigned to individual ", assign_blob[b], " (", c,")...");
                        } else {
                            auto oid = assign_blob[b].first;
                            if(out) {
                                Log(out, "\t\tBlob ", b," is already assigned to ", assign_blob[b],", but fish ", c," is closer (need to check combinations of fish ", oid," again)");
                                Log(out, "\t\t", b,"(", c,"): ", std::get<0>(*combinations.begin()));
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
            std::queue<Idx_t> checks;
            for(auto c : clique) {
                decltype(probs_per_fish)::mapped_type combinations;
                for(auto && [bdx, d] : paired.at(c)) {
                    combinations.insert({Match::prob_t(d), bdx});
                }
                
                probs_per_fish[c] = combinations;
                all_probs_per_fish[c] = combinations;
                
                checks.push(c);
            }
            
            while(!checks.empty()) {
                auto c = checks.front();
                checks.pop();
                
                auto &combinations = all_probs_per_fish.at(c);
                if(!combinations.empty() && !check_combinations(c, combinations, checks))
                    checks.push(c);
            }
            
            UnorderedVectorSet<pv::bid> to_delete;
            size_t counter = 0;
            for(auto && [fdx, set] : all_probs_per_fish) {
                if(out) {
                    Log(out, "Combinations ", fdx,": ", set);
                }
                
                if(set.empty()) {
                    ++counter;
                    Log(out, "No more alternatives for ", fdx);
                    
                    if(!probs_per_fish.at(fdx).empty()) {
                        pv::bid max_id;
                        
                        if(out) {
                            for(auto && [d, bdx] : probs_per_fish.at(fdx)) {
                                Log(out, "\t", bdx,": ", d);
                            }
                        }
                        
                        max_id = std::get<1>(*probs_per_fish.at(fdx).begin());
                        if(max_id.valid()) {
                            frame.split_blobs.insert(max_id);
                            auto ptr = frame.erase_regular(max_id);
                            
                            if(ptr) {
                                Log(out, "Splitting blob ", max_id);
                                to_delete.insert(max_id);
                                
                                /*for(auto && [ind, blobs] : paired) {
                                    auto it = blobs.find(max_id);
                                    if(it != blobs.end()) {
                                        blobs.erase(it);
                                    }
                                }*/
                                
                                ++expect[ptr.get()].number;
                                big_blobs.push_back(ptr);
                            }
                            else if((ptr = frame.find_bdx(max_id))) {
                                if(expect.contains(ptr.get())) {
                                    Log(out, "Increasing expect number for blob ", max_id);
                                    ++expect[ptr.get()].number;
                                }
                                
                                Log(out, "Would split blob ", max_id,", but its part of additional.");
                            }
                            
                            if(allow_less_than)
                                expect[ptr.get()].allow_less_than = allow_less_than;
                        }
                    }
                }
            }
            
            distribute_vector([&](auto, auto start, auto end, auto){
                for(auto k = start; k != end; ++k) {
                    auto &blobs = k->second;
                    auto it = blobs.begin();
                    for(; it != blobs.end();) {
                        if(contains(to_delete, it->first))
                            it = blobs.erase(it);
                        else
                            ++it;
                    }
                }
            }, _thread_pool, paired.begin(), paired.end());
            /*for(auto & [ind, blobs] : paired) {
                std::remove_if(blobs.begin(), blobs.end(), [&to_delete](auto &v){
                    return contains(to_delete, v.first);
                });
                auto it = blobs.begin();
                for(; it != blobs.end();) {
                    if(contains(to_delete, it->first))
                        it = blobs.erase(it);
                    else
                        ++it;
                }
            }*/
            
            if(out) {
                auto str = Meta::toStr(expect);
                Log(out, "expect: ", expect);
                if(counter > 1) {
                    Log(out, "Lost ", counter," fish (", expect, ")");
                }
            }
        }
        
        for(auto && [blob, e] : expect)
            ++e.number;
        
        if(!manual_splits_frame.empty()) {
            for(auto &bdx : manual_splits_frame) {
                auto ptr = frame.find_bdx(bdx);
                if(ptr) {
                    expect[ptr.get()].allow_less_than = false;
                    expect[ptr.get()].number = 2;
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
            throw U_EXCEPTION("Cannot assign identity (",ID,") twice.");
        
        Individual *fish = new Individual();
        fish->identity().set_ID(ID);
        
        _individuals[fish->identity().ID()] = fish;
        active_individuals.insert(fish);
        
        if(ID >= Identity::running_id()) {
            Identity::set_running_id(Idx_t(ID + 1));
        }
        
        return fish;
    }

const FrameProperties* Tracker::add_next_frame(const FrameProperties & props) {
    auto &frames = instance()->frames();
    auto capacity = frames.capacity();
    instance()->_added_frames.emplace_back(std::make_unique<FrameProperties>(props));
    
    if(frames.capacity() != capacity) {
        std::unique_lock guard(_properties_mutex);
        _properties_cache.clear();
        
        auto it = frames.rbegin();
        while(it != frames.rend() && !_properties_cache.full())
        {
            _properties_cache.push((*it)->frame, (*it).get());
            ++it;
        }
        assert((frames.empty() && !end_frame().valid()) || (end_frame().valid() && (*frames.rbegin())->frame == end_frame()));
        
    } else {
        std::unique_lock guard(_properties_mutex);
        _properties_cache.push(props.frame, frames.back().get());
    }
    
    return frames.back().get();
}

void Tracker::clear_properties() {
    std::unique_lock guard(_properties_mutex);
    _properties_cache.clear();
}

Match::PairedProbabilities Tracker::calculate_paired_probabilities
 (
    const PPFrame& frame,
    const Tracker::set_of_individuals_t& active_individuals,
    const ska::bytell_hash_map<Individual*, bool>& fish_assigned,
    const ska::bytell_hash_map<pv::Blob*, bool>& blob_assigned,
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
        std::vector<int> blob_labels(frame.blobs().size());
        std::vector<size_t> unassigned_blobs;
        unassigned_blobs.reserve(frame.blobs().size());
        //std::vector<std::tuple<const pv::BlobPtr*, int>> unassigned_blobs;
        //unassigned_blobs.reserve(frame.blobs().size());
        
        const bool enable_labels = FAST_SETTINGS(track_consistent_categories) || !FAST_SETTINGS(track_only_categories).empty();
        if(enable_labels) {
            for(size_t i=0; i<frame.blobs().size(); ++i) {//auto &p : frame.blobs) {
                if(!blob_assigned.at(frame.blobs()[i].get())) {
                    auto bdx = frame.blobs()[i]->blob_id();
                    auto label = Categorize::DataStore::ranged_label(Frame_t(frameIndex), bdx);
                    blob_labels[i] = label ? label->id : -1;
                    unassigned_blobs.push_back(i);
                } else {
                    blob_labels[i] = -1;
                }
            }
        } else {
            for(size_t i=0; i<frame.blobs().size(); ++i) {
                if(!blob_assigned.at(frame.blobs()[i].get()))
                    unassigned_blobs.push_back(i);
            }
        }
        
        const auto N_blobs = unassigned_blobs.size();
        const auto N_fish  = unassigned_individuals.size();
        const auto matching_probability_threshold = FAST_SETTINGS(matching_probability_threshold);
        
        auto work = [&](auto, auto start, auto end, auto N){
            size_t pid = 0;
            std::vector< Match::pairing_map_t<Match::Blob_t, Match::prob_t> > _probs(N);
            
            for (auto it = start; it != end; ++it, ++pid) {
                auto fish = *it;
                auto cache = frame.cached(fish->identity().ID());
                auto &probs = _probs[pid];
                
                for (size_t i = 0; i < N_blobs; ++i) {
                    auto &bdx = unassigned_blobs[i];
                    auto &blob = frame.blobs()[bdx];
                    auto p = fish->probability(blob_labels[bdx], *cache, frameIndex, blob);
                    if (p > matching_probability_threshold)
                        probs[&blob] = p;
                }
            }
            
            pid = 0;
            std::lock_guard<std::mutex> guard(paired_mutex);
            for (auto it = start; it != end; ++it, ++pid)
                paired_blobs.add(*it, std::move(_probs[pid]));
            
        };
        
#if defined(TREX_THREADING_STATS)
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
            distribute_vector(work, *pool, unassigned_individuals.begin(), unassigned_individuals.end());
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
                auto path = pv::DataLocation::parse("output", (std::string)SETTING(filename).value<file::Path>().filename()+"_threading_stats.npz").str();
                npz_save(path, "values", values.data(), std::vector<size_t>{bins.size(), 3});
                print("Saved threading stats at ", path,".");
            } catch(...) {
                FormatWarning("Error saving threading stats.");
            }
        }
#else
        if(pool && N_fish > 100)
            distribute_vector(work, *pool, unassigned_individuals.begin(), unassigned_individuals.end());
        else
            work(0, unassigned_individuals.begin(), unassigned_individuals.end(), N_fish);
#endif
    }
    
    return paired_blobs;
}

    void Tracker::add(Frame_t frameIndex, PPFrame& frame) {
        static const unsigned concurrentThreadsSupported = cmn::hardware_concurrency();
        double time = double(frame.frame().timestamp()) / double(1000*1000);
        
        if (!start_frame().valid() || start_frame() > frameIndex) {
            _startFrame = frameIndex;
        }
        
        if (!end_frame().valid() || end_frame() < frameIndex) {
            if(end_frame().valid() && end_frame() < start_frame())
              FormatError("end frame is ", end_frame()," < ",start_frame());
            _endFrame = frameIndex;
        }
        
        auto props = add_next_frame(FrameProperties(frame.index(), time, frame.frame().timestamp()));
        const FrameProperties* prev_props = nullptr;
        {
            auto it = --_added_frames.end();
            if(it != _added_frames.begin()) {
                --it;
                if((*it)->frame == frame.index() - 1_f)
                    prev_props = (*it).get();
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
        std::queue<std::tuple<Individual*, Individual::BasicStuff*>> need_postures;
        
        ska::bytell_hash_map<pv::Blob*, bool> blob_assigned;
        ska::bytell_hash_map<Individual*, bool> fish_assigned;
        
        const uint32_t number_fish = (uint32_t)FAST_SETTINGS(track_max_individuals);
        const BlobSizeRange minmax = FAST_SETTINGS(blob_size_ranges);
        
        size_t assigned_count = 0;
        
        std::vector<tags::blob_pixel> tagged_fish, noise;
        ska::bytell_hash_map<pv::bid, Individual*> blob_fish_map;
//#define TREX_DEBUG_MATCHING
#ifdef TREX_DEBUG_MATCHING
        std::vector<std::pair<Individual*, Match::Blob_t>> pairs;
#endif
        
        //auto blob_to_pixels = filter_blobs(frame);
        auto assign_blob_individual = [props, &tagged_fish, &blob_fish_map, &fish_assigned, &blob_assigned, &assigned_count, &do_posture, &need_postures, save_tags
#ifdef TREX_DEBUG_MATCHING
                                        ,&pairs
#endif
        ]
            (Frame_t frameIndex, const PPFrame& frame, Individual* fish, const pv::BlobPtr& blob, default_config::matching_mode_t::Class match_mode)
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
                throw U_EXCEPTION("Cannot find blob in frame.");
#endif*/
#ifndef NDEBUG
            if(!contains(frame.blobs(), blob)
               && !contains(frame.noise(), blob))
            {
                FormatExcept("Cannot find blob ", blob->blob_id()," in frame ", frameIndex,".");
            }
#endif
            
#ifdef TREX_DEBUG_MATCHING
            for(auto &[i, b] : pairs) {
                if(i == fish) {
                    if(b != &blob) {
                        FormatWarning("Frame ",frameIndex,": Assigning individual ",i->identity().ID()," to ",blob ? blob->blob_id() : 0," instead of ", b ? (*b)->blob_id() : 0);
                    }
                    break;
                }
            }
#endif
            
            //auto &pixels = *blob_to_pixels.at(blob);
            assert(blob->properties_ready());
            if(!blob->moments().ready) {
                blob->calculate_moments();
            }
            auto index = fish->add(props, frameIndex, frame, blob, -1, match_mode);
            if(index == -1) {
                FormatExcept("Was not able to assign individual ", fish->identity().ID()," with blob ", blob->blob_id(),"");
                return;
            }
            
            auto &basic = fish->basic_stuff()[size_t(index)];
            fish_assigned[fish] = true;
            blob_assigned[blob.get()] = true;
            
            if(save_tags) {
                if(!blob->split()){
                    blob_fish_map[blob->blob_id()] = fish;
                    if(blob->parent_id().valid())
                        blob_fish_map[blob->parent_id()] = fish;
                    
                    //pv::BlobPtr copy = std::make_shared<pv::Blob>((Blob*)blob.get(), std::make_shared<std::vector<uchar>>(*blob->pixels()));
                    tagged_fish.push_back(std::make_shared<pv::Blob>(*blob->lines(), *blob->pixels()));
                }
            }
            
            if (do_posture)
                need_postures.push({fish, basic.get()});
            else {
                basic->pixels = nullptr;
            }
            //else //if(!Recognition::recognition_enabled())
            //    basic->pixels = nullptr;
            
            assigned_count++;
        };
        
        if(save_tags) {
            for(auto &blob : frame.noise()) {
                if(blob->recount(-1) <= minmax.max_range().start) {
                    pv::BlobPtr copy = std::make_shared<pv::Blob>(*blob);
                    noise.emplace_back(std::move(copy));
                }
            }
        }
        
        //blobs = frame.blobs;
        for(auto &blob: frame.blobs())
            blob_assigned[blob.get()] = false;
        
        // collect all the currently active individuals
        Tracker::set_of_individuals_t active_individuals;
        
        //! TODO: Can probably reuse frame.blob_grid here, but need to add noise() as well
        static grid::ProximityGrid blob_grid(_average->bounds().size());
        blob_grid.clear();
        
        const auto manual_identities = FAST_SETTINGS(manual_identities);
        if(!number_fish && !manual_identities.empty()) {
            SETTING(manual_identities) = Settings::manual_identities_t();
        }
        
        for(auto &b : frame.blobs()) {
            //id_to_blob[b->blob_id()] = b;
            blob_grid.insert(b->bounds().x + b->bounds().width * 0.5f, b->bounds().y + b->bounds().height * 0.5f, (uint32_t)b->blob_id());
        }
        for(auto &b : frame.noise()) {
            //id_to_blob[b->blob_id()] = b;
            blob_grid.insert(b->bounds().x + b->bounds().width * 0.5f, b->bounds().y + b->bounds().height * 0.5f, (uint32_t)b->blob_id());
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
        ska::bytell_hash_map<pv::bid, std::set<Idx_t>> cannot_find;
        ska::bytell_hash_map<pv::bid, std::set<Idx_t>> double_find;
        ska::bytell_hash_map<pv::bid, Idx_t> actually_assign;
        
        for(auto && [fdx, bdx] : current_fixed_matches) {
            auto it = _individuals.find(fdx);
            if(it != _individuals.end()) { //&& size_t(fm.second) < blobs.size()) {
                auto fish = it->second;
                
                if(!bdx.valid()) {
                    // dont assign this fish! (bdx == -1)
                    continue;
                }
                
                auto blob = frame.find_bdx(bdx);
                if(!blob) {
                    cannot_find[bdx].insert(fdx);
                    continue;
                }
                
                if(actually_assign.count(bdx) > 0) {
                    FormatError("(fixed matches) Trying to assign blob ",(uint32_t)bdx," twice in frame ",frameIndex," (fish ",fdx," and ",actually_assign.at(bdx),").");
                    double_find[bdx].insert(fdx);
                    
                } else if(blob_assigned[blob.get()]) {
                    FormatError("(fixed matches, blob_assigned) Trying to assign blob ", bdx," twice in frame ", frameIndex," (fish ",fdx,").");
                    // TODO: remove assignment from the other fish as well and add it to cannot_find
                    double_find[bdx].insert(fdx);
                    
                    /*for(auto fish : active_individuals) {
                        auto blob = fish->blob(frameIndex);
                        if(blob && blob->blob_id() == fm.second) {
                            double_find[fm.second].insert(fish->identity().ID());
                        }
                    }*/
                    
                } else if(fish_assigned[fish]) {
                   FormatError("Trying to assign fish ", fish->identity().ID()," twice in frame ",frameIndex,".");
                } else {
                    actually_assign[(uint32_t)blob->blob_id()] = fdx;
                    //active_individuals.push_back(fish);
                    //fish->add_manual_match(frameIndex);
                    //assign_blob_individual(frameIndex, frame, fish, blob);
                }
                
            } else {
                if(frameIndex != start_frame())
                    FormatWarning("Individual number ", fdx," out of range in frame ",frameIndex,". Creating new one.");
                
                auto blob = frame.find_bdx(bdx);
                if(!blob) {
                    //FormatWarning("Cannot find blob ", it->second," in frame ",frameIndex,". Fallback to normal assignment behavior.");
                    cannot_find[bdx].insert(fdx);
                    continue;
                }
                
                if(actually_assign.count((uint32_t)bdx) > 0) {
                    FormatError("(fixed matches) Trying to assign blob ",bdx," twice in frame ",frameIndex," (fish ",fdx," and ",actually_assign.at(bdx),").");
                    double_find[bdx].insert(fdx);
                } else
                    actually_assign[bdx] = fdx;
                
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
            
            std::map<pv::bid, std::vector<std::tuple<Idx_t, Vec2, pv::bid>>> assign_blobs;
            const auto max_speed_px = FAST_SETTINGS(track_max_speed) / FAST_SETTINGS(cm_per_pixel);
            
            for(auto && [bdx, fdxs] : cannot_find) {
                assert(bdx >= 0);
                auto pos = bdx.calc_position();
                auto list = blob_grid.query(pos, max_speed_px);
                //auto str = Meta::toStr(list);
                
                if(!list.empty()) {
                    // blob ids will not be < 0, as they have been inserted into the
                    // grid before directly from the file. so we can assume (uint32_t)
                    for(auto fdx: fdxs)
                        assign_blobs[pv::bid(std::get<1>(*list.begin()))].push_back({fdx, pos, bdx});
                }
            }
            
            //auto str = prettify_array(Meta::toStr(assign_blobs));
            
            robin_hood::unordered_map<Idx_t, pv::bid> actual_assignments;
            
            for(auto && [bdx, clique] : assign_blobs) {
                //if(clique.size() > 1)
                {
                    // have to split blob...
                    auto &blob = frame.bdx_to_ptr(bdx);
                    
                    //std::vector<pv::BlobPtr> additional;
                    robin_hood::unordered_map<pv::Blob*, split_expectation> expect;
                    expect[blob.get()] = split_expectation(clique.size() == 1 ? 2 : clique.size(), false);
                    
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
                        for(auto & [fdx, pos, original_bdx] : clique) {
                            for(auto &b : big_filtered) {
                                if(b->blob_id() == original_bdx) {
#ifndef NDEBUG
                                    print("frame ",frame.index(),": Found perfect match for individual ",fdx,", bdx ",b->blob_id()," after splitting ",b->parent_id());
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
                            print("frame ", frame.index(),": All missing manual matches perfectly matched.");
#endif
                        } else {
                            FormatError("frame ",frame.index(),": Missing some matches (",found_perfect,"/",clique.size(),") for blob ",bdx," (identities ", clique,").");
                        }
                    }
                    
                } //else
                    //actual_assignments[std::get<0>(*clique.begin())] = bdx;
            }
            
            if(!actual_assignments.empty()) {
                auto str = prettify_array(Meta::toStr(actual_assignments));
                print("frame ", frame.index(),": actually assigning:\n",str.c_str());
            }
            
            std::set<FOI::fdx_t> identities;
            
            for(auto && [fdx, bdx] : actual_assignments) {
                auto &blob = frame.bdx_to_ptr(bdx);
                
                Individual *fish = nullptr;
                auto it = _individuals.find(fdx);
                
                // individual doesnt exist yet. create it
                if(it == _individuals.end()) {
                    throw U_EXCEPTION("Should have created it already."); //fish = create_individual(fdx, blob, active_individuals);
                } else
                    fish = it->second;
                
                if(blob_assigned[blob.get()]) {
                    print("Trying to assign blob ",bdx," twice.");
                } else if(fish_assigned[fish]) {
                    print("Trying to assign fish ",fdx," twice.");
                } else {
                    fish->add_manual_match(frameIndex);
                    assign_blob_individual(frameIndex, frame, fish, blob, default_config::matching_mode_t::benchmark);
                    
                    frame.erase_anywhere(blob);
                    active_individuals.insert(fish);
                    
                    identities.insert(FOI::fdx_t(fdx));
                }
            }
            
            FOI::add(FOI(frameIndex, identities, "failed matches"));
        }
        
        if(frameIndex == start_frame() && !manual_identities.empty()) {
            // create correct identities
            //assert(_individuals.empty());
            
            Idx_t max_id(Identity::running_id());
            
            for (auto m : manual_identities) {
                if(_individuals.find(m) == _individuals.end()) {
                    Individual *fish = new Individual(Idx_t(m));
                    //fish->identity().set_ID(m);
                    assert(fish->identity().ID() == m);
                    max_id = Idx_t(max((uint32_t)max_id, (uint32_t)m));
                    
                    _individuals[m] = fish;
                    //active_individuals.push_back(fish);
                }
            }
            
            if(max_id.valid()) {
                Identity::set_running_id(Idx_t(max_id + 1));
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
                FormatError("frame ",frameIndex,": Automatic assignment cannot be executed with fdx ",fdx,"(",fish ? (fish_assigned[fish] ? "assigned" : "unassigned") : "no fish",") and bdx ",bdx,"(",blob ? (blob_assigned[blob.get()] ? "assigned" : "unassigned") : "no blob",")");
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
        
        if(!manual_identities.empty() && manual_identities.size() < (Match::index_t)paired_blobs.n_rows()) {
            using namespace Match;
            
            for (auto &[r, idx] : paired_blobs.row_indexes()) {
                if(r->identity().manual()) {
                    // this is an important fish, check
                    //auto idx = paired_blobs.index(r);
                    
                    if(paired_blobs.degree(idx) == 1) {
                        auto edges = paired_blobs.edges_for_row(idx);
                        
                        // only one possibility!
                        auto &blob = edges.front();
                        Individual *other = NULL;
                        size_t count = 0;
                        
                        for (auto &[f, idx] : paired_blobs.row_indexes()) {
                            if(f == r)
                                continue;
                            
                            auto e = paired_blobs.edges_for_row(idx);
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
                        if(paired_blobs.has(other))
                            paired_blobs.erase(other);
                    }
                }
            }
        }
        
        
        using namespace default_config;
        const Frame_t approximation_delay_time = Frame_t(max(1, FAST_SETTINGS(frame_rate) * 0.25));
        bool frame_uses_approximate = (_approximative_enabled_in_frame.valid() && frameIndex - _approximative_enabled_in_frame < approximation_delay_time);
        
        auto match_mode = frame_uses_approximate
                ? default_config::matching_mode_t::hungarian
                : FAST_SETTINGS(match_mode);
#ifdef TREX_DEBUG_MATCHING
        {
            Match::PairingGraph graph(frameIndex, paired_blobs);
            
            try {
                auto &optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
                pairs = optimal.pairings;
                
            } catch(...) {
                FormatExcept("Failed to generate optimal solution (frame ", frameIndex,").");
            }
        }
#endif
        
        if(match_mode == default_config::matching_mode_t::automatic) {
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

            const auto p_threshold = FAST_SETTINGS(matching_probability_threshold);
            auto N_cliques = cliques.size();
            std::vector<bool> to_merge(N_cliques);
            
            for(auto &[row, idx] : paired_blobs.row_indexes()) {
                if(paired_blobs.degree(idx) > 1) {
                    auto edges = paired_blobs.edges_for_row(idx);
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
                match_mode = matching_mode_t::approximate;
            } else {
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
                            auto blob = paired_blobs.col(cdx);
#endif
                            auto &bedges = paired_blobs.edges_for_col(cdx);
                            
#ifndef NDEBUG
                            if(!frame.find_bdx((*blob)->blob_id())) {
                                print("Frame ", frameIndex,": Cannot find blob ",(*blob)->blob_id()," in map.");
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
                            auto edges = paired_blobs.edges_for_row(i);
#ifdef TREX_DEBUG_MATCHING
                            print("\t\tExploring row ", i," (aka fish", paired_blobs.row(i)->identity().ID(),") with edges=",edges);
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
                std::condition_variable _variable;
                size_t executed{ 0 };

                auto work_clique = [&] (const IndexClique& clique, Frame_t frameIndex)
                {
                    constexpr bool do_lock = true;
                    using namespace Match;
                    Match::PairedProbabilities paired;

                    for (auto& [fish, fdi] : paired_blobs.row_indexes()) {
                        if (!clique.fids.contains(fdi))
                            continue;

                        auto assigned = fish_assigned.find(fish);
                        if (assigned != fish_assigned.end() && assigned->second)
                            continue;

                        auto edges = paired_blobs.edges_for_row(fdi);

                        Match::pairing_map_t<Match::Blob_t, prob_t> probs;
                        for (auto& e : edges) {
                            auto blob = paired_blobs.col(e.cdx);
                            if (!blob_assigned.count(blob->get()) || !blob_assigned.at(blob->get()))
                                probs[blob] = e.p;
                        }

                        if (!probs.empty())
                            paired.add(fish, probs);
                    }

                    PairingGraph graph(*props, frameIndex, paired);

                    try {
                        auto& optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);

                        if constexpr (do_lock) {
                            std::unique_lock g(thread_mutex);
                            for (auto& p : optimal.pairings) {
                                assign_blob_individual(frameIndex, frame, p.first, *p.second, matching_mode_t::hungarian);
                                active_individuals.insert(p.first);
                            }
                        }
                        else {
                            for (auto& p : optimal.pairings) {
                                assign_blob_individual(frameIndex, frame, p.first, *p.second, matching_mode_t::hungarian);
                                active_individuals.insert(p.first);
                            }
                        }

                    }
                    catch (...) {
                        FormatExcept("Failed to generate optimal solution (frame ", frameIndex,").");
                    }

                    if constexpr (do_lock) {
                        std::unique_lock g(thread_mutex);
                        _variable.notify_one();
                        ++executed;
                    } else
                        ++executed;
                };
                

                /*std::vector<IndexClique*> small;
                small.reserve(cliques.size());
                
                for(auto &clique : cliques) {
                    if (clique.fids.size() < 3)
                        work_clique(clique, frameIndex, false);
                    else
                        small.push_back(&clique);
                }*/

                for(auto &c : cliques)
                    _thread_pool.enqueue(work_clique, c, frameIndex);

                Clique translated;
                _cliques[frameIndex].clear();

                for (auto& clique : cliques) {
                    translated.bids.clear();
                    translated.fishs.clear();

                    for (auto bdi : clique.bids)
                        translated.bids.insert((*paired_blobs.col(bdi))->blob_id());
                    for (auto fdi : clique.fids)
                        translated.fishs.insert(paired_blobs.row(fdi)->identity().ID());

                    _cliques[frameIndex].emplace_back(std::move(translated));
                }

#ifdef TREX_DEBUG_MATCHING
                size_t index = 0;
                for(auto &clique : cliques) {
                    std::set<uint32_t> fishs, blobs;
                    for(auto fdi : clique.fishs)
                        fishs.insert(paired_blobs.row(fdi)->identity().ID());
                    for(auto bdi : clique.bids)
                        blobs.insert((*paired_blobs.col(bdi))->blob_id());
                    
                    auto str = Meta::toStr(fishs);
                    auto str1 = Meta::toStr(blobs);
                    print("Frame ",frameIndex,": Clique ",index,", Matching fishs ",str.c_str()," and blobs ",str1.c_str()," together.");
                    ++index;
                    
                    for(auto &cdx : clique.bids) {
                        print("\tBlob ", (*paired_blobs.col(cdx))->blob_id()," edges:");
                        for(auto &e : paired_blobs.edges_for_col(cdx)) {
                            print("\t\tFish", paired_blobs.row(e)->identity().raw_name().c_str());
                        }
                    }
                }
#endif
                {
                    std::unique_lock g(thread_mutex);
                    const auto N_cliques = cliques.size();
                    //if(frameIndex >= 6600_f)

                    while(executed < N_cliques)
                        _variable.wait(g);
                }
                
                Match::PairedProbabilities paired;
                for(auto &[fish, idx] : paired_blobs.row_indexes()) {
                    auto fit = fish_assigned.find(fish);
                    if(fit == fish_assigned.end() || !fit->second) {
                        auto edges = paired_blobs.edges_for_row(idx);
                        
                        Match::pairing_map_t<Match::Blob_t, Match::prob_t> probs;
                        for(auto &e : edges) {
                            auto blob = paired_blobs.col(e.cdx);
#ifndef NDEBUG
                            if(!frame.find_bdx((*blob)->blob_id())) {
                                print("Frame ", frameIndex,": Cannot find blob ",(*blob)->blob_id()," in map.");
                                continue;
                            }
#endif
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
        
        
        {
            // calculate optimal permutation of blob assignments
            static Timing perm_timing("PairingGraph", 30);
            TakeTiming take(perm_timing);
            
            using namespace Match;
            PairingGraph graph(*props, frameIndex, paired_blobs);
            
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
                    throw U_EXCEPTION("FU");
                
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
            size_t nedges = 0;
            size_t max_edges_per_fish = 0, max_edges_per_blob = 0;
            double mean_edges_per_blob = 0, mean_edges_per_fish = 0;
            size_t fish_with_one_edge = 0, fish_with_more_edges = 0;
            
            std::map<long_t, size_t> edges_per_blob;
            
            double average_probability = 0;
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
                    print("frame ",frameIndex,": ",optimal.improvements_made," of ",optimal.leafs_visited," / ",optimal.objects_looked_at," objects. ",average_improvements / samples," improvements on average, ",average_leafs / samples," leafs visited on average, ",average_objects / samples," objects on average (",mean_edges_per_fish," mean edges per fish and ",mean_edges_per_blob," mean edges per blob). On average we encounter ",average_bad_probabilities / samples," bad probabilities below 0.5 (currently ",bad_probs,").");
                    print("g fish_has_one_edge * mean_edges_per_fish = ", one_edge_probability," * ", mean_edges_per_fish," = ",one_edge_probability * (mean_edges_per_fish));
                    print("g fish_has_one_edge * mean_edges_per_blob = ", one_edge_probability," * ", mean_edges_per_blob," = ",one_edge_probability * (mean_edges_per_blob));
                    print("g blob_has_one_edge * mean_edges_per_fish = ", blob_one_edge," * ", mean_edges_per_fish," = ",blob_one_edge * mean_edges_per_fish);
                    print("g blob_has_one_edge * mean_edges_per_blob = ", blob_one_edge," * ", mean_edges_per_blob," = ",blob_one_edge * mean_edges_per_blob);
                    print("g mean_edges_per_fish / mean_edges_per_blob = ", mean_edges_per_fish / mean_edges_per_blob);
                    print("g one_to_one = ",one_to_one,", one_to_one * mean_edges_per_fish = ",one_to_one * mean_edges_per_fish," / blob: ",one_to_one * mean_edges_per_blob," /// ",average_probability,", ",average_probability * mean_edges_per_fish);
                    print("g --");
                    timer.reset();
                }
            };
            
            if(average_probability * mean_edges_per_fish <= 1) {
                FormatWarning("(", frameIndex,") Warning: ",average_probability * mean_edges_per_fish);
            }
    #endif
            
            try {
                auto &optimal = graph.get_optimal_pairing(false, match_mode);
                
                if(!frame_uses_approximate) {
                    std::lock_guard<std::mutex> guard(_statistics_mutex);
                    _statistics[frameIndex].match_number_blob = (Match::index_t)paired_blobs.n_cols();
                    _statistics[frameIndex].match_number_fish = (Match::index_t)paired_blobs.n_rows();
                    //_statistics[frameIndex].match_number_edges = nedges;
                    _statistics[frameIndex].match_stack_objects = optimal.objects_looked_at;
                    /*_statistics[frameIndex].match_max_edges_per_blob = max_edges_per_blob;
                    _statistics[frameIndex].match_max_edges_per_fish = max_edges_per_fish;
                    _statistics[frameIndex].match_mean_edges_per_blob = mean_edges_per_blob;
                    _statistics[frameIndex].match_mean_edges_per_fish = mean_edges_per_fish;*/
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
                                FormatWarning("Frame ",frameIndex,": Assigning individual ",i->identity().ID()," to ",p.second ? (*p.second)->blob_id() : 0," instead of ", b ? (*b)->blob_id() : 0);
                            }
                            break;
                        }
                    }
#endif
                    
                    assign_blob_individual(frameIndex, frame, p.first, *p.second, match_mode);
                    active_individuals.insert(p.first);
                }
                
            } catch (const UtilsException&) {
    #if !defined(NDEBUG) && defined(PAIRING_PRINT_STATS)
                if(graph.optimal_pairing())
                    print_statistics(*graph.optimal_pairing(), true);
                else
                    FormatWarning("No optimal pairing object.");
                
                graph.print_summary();
    #endif
                            
#if defined(PAIRING_PRINT_STATS)
                // matching did not work
                FormatWarning("Falling back to approximative matching in frame ",frameIndex,". (p=",one_edge_probability,",",mean_edges_per_fish,", ",one_edge_probability * (mean_edges_per_fish),", ",one_edge_probability * mean_edges_per_blob,")");
                FormatWarning("frame ",frameIndex,": (",mean_edges_per_fish," mean edges per fish and ",mean_edges_per_blob," mean edges per blob).");
                
                print("gw Probabilities: fish_has_one_edge=", one_edge_probability," blob_has_one_edge=",blob_one_edge);
                print("gw fish_has_one_edge * mean_edges_per_fish = ", one_edge_probability," * ", mean_edges_per_fish," = ",one_edge_probability * (mean_edges_per_fish));
                print("gw fish_has_one_edge * mean_edges_per_blob = ", one_edge_probability," * ", mean_edges_per_blob," = ",one_edge_probability * (mean_edges_per_blob));
                print("gw blob_has_one_edge * mean_edges_per_fish = ", blob_one_edge," * ", mean_edges_per_fish," = ",blob_one_edge * mean_edges_per_fish);
                print("gw blob_has_one_edge * mean_edges_per_blob = ", blob_one_edge," * ", mean_edges_per_blob," = ",blob_one_edge * mean_edges_per_blob);
                print("gw one_to_one = ",one_to_one,", one_to_one * mean_edges_per_fish = ",one_to_one * mean_edges_per_fish," / blob: ",one_to_one * mean_edges_per_blob," /// ",average_probability,", ",average_probability * mean_edges_per_fish);
                print("gw mean_edges_per_fish / mean_edges_per_blob = ", mean_edges_per_fish / mean_edges_per_blob);
                print("gw ---");
#endif
                
                auto &optimal = graph.get_optimal_pairing(false, default_config::matching_mode_t::hungarian);
                for (auto &p: optimal.pairings) {
                    assign_blob_individual(frameIndex, frame, p.first, *p.second, default_config::matching_mode_t::hungarian);
                    active_individuals.insert(p.first);
                }
                
                _approximative_enabled_in_frame = frameIndex;
                
                FOI::add(FOI(Range<Frame_t>(frameIndex, frameIndex + approximation_delay_time - 1_f), "apprx matching"));
            }
        }
        
        static Timing rest("rest", 30);
        TakeTiming take(rest);
        // see how many are missing
        /*std::vector<Individual*> unassigned_individuals;
        for(auto &p : fish_assigned) {
            if(!p.second) {
                unassigned_individuals.push_back(p.first);
            }
        }*/
        
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
            /*if(frameIndex > 1) {
                static std::random_device rng;
                static std::mt19937 urng(rng());
                std::shuffle(unassigned_blobs.begin(), unassigned_blobs.end(), urng);
            }*/
            
            for(auto fish :_active_individuals)
                if(active_individuals.find(fish) == active_individuals.end())
                    _inactive_individuals.insert(fish->identity().ID());
            
            for (auto &blob: unassigned_blobs) {
                // we measure the number of currently assigned fish based on whether a maximum number has been set. if there is a maximum, then we only look at the currently active individuals and extend that array with new individuals if necessary.
                const size_t number_individuals = number_fish ? _individuals.size() : active_individuals.size();
                if(number_fish && number_individuals >= number_fish) {
                    static bool warned = false;
                    if(!warned) {
                        FormatWarning("Running out of assignable fish (track_max_individuals ", active_individuals.size(),"/",number_fish,")");
                        warned = true;
                    }
                    break;
                }
                
                if(number_fish)
                    FormatWarning("Frame ",frameIndex,": Creating new individual (",Identity::running_id(),") for blob ",blob->blob_id(),".");
                
                Individual *fish = nullptr;
                if(!_inactive_individuals.empty()) {
                    fish = _individuals.at(*_inactive_individuals.begin());
                    _inactive_individuals.erase(_inactive_individuals.begin());
                } else {
                    fish = new Individual;
                    if(_individuals.find(fish->identity().ID()) != _individuals.end()) {
                        throw U_EXCEPTION("Cannot assign identity (",fish->identity().ID(),") twice.");
                        //assert(_individuals[fish->identity().ID()] != fish);
                        //mark_to_delete.insert(_individuals[fish->identity().ID()]);
                    }
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
                     
                                print("Assigning individual because its the most likely (fixed_count, ",p.first->identity().ID(),"-",p.second->blob_id()," in frame ",frameIndex,", p:",new_pairings.at(p.first).at(p.second),").");
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
                            
                            
                            assign_blob_individual(frameIndex, frame, fish, r.bdx(), default_config::matching_mode_t::benchmark);
                            active_individuals.insert(fish);
                            
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
                    auto basic = fish->empty() ? nullptr : fish->find_frame(frameIndex).get();
                    
                    if(basic && basic->frame == frameIndex) {
                        FormatWarning("Fish ", fdx," not in any of the arrays, but has frame ",frameIndex,".");
                    } else
                        FormatWarning("Fish ", fdx," is gone (",basic ? basic->frame : -1_f,")");
                } else if(lost_ids.find(fdx) != lost_ids.end()) {
                    lost_ids.erase(fdx);
                    FormatWarning("Fish ", fdx," found again in frame ",frameIndex,".");
                }
            }
        }
#endif
        
        if(save_tags) {
           _thread_pool.enqueue([&, bmf = blob_fish_map](){
                this->check_save_tags(frameIndex, bmf, tagged_fish, noise, tags_path);
            });
        }
        
        Timer posture_timer;
        
        {
            static Timing timing("Tracker::need_postures", 30);
            TakeTiming take(timing);
            
            if(do_posture && !need_postures.empty()) {
                static std::vector<std::tuple<Individual*, Individual::BasicStuff*>> all;
                
                while(!need_postures.empty()) {
                    all.emplace_back(std::move(need_postures.front()));
                    need_postures.pop();
                }
                
                distribute_vector([frameIndex](auto, auto start, auto end, auto){
                    Timer t;
                    double collected = 0;
                    
                    for(auto it = start; it != end; ++it) {
                        t.reset();
                        
                        auto fish = std::get<0>(*it);
                        auto basic = std::get<1>(*it);
                        fish->save_posture(*basic, frameIndex);
                        basic->pixels = nullptr;
                        
                        collected += t.elapsed();
                    }
                    
                    std::lock_guard<std::mutex> guard(Tracker::instance()->_statistics_mutex);
                    Tracker::instance()->_statistics[frameIndex].combined_posture_seconds += narrow_cast<float>(collected);
                    
                }, _thread_pool, all.begin(), all.end());
                
                all.clear();
                assert(need_postures.empty());
            }
            
            //if(assigned_count < 5)
            //    generate_pairdistances(frameIndex);
            
            //_thread_pool.wait();
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
        _added_frames.back()->active_individuals = assigned_count;
        
        uint32_t n = 0;
        uint32_t prev = 0;
        if(manual_identities.empty()) {
            for(auto fish : _active_individuals) {
                assert((fish->end_frame() == frameIndex) == (fish->has(frameIndex)));
                
                if(fish->end_frame() == frameIndex)
                    ++n;
                if(fish->has(frameIndex - 1_f))
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
                    if(fish->has(frameIndex - 1_f))
                        ++prev;
                }
            }
        }
        
        update_warnings(frameIndex, frame.time, number_fish, n, prev, props, prev_props, _active_individuals, _individual_add_iterator_map);
        
        if (save_tags || !frame.tags().empty()) {
            // find match between tags and individuals
            Match::PairedProbabilities paired;
            const Match::prob_t p_threshold = FAST_SETTINGS(matching_probability_threshold);

            for (auto fish : active_individuals) {
                Match::pairing_map_t<Match::Blob_t, prob_t> probs;
                
                auto cache = frame.cached(fish->identity().ID());
                if (!cache)
                    continue;

                for (const auto &blob : frame.tags()) {
                    auto p = fish->probability(-1, *cache, frameIndex, blob);
                    if (p >= p_threshold)
                        probs[&blob] = p;
                }

                if(!probs.empty())
                    paired.add(fish, probs);
            }

            /*for (auto& [fish, fdi] : paired_blobs.row_indexes()) {
                if (!clique.fids.contains(fdi))
                    continue;

                auto assigned = fish_assigned.find(fish);
                if (assigned != fish_assigned.end() && assigned->second)
                    continue;

                auto edges = paired_blobs.edges_for_row(fdi);

                Match::pairing_map_t<Match::Blob_t, prob_t> probs;
                for (auto& e : edges) {
                    auto blob = paired_blobs.col(e.cdx);
                    if (!blob_assigned.count(blob->get()) || !blob_assigned.at(blob->get()))
                        probs[blob] = e.p;
                }

                if (!probs.empty())
                    paired.add(fish, probs);
            }*/

            Match::PairingGraph graph(*props, frameIndex, paired);
            try {
                auto& optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
                for (auto& [fish, blob] : optimal.pairings) {
                    if (!fish->add_qrcode(frameIndex, std::move(*const_cast<pv::BlobPtr*>(blob)))) {
                        //FormatWarning("Fish ", fish->identity(), " rejected tag at ", (*blob)->bounds());
                    }
                }
            }
            catch (...) {
                FormatExcept("Exception during tags to individuals matching.");
            }
            frame.tags().clear(); // is invalidated now
            _thread_pool.wait();
        }
        
        std::lock_guard<std::mutex> guard(_statistics_mutex);
        _statistics[frameIndex].number_fish = assigned_count;
        _statistics[frameIndex].posture_seconds = posture_seconds;
    }

void Tracker::update_iterator_maps(Frame_t frame, const Tracker::set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators)
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
            
    void Tracker::update_warnings(Frame_t frameIndex, double time, long_t /*number_fish*/, long_t n_found, long_t n_prev, const FrameProperties *props, const FrameProperties *prev_props, const Tracker::set_of_individuals_t& active_individuals, ska::bytell_hash_map<Idx_t, Individual::segment_map::const_iterator>& individual_iterators) {
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
        
        update_iterator_maps(frameIndex - 1_f, active_individuals, individual_iterators);
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
            if(_warn_individual_status.size() <= fish->identity().ID()) {
                assert(!fish->has(frameIndex - 1_f));
                continue;
            }
            
            auto &property = _warn_individual_status.at(fish->identity().ID());
            if(property.prev == nullptr) {
                assert(!fish->has(frameIndex - 1_f));
            } else {
                assert((property.prev != nullptr) == fish->has(frameIndex - 1_f));
                if(property.prev != nullptr) {
                    if(property.current == nullptr) {
                        assert(fish->segment_for(frameIndex - 1_f) != fish->segment_for(frameIndex));
                    } else
                        assert(fish->segment_for(frameIndex - 1_f) == fish->segment_for(frameIndex));
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
            for(auto id : segment_end) {
                assert(individuals().at(Idx_t(id.id))->segment_for(frameIndex) != individuals().at(Idx_t(id.id))->segment_for(frameIndex - 1_f));
            }
            for(auto id : fdx) {
                assert(!individuals().at(Idx_t(id.id))->has(frameIndex));
                assert(frameIndex != start_frame() && _individuals.at(Idx_t(id.id))->has(frameIndex - 1_f));
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

    void Tracker::update_consecutive(const Tracker::set_of_individuals_t &active, Frame_t frameIndex, bool update_dataset) {
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
            FormatWarning("Letting frame ", manual_approval.start,"-",manual_approval.end," slip because its manually approved.");
        }*/
        
        if(all_good) {
            if(!_consecutive.empty() && _consecutive.back().end == frameIndex - 1_f) {
                _consecutive.back().end = frameIndex;
                if(frameIndex == analysis_range().end)
                    _recognition->update_dataset_quality();
            } else {
                if(!_consecutive.empty()) {
                    FOI::add(FOI(_consecutive.back(), "global segment"));
                }
                
                _consecutive.push_back(Range<Frame_t>(frameIndex, frameIndex));
                if(update_dataset)
                    _recognition->update_dataset_quality();
            }
        }
    }

    void Tracker::generate_pairdistances(Frame_t frameIndex) {
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
                d.set_d(euclidean_distance(frame_individuals.at(i)->centroid(frameIndex)->pos<Units::PX_AND_SECONDS>(), frame_individuals.at(idx1)->centroid(frameIndex)->pos<Units::PX_AND_SECONDS>()));
                
                distances.insert(d);
            }
        }

        throw U_EXCEPTION("Pair distances need to implement the new properties.");
    }
    
    void Tracker::_remove_frames(Frame_t frameIndex) {
        Categorize::DataStore::reanalysed_from(Frame_t(frameIndex));
        
        LockGuard guard("_remove_frames("+Meta::toStr(frameIndex)+")");
        recognition_pool.wait();
        _thread_pool.wait();
        
        _individual_add_iterator_map.clear();
        _segment_map_known_capacity.clear();
        
        if(_approximative_enabled_in_frame >= frameIndex)
            _approximative_enabled_in_frame.invalidate();
        
        print("Removing frames after and including ", frameIndex);
        
        if (end_frame() < frameIndex || start_frame() > frameIndex)
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
            print("Last remaining ", _consecutive.size());
            if(!_consecutive.empty()) {
                if(_consecutive.back().end >= frameIndex)
                    _consecutive.back().end = frameIndex - 1_f;
                print(_consecutive.back().start,"-",_consecutive.back().end);
            }
        }
        
        auto manual_identities = FAST_SETTINGS(manual_identities);
        std::vector<Idx_t> to_delete;
        std::vector<Individual*> ptrs;
        for(auto & [fdx, fish] : _individuals) {
            fish->remove_frame(frameIndex);
            
            if(FAST_SETTINGS(track_max_individuals) == 0 || manual_identities.find(fdx) == manual_identities.end()) {
                if(fish->empty()) {
                    to_delete.push_back(fdx);
                    ptrs.push_back(fish);
                }
            }
        }
        
        for(auto& fdx : to_delete)
            _individuals.erase(fdx);
        
        for(auto it = _active_individuals_frame.begin(); it != _active_individuals_frame.end();) {
            if(it->first >= frameIndex)
                it = _active_individuals_frame.erase(it);
            else
                ++it;
        }
        
        while(!_added_frames.empty()) {
            if((*(--_added_frames.end()))->frame < frameIndex)
                break;
            _added_frames.erase(--_added_frames.end());
        }
        
        for (auto it=_statistics.begin(); it != _statistics.end();) {
            if(it->first < frameIndex)
                ++it;
            else
                it = _statistics.erase(it);
        }
        
        _endFrame = frameIndex - 1_f;
        while (!properties(end_frame())) {
            if (end_frame() < start_frame()) {
                _endFrame = _startFrame = Frame_t();
                break;
            }
            
            _endFrame = end_frame() - 1_f;
        }
        
        if(end_frame().valid() && end_frame() < analysis_range().start)
            _endFrame = _startFrame = Frame_t();
        
        if(properties(end_frame()))
            _active_individuals = _active_individuals_frame.at(end_frame());
        else
            _active_individuals = {};
        
        _inactive_individuals.clear();
        //! assuming that most of the active / inactive individuals will stay the same, this should actually be more efficient
        for(auto& [id, fish] : _individuals) {
            if(_active_individuals.find(fish) == _active_individuals.end())
                _inactive_individuals.insert(id);
        }
        
        for (auto ptr : ptrs) {
            assert (_individual_add_iterator_map.find(ptr->identity().ID()) == _individual_add_iterator_map.end() );
            delete ptr;
        }
        
        if(_individuals.empty())
            Identity::set_running_id(Idx_t(0));
        
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
            
            auto it = _added_frames.rbegin();
            while(it != _added_frames.rend() && !_properties_cache.full())
            {
                _properties_cache.push((*it)->frame, (*it).get());
                ++it;
            }
            assert((_added_frames.empty() && !end_frame().valid()) || (end_frame().valid() && (*_added_frames.rbegin())->frame == end_frame()));
        }
        
        FOI::remove_frames(frameIndex);
        global_segment_order_changed();
        
        print("Inactive individuals: ", _inactive_individuals);
        print("Active individuals: ", _active_individuals);
        
        print("After removing frames: ", gui::CacheObject::memory());
        print("posture: ", Midline::saved_midlines());
        print("all blobs: ", Blob::all_blobs());
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

    void Tracker::wait() {
        recognition_pool.wait();
    }

    void Tracker::global_segment_order_changed() {
        LockGuard guard("Tracker::global_segment_order_changed");
        _global_segment_order.clear();
    }
    
    std::vector<Range<Frame_t>> Tracker::global_segment_order() {
        LockGuard guard("Tracker::max_range()");
        if(_global_segment_order.empty()) {
            std::set<Range<Frame_t>> manuals;
            auto manually_approved = FAST_SETTINGS(manually_approved);
            for(auto && [from, to] : manually_approved)
                manuals.insert(Range<Frame_t>(Frame_t(from), Frame_t(to)));
            
            std::set<Range<Frame_t>, std::function<bool(Range<Frame_t>, Range<Frame_t>)>> ordered([&manuals](Range<Frame_t> A, Range<Frame_t> B) -> bool {
                if(manuals.find(A) != manuals.end() && manuals.find(B) == manuals.end())
                    return true;
                if(manuals.find(B) != manuals.end() && manuals.find(A) == manuals.end())
                    return false;
                return (recognition() && recognition()->dataset_quality() ? ((recognition()->dataset_quality()->has(A) ? recognition()->dataset_quality()->quality(A) : DatasetQuality::Quality()) > (recognition()->dataset_quality()->has(B) ? recognition()->dataset_quality()->quality(B) : DatasetQuality::Quality())) : (A.length() > B.length()));
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
        }
        
        return _global_segment_order;
    }
    
    struct IndividualImages {
        std::vector<Frame_t> frames;
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
        void add_frame(Frame_t frame, long_t id, Image::Ptr image);
    };
    
    SplitData::SplitData() : _normalized(SETTING(recognition_normalize_direction).value<default_config::recognition_normalization_t::Class>()) {
        
    }
    
    void SplitData::add_frame(Frame_t frame, long_t id, Image::Ptr image) {
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
        LockGuard guard("clear_segments_identities");
        
        recognition_pool.wait();
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(Frame_t(0), fid);
        
        for(auto && [fdx, fish] : _individuals) {
            fish->clear_recognition();
        }
        
        _automatically_assigned_ranges.clear();
    }
    
    void Tracker::check_segments_identities(bool auto_correct, std::function<void(float)> callback, const std::function<void(const std::string&, const std::function<void()>&, const std::string&)>& add_to_queue, Frame_t after_frame) {
        
        print("Waiting for lock...");
        LockGuard guard("check_segments_identities");
        print("Updating automatic ranges starting from ", !after_frame.valid() ? Frame_t(0) : after_frame);
        
        const auto manual_identities = FAST_SETTINGS(manual_identities);
        size_t count=0;
        
        //recognition_pool.wait();
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(after_frame.valid() ? Frame_t(0) : after_frame, fid);
        
#ifdef TREX_DEBUG_IDENTITIES
        auto f = fopen(pv::DataLocation::parse("output", "identities.log").c_str(), "wb");
#endif
        distribute_vector([this, &count, &callback, &manual_identities](auto, auto it, auto, auto){
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
            std::map<Range<Frame_t>, Idx_t> track_ids;
        };
        
        Settings::manual_matches_t automatic_matches;
        std::map<fdx_t, VirtualFish> virtual_fish;
        
        // wrong fish -> set of unassigned ranges
        std::map<fdx_t, std::set<Range<Frame_t>>> unassigned_ranges;
        std::map<fdx_t, std::map<Range<Frame_t>, fdx_t>> assigned_ranges;
        
        decltype(_automatically_assigned_ranges) tmp_assigned_ranges;
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
                    
                    if(after_frame.valid() && segment.range.end < after_frame)
                        continue;
                    
                    if(n >= n_lower_bound || (segment.start() == fish->start_frame() && n > 0)) {
#ifdef TREX_DEBUG_IDENTITIES
                        log(f, "fish ",fdx,": segment ",segment.start(),"-",segment.end()," has ",n," samples");
#endif
                        //print("fish ",fdx,": segment ",segment.start(),"-",segment.end()," has ",n," samples");
                        
                        std::set<std::pair<Idx_t, Match::prob_t>, decltype(compare_greatest)> sorted(compare_greatest);
                        sorted.insert(average.begin(), average.end());
                        
                        // check if the values for this segment are too close, this probably
                        // means that we shouldnt correct here.
                        if(sorted.size() >= 2) {
                            Match::prob_t ratio = sorted.begin()->second / ((++sorted.begin())->second);
                            if(ratio > 1)
                                ratio = 1 / ratio;
                            
                            if(ratio >= 0.6) {
#ifdef TREX_DEBUG_IDENTITIES
                                log(f, "\ttwo largest probs ",sorted.begin()->second," and ",(++sorted.begin())->second," are too close (ratio ",ratio,")");
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
                                log(f, "\t",fdx," (as ",it->first,") Found range(s) ",*rit," for search range ",segment," p:",fit->second.probs.at(*rit)," n:",fit->second.samples.at(*rit)," (self:",it->second,",n:",n,")");
#endif
                                
                                Match::prob_t n_me = n;//segment.end() - segment.start();
                                Match::prob_t n_he = fit->second.samples.at(*rit);//rit->end() - rit->start();
                                const Match::prob_t N = n_me + n_he;
                                
                                n_me /= N;
                                n_he /= N;
                                
                                Match::prob_t sum_me = sigmoid(it->second) * sigmoid(n_me);
                                Match::prob_t sum_he = sigmoid(fit->second.probs.at(*rit)) * sigmoid(n_he);
                                
#ifdef TREX_DEBUG_IDENTITIES
                                log(f, "\tself:",segment.length()," ",it->second," other:",rit->length()," ",fit->second.probs.at(*rit)," => ",sum_me," / ", sum_he);
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
                        log(f, "\tassigning ",it->first," to ",fdx," with p ",it->second," for ", segment.start(), segment.end());
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
            log(f, "\t", fdx,":");
#endif
            // manual_match for first segment
            if(!fish.segments.empty()) {
                auto segment = *fish.segments.begin();
                
                if(!fish.probs.count(segment))
                    throw U_EXCEPTION("Cannot find ",segment.start(),"-",segment.end()," in fish.probs");
                if(!fish.track_ids.count(segment.range))
                    throw U_EXCEPTION("Cannot find ",segment.start(),"-",segment.end()," in track_ids");
                
                auto track = _individuals.at(fish.track_ids.at(segment.range));
                
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
                auto track = _individuals.at(fish.track_ids.at(segment.range));
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
                add_assigned_range(tmp_assigned_ranges, fdx, segment.range, std::move(blob_ids));
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
                if(after_frame.valid() && segment.range.end < after_frame)
                    continue;
                //if(start == 741 && fish->identity().ID() == 1)
                
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
                
                if(next != fish->recognition_segments().end() && /*previous.start() != -1 &&*/ next->second.start().valid()) {
                    Idx_t prev_id, next_id;
                    MotionRecord *prev_pos = nullptr, *next_pos = nullptr;
                    Frame_t prev_blob_frame;
                    
                    auto it = assigned_ranges.find(fdx);
                    if(it != assigned_ranges.end()) {
                        decltype(it->second.begin()) rit;
                        const Frame_t max_frames{ FAST_SETTINGS(frame_rate)*15 };
                        
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
                                        prev_blob_frame = previous->second.end();
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
                        Vec2 pos_start(FLT_MAX), pos_end(FLT_MAX);
                        auto blob_start = fish->centroid_weighted(segment.start());
                        auto blob_end = fish->centroid_weighted(segment.end());
                        if(blob_start)
                            pos_start = blob_start->pos<Units::CM_AND_SECONDS>();
                        if(blob_end)
                            pos_end = blob_end->pos<Units::CM_AND_SECONDS>();
                        
                        if(blob_start && blob_end) {
                            auto dprev = euclidean_distance(prev_pos->pos<Units::CM_AND_SECONDS>(), pos_start) 
                                / abs(blob_start->time() - prev_pos->time());
                            auto dnext = euclidean_distance(next_pos->pos<Units::CM_AND_SECONDS>(), pos_end) 
                                / abs(next_pos->time() - blob_end->time());
                            Idx_t chosen_id;
                            
                            if(dnext < dprev) {
                                if(dprev < FAST_SETTINGS(track_max_speed) * 0.1)
                                    chosen_id = next_id;
                            } else if(dnext < FAST_SETTINGS(track_max_speed) * 0.1)
                                chosen_id = prev_id;
                            
                            if(chosen_id.valid()) {
#ifdef TREX_DEBUG_IDENTITIES
                                if(segment.start() == 0) {
                                    log(f, "Fish ",fdx,": chosen_id ",chosen_id,", assigning ",segment," (",dprev," / ",dnext,")...");
                                }
#endif
                                
                                if(prev_blob_frame.valid() && prev_id.valid()) {
                                    // we found the previous blob/segment quickly:
                                    auto range = _individuals.at(prev_id)->get_segment_safe(prev_blob_frame);
                                    if(!range.empty()) {
                                        Frame_t frame = range.end();
                                        while(frame >= range.start()) {
                                            auto blob = _individuals.at(prev_id)->compressed_blob(frame);
                                            if(blob->split()) {
                                                //if(blob->parent_id != pv::bid::invalid) {
                                                    //manual_splits[frame].insert(blob->parent_id());
                                                //}
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
                                
                                std::set<Range<Frame_t>> remove_from;
                                
                                std::vector<pv::bid> blob_ids;
                                for(Frame_t frame=segment.start(); frame<=segment.end(); ++frame) {
                                    auto blob = fish->compressed_blob(frame);
                                    
                                    if(blob) {
                                        //automatically_assigned_blobs[frame][blob->blob_id()] = fdx;
                                        blob_ids.push_back(blob->blob_id());
                                        //if(blob->split() && blob->parent_id().valid())
                                        //    manual_splits[frame].insert(blob->parent_id());
                                    } else
                                        blob_ids.push_back(pv::bid::invalid);
                                    
                                    auto it = std::find(tmp_assigned_ranges.begin(), tmp_assigned_ranges.end(), chosen_id);
                                    if(it != tmp_assigned_ranges.end()) {
                                        for(auto && [range, blobs] : it->ranges)
                                        {
                                            if(range != segment.range && range.contains(frame)) {
                                                remove_from.insert(range);
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if(!remove_from.empty()) {
                                    //for(auto range : remove_from)
                                    //    automatically_assigned_ranges[chosen_id].erase(range);
                                    
                                    FormatWarning("[ignore] While assigning ",segment.range.start,"-",segment.range.end," to ",(uint32_t)chosen_id," -> same fish already assigned in ranges ",remove_from);
                                } else {
                                    assert((int64_t)blob_ids.size() == (segment.range.end - segment.range.start + 1_f).get());
                                    add_assigned_range(tmp_assigned_ranges, chosen_id, segment.range, std::move(blob_ids));
                                    
                                    auto blob = fish->blob(segment.start());
                                    if(blob && blob->split() && blob->parent_id().valid())
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
        print("auto_assign is ", auto_correct ? 1 : 0);
        if(auto_correct) {
            add_to_queue("", [after_frame, automatic_matches, manual_splits, tmp_assigned_ranges](){
                print("Assigning to queue from frame ", after_frame);
                
                //std::lock_guard<decltype(GUI::instance()->gui().lock())> guard(GUI::instance()->gui().lock());
                
                {
                    Tracker::LockGuard guard("check_segments_identities::auto_correct");
                    Tracker::instance()->_remove_frames(!after_frame.valid() ? Tracker::analysis_range().start : after_frame);
                    for(auto && [fdx, fish] : instance()->individuals()) {
                        fish->clear_recognition();
                    }
                    
                    print("automatically_assigned_ranges ", tmp_assigned_ranges.size());
                    _automatically_assigned_ranges = tmp_assigned_ranges;
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

pv::BlobPtr Tracker::find_blob_noisy(const PPFrame& pp, pv::bid bid, pv::bid pid, const Bounds&)
{
    auto blob = pp.find_bdx(bid);
    if(!blob) {
        return nullptr;
        
        if(pid.valid()) {
            blob = pp.find_bdx(pid);
            if(blob) {
                auto blobs = pixel::threshold_blob(blob, FAST_SETTINGS(track_threshold), Tracker::instance()->background());
                
                for(auto & sub : blobs) {
                    if(sub->blob_id() == bid) {
                        //print("Found perfect match for ", bid," in blob ",b->blob_id());//blob_to_id[bid] = sub;
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
                        print("Found blob ",bid," in parent ",pid," within thresholds [",first_found," - ",last_found,"]");
                    } else {
                        //FormatWarning("Cannot find blob ",bid," in it, but can find the parent ",pid," in frame ",frame," (threshold=",FAST_SETTINGS(track_threshold),").");
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

    void Tracker::check_save_tags(Frame_t frameIndex, const ska::bytell_hash_map<pv::bid, Individual*>& blob_fish_map, const std::vector<tags::blob_pixel> &tagged_fish, const std::vector<tags::blob_pixel> &noise, const file::Path &) {
        static Timing tag_timing("tags", 0.1);
        TakeTiming take(tag_timing);
        
        auto result = tags::prettify_blobs(tagged_fish, noise, {}, *_average);
        for (auto &r : result) {
            auto && [var, bid, ptr, f] = tags::is_good_image(r);
            if(ptr) {
                auto it = blob_fish_map.find(r.blob->blob_id());
                if(it != blob_fish_map.end())
                    it->second->add_tag_image(tags::Tag{var, r.blob->blob_id(), ptr, frameIndex} /* ptr? */);
                else
                    FormatWarning("Blob ", r.blob->blob_id()," in frame ",frameIndex," contains a tag, but is not associated with an individual.");
            }
        }
        
        if(_active_individuals_frame.find(frameIndex - 1_f) != _active_individuals_frame.end())
        {
            for(auto fish : _active_individuals_frame.at(frameIndex - 1_f)) {
                if(fish->start_frame() < frameIndex && fish->has(frameIndex - 1_f) && !fish->has(frameIndex))
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
                print("Determining blob size in ", video.length()," frames...");
            
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
                        print(i / step,": ",*frame_values.rbegin()," ",str0.c_str());
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
