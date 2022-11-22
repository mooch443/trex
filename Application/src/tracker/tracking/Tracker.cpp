#include "Tracker.h"
#include <misc/GlobalSettings.h>
#include <tracking/SplitBlob.h>
#include <misc/Timer.h>
#include "PairingGraph.h"
#include <misc/OutputLibrary.h>
#include <tracking/DetectTag.h>
#include <misc/cnpy_wrapper.h>
#include <processing/CPULabeling.h>
#include <misc/ReverseAdapter.h>
#include <misc/ProximityGrid.h>
#include <misc/default_settings.h>
#include <misc/pretty.h>
#include <tracking/DatasetQuality.h>
#include <misc/PixelTree.h>
#include <misc/CircularGraph.h>
#include <misc/MemoryStats.h>
#include <tracking/Categorize.h>
#include <tracking/VisualField.h>
#include <file/DataLocation.h>

#include <tracking/TrackingSettings.h>
#include <tracking/AutomaticMatches.h>

#if !COMMONS_NO_PYTHON
#include <tracking/PythonWrapper.h>
#include <tracking/RecTask.h>
#include <tracking/Accumulation.h>
namespace py = Python;
#endif

#ifndef NDEBUG
//#define PAIRING_PRINT_STATS
#endif
//#define TREX_DEBUG_IDENTITIES

namespace track {
    auto *tracker_lock = new std::shared_timed_mutex;

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
    
    inline void analyse_posture_pack(Frame_t frameIndex, const std::vector<std::tuple<Individual*, const BasicStuff*>>& p) {
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


void Tracker::predicted(Frame_t frame, pv::bid bdx, std::vector<float> && ps) {
    auto &ff = _vi_predictions[frame];
#ifndef NDEBUG
    if(ff.count(bdx)) {
        FormatWarning("bdx ", bdx, " already in predictions (forgot to clear?).");
    }
#endif
    
    ff[bdx] = std::move(ps);
}

const std::vector<float>& Tracker::get_prediction(Frame_t frame, pv::bid bdx) const {
    auto ptr = find_prediction(frame, bdx);
    if(!ptr)
        throw U_EXCEPTION("Assumed that prediction in ", frame, " for ", bdx, " exists. It does not.");
    return *ptr;
}

const std::vector<float>* Tracker::find_prediction(Frame_t frame, pv::bid bdx) const
{
    auto it = _vi_predictions.find(frame);
    if(it == _vi_predictions.end())
        return nullptr;
    
    auto kit = it->second.find(bdx);
    if(kit == it->second.end())
        return nullptr;
    
    return &kit->second;
}

std::map<Idx_t, float> Tracker::prediction2map(const std::vector<float>& pred) {
    std::map<Idx_t, float> map;
    for (size_t i=0; i<pred.size(); i++) {
        map[Idx_t(i)] = pred[i];
    }
    return map;
}
    
static std::string _last_thread = "<none>", _last_purpose = "";
static Timer _thread_holding_lock_timer;
static std::map<std::string, Timer> _last_printed_purpose;
static std::thread::id _writing_thread_id;

std::mutex thread_switch_mutex;

static std::mutex read_mutex;
static std::unordered_set<std::thread::id> read_locks;

Tracker::LockGuard::~LockGuard() {
    if(_write && _set_name) {
        std::unique_lock tswitch(thread_switch_mutex);
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
    }
    
    _locked = false;
        
    if(_write) {
        if(_owns_write) {
            {
                std::unique_lock tswitch(thread_switch_mutex);
                //std::stringstream ss, ss1;
                //ss << _writing_thread_id;
                //ss1 << std::this_thread::get_id();
                
                //print("[TG] ",_purpose, " resets _writing_thread_id(old=", ss.str()," vs. mine=", ss1.str(),") write=", _write, " regain=", _regain_read, " owned=", _owns_write);
                _writing_thread_id = std::thread::id();
            }
            
            tracker_lock->unlock();
            
            if(_regain_read) {
                //std::stringstream ss;
                //ss << std::this_thread::get_id();
                //print("[TG] ", _purpose, " reacquired shared_lock in thread ", ss.str(), " temporarily for write lock");
                
                tracker_lock->lock_shared();
                
                std::unique_lock rm(read_mutex);
                read_locks.insert(std::this_thread::get_id());
            }
            
        }
    } else if(_owns_write) {
        //std::stringstream ss;
        //ss << std::this_thread::get_id();
        //print("[TG] ", _purpose, " released shared_lock in thread ", ss.str());
        
        {
            std::unique_lock rm(read_mutex);
            read_locks.erase(std::this_thread::get_id());
        }
        
        tracker_lock->unlock_shared();
    }
        
}

//Tracker::LockGuard::LockGuard(std::string purpose, uint32_t timeout_ms) : LockGuard(w_t{}, purpose, timeout_ms)
//{ }

Tracker::LockGuard::LockGuard(w_t, std::string purpose, uint32_t timeout_ms) : _write(true), _purpose(purpose)
{
    init(timeout_ms);
}

Tracker::LockGuard::LockGuard(ro_t, std::string purpose, uint32_t timeout_ms) : _write(false), _purpose(purpose)
{
    init(timeout_ms);
}

bool Tracker::LockGuard::locked() const {
    //std::unique_lock tswitch(thread_switch_mutex);
    return _locked;//(!_write && _writing_thread_id == std::thread::id())
        //|| std::this_thread::get_id() == _writing_thread_id;
}

bool Tracker::LockGuard::init(uint32_t timeout_ms)
{
    assert(Tracker::instance());
    assert(!_purpose.empty());
    
    auto my_id = std::this_thread::get_id();
    
    {
        std::unique_lock tswitch(thread_switch_mutex);
        if(my_id == _writing_thread_id) {
            _locked = true;
            //std::stringstream ss;
            //ss << _writing_thread_id;
            //print("[TG] ",_purpose, " already has writing lock at ", ss.str());
            return true;
        }
    }
    
    if(!_write) {
        std::unique_lock rm(read_mutex);
        if(read_locks.contains(my_id)) {
            //! we are already reading in this thread, dont
            //! reacquire the lock
            _locked = true;
            return true;
        }
        
    } else {
        std::unique_lock rm(read_mutex);
        if(read_locks.contains(my_id)) {
            read_locks.erase(my_id);
            tracker_lock->unlock_shared();
            
            //std::stringstream ss;
            //ss << std::this_thread::get_id();
            //print("[TG] ", _purpose, " released shared_lock in thread ", ss.str(), " temporarily for write lock");
            
            _regain_read = true;
        }
    }
    
    if(timeout_ms) {
        auto duration = std::chrono::milliseconds(timeout_ms);
        if(_write && !tracker_lock->try_lock_for(duration)) {
            // did not get the write lock... :(
            if(_regain_read) {
                _regain_read = false;
                //std::stringstream ss;
                //ss << std::this_thread::get_id();
                //print("[TG] ", _purpose, " reacquired shared_lock in thread ", ss.str(), " temporarily for write lock");
                
                tracker_lock->lock_shared();
                
                std::unique_lock rm(read_mutex);
                read_locks.insert(my_id);
            }
            
            return false;
        } else if(!_write && !tracker_lock->try_lock_shared_for(duration)) {
            return false;
        }
        
    } else {
        constexpr auto duration = std::chrono::milliseconds(10);
        Timer timer, print_timer;
        while(true) {
            if((_write && tracker_lock->try_lock_for(duration))
               || (!_write && tracker_lock->try_lock_shared_for(duration)))
            {
                // acquired the lock :)
                break;
                
            } else if(timer.elapsed() > 60 && print_timer.elapsed() > 120) {
                std::unique_lock tswitch(thread_switch_mutex);
                auto name = _last_thread;
                auto myname = get_thread_name();
                FormatWarning("(",myname.c_str(),") Possible dead-lock with ",name," (",_last_purpose,") thread holding the lock for ",dec<2>(_thread_holding_lock_timer.elapsed()),"s (waiting for ",timer.elapsed(),"s, current purpose is ",_purpose,")");
                print_timer.reset();
            }
        }
    }
    
    _locked = true;
    _owns_write = true;
    
    if(_write) {
        
        std::unique_lock tswitch(thread_switch_mutex);
        _set_name = true;
        _last_thread = get_thread_name();
        _last_purpose = _purpose;
        _thread_holding_lock_timer.reset();
        _timer.reset();
        
        //std::stringstream ss, ss1;
        //ss << my_id;
        //ss1 << _writing_thread_id;
        //print("[TG] ",_purpose," sets writing thread id ", ss.str(), " from ", ss1.str());
        
        _writing_thread_id = my_id;
    } else {
        //std::stringstream ss;
        //ss << my_id;
        //print("[TG] ",_purpose," acquire read lock in thread ", ss.str());
        
        std::unique_lock rm(read_mutex);
        read_locks.insert(my_id);
    }
    
    return true;
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
        LockGuard guard(ro_t{}, "print_memory");
        mem::TrackerMemoryStats stats;
        stats.print();
    }

    bool callback_registered = false;
    
void Tracker::analysis_state(AnalysisState pause) {
    if(!instance())
        throw U_EXCEPTION("No tracker instance can be used to pause.");

    std::packaged_task<void(bool)> task([](bool value) {
        SETTING(analysis_paused) = value;
    });
    
    std::thread tmp(std::move(task), pause == AnalysisState::PAUSED);
    tmp.detach();
}

Tracker::Tracker()
      : _thread_pool(max(1u, cmn::hardware_concurrency()), "Tracker::thread_pool"),
        recognition_pool(max(1u, cmn::hardware_concurrency()), "RecognitionPool"),
        _max_individuals(0),
        _background(NULL),
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
        DatasetQuality::update();
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
                {
                    LockGuard guard(w_t{}, "Updating midlines in changed_setting("+key+")");
                    
                    for (auto && [id, fish] : Tracker::individuals()) {
                        Tracker::instance()->_thread_pool.enqueue([](long_t id, Individual *fish){
                            print("\t", id);
                            fish->clear_post_processing();
                            fish->update_midlines(nullptr);
                        }, id, fish);
                    }
                    
                    Tracker::instance()->_thread_pool.wait();
                }
                DatasetQuality::update();
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
                Tracker::LockGuard guard(ro_t{}, "changed_settings");
                Settings :: variable_changed(signal, map, key, value);
            }
        };
        cmn::GlobalSettings::map().register_callback(ptr, variable_changed);
        for(auto &n : Settings :: names())
            variable_changed(sprite::Map::Signal::NONE, cmn::GlobalSettings::map(), n, cmn::GlobalSettings::get(n).get());
        
    }
}

Tracker::~Tracker() {
    assert(_instance);
    Settings::clear_callbacks();
    
#if !COMMONS_NO_PYTHON
    Accumulation::on_terminate();
    RecTask::deinit();
#endif
    Individual::shutdown();
    
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
    Match::PairingGraph::prepare_shutdown();
#if !COMMONS_NO_PYTHON
    Accumulation::on_terminate();
    try {
        py::deinit().get();
    } catch(...) {
        FormatWarning("Exception during py::deinit().");
    }
#endif
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
    
    if(first_change.valid() && first_change <= Tracker::end_frame()) {
        Tracker::analysis_state(Tracker::AnalysisState::UNPAUSED);
    }
    
    //SETTING(manual_matches) = next;
    //FAST_SETTINGS(manual_matches) = next;
    //auto str = Meta::toStr(FAST_SETTINGS(manual_matches));
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
    static Timing timing("Tracker::add(PPFrame)", 10);
    TakeTiming take(timing);
    
    Timer overall_timer;
    LockGuard guard(w_t{}, "Tracker::add(PPFrame)");
    
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
    
    std::lock_guard<std::mutex> lguard(_statistics_mutex);
    _statistics[frame.index()].adding_seconds = (float)overall_timer.elapsed();
    _statistics[frame.index()].loading_seconds = (float)frame.frame().loading_time();
    
    auto samples = _time_samples.load();
    samples.add(_statistics[frame.index()].adding_seconds, _statistics[frame.index()].number_fish);
    _time_samples = samples;
}

double Tracker::average_seconds_per_individual() {
    auto samples = _time_samples.load();
    if(samples._frames_sampled == 0)
        return 0;
    return samples._seconds_per_frame / samples._frames_sampled;
}

class PairProbability {
private:
    GETTER_PTR(Individual*, idx)
    GETTER_PTR(pv::BlobPtr, bdx)
    GETTER(Match::prob_t, p)
    
public:
    PairProbability() = default;
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
    Tracker::LockGuard guard(ro_t{}, "update_history_log");
    if(history_log == nullptr && !SETTING(history_matching_log).value<file::Path>().empty()) {
        history_log = std::make_shared<std::ofstream>();
        
        auto path = SETTING(history_matching_log).value<file::Path>();
        if(!path.empty()) {
            path = file::DataLocation::parse("output", path);
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
    static std::once_flag flag;
    std::call_once(flag, [](){
        if(!GlobalSettings::has("meta_real_width"))
            throw U_EXCEPTION("Please load default settings before attempting to preprocess any frames (`meta_real_width` was not set, this is usually in there).");
        
        if(SETTING(meta_real_width).value<float>() == 0) {
            FormatWarning("This video does not set `meta_real_width`. Please set this value during conversion (see https://trex.run/docs/parameters_trex.html#meta_real_width for details). Defaulting to 30cm.");
            SETTING(meta_real_width) = float(30.0);
        }
        
        // setting cm_per_pixel after average has been generated (and offsets have been set)
        if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
            SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / float(average().cols);
    });
    
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
    
    const auto tags_dont_track = SETTING(tags_dont_track).value<bool>();
    
    auto check_blob = [&tags_dont_track, &track_ignore, &track_include, &result, &cm_sqr](const pv::BlobPtr& b)
    {
        // TODO: magic numbers
        if(b->pixels()->size() * cm_sqr > result->fish_size.max_range().end * 100)
            b->force_set_recount(result->threshold);
        else
            b->recount(result->threshold, *result->background);
        
        if(b->is_tag() && tags_dont_track) {
            result->filter_out(b);
            return false;
        }
        
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
        auto &b = *it;
        
        if(!check_blob(b))
            continue;
        
        float recount = b->recount(-1);
        
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
                
#if !COMMONS_NO_PYTHON
                if(!only_allowed.empty()) {
                    auto ldx = Categorize::DataStore::_ranged_label_unsafe(Frame_t(result->frame_index), ptr->blob_id());
                    if(ldx == -1 || !contains(only_allowed, Categorize::DataStore::label(ldx)->name)) {
                        result->filter_out(ptr);
                        continue;
                    }
                }
#endif
                
                //! only after all the checks passed, do we commit the blob
                /// to the "filtered" array:
                result->commit(ptr);
                
            } else if(recount < result->fish_size.max_range().start) {
                result->filter_out(ptr);
            } else
                big_blobs.push_back(ptr);
        }
        
        ptrs.clear();
    }
    
    for(auto &blob : filtered)
        blob->calculate_moments();
    
    if (result->frame_index == Tracker::start_frame() || !Tracker::start_frame().valid())
        big_blobs = Tracker::instance()->split_big(
                BlobReceiver(*result, BlobReceiver::noise),
                big_blobs,
                {});

#if !COMMONS_NO_PYTHON
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
#endif

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
    
    size_t available_threads = 1 + (pool ? pool->num_threads() : 0);
    size_t maximal_threads = frame.blobs().size();
    size_t needed_threads = min(maximal_threads / (size_t)FAST_SETTINGS(blobs_per_thread), available_threads);
    std::shared_lock guard(Categorize::DataStore::range_mutex());
    
    if (maximal_threads > 1
        && needed_threads > 1
        && available_threads > 1
        && pool)
    {
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
    const robin_hood::unordered_map<pv::bid, split_expectation> &expect,
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
            
            auto bdx = b->blob_id();

            split_expectation ex(2, false);
            if(!expect.empty() && expect.count(bdx))
                ex = expect.at(bdx);
            
            auto rec = b->recount(threshold, *_background);
            if(!fish_size.close_to_maximum_of_one(rec, 10 * ex.number)) {
                noise.push_back(b);
                continue;
            }
            
            SplitBlob s(*_background, b);
            std::vector<pv::BlobPtr> copy;
            auto ret = s.split(ex.number, ex.centers);
            
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
        Tracker::LockGuard guard(ro_t{}, "history_split#1");
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
    robin_hood::unordered_map<Idx_t, Vec2> last_positions;

    const auto frame_limit = FAST_SETTINGS(frame_rate) * FAST_SETTINGS(track_max_reassign_time);
    const auto N = active_individuals.size();

    {
        const size_t num_threads = pool ? min(hardware_concurrency(), N / 200u) : 0;
        const auto space_limit = SQR(Individual::weird_distance() * 0.5);
        const auto frame_rate = FAST_SETTINGS(frame_rate);
        const auto track_max_reassign_time = FAST_SETTINGS(track_max_reassign_time);

        CacheHints hints;
        if(frame.index().valid() && frame.index() > Tracker::start_frame())
            hints.push(frame.index() - 1_f, properties(frame.index() - 1_f));
        hints.push(frame.index(), properties(frame.index()));

        // mutex protecting count and global paired + fish_mappings/blob_mappings
        std::mutex mutex;
        std::condition_variable variable;
        size_t count = 0;

        auto fn = [&](const Tracker::set_of_individuals_t& active_individuals,
                      size_t start,
                      size_t N)
        {
            struct FishAssignments {
                Idx_t fdx;
                std::vector<pv::bid> blobs;
                std::vector<float> distances;
                Vec2 last_pos;
            };
            struct BlobAssignments {
                UnorderedVectorSet<Idx_t> idxs;
            };

            std::vector<FishAssignments> fish_assignments(N);
            ska::bytell_hash_map<pv::bid, BlobAssignments> blob_assignments;

            auto it = active_individuals.begin();
            std::advance(it, start);
            
            //! go through individuals (for this pack/thread)
            for(auto i = start; i < start + N; ++i, ++it) {
                auto fish = *it;
                
                Vec2 last_pos(-1,-1);
                Frame_t last_frame;
                long_t last_L = -1;

                // IndividualCache is in the same position as the indexes here
                auto cache = fish->cache_for_frame(frame.index(), frame.time, &hints);
                const auto time_limit = cache.previous_frame.get() - frame_limit; // dont allow too far to the past
                    
                // does the current individual have the frame previous to the current frame?
                //! try to find a frame thats close in time AND space to the current position
                size_t counter = 0;
                auto sit = fish->iterator_for(cache.previous_frame);
                if (sit != fish->frame_segments().end() && (*sit)->contains(cache.previous_frame))
                {
                    for (; sit != fish->frame_segments().end()
                            && min((*sit)->end(), cache.previous_frame).get() >= time_limit
                            && counter < frame_limit; // shouldnt this be the same as the previous?
                        ++counter)
                    {
                        const auto index = (*sit)->basic_stuff((*sit)->end());
                        const auto pos = fish->basic_stuff().at(index)->centroid.pos<Units::DEFAULT>();

                        if ((*sit)->length() > frame_rate * track_max_reassign_time * 0.25)
                        {
                            //! segment is long enough, we can stop. but only actually use it if its not too far away:
                            if (last_pos.x == -1
                                || sqdistance(pos, last_pos) < space_limit)
                            {
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
                    Log(out, "\tNot processing fish ", fish->identity()," because its last measured frame is ", last_frame,", best segment length is ", last_L," and we are in frame ", frame.index(),".");
                    
                } else {
                    auto set = frame.blob_grid().query(cache.estimated_px, max_d);
                    
                    if(!set.empty()) {
                        auto fdx = fish->identity().ID();
                        
                        //std::unique_lock guard(thread_mutex);
                        //auto &map = fish_mappings[fdx];
                        //auto &pair_map = paired[fdx];
                        auto& map = fish_assignments[i - start];
                        map.fdx = fdx;
                        map.last_pos = last_pos.x == -1 ? cache.estimated_px : last_pos;
                        
                        for(auto && [d, bdx] : set) {
                            if(!frame.find_bdx(uint32_t(bdx)))
                                continue;
                            
                            map.blobs.push_back(uint32_t(bdx));
                            map.distances.push_back(d);
                            blob_assignments[uint32_t(bdx)].idxs.insert(fdx);
                        }
                    }
                    
                    Log(out, "\tFish ", fish->identity()," (", cache.estimated_px.x, ",", cache.estimated_px.y, ") proximity: ", set);
                }
                
                frame.set_cache(fish->identity().ID(), std::move(cache));
            }

            std::unique_lock lock(mutex);
            for (auto&& [fdx, blobs, distances, last_pos] : fish_assignments) {
                fish_mappings[fdx].insert(std::make_move_iterator(blobs.begin()), std::make_move_iterator(blobs.end()));
                auto N = blobs.size();
                for(size_t i=0; i<N; ++i)
                    paired[fdx][blobs[i]] = distances[i];
                last_positions[fdx] = last_pos;
            }
            for (auto& [bdx, assign] : blob_assignments) {
                blob_mappings[bdx].insert(assign.idxs.begin(), assign.idxs.end());
            }

            ++count;
            variable.notify_one();
        };
        
        frame.init_cache(active_individuals);
        
        if(num_threads < 2 || !pool || N < num_threads) {
            Tracker::LockGuard guard(ro_t{}, "history_split#2");
            fn(active_individuals, 0, N);
            
        } else if(N) {
            size_t last = N % num_threads;
            size_t per_thread = (N - last) / num_threads;
            size_t i = 0;

            Tracker::LockGuard guard(ro_t{}, "history_split#2");
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
    UnorderedVectorSet<pv::BlobPtr> big_blobs;
    robin_hood::unordered_map<pv::bid, split_expectation> expect;
    
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
                blob_mappings[bdx] = { };
            } else{
                //it->second.insert(Idx_t());
            }
            
            Log(out, "\t\tManually splitting ", (uint32_t)bdx);
            auto ptr = frame.erase_anywhere(bdx);
            if(ptr) {
                big_blobs.insert(ptr);
                
                expect[bdx].number = 2;
                expect[bdx].allow_less_than = false;
                
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
    
    /**
     * Now we have found all the mappings from fish->blob and vice-a-versa,
     * lets do the actual splitting (if enabled).
     * -------------------------------------------------------------------------------------
     */

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
                if(!fdx.valid())
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
        std::unordered_map<pv::bid, std::pair<Idx_t, Match::prob_t>> assign_blob; // blob: individual
        std::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> all_probs_per_fish;
        std::unordered_map<Idx_t, std::set<std::tuple<Match::prob_t, pv::bid>>> probs_per_fish;
        
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
        
        auto check_combinations =
            [&assign_blob, out](Idx_t c, decltype(probs_per_fish)::mapped_type& combinations, std::queue<Idx_t>& q)
          -> bool
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
        std::map<pv::bid, std::vector<Vec2>> centers;
        size_t counter = 0;
        if(out)
            Log(out, "Final assign blob:", assign_blob);

        for(auto && [fdx, set] : all_probs_per_fish) {
            if(out) {
                Log(out, "Combinations ", fdx,": ", set);
            }

            auto last_pos = last_positions.at(fdx);
            
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

                        auto ptr = frame.find_bdx(max_id);
                        if(ptr) {
                            if(assign_blob.count(max_id)) {
                                ++expect[max_id].number;
                                expect[max_id].centers.push_back(last_positions.at(assign_blob.at(max_id).first) - ptr->bounds().pos());
                                assign_blob.erase(max_id);
                            }

                            ++expect[max_id].number;
                            big_blobs.insert(ptr);
                            expect[max_id].centers.push_back(last_pos - ptr->bounds().pos());
                            Log(out, "Increasing split number in ", *ptr, " to ", expect[max_id]);
                        } else
                            Log(out, "Cannot split blob ", max_id, " since it cannot be found.");

                        /*auto ptr = frame.erase_regular(max_id);
                        
                        if(ptr) {
                            Log(out, "Splitting blob ", max_id);
                            to_delete.insert(max_id);
                            
                            ++expect[max_id].number;
                            big_blobs.insert(ptr);
                            expect[max_id].centers.push_back(last_pos);
                        }
                        else if((ptr = frame.find_bdx(max_id))) {
                            if(expect.contains(max_id)) {
                                Log(out, "Increasing expect number for blob ", max_id);
                                ++expect[max_id].number;
                                expect[max_id].centers.push_back(last_pos);
                            }
                            
                            Log(out, "Would split blob ", max_id,", but its part of additional.");
                        } else
                            Log(out, "Cannot split blob ", max_id, " since it cannot be found.");*/
                        
                        if(allow_less_than)
                            expect[max_id].allow_less_than = allow_less_than;
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
        
        if(out) {
            Log(out, "expect: ", expect);
            if(counter > 1) {
                Log(out, "Lost ", counter," fish (", expect, ")");
            }
        }
    }
    
    //for(auto && [blob, e] : expect)
    //    ++e.number;
    
    if(!manual_splits_frame.empty()) {
        for(auto &bdx : manual_splits_frame) {
            auto ptr = frame.find_bdx(bdx);
            if(ptr) {
                expect[bdx].allow_less_than = false;
                expect[bdx].number = 2;
            }
        }
    }

    for(auto& b : big_blobs)
        frame.erase_regular(b->blob_id());
    
    auto big_filtered = split_big(BlobReceiver(frame, BlobReceiver::noise), big_blobs._data, expect, true, out, pool);
    
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
    auto& individuals = instance()->_individuals;
    
    if(individuals.find(ID) != individuals.end())
        throw U_EXCEPTION("Cannot assign identity (",ID,") twice.");
    
    Individual *fish = new Individual();
    fish->identity().set_ID(ID);
    
    individuals[fish->identity().ID()] = fish;
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

bool Tracker::has_identities() {
    return FAST_SETTINGS(track_max_individuals) > 0;
}

const std::set<Idx_t> Tracker::identities() {
    if(!has_identities())
        return {};
    
    static std::set<Idx_t> set;
    static std::mutex mutex;
    
    std::unique_lock guard(mutex);
    if(set.empty()) {
        //LockGuard guard("Tracker::identities");
        //for(auto &[id, fish] : Tracker::individuals())
        //    set.insert(id);
        
        //if(set.empty()) {
            for(Idx_t i = Idx_t(0); i < Idx_t(FAST_SETTINGS(track_max_individuals)); i = Idx_t(i._identity + 1)) {
                set.insert(i);
            }
        //}
    }
    
    return set;
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
        
#if !COMMONS_NO_PYTHON
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
#endif
            for(size_t i=0; i<frame.blobs().size(); ++i) {
                if(!blob_assigned.at(frame.blobs()[i].get()))
                    unassigned_blobs.push_back(i);
            }
#if !COMMONS_NO_PYTHON
        }
#endif

        const auto N_blobs = unassigned_blobs.size();
        const auto N_fish  = unassigned_individuals.size();
        const auto matching_probability_threshold = FAST_SETTINGS(matching_probability_threshold);
        
        auto work = [&](auto, auto start, auto end, auto N){
            size_t pid = 0;
            std::vector< Match::pairing_map_t<Match::Blob_t, Match::prob_t> > _probs(N);
            
            for (auto it = start; it != end; ++it, ++pid) {
                auto fish = *it;
                auto cache = frame.cached(fish->identity().ID());
                assert(cache->_idx == fish->identity().ID());
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
                auto path = file::DataLocation::parse("output", (std::string)SETTING(filename).value<file::Path>().filename()+"_threading_stats.npz").str();
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

void collect_matching_cliques(TrackingSettings& s, GenericThreadPool& thread_pool) {
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

    const auto p_threshold = FAST_SETTINGS(matching_probability_threshold);
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
                    if(!s.frame.find_bdx((*blob)->blob_id())) {
                        print("Frame ", s.frame.index(),": Cannot find blob ",(*blob)->blob_id()," in map.");
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
                
                for (auto& [fish, fdi] : s.paired.row_indexes()) {
                    if (!clique.fids.contains(fdi))
                        continue;
                    
                    auto assigned = s.fish_assigned.find(fish);
                    if (assigned != s.fish_assigned.end() && assigned->second)
                        continue;
                    
                    auto edges = s.paired.edges_for_row(fdi);
                    
                    Match::pairing_map_t<Match::Blob_t, prob_t> probs;
                    for (auto& e : edges) {
                        auto blob = s.paired.col(e.cdx);
                        if (!s.blob_assigned.count(blob->get()) || !s.blob_assigned.at(blob->get()))
                            probs[blob] = e.p;
                    }
                    
                    if (!probs.empty())
                        paired.add(fish, probs);
                }
                
                PairingGraph graph(*s.props, frameIndex, paired);
                
                try {
                    // we use hungarian matching here by default, although
                    // the tree-based method is basically equal:
                    auto& optimal = graph.get_optimal_pairing(false, matching_mode_t::hungarian);
                    
                    std::unique_lock g(thread_mutex);
                    for (auto& p : optimal.pairings) {
                        s.assign_blob_individual(p.first, *p.second, matching_mode_t::hungarian);
                        s.active_individuals.insert(p.first);
                    }
                    
                }
                catch (...) {
                    FormatExcept("Failed to generate optimal solution (frame ", frameIndex,").");
                }
            }
        };
        
        const auto frameIndex = s.frame.index();
        distribute_vector(work_cliques, thread_pool, cliques.begin(), cliques.end());
        
        //! update cliques in the global array:
        Tracker::Clique translated;
        Tracker::instance()->_cliques[frameIndex].clear();

        for (auto& clique : cliques) {
            translated.bids.clear();
            translated.fishs.clear();

            for (auto bdi : clique.bids)
                translated.bids.insert((*s.paired.col(bdi))->blob_id());
            for (auto fdi : clique.fids)
                translated.fishs.insert(s.paired.row(fdi)->identity().ID());

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
            auto fit = s.fish_assigned.find(fish);
            if(fit == s.fish_assigned.end() || !fit->second) {
                auto edges = s.paired.edges_for_row(idx);
                
                Match::pairing_map_t<Match::Blob_t, Match::prob_t> probs;
                for(auto &e : edges) {
                    auto blob = s.paired.col(e.cdx);
#ifndef NDEBUG
                    if(!s.frame.find_bdx((*blob)->blob_id())) {
                        print("Frame ", frameIndex,": Cannot find blob ",(*blob)->blob_id()," in map.");
                        continue;
                    }
#endif
                    auto it = s.blob_assigned.find(blob->get());
                    if(it == s.blob_assigned.end() || !it->second) {
                        probs[blob] = e.p;
                    }
                }
                
                if(!probs.empty())
                    paired.add(fish, probs);
            }
        }
        
        s.paired = std::move(paired);
        s.match_mode = matching_mode_t::approximate;
    }
}

/**
 * Adding a frame that has been preprocessed previously in a different thread.
 */
void Tracker::add(Frame_t frameIndex, PPFrame& frame) {
    if (!start_frame().valid() || start_frame() > frameIndex) {
        _startFrame = frameIndex;
    }
    
    if (!end_frame().valid() || end_frame() < frameIndex) {
        if(end_frame().valid() && end_frame() < start_frame())
          FormatError("end frame is ", end_frame()," < ",start_frame());
        _endFrame = frameIndex;
    }
    
    static Timing timing("Tracker::add(frameIndex,PPFrame)", 10);
    TakeTiming assign(timing);
    
    TrackingSettings s(frame, _added_frames);
    
    // see if there are manually fixed matches for this frame
    s.apply_manual_matches(_individuals);
    s.apply_automatic_matches();
    
    const auto track_max_reassign_time = FAST_SETTINGS(track_max_reassign_time);
        
    for(auto fish: _active_individuals) {
        // jump over already assigned individuals
        if(!s.fish_assigned[fish]) {
            if(fish->empty()) {
                //fish_assigned[fish] = false;
                //active_individuals.push_back(fish);
            } else {
                auto found_idx = fish->find_frame(frameIndex)->frame;
                float tdelta = cmn::abs(frame.time - properties(found_idx)->time);
                if (tdelta < track_max_reassign_time)
                    s.active_individuals.insert(fish);
            }
        }
    }
    
    // now that the blobs array has been cleared of all the blobs for fixed matches,
    // get pairings for all the others:
    //std::unordered_map<pv::Blob*, pv::BlobPtr> ptr2ptr;
    s.paired = calculate_paired_probabilities( frame,
                                               s.active_individuals,
                                               s.fish_assigned,
                                               s.blob_assigned,
                                               &_thread_pool);
    
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
    //! solved fully (e.g. hungarian algorithm) for each subset.
    if(s.match_mode == default_config::matching_mode_t::automatic) {
        collect_matching_cliques(s, _thread_pool);
    }
    
    //! now actually apply the matching
    //! the below method basically calls PairingGraph.find_optimal_pairing
    s.apply_matching();
    
    static Timing rest("rest", 30);
    TakeTiming take(rest);
    
    // Create Individuals for unassigned blobs
    std::vector<pv::BlobPtr> unassigned_blobs;
    for(auto &p: frame.blobs()) {
        if(!s.blob_assigned[p.get()])
            unassigned_blobs.emplace_back(p);
    }
    
    if(!s.number_fish /*|| (number_fish && number_individuals < number_fish)*/) {
        // the number of individuals is limited
        // fallback to creating new individuals if the blobs cant be matched
        // to existing ones
        /*if(frameIndex > 1) {
            static std::random_device rng;
            static std::mt19937 urng(rng());
            std::shuffle(unassigned_blobs.begin(), unassigned_blobs.end(), urng);
        }*/
        
        for(auto fish :_active_individuals)
            if(s.active_individuals.find(fish) == s.active_individuals.end())
                _inactive_individuals.insert(fish->identity().ID());
        
        for (auto &blob: unassigned_blobs) {
            // we measure the number of currently assigned fish based on whether a maximum number has been set. if there is a maximum, then we only look at the currently active individuals and extend that array with new individuals if necessary.
            const size_t number_individuals = s.number_fish ? _individuals.size() : s.active_individuals.size();
            if(s.number_fish && number_individuals >= s.number_fish) {
                static bool warned = false;
                if(!warned) {
                    FormatWarning("Running out of assignable fish (track_max_individuals ", s.active_individuals.size(),"/",s.number_fish,")");
                    warned = true;
                }
                break;
            }
            
            if(s.number_fish)
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
            s.assign_blob_individual(fish, blob, default_config::matching_mode_t::benchmark);
            s.active_individuals.insert(fish);
        }
    }
    
    if(s.number_fish && s.active_individuals.size() < s.number_fish) {
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
            for(auto id : identities()) {
                if(!_individuals.count(id)) {
                    set_of_individuals_t set;
                    create_individual(id, set);
                }
            }
            
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
                    if(!s.blob_assigned.at(r.bdx().get()) && contains(lost_individuals, r.idx())) {
                        s.blob_assigned.at(r.bdx().get()) = true;
                        
                        auto it = std::find(lost_individuals.begin(), lost_individuals.end(), r.idx());
                        assert(it != lost_individuals.end());
                        lost_individuals.erase(it);
                        
                        Individual *fish = r.idx();
                        s.assign_blob_individual(fish, r.bdx(), default_config::matching_mode_t::benchmark);
                        s.active_individuals.insert(fish);
                        
                    }
                }
            }
        }
    }
    
    _active_individuals = s.active_individuals;
    
#ifndef NDEBUG
    if(!s.number_fish) {
        static std::set<Idx_t> lost_ids;
        for(auto && [fdx, fish] : _individuals) {
            if(s.active_individuals.find(fish) == s.active_individuals.end() && _inactive_individuals.find(fdx) == _inactive_individuals.end()) {
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
    
    if(s.save_tags) {
       _thread_pool.enqueue([&](){
            this->check_save_tags(frameIndex, s.blob_fish_map, s.tagged_fish, s.noise, FAST_SETTINGS(tags_path));
        });
    }
    
    
    Timer posture_timer;
    s.process_postures();
    const auto posture_seconds = posture_timer.elapsed();
    
    Output::Library::frame_changed(frameIndex);
    
    if(s.number_fish && s.assigned_count >= s.number_fish) {
        update_consecutive(_active_individuals, frameIndex, true);
    }
    
    _max_individuals = cmn::max(_max_individuals, s.assigned_count);
    _active_individuals_frame[frameIndex] = _active_individuals;
    _added_frames.back()->active_individuals = narrow_cast<long_t>(s.assigned_count);
    
    uint32_t n = 0;
    uint32_t prev = 0;
    if(!has_identities()) {
        for(auto fish : _active_individuals) {
            assert((fish->end_frame() == frameIndex) == (fish->has(frameIndex)));
            
            if(fish->end_frame() == frameIndex)
                ++n;
            if(fish->has(frameIndex - 1_f))
                ++prev;
        }
        
    } else {
        for(auto id : Tracker::identities()) {
            //! TODO: check _individuals?
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
    
    update_warnings(frameIndex, s.frame.time, s.number_fish, n, prev, s.props, s.prev_props, _active_individuals, _individual_add_iterator_map);
    
#if !COMMONS_NO_PYTHON
    if (s.save_tags || !frame.tags().empty()) {
        // find match between tags and individuals
        Match::PairedProbabilities paired;
        const Match::prob_t p_threshold = FAST_SETTINGS(matching_probability_threshold);

        for (auto fish : s.active_individuals) {
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

        /*for (auto& [fish, fdi] : s.paired.row_indexes()) {
            if (!clique.fids.contains(fdi))
                continue;

            auto assigned = fish_assigned.find(fish);
            if (assigned != fish_assigned.end() && assigned->second)
                continue;

            auto edges = s.paired.edges_for_row(fdi);

            Match::pairing_map_t<Match::Blob_t, prob_t> probs;
            for (auto& e : edges) {
                auto blob = s.paired.col(e.cdx);
                if (!blob_assigned.count(blob->get()) || !blob_assigned.at(blob->get()))
                    probs[blob] = e.p;
            }

            if (!probs.empty())
                paired.add(fish, probs);
        }*/

        Match::PairingGraph graph(*s.props, frameIndex, paired);
        try {
            auto& optimal = graph.get_optimal_pairing(false, default_config::matching_mode_t::hungarian);
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
#endif
    
    std::lock_guard<std::mutex> guard(_statistics_mutex);
    _statistics[frameIndex].number_fish = s.assigned_count;
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
        
        //auto manual_identities = FAST_SETTINGS(manual_identities);
        for(auto fish : active) {
            //if(manual_identities.empty() || manual_identities.count(fish->identity().ID()))
            {
                if(!fish->has(frameIndex) /*|| fish->centroid_weighted(frameIndex)->speed() >= FAST_SETTINGS(track_max_speed) * 0.25*/) {
                    all_good = false;
                    break;
                }
            }
        }
        
        if(all_good) {
            if(!_consecutive.empty() && _consecutive.back().end == frameIndex - 1_f) {
                _consecutive.back().end = frameIndex;
                if(frameIndex == analysis_range().end) {
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
#if !COMMONS_NO_PYTHON
        Categorize::DataStore::reanalysed_from(Frame_t(frameIndex));
#endif
        
        LockGuard guard(w_t{}, "_remove_frames("+Meta::toStr(frameIndex)+")");
        recognition_pool.wait();
        _thread_pool.wait();
        
        _individual_add_iterator_map.clear();
        _segment_map_known_capacity.clear();
        
        if(TrackingSettings::_approximative_enabled_in_frame >= frameIndex)
            TrackingSettings::_approximative_enabled_in_frame.invalidate();
        
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
        
        DatasetQuality::remove_frames(frameIndex);
        
        std::vector<Idx_t> to_delete;
        std::vector<Individual*> ptrs;
        for(auto & [fdx, fish] : _individuals) {
            fish->remove_frame(frameIndex);
            
            //! TODO: note that this also deletes "wanted" individuals
            //! and so we need to check if this is OK
            if(fish->empty()) {
                to_delete.push_back(fdx);
                ptrs.push_back(fish);
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
        
        FilterCache::clear();
        //! TODO: MISSING remove_frames
        //_recognition->remove_frames(frameIndex);
        DatasetQuality::update();
        
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
        
        VisualField::remove_frames_after(frameIndex);
        FOI::remove_frames(frameIndex);
        
        global_segment_order_changed();
        
        print("Inactive individuals: ", _inactive_individuals);
        print("Active individuals: ", _active_individuals);
        
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

    void Tracker::wait() {
        recognition_pool.wait();
    }

    void Tracker::global_segment_order_changed() {
        LockGuard guard(w_t{}, "Tracker::global_segment_order_changed");
        _global_segment_order.clear();
    }
    
    std::vector<Range<Frame_t>> Tracker::global_segment_order() {
        LockGuard guard(ro_t{}, "Tracker::max_range()");
        if(_global_segment_order.empty()) {
            LockGuard guard(w_t{}, "Tracker::max_range()::write");
            std::set<Range<Frame_t>> manuals;
            auto manually_approved = FAST_SETTINGS(manually_approved);
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
        
        GETTER(default_config::individual_image_normalization_t::Class, normalized)
        
    public:
        SplitData();
        void add_frame(Frame_t frame, long_t id, Image::Ptr image);
    };
    
    SplitData::SplitData() : _normalized(SETTING(recognition_normalize_direction).value<default_config::individual_image_normalization_t::Class>()) {
        
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
        LockGuard guard(w_t{}, "clear_segments_identities");
        
        recognition_pool.wait();
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(Frame_t(0), fid);
        
        for(auto && [fdx, fish] : _individuals) {
            fish->clear_recognition();
        }
        
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
            const Frame_t max_frames{ FAST_SETTINGS(frame_rate) * 15 };

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
                        auto blob = Tracker::individuals().at(org_id)->centroid_weighted(next->second.start());
                        if (blob)
                            next_pos = blob;
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
                        auto pos = Tracker::individuals().at(org_id)->centroid_weighted(previous->second.end());
                        if (pos) {
                            prev_pos = pos;
                        }
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
            const Frame_t max_frames{ FAST_SETTINGS(frame_rate) * 15 };

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
                        auto blob = Tracker::individuals().at(org_id)->centroid_weighted((*next)->start());
                        if (blob)
                            next_pos = blob;
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
                        auto pos = Tracker::individuals().at(org_id)->centroid_weighted((*previous)->end());
                        if (pos) {
                            prev_pos = pos;
                        }
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
            if (dprev < FAST_SETTINGS(track_max_speed) * 0.1)
                chosen_id = next_id;
        }
        else if (dnext < FAST_SETTINGS(track_max_speed) * 0.1)
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
                FormatWarning("[ignore] While assigning ", segment.range.start, "-", segment.range.end, " to ", (uint32_t)chosen_id, " -> same fish already assigned in ranges ", remove_from);
#else
                FormatWarning("[ignore] While assigning ", segment.range.start, "-", segment.range.end, " to ", (uint32_t)chosen_id, " -> same fish already assigned in another range.");
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
        
        //recognition_pool.wait();
        auto fid = FOI::to_id("split_up");
        if(fid != -1)
            FOI::remove_frames(after_frame.valid() ? Frame_t(0) : after_frame, fid);
        
#ifdef TREX_DEBUG_IDENTITIES
        auto f = fopen(file::DataLocation::parse("output", "identities.log").c_str(), "wb");
#endif
        float N = float(_individuals.size());
        distribute_vector([&count, &callback, N](auto, auto start, auto end, auto)
        {
            for(auto it = start; it != end; ++it) {
                auto & [fdx, fish] = *it;
                
                fish->clear_recognition();
                fish->calculate_average_recognition();
                
                callback(count / N * 0.5f);
                ++count;
            }
            
        }, recognition_pool, _individuals.begin(), _individuals.end());
        
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
        
        const size_t n_lower_bound =  source == IdentitySource::QRCodes ? 2  : max(5, FAST_SETTINGS(frame_rate) * 0.1f);

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
            for(const auto & [fdx, fish] : _individuals) {
                for (auto& [start, assign] : fish->qrcodes()) {
                    auto segment = fish->segment_for(start);
                    if (segment && segment->contains(start)) {
                        collect_virtual_fish(fish->identity().ID(), fish, *segment, assign.samples, {
                            { Idx_t(assign.best_id), assign.p }
                        });
                    }
                }
            }
            
        } else {
            //! When using other ML information (visual identification)...
            assert(source == IdentitySource::MachineLearning);
            
            for(const auto & [fdx, fish] : _individuals) {
                //! TODO: MISSING recalculate recognition for all segments
                //fish->clear_recognition();

                for (auto&& [start, segment] : fish->recognition_segments()) {
                    if (after_frame.valid() && segment.end() < after_frame)
                        continue;

                    auto& [n, average] = fish->processed_recognition(start);
                    collect_virtual_fish(fdx, fish, segment, n, average);
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
                AutoAssign::add_assigned_range(tmp_assigned_ranges, fdx, segment.range, std::move(blob_ids));
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
            for(const auto &[fdx, fish] : _individuals)
                process_qr(after_frame, fdx, fish, assigned_ranges, virtual_fish, _apply);
            
        } else {
            assert(source == IdentitySource::MachineLearning);
            for(const auto &[fdx, fish] : _individuals)
                process_vi(after_frame, fish->recognition_segments(), fdx, fish, assigned_ranges, virtual_fish, _apply);
        }
        
        //auto str = prettify_array(Meta::toStr(still_unassigned));
        print("auto_assign is ", auto_correct ? 1 : 0);
        if(auto_correct) {
            add_to_queue("", [after_frame, automatic_matches, manual_splits, tmp_assigned_ranges = std::move(tmp_assigned_ranges)]() mutable {
                print("Assigning to queue from frame ", after_frame);
                
                //std::lock_guard<decltype(GUI::instance()->gui().lock())> guard(GUI::instance()->gui().lock());
                
                {
                    Tracker::LockGuard guard(w_t{}, "check_segments_identities::auto_correct");
                    Tracker::instance()->_remove_frames(!after_frame.valid() ? Tracker::analysis_range().start : after_frame);
                    for(auto && [fdx, fish] : instance()->individuals()) {
                        fish->clear_recognition();
                    }
                    
                    print("automatically_assigned_ranges ", tmp_assigned_ranges.size());
                    AutoAssign::set_automatic_ranges(std::move(tmp_assigned_ranges));
                    
                    if(!after_frame.valid()) {
                        Settings::set<Settings::Variables::manual_matches>(automatic_matches);
                        Settings::set<Settings::Variables::manual_splits>(manual_splits);
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
