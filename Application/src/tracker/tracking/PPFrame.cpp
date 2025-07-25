#include "PPFrame.h"
#include <tracking/Tracker.h>
#include <tracking/CategorizeDatastore.h>
#include <misc/default_settings.h>
#include <file/DataLocation.h>
#include <misc/ThreadPool.h>

//#define TREX_DEBUG_BLOBS

namespace track {

#if TREX_ENABLE_HISTORY_LOGS
std::shared_ptr<std::ofstream>& PPFrame::history_log() {
    static std::shared_ptr<std::ofstream> history_log;
    return history_log;
}
LOGGED_MUTEX_TYPE& PPFrame::log_mutex() {
    static auto m = new LOGGED_MUTEX("PPFrame::log_mutex");
    return *m;
}
#endif

void PPFrame::UpdateLogs() {
#if TREX_ENABLE_HISTORY_LOGS
    if(history_log() == nullptr && !SETTING(history_matching_log).value<file::Path>().empty()) {
        history_log() = std::make_shared<std::ofstream>();
        
        auto path = SETTING(history_matching_log).value<file::Path>();
        if(!path.empty()) {
            path = file::DataLocation::parse("output", path);
            DebugCallback("Opening history_log at ", path, "...");
            //!TODO: CHECK IF THIS WORKS
            history_log()->open(path.str(), std::ios_base::out | std::ios_base::binary);
            if(history_log()->is_open()) {
                auto &ss = *history_log();
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
                warning { \
                color:rgb(193, 134, 23); display:inline; \
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
                line { \
                display: block; \
                padding-left: 1.5em; \
                text-indent: -1.5em; \
                } \
                string { display:inline; color: red; font-style: italic; }    \
                ref { display:inline; font-weight:bold; } ref:hover { color: gray; } \
                number,nr { display:inline; color: green; } \
                keyword { display:inline; color: purple; } \
                .body { \
                display: table-row-group; \
                }";
                
                ss <<"</style>";
                ss <<"</head><body>";
            }
        }
    }
#endif
}

void PPFrame::CloseLogs() {
#if TREX_ENABLE_HISTORY_LOGS
    auto guard = LOGGED_LOCK(log_mutex());
    if(history_log() != nullptr && history_log()->is_open()) {
        Print("Closing history log.");
        *history_log() << "</body></html>";
        history_log()->flush();
        history_log()->close();
    }
    history_log() = nullptr;
#endif
}

void PPFrame::write_log(std::string str) {
#if TREX_ENABLE_HISTORY_LOGS
    auto guard = LOGGED_LOCK(log_mutex());
    if(!history_log())
        return;
    
    const auto tname = get_thread_name();
    if(not utils::beginsWith(tname, "ConnectedTasks::stage_1_")
       && not utils::contains(tname, "tracking-thread"))
        return;
    
    str = "<line>[<warning>"+tname+"</warning>] "+ settings::htmlify(str) + "</line>";
    
    *history_log() << str << std::endl;
#else
    UNUSED(str);
#endif
}

#define ASSUME_NOT_FINALIZED _assume_not_finalized( __FILE__ , __LINE__ )

inline void insert_line(grid::ProximityGrid& grid, const HorizontalLine* ptr, pv::bid blob_id, ptr_safe_t step_size)
{
    auto d = ptr_safe_t(ptr->x1) - ptr_safe_t(ptr->x0);
    grid.insert(ptr->x0, ptr->y, blob_id);
    grid.insert(ptr->x1, ptr->y, blob_id);
    grid.insert(ptr->x0 + d * 0.5, ptr->y, blob_id);

    if(d >= step_size * 2 && step_size >= 5) {
        for(auto x = ptr_safe_t(ptr->x0) + step_size; x <= ptr_safe_t(ptr->x1) - step_size; x += step_size) {
            grid.insert(x, ptr->y, blob_id);
        }
    }
}

PPFrame::PPFrame(const Size2& size)
    : _resolution(size), _blob_grid(size)
    //: _blob_grid(Tracker::average().bounds().size())
{
}

std::string PPFrame::toStr() const {
    return "PPFrame<"+index().toStr()
        +" t:"+Meta::toStr(time)
        +" "+(_finalized ? "final":"")
        +" "+Meta::toStr(_blob_owner.size())+" blobs "
        +" "+Meta::toStr(_noise_owner.size())+" noise "
        +">";
}

const IndividualCache* PPFrame::cached(Idx_t id) const {
    auto it = _individual_cache.find(id);
    if(it != _individual_cache.end())
        return &it->second;
    return nullptr;
}

bool operator==(const pv::BlobPtr& blob, pv::bid bdx) {
    return blob ? blob->blob_id() == bdx : false;
}

bool PPFrame::has_bdx(pv::bid bdx) const noexcept {
    return bdx_to_ptr(bdx) != nullptr;
}

void PPFrame::init_cache(GenericThreadPool* pool, NeedGrid need)
{
    ASSUME_NOT_FINALIZED;
    
    Settings::manual_matches_t::mapped_type current_fixed_matches;
    {
        auto manual_matches = Settings::get<Settings::manual_matches>();
        auto it = manual_matches->find(index());
        if (it != manual_matches->end())
            fixed_matches = it->second;
    }
    
    assert(index().valid());
    if(index() == 0_f)
        return;
    
    const auto previous_frame = index() - 1_f;
    _individual_cache.clear();
    _previously_active_identities.clear();
    
    float tdelta;
    
    LockGuard guard(ro_t{}, "history_split#1");
    auto props = Tracker::properties(previous_frame);
    if(props == nullptr) {
        //! initial frame
        assert(previous_frame.valid());
        if(Tracker::start_frame().valid()
           && FrameRange(Range<Frame_t>(Tracker::start_frame(), Tracker::end_frame())).contains(previous_frame))
        {
            FormatWarning("Previous frame has already been processed: ", Range(Tracker::start_frame(), Tracker::end_frame()), " and previous:", previous_frame);
        }
        assert(not Tracker::start_frame().valid()
               or previous_frame < Tracker::start_frame()
               or previous_frame > Tracker::end_frame());
        return;
    }
    
    tdelta = time - props->time();
    assert(tdelta > 0);
    
    hints.push(previous_frame, props);
    hints.push(index(), Tracker::properties(index()));

    const auto last_active = Tracker::active_individuals(previous_frame);
    const auto N = last_active.size();
    _individual_cache.reserve(N);
    
    const auto max_d = FAST_SETTING(track_max_speed) * tdelta / FAST_SETTING(cm_per_pixel) * 0.5_F;
    const auto frame_limit = FAST_SETTING(frame_rate) * FAST_SETTING(track_max_reassign_time);
    
    const size_t num_threads = pool ? min(hardware_concurrency(), N / 200u) : 0;
    const auto space_limit = SQR(Individual::weird_distance() * 0.5);
    const auto frame_rate = FAST_SETTING(frame_rate);
    const auto track_max_reassign_time = FAST_SETTING(track_max_reassign_time);

    // mutex protecting count and global paired + fish_mappings/blob_mappings
    std::mutex mutex;
    std::condition_variable variable;
    size_t count = 0;
    const bool history_split = FAST_SETTING(track_do_history_split);

    auto fn = [&](auto i,
                  auto start_it,
                  auto end_it,
                  auto)
    {
        const auto start = i;
        const auto N = std::distance(start_it, end_it);
        using DistanceToBdx = map_t<pv::bid, Match::prob_t>::value_type;
        struct FishAssignments {
            Idx_t fdx;
            std::vector<DistanceToBdx> assign;
            std::vector<Vec2> last_pos;
        };
        struct BlobAssignments {
            UnorderedVectorSet<Idx_t> idxs;
        };

        std::vector<FishAssignments> fish_assignments(N);
        ska::bytell_hash_map<pv::bid, BlobAssignments> blob_assignments;
        PPFrame::cache_map_t cache_map;

        //auto it = active_individuals.begin();
        //std::advance(it, start);
        auto previous = Tracker::properties(index() - 1_f);
        
        //! go through individuals (for this pack/thread)
        for(auto it = start_it; it != end_it; ++i, ++it) {
            auto fish = *it;
            
            // IndividualCache is in the same position as the indexes here
            auto result = fish->cache_for_frame(previous, index(), time, &hints);
            if(not result)
                continue;
            
            auto& cache = cache_map[fish->identity().ID()];
            cache_map[fish->identity().ID()] = std::move(result.value());
            
            const auto time_limit = cache.previous_frame.get() - frame_limit; // dont allow too far to the past
            assert(cache.previous_frame.valid());
                
            // does the current individual have the frame previous to the current frame?
            //! try to find a frame thats close in time AND space to the current position
            size_t counter = 0;
            std::vector<Vec2> last_positions;
            Frame_t last_frame = cache.previous_frame;
            long_t last_L = -1;
            
            auto sit = fish->iterator_for(cache.previous_frame);
            if (sit != fish->tracklets().end() && (*sit)->contains(cache.previous_frame))
            {
                for (; sit != fish->tracklets().end()
                        && min((*sit)->end(), cache.previous_frame).get() >= time_limit
                        && counter < frame_limit; // shouldnt this be the same as the previous?
                    ++counter)
                {
                    auto range = arange<Frame_t>(
						time_limit > 0 ? max(Frame_t((uint32_t)time_limit), (*sit)->start()) : (*sit)->start(),
						(last_frame.valid() ? min((*sit)->end(), last_frame) : (*sit)->end()) + 1_f
                    );

                    const auto index = (*sit)->basic_stuff((*sit)->end());
                    const auto pos = fish->basic_stuff().at(index)->centroid.template pos<Units::DEFAULT>();
#ifndef NDEBUG
                    std::vector<Frame_t> considered;
					considered.reserve(range.size());
                    for(auto frame : range) {
                        if (not last_frame.valid() || frame > last_frame) {
                            //Print("Updating last frame for fish ", fish->identity(), " at ", pos, " to ", frame, " from ", last_frame);
                            break;
                        }
                        if (frame.get() < time_limit) {
                            //Print("Skipping fish ", fish->identity(), " at frame ", frame, " because it is before time limit ", time_limit);
                            continue;
                        }
                        considered.push_back(frame);
					}

                    /// traditional method
					std::vector<Frame_t> old_considered;
					old_considered.reserve(considered.size());

                    for (auto frame : (*sit)->iterable()) {
                        if (not last_frame.valid() || frame > last_frame) {
                            //Print("Updating last frame for fish ", fish->identity(), " at ", pos, " to ", frame, " from ", last_frame);
                            break;
                        }
                        if (frame.get() < time_limit) {
                            //Print("Skipping fish ", fish->identity(), " at frame ", frame, " because it is before time limit ", time_limit);
                            continue;
                        }

						old_considered.push_back(frame);
                    }

                    if(considered != old_considered) {
                        Print("Considered frames differ: ", considered, " vs. ", old_considered);
                        Print("Range: ", range.first, "-", range.last, " for fish ", fish->identity(), " at ", pos, " with last frame ", last_frame, " and time limit ", time_limit, " in range ", (*sit)->start(), "-", (*sit)->end());
					}
#endif
                    //Print("Would iterate ", max(0, int64_t(range.first.get()) - int64_t((*sit)->start().get())), " from ", (*sit)->start(), " to ", range.first, " - ", range.last);

                    for(auto frame : range) {
                        if (not last_frame.valid() || frame > last_frame) {
							//Print("Updating last frame for fish ", fish->identity(), " at ", pos, " to ", frame, " from ", last_frame);
                            break;
                        }
                        if (frame.get() < time_limit) {
							//Print("Skipping fish ", fish->identity(), " at frame ", frame, " because it is before time limit ", time_limit);
                            continue;
                        }
                        //Print("Looking at ", frame, " for fish ", fish->identity(), " at ", pos, " with last frame ", last_frame, " and time limit ", time_limit, " in range ", *sit);
                        
                        auto bindex = (*sit)->basic_stuff(frame);
                        auto pindex = (*sit)->posture_stuff(frame);
                        if(const MotionRecord* posture{nullptr};
                           pindex != -1 &&
                           (posture = fish->posture_stuff().at(pindex)->head.get()))
                        {
                            last_positions.emplace_back(posture->template pos<Units::DEFAULT>());
                            //last_pos = posture->template pos<Units::DEFAULT>();
                        } else {
                            last_positions.emplace_back(fish->basic_stuff().at(bindex)->centroid.template pos<Units::DEFAULT>());
                            //last_pos = fish->basic_stuff().at(bindex)->centroid.template pos<Units::DEFAULT>();
                        }
                    }

                    if ((*sit)->length().get() > frame_rate * track_max_reassign_time * 0.25)
                    {
                        //! tracklet is long enough, we can stop. but only actually use it if its not too far away:
                        if (last_positions.empty()
                            || sqdistance(pos, last_positions.back()) < space_limit)
                        {
                            last_frame = min((*sit)->end(), cache.previous_frame);
                            assert(last_frame.valid());
                            last_L = (last_frame - (*sit)->start()).get();
                        }
                        break;
                    }

                    if (sit != fish->tracklets().begin())
                        --sit;
                    else
                        break;
                }
            }
            
            if(last_frame.get() < time_limit) {
                Log("\tNot processing fish ", fish->identity()," because its last measured frame is ", last_frame,", best tracklet length is ", last_L," and we are in frame ", index(),".");
                
            } else {
                auto set = blob_grid().query(cache.estimated_px, max_d);
                
                if(!set.empty()) {
                    auto fdx = fish->identity().ID();
                    auto& map = fish_assignments[i - start];
                    map.fdx = fdx;
                    if(last_positions.empty()) {
                        map.last_pos = {cache.estimated_px};
                        //last_pos.x == -1 ? cache->estimated_px : last_pos;
                    } else {
                        map.last_pos = last_positions;
                        map.last_pos.emplace_back(cache.estimated_px);
                    }
                    
                    for(auto && [d, bdx] : set) {
                        if(!has_bdx(bdx))
                            continue;
                        
                        map.assign.push_back({bdx, d});
                        blob_assignments[bdx].idxs.insert(fdx);
                    }
                }
                
                Log("\tFish ", fish->identity()," (", cache.estimated_px.x, ",", cache.estimated_px.y, ") proximity: ", set);
            }
        }

        std::unique_lock guard(mutex);
        for(auto&& [fdx, cache] : cache_map)
            set_cache(fdx, std::move(cache));
        
        if(history_split) {
            for (auto&& [fdx, assign, last_pos] : fish_assignments) {
                //fish_mappings[fdx].insert(std::make_move_iterator(blobs.begin()), std::make_move_iterator(blobs.end()));
                auto N = assign.size();
                for(size_t i=0; i<N; ++i)
                    paired[fdx].insert(std::move(assign[i]));
                last_positions[fdx] = last_pos;
            }
            for (auto& [bdx, assign] : blob_assignments) {
                blob_mappings[bdx].insert(assign.idxs.begin(), assign.idxs.end());
            }
        }

        ++count;
        variable.notify_one();
    };
    
    //LockGuard guard(ro_t{}, "history_split#2");
    _previously_active_identities.reserve(N);
    for(auto fish : last_active)
        _previously_active_identities.push_back(fish->identity().ID());
    
    if(num_threads < 2 || !pool || N < num_threads) {
        fn(0, last_active.begin(), last_active.end(), N);
    } else if(N) {
        distribute_indexes(fn, *pool, last_active.begin(), last_active.end());
    }
    
    if(need == NeedGrid::NoNeed) {
        std::unique_lock guard(_blob_grid_mutex);
        _blob_grid.clear();
    }
}



void PPFrame::set_cache(Idx_t id, IndividualCache&& cache) {
    ASSUME_NOT_FINALIZED;
    //static std::mutex mutex;
    //std::unique_lock guard(mutex);
   // mutex.lock();
    _individual_cache[id] = std::move(cache);
    //mutex.unlock();
}

pv::bid PPFrame::_add_ownership(bool regular, pv::BlobPtr && blob) {
    assert(blob != nullptr);
    auto bdx = blob->blob_id();
    
    //! see if this blob is already part of the frame
    if(has_bdx(blob->blob_id())) {
#ifdef TREX_DEBUG_BLOBS
        auto blob1 = bdx_to_ptr(blob->blob_id());
        
        Print("Blob0 ", uint32_t(blob->bounds().x) & 0x00000FFF," << 24 = ", (uint32_t(blob->bounds().x) & 0x00000FFF) << 20," (mask ", (uint32_t(blob->lines()->front().y) & 0x00000FFF) << 8,", max=", std::numeric_limits<uint32_t>::max(),")");
        
        Print("Blob1 ", uint32_t(blob1->bounds().x) & 0x00000FFF," << 24 = ", (uint32_t(blob1->bounds().x) & 0x00000FFF) << 20," (mask ", (uint32_t(blob1->lines()->front().y) & 0x00000FFF) << 8,", max=", std::numeric_limits<uint32_t>::max(),")");
        
        auto bid0 = pv::bid::from_blob(*blob);
        auto bid1 = pv::bid::from_blob(*bdx_to_ptr(blob->blob_id()));
        
        FormatExcept("Frame ", _index,": Blob ", blob->blob_id()," already in map (", blob.get() == bdx_to_ptr(blob->blob_id()),"), at ",blob->bounds().pos()," bid=", bid0," vs. ", bdx_to_ptr(blob->blob_id())->bounds().pos()," bid=", bid1);
#endif
        return pv::bid::invalid;
    }
    
    //! update metadata
    _pixel_samples++;
    _num_pixels += blob->num_pixels();
    
    assert((not blob->pixels() && blob->is_binary()) || size_t(blob->num_pixels()) * size_t(blob->channels()) == blob->pixels()->size());
    //assert(blob->is_rgb() == (Background::image_mode() == ImageMode::RGB));
    
    //! add to the ownership vector and map
    //_bdx_to_ptr[bdx] = _owner.size();
    /*_owner.emplace_back(Container{
        .regular = regular,
        .blob = std::move(blob)
    });*/
#ifdef TREX_DEBUG_BLOBS
    Print(this->index(), " Added ", blob, " with regular=", regular);
#endif
    
    if(regular) {
        _blob_map[blob->blob_id()] = blob.get();
        _blob_owner.emplace_back(std::move(blob));
    } else {
        _noise_map[blob->blob_id()] = blob.get();
        _noise_owner.emplace_back(std::move(blob));
    }
    
    return bdx;
}

void PPFrame::_assume_not_finalized(const char* file, int line) {
#ifndef NDEBUG
    if(_finalized) {
        throw U_EXCEPTION("PPFrame already finalized @ [",file,":",line,"]. Finalized at ", _finalized_at.file_name(),":", _finalized_at.line(), " in function ", _finalized_at.function_name(), ".");
    }
#else
    UNUSED(file);
    UNUSED(line);
#endif
}

MaybeLabel PPFrame::label(const pv::bid& bdx) const {
#if !COMMONS_NO_PYTHON
    auto l = Categorize::DataStore::ranged_label(Frame_t(index()), bdx);
    if(l)
        return l->id;
#endif
    return {};
}

void PPFrame::add_noise(pv::BlobPtr && blob) {
    ASSUME_NOT_FINALIZED;
    //Print("Frame ",index()," has added 1 noise blobs.");
    _add_ownership(false, std::move(blob));
}

void PPFrame::add_noise(std::vector<pv::BlobPtr>&& v) {
    ASSUME_NOT_FINALIZED;
    
    //Print("Frame ",index()," has added ", v.size(), " noise blobs.");
    _noise_owner.reserve(_noise_owner.size() + v.size());
    
    for(auto it = std::make_move_iterator(v.begin());
        it.base() != v.end(); ++it)
    {
        _add_ownership(false, std::move(*it));
    }
    
    //_pixel_samples += v.size();
    v.clear();
    //_noise.insert(_noise.end(), std::make_move_iterator( v.begin() ), std::make_move_iterator( v.end() ));
}

void PPFrame::move_to_noise(size_t blob_index) {
    ASSUME_NOT_FINALIZED;
    assert(blob_index < _blob_owner.size());
    
    //Print("Frame ", index(), " moving ", blob_index, " to noise");
    
    // no update of pixels or maps is required
    auto ptr = _blob_owner.at(blob_index).get();
    _blob_map.erase(ptr->blob_id());
    _noise_map[ptr->blob_id()] = ptr;
    
    _noise_owner.insert(_noise_owner.end(), std::make_move_iterator(_blob_owner.begin() + blob_index), std::make_move_iterator(_blob_owner.begin() + blob_index + 1));
    _blob_owner.erase(_blob_owner.begin() + blob_index);
}

pv::BlobPtr PPFrame::extract(pv::bid bdx) {
    auto ptr = _extract_from(std::move(_blob_owner), bdx);
    if(!ptr)
        return _extract_from(std::move(_noise_owner), bdx);
    return ptr;
}

const grid::ProximityGrid& PPFrame::blob_grid() {
    std::scoped_lock guard(_blob_grid_mutex);
    if(_blob_grid.empty()) {
        // have to fill the grid
        if(_resolution.empty())
            throw U_EXCEPTION("Resolution not set at time of use.");
        fill_proximity_grid(_resolution);
    }
    
    return _blob_grid;
}

pv::BlobPtr PPFrame::_extract_from(std::vector<pv::BlobPtr>&& range, pv::bid bdx) {
    assert(bdx.valid());
    
    for(auto it = range.begin(); it != range.end(); ) {
        auto&& own = *it;
        if(!own) {
            ++it;
            continue;
        }
        
        if(own->blob_id() == bdx) {
            //! we found the blob, so remove it everywhere...
            {
                std::scoped_lock guard(_blob_grid_mutex);
                _blob_grid.erase(bdx);
            }
            
            _num_pixels -= own->num_pixels();
            _pixel_samples--;
            
        #ifdef TREX_DEBUG_BLOBS
            Print(this->index(), " Removing ", own->blob_id());
        #endif
            
            // move object out and delete
            auto object = std::move(own);
            it = range.erase(it);
            
            if(&_blob_owner == &range) {
                _blob_map.erase(object->blob_id());
            } else {
                _noise_map.erase(object->blob_id());
            }
            
            _check_owners();
            return object;
        } else
            ++it;
    }
    
    [[unlikely]];
#ifdef TREX_DEBUG_BLOBS
    Print("Cannot find ", bdx, " in _bdx_to_ptr");
#endif
    return nullptr;
}

pv::BlobPtr PPFrame::create_copy(pv::bid bdx) const {
    auto ptr = bdx_to_ptr(bdx);
    if(!ptr)
        return nullptr;
    return pv::Blob::Make(*ptr);
}

void PPFrame::add_regular(pv::BlobPtr&& blob) {
    ASSUME_NOT_FINALIZED;
    assert(blob != nullptr);
    _add_ownership(true, std::move(blob));
}

void PPFrame::add_regular(std::vector<pv::BlobPtr>&& v) {
    ASSUME_NOT_FINALIZED;
    
    _blob_owner.reserve(_blob_owner.size() + v.size());
    for(auto it = v.begin(); it != v.end(); ++it) {
        assert(*it != nullptr);
        _add_ownership(true, std::move(*it));
    }
    
    //_blobs.insert(_blobs.end(), std::make_move_iterator( v.begin() ), std::make_move_iterator( v.end() ));
}

bool PPFrame::is_regular(pv::bid bdx) const {
    return _blob_map.find(bdx) != _blob_map.end();
}

pv::BlobWeakPtr PPFrame::bdx_to_ptr(pv::bid bdx) const noexcept {
    auto it = _blob_map.find(bdx);
    if(it != _blob_map.end())
        return it->second;
    
    it = _noise_map.find(bdx);
    if(it != _noise_map.end())
        return it->second;
    
    return nullptr;
}

void PPFrame::set_tags(std::vector<pv::BlobPtr>&& tags) {
    _tags = std::move(tags);
}

void PPFrame::clear_blobs() {
    ASSUME_NOT_FINALIZED;
    
    _blob_map.clear();
    _noise_map.clear();
    _blob_owner.clear();
    _noise_owner.clear();
    _num_pixels = 0;
    _pixel_samples = 0;
    
    _check_owners();
}

void PPFrame::_check_owners() {
#if !defined(NDEBUG) && defined(TREX_DEBUG_BLOBS)
    for(auto& [bdx, ptr] : _blob_map) {
        auto it = std::find(_blob_owner.begin(), _blob_owner.end(), bdx);
        if(it == _blob_owner.end())
            throw U_EXCEPTION("Cannot find ", bdx, " in _blob_owner.");
    }
    
    for(auto& [bdx, ptr] : _noise_map) {
        auto it = std::find(_noise_owner.begin(), _noise_owner.end(), bdx);
        if(it == _noise_owner.end())
            throw U_EXCEPTION("Cannot find ", bdx, " in _blob_owner.");
    }
#endif
    
#ifndef NDEBUG
    /*size_t i=0;
    for(; i < _blob_owner.size(); ++i) {
        auto &o = _blob_owner.at(i);
        assert(o != nullptr);*/
        //assert(_bdx_to_ptr.at(o.blob->blob_id()) == i);
        /*assert((o.regular && std::find(_blobs.begin(), _blobs.end(), o.blob->blob_id()) != _blobs.end())
               || (!o.regular && std::find(_noise.begin(), _noise.end(), o.blob->blob_id()) != _noise.end()));*/
    //}
#endif
}

void PPFrame::add_blobs(std::vector<pv::BlobPtr>&& blobs,
                        std::vector<pv::BlobPtr>&& noise,
                        robin_hood::unordered_flat_set<pv::bid>&& big_ids,
                        size_t /*pixels*/,
                        size_t /*samples*/)
{
    ASSUME_NOT_FINALIZED;
    
    //assert(samples == blobs.size() + noise.size());
    //_num_pixels += pixels;
    //_pixel_samples += samples;
    
#ifdef TREX_DEBUG_BLOBS
    thread_print("Frame ", index(), " adding ", noise.size(), " noise and ", blobs.size(), " blobs (big ids =", big_ids,")");
#endif
    
    _big_ids = std::move(big_ids);
    _blob_owner.reserve(_blob_owner.size() + blobs.size());
    _noise_owner.reserve(_noise_owner.size() + noise.size());
    
    if(_blob_owner.empty()
       && _noise_owner.empty())
    {
        auto integrate_blob = [this](const pv::BlobPtr& blob){
            assert(blob != nullptr);
            //auto bdx = blob->blob_id();
            
            //! see if this blob is already part of the frame
            /*if(has_bdx(blob->blob_id())) {
        #ifdef TREX_DEBUG_BLOBS
                auto blob1 = bdx_to_ptr(blob->blob_id());
                
                Print("Blob0 ", uint32_t(blob->bounds().x) & 0x00000FFF," << 24 = ", (uint32_t(blob->bounds().x) & 0x00000FFF) << 20," (mask ", (uint32_t(blob->lines()->front().y) & 0x00000FFF) << 8,", max=", std::numeric_limits<uint32_t>::max(),")");
                
                Print("Blob1 ", uint32_t(blob1->bounds().x) & 0x00000FFF," << 24 = ", (uint32_t(blob1->bounds().x) & 0x00000FFF) << 20," (mask ", (uint32_t(blob1->lines()->front().y) & 0x00000FFF) << 8,", max=", std::numeric_limits<uint32_t>::max(),")");
                
                auto bid0 = pv::bid::from_blob(blob);
                auto bid1 = pv::bid::from_blob(*bdx_to_ptr(blob->blob_id()));
                
                FormatExcept("Frame ", _index,": Blob ", blob->blob_id()," already in map (", blob.get() == bdx_to_ptr(blob->blob_id()),"), at ",blob->bounds().pos()," bid=", bid0," vs. ", bdx_to_ptr(blob->blob_id())->bounds().pos()," bid=", bid1);
        #endif
                return pv::bid::invalid;
            }*/
            
            //! update metadata
            _pixel_samples++;
            _num_pixels += blob->num_pixels();
            
            assert((not blob->pixels() && blob->is_binary()) || size_t(blob->num_pixels()) * size_t(blob->channels()) == blob->pixels()->size());
            //assert(blob->is_rgb() == (Background::image_mode() == ImageMode::RGB));
            
        #ifdef TREX_DEBUG_BLOBS
            //Print(this->index(), " Added ", blob, " with regular=", blobs);
        #endif
        };
        
        for(auto &blob : blobs)
            integrate_blob(blob);
        
        for(auto& blob : blobs)
            _blob_map[blob->blob_id()] = blob.get();
            
        _blob_owner.insert(_blob_owner.end(), std::make_move_iterator(blobs.begin()), std::make_move_iterator(blobs.end()));
        
        for(auto &blob : noise)
            integrate_blob(blob);
        
        for(auto& blob: noise)
            _noise_map[blob->blob_id()] = blob.get();
        
        _noise_owner.insert(_noise_owner.end(), std::make_move_iterator(noise.begin()), std::make_move_iterator(noise.end()));
        
        blobs.clear();
        noise.clear();
        
        _check_owners();
        
        return;
    }
    
    for(auto it = blobs.begin(); it != blobs.end(); ++it) {
        _add_ownership(true, std::move(*it));
    }
    
    for(auto it = noise.begin(); it != noise.end(); ++it) {
        _add_ownership(false, std::move(*it));
    }
    
    noise.clear();
    blobs.clear();
    
    _check_owners();
}

void PPFrame::finalize(source_location loc) {
    ASSUME_NOT_FINALIZED;
    _finalized = true;
    _finalized_at = std::move(loc);
    _check_owners();
}

void PPFrame::unfinalize(source_location) {
    //ASSUME_FINALIZED;
    _finalized = false;
    _finalized_at = {};
    _check_owners();
}

void PPFrame::init_from_blobs(std::vector<pv::BlobPtr>&& vec) {
    ASSUME_NOT_FINALIZED;
    
    add_regular(std::move(vec));
    //_original_blobs = _blobs; // also save to copy to original.
    _check_owners();
}



void PPFrame::clear() {
    _finalized = false;
    _blob_owner.clear();
    _noise_owner.clear();
    _blob_map.clear();
    _noise_map.clear();
    _individual_cache.clear();
    {
        std::scoped_lock guard(_blob_grid_mutex);
        _blob_grid.clear();
    }
    hints.clear();
    //! TODO: original_blobs
    //_original_blobs.clear();
    //clique_for_blob.clear();
    //clique_second_order.clear();
    //split_blobs.clear();
    _num_pixels = 0;
    _pixel_samples = 0;
    //_split_objects = _split_pixels = 0;
    
    //fish_mappings.clear();
    blob_mappings.clear();
    paired.clear();
    last_positions.clear();
    fixed_matches.clear();
    _index.invalidate();
    _source_index.invalidate();
    
    // Clear additional fields
    _tags.clear();
    _big_ids.clear();
    _loading_time = 0;
    _previously_active_identities.clear();
    
    _check_owners();
}

bool PPFrame::has_fixed_matches() const {
    return !fixed_matches.empty();
}

void PPFrame::fill_proximity_grid(const Size2& size) {
    ASSUME_NOT_FINALIZED;
    
    /*if(!SETTING(gui_show_pixel_grid).value<bool>())
    {
        // do not need a blob_grid, so dont waste time here
        return;
    }*/
    _resolution = size;
    _blob_grid.set_resolution(size, grid::proximity_res);
    
    auto add_blob = [this](const pv::Blob& b) {
        auto N = b.hor_lines().size();
        auto ptr = b.hor_lines().data();
        const auto end = ptr + N;
        
        const ptr_safe_t step_size = 2;
        const ptr_safe_t step_size_x = (ptr_safe_t)max(1, b.bounds().width * 0.1);
        
        auto bdx = b.blob_id();
        assert(bdx.valid());
        
        if(N >= step_size * 2) {
            insert_line(_blob_grid, ptr, bdx, step_size_x);
            
            for(ptr = ptr + 1; ptr < end-1; ++ptr) {
                if(ptr->y % step_size == 0) {
                    insert_line(_blob_grid, ptr, bdx, step_size_x);
                }
            }
            
            insert_line(_blob_grid, end-1, bdx, step_size_x);
            
        } else {
            for(; ptr != end; ++ptr)
                insert_line(_blob_grid, ptr, bdx, step_size_x);
        }
    };
    
    transform_blobs(add_blob);
    transform_noise_ids(_big_ids, add_blob);
}

}
