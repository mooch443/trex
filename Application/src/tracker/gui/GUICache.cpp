#include "GUICache.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <gui/DrawFish.h>
#include <ml/Categorize.h>
#include <gui/DrawBase.h>
#include <tracking/IndividualManager.h>
#include <misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <gui/DrawPosture.h>
#include <tracking/FilterCache.h>
#include <misc/TimingStatsCollector.h>
#include <gui/Scene.h>
#include <gui/DrawGraph.h>
#include <tracking/TrackingHelper.h>

namespace cmn::gui {
    
std::unique_ptr<PPFrame> GUICache::PPFrameMaker::operator()() const {
    return std::make_unique<PPFrame>();
}

    GUICache*& cache() {
        static GUICache* _cache{ nullptr };
        return _cache;
    }

    GUICache& GUICache::instance() {
        if (!cache())
            throw U_EXCEPTION("No cache created yet.");
        return *cache();
    }

    bool GUICache::exists() {
        return cache() != nullptr;
    }

    GUICache::GUICache(DrawStructure* graph, std::weak_ptr<pv::File> video)
        : _pool(saturate(cmn::hardware_concurrency(), 1u, 5u), "GUICache::_pool"),
            _current_processed_frame(std::make_unique<PPFrame>()),
            _video(video), _graph(graph),
            _preloader([this](Frame_t frameIndex) -> FramePtr {
                FramePtr ptr;
                auto video = _video.lock();
                if(not video)
                    return nullptr;
                
                try {
                    if(frameIndex.valid()
                       && video->is_read_mode())
                    {
                        if(frameIndex >= video->length())
                            return nullptr; // past the end
                        
                        pv::Frame frame;
                        video->read_with_encoding(frame, frameIndex, Background::meta_encoding());
                        
                        ptr = buffers.get(source_location::current());
                        ptr->clear();
                        
                        Tracker::preprocess_frame(std::move(frame), *ptr, &_pool, PPFrame::NeedGrid::Need, video->header().resolution);
                    }
                    
                } catch(...) {
                    FormatExcept("Cannot load frame ", frameIndex, " from file ", video->filename());
                }
                
                return ptr;
            },
            [this](FramePtr&& ptr) {
                buffers.move_back(std::move(ptr));
            }, TimingMetric_t::PVRequest, TimingMetric_t::PVLoad)

    {
        cache() = this;
        globals::Cache::init();
        globals::CachedGUIOptions::init();
        
        _settings_callback = GlobalSettings().map().register_callbacks({"gui_fish_label"}, [this](auto)
        {
            std::unique_lock g{label_mutex};
            _label_text = SETTING(gui_fish_label).value<std::string>();
            _prepared_label_text.reset();
        });
    }

    GUICache::~GUICache() {
        GlobalSettings().map().unregister_callbacks(std::move(_settings_callback));
        
        if(_delete_frame_callback) {
            LockGuard guard(ro_t{}, "Delete Frame Callback Delete");
            if(Tracker::instance()) {
                Tracker::instance()->unregister_delete_callback(*_delete_frame_callback);
            }
            _delete_frame_callback.reset();
        }
        
        _fish_map.clear();
        display_blobs.clear();
        raw_blobs.clear();
        available_blobs_list.clear();
        
        if(_next_tracklet.valid())
            _next_tracklet.get();

        std::lock_guard guard(percentile_mutex);
        if(percentile_ptr) {
            percentile_ptr->join();
            percentile_ptr = nullptr;
        }

        cache() = nullptr;
        
        _pool.force_stop();
    }

    SimpleBlob::SimpleBlob(std::unique_ptr<ExternalImage>&& available, pv::BlobWeakPtr b, int t)
        : blob(b), threshold(t), ptr(std::move(available))
    {
        assert(ptr);
        if (!ptr->source()) {
            ptr->set_source(Image::Make());
        }
        ptr->set_cut_border(true);
    }
    
    void SimpleBlob::convert() {
        assert(blob != nullptr);
        
        auto &percentiles = GUICache::instance().pixel_value_percentiles;
        OutputInfo output {
            .channels = 4u,
            .encoding = meta_encoding_t::rgb8
        };
        
        if(FAST_SETTING(meta_encoding) == meta_encoding_t::gray
           || FAST_SETTING(track_background_subtraction))
        {
            if (GUICache::instance()._equalize_histograms && !percentiles.empty()) {
                image_pos = blob->equalized_luminance_alpha_image(*Tracker::background(), threshold, percentiles.front(), percentiles.back(), ptr->unsafe_get_source(), 0, output);
                
            } else {
                image_pos = blob->luminance_alpha_image(*Tracker::background(), threshold, ptr->unsafe_get_source(), 0, output);
            }
            
        } else {
            image_pos = blob->rgba_image(*Tracker::background(), threshold, ptr->unsafe_get_source(), 0);
        }

        /*if(Background::meta_encoding() == meta_encoding_t::r3g3b2) {
            if(not ptr->empty()) {
                auto mat = ptr->unsafe_get_source().get();
                cv::Mat output;
                convert_from_r3g3b2<4,2>(mat, output);
                //Print("converted ", mat.channels(), " to ", output.channels());
                //tf::imshow("output", output);
                //ptr->unsafe_get_source().create(output);
                
                /// dangling source change in order to *NOT* trigger a parent
                /// update yet since we are using this in parallel and dont want
                /// to produce undefined behavior / data races.
                ptr->unsafe_get_source().create(output);
                //ptr->set_source(Image::Make(output));
            }
        }*/
        
        //ptr->set_pos(image_pos);
        //ptr->updated_source();
        
        ptr->add_custom_data("blob_id", (void*)(uint64_t)(uint32_t)blob->blob_id());
        ptr->add_custom_data("frame", (void*)(uint64_t)frame.get());
        if(ptr->name().empty())
            ptr->set_name("SimpleBlob_"+Meta::toStr(blob->blob_id()));
        
        //blob = nullptr;
    }
    
    bool GUICache::has_selection() const {
        std::unique_lock guard(individuals_mutex);
        return !selected.empty() && individuals.count(selected.front()) != 0;
    }
    
    Individual * GUICache::primary_selection() const {
        std::unique_lock guard(individuals_mutex);
        return has_selection() && individuals.count(selected.front())
                ? individuals.at(selected.front())
                : nullptr;
    }

    Idx_t GUICache::primary_selected_id() const {
        return has_selection() ? selected.front() : Idx_t();
    }
    
    void GUICache::set_dt(float dt) {
        _dt = dt;
        _gui_time += dt;
    }
    
    void GUICache::deselect_all() {
        if(!selected.empty()) {
            selected.clear();
            SETTING(gui_focus_group) = selected;
            SETTING(heatmap_ids) = std::vector<Idx_t>();
        }
    }
    
    bool GUICache::is_selected(Idx_t id) const {
        return contains(selected, id);
    }
    
    void GUICache::do_select(Idx_t id) {
        if(!is_selected(id)) {
            selected.push_back(id);
            SETTING(gui_focus_group) = selected;
            SETTING(heatmap_ids) = std::vector<Idx_t>(selected.begin(), selected.end());
        }
    }
    
    void GUICache::deselect(Idx_t id) {
        auto it = std::find(selected.begin(), selected.end(), id);
        if(it != selected.end()) {
            selected.erase(it);
            SETTING(gui_focus_group) = selected;
            SETTING(heatmap_ids) = std::vector<Idx_t>(selected.begin(), selected.end());
        }
    }
    
    void GUICache::deselect_all_select(Idx_t id) {
        selected.clear();
        selected.push_back(id);
        
        SETTING(gui_focus_group) = selected;
        SETTING(heatmap_ids) = std::vector<Idx_t>(selected.begin(), selected.end());
    }
    
    void GUICache::set_tracking_dirty() {
        _tracking_dirty = true;
    }

    void GUICache::set_reload_frame(Frame_t frameIndex) {
        _do_reload_frame = frameIndex;
        
        if(_do_reload_frame.valid()
           && _next_processed_frame
           && _do_reload_frame == _next_processed_frame->index())
        {
            buffers.move_back(std::move(_next_processed_frame));
            
            auto frame = _preloader.get_frame(frameIndex, _preloader.last_increment());
            if(frame.has_value() && frame.value()->index() == frameIndex) {
                _next_processed_frame = std::move(frame.value());
            }
        }
    }
    
    void GUICache::set_blobs_dirty() {
        _blobs_dirty = true;
    }
    
    void GUICache::set_raw_blobs_dirty() {
        _raw_blobs_dirty = true;
    }
    
    void GUICache::set_redraw() {
        //if(GUI::instance())
        //    GUI::instance()->gui().set_dirty(nullptr);
        _dirty = true;
    }
    
    void GUICache::set_mode(const mode_t::Class& mode) {
        if(mode != _mode) {
            _mode = mode;
            
            if(mode == mode_t::blobs)
                set_blobs_dirty();
            else if(mode == mode_t::tracking)
                set_tracking_dirty();
            set_raw_blobs_dirty();
        }
    }
    
    bool GUICache::must_redraw() const {
        if(raw_blobs_dirty() || _dirty || (_mode == mode_t::tracking && _tracking_dirty) || (_mode == mode_t::blobs && _blobs_dirty))
            return true;
        return false;
    }

void GUICache::request_frame_change_to(Frame_t frame) {
    _preloader.announce(frame);
}

const grid::ProximityGrid& GUICache::blob_grid() {
    if(not _current_processed_frame)
        throw InvalidArgumentException("No preprocessed frame available.");
    return _current_processed_frame->blob_grid();
}

bool GUICache::something_important_changed(Frame_t frameIndex) const {
    const auto threshold = FAST_SETTING(track_threshold);
    //const auto posture_threshold = FAST_SETTING(track_posture_threshold);
    //auto& _tracker = *Tracker::instance();
    //auto& _gui = *_graph;
    
    return  not processed_frame().index().valid()
            || frameIndex != frame_idx
            || frameIndex != processed_frame().index()
            || not _current_processed_frame
            || last_threshold != threshold
            || selected != previous_active_fish
            || active_blobs != previous_active_blobs
            //|| _gui.mouse_position() != previous_mouse_position
            || is_tracking_dirty()
            || raw_blobs_dirty()
            //|| _blobs_dirty
            || _frame_contained != tracked_frames.contains(frameIndex)
            || _do_reload_frame.valid();
}



void GUICache::draw_posture(DrawStructure &base, Frame_t) {
    static Timing timing("posture draw", 0.1);
    TakeTiming take(timing);
    if(not _posture_window)
        return;
    
    if(_posture_window->valid())
        base.wrap_object(*_posture_window);
}

std::optional<std::vector<Range<Frame_t>>> GUICache::update_slow_tracker_stuff() {
    if(bool compared = false;
       _last_consecutive_update.elapsed() > (GUI_SETTINGS(track_pause) ? 1.0 : 10.0)
       && _updating_consecutive.compare_exchange_strong(compared, true))
    {
        _next_tracklet = std::async(std::launch::async, [](){
            return Tracker::instance()->global_tracklet_order();
        });
    }
    
    if(_next_tracklet.valid()
       && _next_tracklet.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
    {
        _global_tracklet_order = _next_tracklet.get();
        _last_consecutive_update.reset();
        _updating_consecutive = false;
        
        return _global_tracklet_order;
    }
    
    return std::nullopt;
}
    
    Frame_t GUICache::update_data(const Frame_t frameIndex) {
        const auto threshold = FAST_SETTING(track_threshold);
        const bool output_normalize_midline_data = SETTING(output_normalize_midline_data);
        //const auto posture_threshold = FAST_SETTING(track_posture_threshold);
        auto& _gui = *_graph;
        _equalize_histograms = GUI_SETTINGS(gui_equalize_blob_histograms);
        
        /*std::vector<std::string> reasons;
        if(last_threshold != threshold)
            reasons.emplace_back("detect_threshold");
        if(raw_blobs_dirty())
            reasons.emplace_back("raw_blobs_dirty");*/
        
        _fish_dirty = false;
        //if(not _background)
        if (Tracker::instance()) {
            if (Tracker::background() != _background) {
                _background = Tracker::background();
                _border = Tracker::instance()->border();
            }
        }
        
        if(frameIndex != _do_reload_frame)
            _do_reload_frame.invalidate();
        
        bool current_frame_matches = _current_processed_frame
            && _current_processed_frame->index() == frameIndex && not _do_reload_frame.valid();
        bool next_frame_matches = _next_processed_frame
            && _next_processed_frame->index() == frameIndex;
        
        /*if(not current_frame_matches)
        {
            reasons.emplace_back("No matching next_frame found current="+Meta::toStr(current_frame_matches)+" next="+Meta::toStr(next_frame_matches)+". frame="+Meta::toStr(frameIndex)+"  current="+Meta::toStr(_current_processed_frame)+" next="+Meta::toStr(_next_processed_frame));
        }*/
        
        bool reload_blobs = not current_frame_matches
                            || frame_idx != frameIndex
                            || last_threshold != threshold
                            || _do_reload_frame.valid();
                            //|| raw_blobs_dirty();
        
        if(not current_frame_matches) {
            if(_next_processed_frame
               && _last_success.elapsed() < 0.1
               && not next_frame_matches)
            {
                /// we dont have a timeout, and we have a next frame
                /// but it doesnt match the requested frame:
                /// discard it.
                buffers.move_back(std::move(_next_processed_frame));
            }
            
            if(not next_frame_matches) {
                /// the next frame does *not* match - at least should
                /// nudge the preloader:
                auto maybe_frame = _load_frames_blocking //&& _mistakes_count >= 1u
                    ? _preloader.load_exactly(frameIndex, 1_f)
                    : _preloader.get_frame(frameIndex, 1_f, std::chrono::milliseconds(10));
                _preloader.notify();
                // _preloader.get_frame(frameIndex, Frame_t(saturate(static_cast<uint8_t>(GUI_SETTINGS(gui_playback_speed)), 1, 255)), std::chrono::milliseconds(50));
                
                //if(maybe_frame.has_value())
                //    timer.reset();
                
                if(not maybe_frame.has_value()
                   && not _next_processed_frame)
                {
                    /// we tried getting an image, but we didnt get any.
                    /// we also have no next frame ready.
                    /// means we cant do anything.
                    return {};
                }
                
                if(maybe_frame.has_value()) {
                    /// reset next processed frame since we have a more
                    /// up-to-date version here:
                    if(_next_processed_frame)
                        buffers.move_back(std::move(_next_processed_frame));
                    
                    if(maybe_frame.value()->index() != frameIndex
                        && _last_success.elapsed() < 0.1)
                    {
                        /// we dont have a timeout, and the indexes
                        /// dont match. discard and return...
                        buffers.move_back(std::move(maybe_frame.value()));
                        return {};
                        
                    } else {
                        /// we have a timeout, se just use what we have:
                        //frameIndex = maybe_frame.value()->index();
                        if(frameIndex != maybe_frame.value()->index()) {
                            //Print("Using maybe_frame anyway for ", maybe_frame.value()->index(), " != ", frameIndex, " since we waited ", _last_success.elapsed());
                            
                            buffers.move_back(std::move(maybe_frame.value()));
                            return {};
                        } else {
                            /// got correct frameIndex
                            //Print("Got frameIndex ", maybe_frame.value()->index()," (", frameIndex, ")");
                        }
                    }
                    
                    /// if we reached this point, use the maybe_frame value:
                    _next_processed_frame = std::move(maybe_frame.value());
                    
                } else {
                    /// we got nothing. return.
                    return {};
                }
            }
            
            if(not _next_processed_frame)
                throw InvalidArgumentException("Frame returned for ", frameIndex, " was null.");
            
            auto source_index = _next_processed_frame->source_index();
            SETTING(gui_source_video_frame) = source_index.valid()
                                                ? source_index
                                                : frameIndex;
        }
        
        if(reload_blobs
           || selected != previous_active_fish
           || active_blobs != previous_active_blobs
           //|| _gui.mouse_position() != previous_mouse_position
           || is_tracking_dirty()
           //|| _blobs_dirty
           || _raw_blobs_dirty
           || _dirty)
        {
            
        } else 
            return {};
        
        //Print("reload_blobs = ", reload_blobs, " current_frame_matches=", current_frame_matches, " next_frame_matches=", next_frame_matches, " last_threshold=", last_threshold, " threshold=", threshold, " raw_blobs_dirty=", raw_blobs_dirty(), " frameIndex=", frameIndex, " current=", _current_processed_frame ? _current_processed_frame->index() : Frame_t{}, " next=", _next_processed_frame ? _next_processed_frame->index() : Frame_t{}, " selected=", selected, " previous_active_fish=", previous_active_fish, " active_blobs=", active_blobs, " previous_active_blobs=", previous_active_blobs, " mouse_position=", _gui.mouse_position(), " previous_mouse_position=", previous_mouse_position, " is_tracking_dirty=", is_tracking_dirty(), " _blobs_dirty=", _blobs_dirty, " _dirty=",_dirty);
        
        LockGuard guard(ro_t{}, "update_cache", 10);
        if(not guard.locked())
            return {};
        
        auto stats = TimingStatsCollector::getInstance();
        auto handle = stats->startEvent(TimingMetric_t::FrameRender, frameIndex);
        
        TimingStatsCollector::HandleGuard _handleguard(stats, handle);
        
        auto& _tracker = *Tracker::instance();
        frame_idx = frameIndex;
        
        if(not _delete_frame_callback) {
            _delete_frame_callback = _tracker.register_delete_callback([this](){
                {
                    /// make sure that the gui elements are actually deleted in the
                    /// gui thread, not in the callback thread
                    std::unique_lock guard{_fish_map_mutex};
                    SceneManager::enqueue([fish_map = std::move(_fish_map)]() mutable {
                        fish_map.clear();
                    });
                    assert(_fish_map.empty());
                }
                
                {
                    std::unique_lock guard(_tracklet_cache_mutex);
                    _processed_tracklet_caches.clear();
                    _tracklet_caches.clear();
                }
                
                {
                    std::unique_lock guard(_next_frame_cache_mutex);
                    _next_frame_caches.clear();
                }
            });
        }
        
        {
            auto lock = _video.lock();
            if(lock)
                _video_resolution = lock->size();
        }
        
        if(not GUI_SETTINGS(nowindow)) {
            //! Calculate average pixel values. This is not a high-priority action, especially if the GUI is disabled. Only used for `gui_equalize_blob_histograms`.
            std::call_once(_percentile_once, [this](){
                percentile_ptr = std::make_unique<std::thread>([this](){
                    cmn::set_thread_name("percentile_thread");
                    auto video = _video.lock();
                    if(not video)
                        return; // abort! video does not exist
                    
                    Print("open for writing: ", video->is_write_mode());
                    if(video->is_read_mode()) {
                        auto percentiles = video->calculate_percentiles({0.05f, 0.95f});
                        
                        if(_graph) {
                            //auto guard = GUI_LOCK(_graph->lock());
                            pixel_value_percentiles = percentiles;
                        }
                    }
                    
                    done_calculating = true;
                });
            });
            
            
            if(done_calculating) {
                std::lock_guard guard(percentile_mutex);
                if(percentile_ptr) {
                    percentile_ptr->join();
                    percentile_ptr = nullptr;
                }
                done_calculating = false; // reset, so we dont have to check again
            }
        }
        
        if(_statistics.size() < _tracker.statistics().size()) {
            auto start = _tracker.statistics().end();
            std::advance(start, (int64_t)_statistics.size() - (int64_t)_tracker.statistics().size());
            
            for (; start != _tracker.statistics().end(); ++start)
                _statistics[start->first] = start->second;
            
        } else if(_statistics.size() > _tracker.statistics().size()) {
            auto start = _statistics.begin();
            std::advance(start, (int64_t)_tracker.statistics().size());
            _statistics.erase(start, _statistics.end());
        }
        
        auto properties = _tracker.properties(frameIndex);
        if(properties)
            _props = *properties;
        else
            _props.reset();
        
        auto next_properties = _tracker.properties(frameIndex + 1_f);
        if(next_properties)
            _next_props = *next_properties;
        else
            _next_props.reset();
        
        active_blobs.clear();
        active.clear();
        selected_blobs.clear();
        inactive_ids.clear();
        active_ids.clear();
        fish_selected_blobs.clear();
        inactive_estimates.clear();
        blob_selected_fish.clear();
        
        while(fish_last_bounds.size() > 10000) {
            fish_last_bounds.erase(fish_last_bounds.begin());
        }
        
        if(properties) {
            active = _tracker.active_individuals(frameIndex);
            {
                std::unique_lock guard(individuals_mutex);
                individuals = IndividualManager::copy();
            }
            selected = SETTING(gui_focus_group).value<std::vector<Idx_t>>();
            tracked_frames = Range<Frame_t>(_tracker.start_frame(), _tracker.end_frame());
            
            auto delete_callback = [this](Individual* fish) {
                //if(!cache() || !_graph)
                //    return;
                
                //auto guard = GUI_LOCK(_graph->lock());
                
                auto id = fish->identity().ID();
                
                {
                    std::unique_lock guard(individuals_mutex);
                    auto it = individuals.find(id);
                    if(it != individuals.end())
                        individuals.erase(it);
                    
                    auto cit = _registered_callback.find(fish);
                    if(cit != _registered_callback.end())
                        _registered_callback.erase(cit);
                }
                
                /*active.clear();
                
                auto kit = active_ids.find(id);
                if(kit != active_ids.end())
                    active_ids.erase(kit);
                
                kit = inactive_ids.find(id);
                if(kit != inactive_ids.end())
                    inactive_ids.erase(kit);
                
                auto bit = fish_selected_blobs.find(id);
                if(bit != fish_selected_blobs.end())
                    fish_selected_blobs.erase(bit);
                
                */
            };
            //_individual_ranges.clear();
            all_ids.clear();
            
            IndividualManager::transform_all([&](auto idx, Individual* fish){
                if(std::unique_lock guard(individuals_mutex);
                   !contains(_registered_callback, fish))
                {
                    fish->register_delete_callback((void*)12341337, delete_callback);
                    _registered_callback.insert(fish);
                }
                
                auto &ranges = _individual_ranges[idx];
                ranges.clear();
                for(auto& tracklet : fish->tracklets()) {
                    ranges.emplace_back(ShadowTracklet{
                        .frames = *tracklet,
                        .error_code=tracklet->error_code
                    });
                }
                all_ids.insert(idx);
            });
            
            for(auto it = _individual_ranges.begin(); it != _individual_ranges.end();) {
                if(not all_ids.contains(it->first)) {
                    it = _individual_ranges.erase(it);
                } else
                    ++it;
            }
            
            auto connectivity_map = SETTING(gui_connectivity_matrix).value<std::map<long_t, std::vector<float>>>();
            if(connectivity_map.count(frameIndex.get()))
                connectivity_matrix = connectivity_map.at(frameIndex.get());
            else
                connectivity_matrix.clear();
            
            double time = properties ? properties->time() : 0;
            std::mutex map_mutex;
            
            distribute_indexes([this, &map_mutex, frameIndex, output_normalize_midline_data](int64_t, auto start, auto end, int64_t){
                for(auto it = start; it != end; ++it) {
                    auto fish = *it;
                    Range<Frame_t> tracklet_range;
                    std::shared_ptr<track::constraints::FilterCache> filters;
                    
                    auto tracklet = fish->tracklet_for(frameIndex);
                    BasicStuff* basic{nullptr};
                    PostureStuff* posture{nullptr};
                    
                    if(tracklet) {
                        auto basic_index = tracklet->basic_stuff(frameIndex);
                        auto posture_index = tracklet->posture_stuff(frameIndex);
                        basic = basic_index != -1 ? fish->basic_stuff().at(basic_index).get() : nullptr;
                        posture = posture_index != -1 ? fish->posture_stuff().at(posture_index).get() : nullptr;
                    }
                    
                    //if(fish->identity().ID() == primary_selected_id())
                    {
                        if(tracklet) {
                            filters = constraints::local_midline_length(fish, tracklet->range);
                            tracklet_range = tracklet->range;
                        }
                    }
                    
                    if(basic) {
                        BdxAndPred blob{
                            .bdx = basic->blob.blob_id(),
                            .basic_stuff = *basic,
                            .automatic_match = fish->is_automatic_match(frameIndex),
                            .tracklet = tracklet_range
                        };
                        if(posture) {
                            blob.posture_stuff = posture->clone();
                            
                            /// this could be optimized by using the posture stuff
                            /// in the fixed midline function + SETTING()
                            blob.midline = output_normalize_midline_data ? fish->fixed_midline(frameIndex) : fish->calculate_midline_for(*posture);
                        }
                        
                        std::unique_lock g{map_mutex};
                        active_blobs.insert(blob.bdx);
                        active_ids.insert(fish->identity().ID());
                        blob_selected_fish[blob.bdx] = fish->identity().ID();
                        fish_last_bounds[fish->identity().ID()] = basic->blob.calculate_bounds();
                        fish_selected_blobs[fish->identity().ID()] = std::move(blob);
                        
                        /// dont forget the filters
                        if(filters)
                            filter_cache[fish->identity().ID()] = std::move(filters);
                        
                    } else {
                        /// try a slightly more efficient way of getting the basic stuff
                        /// for the previous frame (in order to allow the gui to quickly
                        /// switch selected individuals if necessary, even if not all
                        /// individuals are currently visible)
                        const BasicStuff *pstuff{nullptr};
                        if(tracklet && tracklet->contains(frameIndex - 1_f)) {
                            auto index = tracklet->basic_stuff(frameIndex - 1_f);
                            if(index != -1)
                                pstuff = fish->basic_stuff().at(index).get();
                            
                        }
                        
                        if(not pstuff) {
                            pstuff = fish->find_frame(frameIndex - 1_f);
                        }
                        
                        std::unique_lock g{map_mutex};
                        inactive_ids.insert(fish->identity().ID());
                        if(pstuff) {
                            fish_last_bounds[fish->identity().ID()] = pstuff->blob.calculate_bounds();
                            
                        } else if(auto fit = fish_last_bounds.find(fish->identity().ID());
                                fit != fish_last_bounds.end())
                        {
                            fish_last_bounds.erase(fit);
                        }
                        
                        /// dont forget the filters
                        if(filters)
                            filter_cache[fish->identity().ID()] = std::move(filters);
                        
                        if(pstuff)
                            active_blobs.insert(pstuff->blob.blob_id());
                    }
                }
            }, pool(), active.begin(), active.end());
            
            if(has_selection()) {
                auto previous = Tracker::properties(frameIndex - 1_f);
                auto pid = selected.empty() ? Idx_t() : selected.front();
                
                std::unique_lock guard(individuals_mutex);
                if(individuals.contains(pid)) {
                    if(not _posture_window) {
                        _posture_window = std::make_unique<gui::Posture>();
                        
                        //_posture_window->set_scale(base.scale().reciprocal());
                        //auto coords = FindCoord::get();
                        //_posture_window->set_pos(Vec2(coords.screen_size().width - 600, 150));
                        _posture_window->set_origin(Vec2(1, 0));
                        //_posture_window.set_fish(fish);
                        //_posture_window->set_frameIndex(frameNr);
                        _posture_window->set_draggable();
                    }
                    _posture_window->set_fish(individuals.at(pid), frameIndex);
                }
                
                IndividualCache _tmp;
                for(auto id : selected) {
                    if(individuals.count(id)) {
                        auto fish = individuals.at(id);
                        if(!fish->has(frameIndex) && !fish->empty() && frameIndex >= fish->start_frame()) {
                            auto c = fish->cache_for_frame(previous, frameIndex, time);
                            if(c) {
                                inactive_estimates.push_back(c.value().estimated_px);
                                //inactive_ids.insert(fish->identity().ID());
                            } else {
                                FormatError("Cannot create cache_for_frame of ", fish->identity(), " in frame ", frameIndex," because: ", c.error());
                            }
                        }
                    }
                }
            }
            
            if(!has_selection() || !SETTING(gui_auto_scale_focus_one)) {
                selected_blobs = active_blobs;
            } else {
                // display blobs that are selected
                std::unique_lock guard(individuals_mutex);
                for(auto id : selected) {
                    auto it = individuals.find(id);
                    if(it != individuals.end()) {
                        auto blob = it->second->compressed_blob(frameIndex);
                        if(blob)
                            selected_blobs.insert(blob->blob_id());
                    }
                }
            }
            
        }
        
        if(not something_important_changed(frameIndex))
            return {};
        
        bool contained = tracked_frames.contains(frameIndex);
        if(contained != _frame_contained) {
            _frame_contained = contained;
            _fish_dirty = true;
            //Print("frameIndex ", frameIndex, " contained=",contained);
        }
        
        if((_current_processed_frame && _current_processed_frame->index() != frameIndex) || _do_reload_frame.valid()) {
            buffers.move_back(std::move(_current_processed_frame));
            
            //Print("current_processed_frame moved out for ", frameIndex);
        } else if(_current_processed_frame) {
            reload_blobs = false;
            //reasons.emplace_back("-");
            //Print("current_processed_frame is fine for ", frameIndex, " = ", _current_processed_frame->index());
            if(_next_processed_frame)
                buffers.move_back(std::move(_next_processed_frame));
        }
        
        if(_next_processed_frame) {
            if(not reload_blobs) {
                reload_blobs = true;
                //reasons.emplace_back("next_processed_frame was != nullptr");
            }
            //Print("current_processed_frame moved out for ", frameIndex, " = ", _next_processed_frame->index());
            if(_current_processed_frame)
				buffers.move_back(std::move(_current_processed_frame));
            
            _current_processed_frame = std::move(_next_processed_frame);
            assert(_current_processed_frame->index() == frameIndex);
            
            if(_do_reload_frame.valid()
               && _do_reload_frame == _current_processed_frame->index())
            {
                _do_reload_frame = {};
            }
        } else {
            //Print("No next frame: ", _do_reload_frame, " @ ", frameIndex);
        }
        
        //Print("reload_blobs = ", reload_blobs);
        
        if(reload_blobs
           || selected != previous_active_fish
           || active_blobs != previous_active_blobs) 
        {
            set_tracking_dirty();
        }
        
        _global_tracklet_order = _tracker.unsafe_global_tracklet_order();
        previous_active_fish = selected;
        previous_active_blobs = active_blobs;
        previous_mouse_position = _gui.mouse_position();
        
        //set_blobs_dirty();
        
        Vec2 min_vec(FLT_MAX, FLT_MAX);
        Vec2 max_vec(-FLT_MAX, -FLT_MAX);
        
        //if(active_blobs.empty()) {
        for(auto &pos : inactive_estimates) {
            min_vec = min(min_vec, pos);
            max_vec = max(max_vec, pos + Vec2(1));
        }
        //}
        
        const bool nothing_to_zoom_on = !has_selection() || (inactive_estimates.empty() && selected_blobs.empty());
        
        _num_pixels = 0;
        auto L = processed_frame().number_objects();
        
        if(reload_blobs) {
            probabilities.clear();
            checked_probs.clear();
            display_blobs.clear();
            
            //Print("Reloading blobs ", frameIndex);
            if(L < raw_blobs.size()) {
                std::move(raw_blobs.begin() + L, raw_blobs.end(), std::back_inserter(available_blobs_list));
                raw_blobs.erase(raw_blobs.begin() + L, raw_blobs.end());
                
            } else if(L != raw_blobs.size()) {
                raw_blobs.reserve(L);
            }
        } //else
            //Print("Not reloading blobs ", frameIndex);
        
        //! count the actual number of objects
        size_t i = 0;
        
        processed_frame().transform_blobs([&](pv::Blob& blob) {
            if(nothing_to_zoom_on || selected_blobs.find(blob.blob_id()) != selected_blobs.end())
            {
                min_vec = min(min_vec, blob.bounds().pos());
                max_vec = max(max_vec, blob.bounds().pos() + blob.bounds().size());
            }
            
            _num_pixels += size_t(blob.bounds().width * blob.bounds().height);
            
            if(reload_blobs) {
                if(i < raw_blobs.size()) {
                    auto &obj = raw_blobs[i];
                    obj->blob = &blob;
                    obj->threshold = threshold;
                    obj->frame = frameIndex;
                    
                } else {
                    std::unique_ptr<SimpleBlob> ptr;
                    if(!available_blobs_list.empty()) {
                        ptr = std::move(available_blobs_list.back());
                        available_blobs_list.pop_back();
                        
                        ptr->blob = &blob;
                        ptr->threshold = threshold;
                    } else
                        ptr = std::make_unique<SimpleBlob>(std::make_unique<ExternalImage>(), &blob, threshold);
                    
                    ptr->frame = frameIndex;
                    raw_blobs.emplace_back(std::move(ptr));
                }
            }
            
            ++i;
        });
        
        processed_frame().transform_noise([&](pv::Blob& blob) {
            if(not blob.blob_id().valid()) {
                FormatWarning("Empty blob here.");
                return;
            }
            
            blob.calculate_moments();
            
            if((nothing_to_zoom_on
                && (not FAST_SETTING(track_size_filter)
                    || blob.recount(-1) >= FAST_SETTING(track_size_filter).max_range().start))
               || selected_blobs.find(blob.blob_id()) != selected_blobs.end())
            {
                min_vec = min(min_vec, blob.bounds().pos());
                max_vec = max(max_vec, blob.bounds().pos() + blob.bounds().size());
            }
            
            if(reload_blobs) {
                if(i < raw_blobs.size()) {
                    auto &obj = raw_blobs[i];
                    obj->blob = &blob;
                    obj->threshold = threshold;
                    obj->frame = frameIndex;
                    
                } else {
                    std::unique_ptr<SimpleBlob> ptr;
                    if(!available_blobs_list.empty()) {
                        ptr = std::move(available_blobs_list.back());
                        available_blobs_list.pop_back();
                        
                        ptr->blob = &blob;
                        ptr->threshold = threshold;
                    } else
                        ptr = std::make_unique<SimpleBlob>(std::make_unique<ExternalImage>(), &blob, threshold);
                    
                    ptr->frame = frameIndex;
                    raw_blobs.emplace_back(std::move(ptr));
                }
            }
            
            ++i;
        });
        
        //! raw_blobs can be exaggerated because nullptrs within the
        //! blob and noise arrays are also counted (but not provided
        //! within the above loop). shrink:
        if(reload_blobs) {
            for(size_t j=i; j<raw_blobs.size(); ++j)
                available_blobs_list.emplace_back(std::move(raw_blobs[j]));
            raw_blobs.resize(i);
        }
        
#ifndef NDEBUG
        for(auto &obj : raw_blobs) {
            assert(obj->frame == frameIndex);
        }
#endif
        
        if(reload_blobs) {
            /**
             * Delete what we know about cliques and replace it
             * with current information.
             */
            if(frameIndex.valid() && Tracker::instance()->_cliques.count(frameIndex)) {
                _cliques = Tracker::instance()->_cliques.at(frameIndex);
            } else
                _cliques.clear();
            
            /**
             * Reload all ranged label information.
             */
            _ranged_blob_labels.clear();
            _blob_labels.clear();
            _individual_avg_categories.clear();
            
#if !COMMONS_NO_PYTHON
            if(std::shared_lock guard(Categorize::DataStore::cache_mutex());
               frameIndex.valid())
            {
                for(auto &b: raw_blobs) {
                    auto label = Categorize::DataStore::_label_unsafe(frameIndex, b->blob->blob_id());
                    if(label.has_value())
                        _blob_labels[b->blob->blob_id()] = label.value();
                }
                
                auto lock = lock_individuals();
                for(auto &[id, ptr]: lock.individuals) {
                    auto label = Categorize::DataStore::_label_averaged_unsafe(ptr, frameIndex);
                    if(label && label->id.has_value())
                        _individual_avg_categories[id] = label->id.value();
                }
            }
            
            if(std::shared_lock guard(Categorize::DataStore::range_mutex());
               frameIndex.valid()
               && !Categorize::DataStore::_ranges_empty_unsafe())
            {
                Frame_t f(frameIndex);
                
                if(raw_blobs.size() > 50) {
                    std::vector<MaybeLabel> labels(raw_blobs.size());
                    
                    distribute_indexes([&](auto i, auto start, auto end, auto){
                        for(auto it = start; it != end; ++it, ++i) {
                            labels[i] = Categorize::DataStore::_ranged_label_unsafe(f, (*it)->blob->blob_id());
                        }
                    }, _pool, raw_blobs.begin(), raw_blobs.end());
                    
                    for(size_t i=0; i<raw_blobs.size(); ++i) {
                        auto &b = raw_blobs[i];
                        if(labels[i].has_value())
                            _ranged_blob_labels[b->blob->blob_id()] = labels[i].value();
                    }
                    
                } else {
                    for(auto &b: raw_blobs) {
                        auto label = Categorize::DataStore::_ranged_label_unsafe(f, b->blob->blob_id());
                        if(label.has_value())
                            _ranged_blob_labels[b->blob->blob_id()] = label.value();
                    }
                }
            }
#endif
        }
        
        boundary = Bounds(min_vec, max_vec - min_vec);
        last_threshold = threshold;
        
        if(reload_blobs || _raw_blobs_dirty) {
            size_t gpixels = 0;
            double gaverage_pixels = 0, gsamples = 0;
            display_blobs.clear();
            
            const bool gui_show_only_unassigned = SETTING(gui_show_only_unassigned).value<bool>();
            const bool tags_dont_track = SETTING(tags_dont_track).value<bool>();
            
            distribute_indexes([&](auto, auto start, auto end, auto){
                std::unordered_map<pv::bid, SimpleBlob*> map;
                //std::vector<std::unique_ptr<gui::ExternalImage>> vector;
                size_t pixels = 0;
                double average_pixels = 0, samples = 0;
                
                for(auto it = start; it != end; ++it) {
                    if(!*it || (tags_dont_track && (*it)->blob && (*it)->blob->is_tag())) {
                        continue;
                    }
                    
                    //bool found = copy.count((*it)->blob.get());
                    //if(!found) {
                        //auto bds = bowl.transformRect((*it)->blob->bounds());
                        //if(bds.overlaps(screen_bounds))
                        //{
                    auto& blob = (*it)->blob;
                    if(blob
                       && (!gui_show_only_unassigned ||
                           (!display_blobs.contains(blob->blob_id())
                            && !contains(active_blobs, blob->blob_id()))))
                    {
                        (*it)->convert();
                        //vector.push_back((*it)->convert());
                        map[blob->blob_id()] = it->get();
                    }
                        //}
                    //}
                    
                    if(blob) {
                        pixels += blob->num_pixels();
                        average_pixels += blob->num_pixels();
                    }
                    ++samples;
                }
                
                auto guard = LOGGED_LOCK(vector_mutex);
                gpixels += pixels;
                gaverage_pixels += average_pixels;
                gsamples += samples;
                display_blobs.insert(map.begin(), map.end());
                //std::move(vector.begin(), vector.end(), std::back_inserter(PD(cache).display_blobs_list));
                //PD(cache).display_blobs_list.insert(PD(cache).display_blobs_list.end(), vector.begin(), vector.end());
                
            }, _pool, raw_blobs.begin(), raw_blobs.end());
            
            //for(auto &b : raw_blobs)
            //    b->ptr->updated_source();
            
            const uint8_t max_alpha = _graph->is_key_pressed(Codes::LSystem) ? 200 : 255;
            for(auto &b : display_blobs) {
                b.second->ptr->set_pos(b.second->image_pos);
                b.second->ptr->updated_source();
                b.second->ptr->set_color(White.alpha(max_alpha));
            }
            _current_pixels = gpixels;
            _average_pixels = gsamples > 0 ? gaverage_pixels / gsamples : 0;
            updated_raw_blobs();
            updated_blobs();
        }
        
        _last_success.reset();
        
        if(properties && (reload_blobs || _fish_dirty || _tracking_dirty))
        {
            update_graphs(frameIndex);
            
            set_of_individuals_t source;
            if(Tracker::has_identities()
                && GUI_SETTINGS(gui_show_inactive_individuals))
            {
                std::unique_lock guard(individuals_mutex);
                for (auto [id, fish] : individuals) {
                    source.insert(fish);
                }
                //! TODO: Tracker::identities().count(id) ?
                
            } else {
                for (auto fish : active) {
                    source.insert(fish);
                    //Print("active: ", fish->identity().ID());
                }
            }
            
            auto pred = _tracker.get_prediction(frameIndex);
            if(pred) {
                _current_predictions = pred.value();
            } else
                _current_predictions.reset();
            
            const pattern::UnresolvedStringPattern* current_label_text{nullptr};
            if(std::unique_lock g{label_mutex};
               not _prepared_label_text)
            {
                _prepared_label_text = pattern::UnresolvedStringPattern::prepare(_label_text);
                
                for(auto &[id, fish] : _fish_map)
                    fish->set_label_text(_prepared_label_text.value());
                
                current_label_text = &_prepared_label_text.value();
            } else {
                current_label_text = &_prepared_label_text.value();
            }
            
            assert(current_label_text);
            
            std::unordered_set<Idx_t> ids;
            std::unordered_map<Idx_t, Individual*> actives;
            
            //std::unique_lock g(Categorize::DataStore::cache_mutex());
            for (auto& fish : (source.empty() ? active : source)) {
                auto id = fish->identity().ID();
                ids.insert(id);
                actives[id] = fish;

                if (fish->empty()
                    || fish->start_frame() > frameIndex) 
                {
                    continue;
                }

                auto tracklet = fish->tracklet_for(frameIndex);
                if (!GUI_SETTINGS(gui_show_inactive_individuals)
                    && (!tracklet || (tracklet->end() != Tracker::end_frame()
                        && tracklet->length().get() < sign_cast<uint32_t>(GUI_SETTINGS(output_min_frames)))))
                {
                    continue;
                }

                /*auto it = container->map().find(fish);
                if (it != container->map().end())
                    empty_map = &it->second;
                else
                    empty_map = NULL;*/

                std::unique_lock guard(_fish_map_mutex);
                if (not _fish_map.contains(id)) {
                    std::shared_lock g{label_mutex};
                    _fish_map[id] = std::make_unique<gui::Fish>(*fish, *current_label_text);
                    
                    /*fish->register_delete_callback(_fish_map[id].get(), [gui = ](Individual* f) {
                            std::unique_lock guard(_fish_map_mutex);
                            auto it = _fish_map.find(f->identity().ID());
                            if (it != _fish_map.end()) {
                                _fish_map.erase(it);
                            }
                            
                            set_tracking_dirty();
                    });*/
                }
                
                
                //base.wrap_object(*PD(cache)._fish_map[fish]);
                //PD(cache)._fish_map[fish]->label(ptr, e);
            }
            
            UpdateSettings update_settings{
                .gui_show_outline = GUIOPTION(gui_show_outline),
                .gui_show_midline = GUIOPTION(gui_show_midline),
                .gui_happy_mode = GUIOPTION(gui_happy_mode),
                .gui_show_probabilities = GUIOPTION(gui_show_probabilities),
                .gui_show_shadows = GUIOPTION(gui_show_shadows),
                .gui_show_texts = GUIOPTION(gui_show_texts),
                .gui_show_selections = GUIOPTION(gui_show_selections),
                .gui_show_boundary_crossings = GUIOPTION(gui_show_boundary_crossings),
                .gui_show_paths = GUIOPTION(gui_show_paths),
                .gui_highlight_categories = GUIOPTION(gui_highlight_categories),
                .gui_show_match_modes = GUIOPTION(gui_show_match_modes),
                .gui_show_cliques = GUIOPTION(gui_show_cliques),
                .gui_fish_label = GUIOPTION(gui_fish_label),
                .panic_button = GUIOPTION(panic_button),
                .gui_outline_thickness = GUIOPTION(gui_outline_thickness),
                .gui_fish_color = GUIOPTION(gui_fish_color),
                .gui_single_identity_color = GUIOPTION(gui_single_identity_color),
                .gui_pose_smoothing = GUIOPTION(gui_pose_smoothing),
                .gui_max_path_time = GUIOPTION(gui_max_path_time),
                .pose_midline_indexes = GUIOPTION(pose_midline_indexes)
            };
            
            {
                std::vector<Idx_t> ids_to_check;
                
                {
                    std::scoped_lock guard(_fish_map_mutex, _next_frame_cache_mutex);
                    for (auto it = _fish_map.begin(); it != _fish_map.end();) {
                        if (not ids.contains(it->first)) {
                            //Print("erasing from map ", it->first);
                            _next_frame_caches.erase(it->first);
                            it = _fish_map.erase(it);
                        } else {
                            ids_to_check.emplace_back(it->first);
                            ++it;
                        }
                    }
                }
                
                /// if possible, try to get information about probabilities and such
                /// for the next frame already. this is whats displayed in the gui in the
                /// blob probabilities for example.
                if(_props && _next_props) {
                    std::scoped_lock guard(individuals_mutex, _next_frame_cache_mutex, _tracklet_cache_mutex);
                    auto current_time = _props->time();
                    auto next_props = _next_props ? &_next_props.value() : nullptr;
                    auto next_time = next_props ? next_props->time() : (current_time + 1.f/float(GUI_SETTINGS(frame_rate)));
                    /// cache cache_for_frame(frame + 1)
                    std::mutex map_mutex;
                    
                    for(auto it = _processed_tracklet_caches.begin(); it != _processed_tracklet_caches.end();) {
                        if(not contains(ids_to_check, it->first)) {
                            if(auto kit = _tracklet_caches.find(it->first);
                               kit != _tracklet_caches.end())
                            {
                                _tracklet_caches.erase(kit);
                            }
                            it = _processed_tracklet_caches.erase(it);
                        } else
                            ++it;
                    }
                    
                    distribute_indexes([this, &map_mutex, next_time, frameIndex](int64_t, auto start, auto end, int64_t)
                    {
                        [[maybe_unused]] track::TrackingThreadG g{};
                        
                        for(auto it = start; it != end; ++it) {
                            Individual* fish = individuals.at(*it);
                            auto cache = fish->cache_for_frame(&_props.value(), frameIndex + 1_f, next_time);
                            auto ptr = fish->tracklet_for(frameIndex);
                            
                            if(std::unique_lock g(map_mutex);
                               cache)
                            {
                                _next_frame_caches[*it] = std::move(cache.value());
                                _processed_tracklet_caches[*it] = fish->has_processed_tracklet(frameIndex);
                                if(ptr)
                                    _tracklet_caches[*it] = std::make_shared<TrackletInformation>(*ptr);
                                else if(auto sit = _tracklet_caches.find(*it);
                                        sit != _tracklet_caches.end())
                                {
                                    _tracklet_caches.erase(sit);
                                }
                                
                            } else {
                                auto kit = _next_frame_caches.find(*it);
                                if(kit != _next_frame_caches.end())
                                    _next_frame_caches.erase(kit);
                                _processed_tracklet_caches[*it] = fish->has_processed_tracklet(frameIndex);
                                if(ptr)
                                    _tracklet_caches[*it] = std::make_shared<TrackletInformation>(*ptr);
                                else if(auto sit = _tracklet_caches.find(*it);
                                        sit != _tracklet_caches.end())
                                {
                                    _tracklet_caches.erase(sit);
                                }
                                
                                FormatWarning("Cannot create cache_for_frame of ", *it, " for frame ", frameIndex + 1_f, " because: ", cache.error());
                            }
                        }
                        
                    }, pool(), ids_to_check.begin(), ids_to_check.end());
                }
                
                /// update the GUI objects with the new data
                /// this might take a long time for many objects, but is parallelizable
                /// we just need to make sure not to push any actual gui objects to the actual layouts yet.
                std::unique_lock guard(_fish_map_mutex);
                distribute_indexes([this, frameIndex, &update_settings, &properties](int64_t, auto start, auto end, int64_t) {
                    for(auto fit = start; fit != end; ++fit) {
                        auto id = *fit;
                        auto it = _fish_map.find(id);
                        if(it == _fish_map.end())
                            continue;
                        
                        std::unique_lock guard(individuals_mutex);
                        auto fish = individuals.at(it->first);
                        it->second->set_data(update_settings, *fish, frameIndex, properties->time(), nullptr);
                    }
                }, pool(), ids_to_check.begin(), ids_to_check.end());
            }
            
            updated_tracking();
        }
        
        //updated_raw_blobs();
        _dirty = false;
        
        //if(reload_blobs)
        //    Print("reloading: ", reasons);
        return reload_blobs ? processed_frame().index() : Frame_t{};
    }
    
    bool GUICache::has_probs(Idx_t fdx) {
        if(checked_probs.find(fdx) != checked_probs.end()) {
            return probabilities.find(fdx) != probabilities.end();
        }
        
        return probs(fdx) != nullptr;
    }

    const ska::bytell_hash_map<pv::bid, DetailProbability>* GUICache::probs(Idx_t fdx) {
        if(checked_probs.find(fdx) != checked_probs.end()) {
            auto it = probabilities.find(fdx);
            if(it  != probabilities.end())
                return &it->second;
            return nullptr;
        }
        
        checked_probs.insert(fdx);
        
        if (frame_idx.valid()) {
            if(auto c = processed_frame().cached(fdx);
               c != nullptr)
            {
                /// this is probably(?) safe since the probability does _not_
                /// access anything inside individual. should make this static
                /// to make sure this never happens.
                //std::unique_lock guard(individuals_mutex);
                CachedSettings settings;
                processed_frame().transform_blobs([&](const pv::Blob& blob) {
                    //auto it = active_ids.find(fdx);
                    //if(it == active_ids.end())
                    //    return;
                    //auto it = individuals.find(fdx);
                    //if(it == individuals.end() || it->second->empty() || frame_idx < it->second->start_frame())
                    //    return;
                    
                    auto p = Individual::probability(settings, processed_frame().label(blob.blob_id()), *c, frame_idx, blob);
                    if(p/*.p*/ >= FAST_SETTING(match_min_probability))
                        probabilities[fdx][blob.blob_id()] = {
                            .p = p,
                            .p_time = c->time_probability
                        };
                });
            }
        }
        
        auto it = probabilities.find(fdx);
        if(it != probabilities.end())
            return &it->second;
        return nullptr;
    }

    Size2 screen_dimensions(Base* base, const DrawStructure& graph) {
        if (!base)
            return Size2(1);

        auto gui_scale = graph.scale();
        if (gui_scale.x == 0)
            gui_scale = Vec2(1);
        auto window_dimensions = base
            ? base->window_dimensions().div(gui_scale) * gui::interface_scale()
            : Tracker::average().dimensions();
        return window_dimensions;
    }

    bool GUICache::key_down(Codes code) const {
        return _graph && _graph->is_key_pressed(code);
    }

std::optional<std::vector<float>> GUICache::find_prediction(pv::bid bdx) const {
    if(not _current_predictions)
        return std::nullopt;
    
    auto it = _current_predictions->find(bdx);
    if(it != _current_predictions->end()) {
        return it->second;
    }
    
    return std::nullopt;
}

std::optional<const IndividualCache*> GUICache::next_frame_cache(Idx_t id) const
{
    std::shared_lock guard(_next_frame_cache_mutex);
    auto it = _next_frame_caches.find(id);
    if(it != _next_frame_caches.end()) {
        return &it->second;
    }
    return std::nullopt;
}

std::tuple<bool, FrameRange> GUICache::processed_tracklet_cache(Idx_t id) const
{
    std::shared_lock guard(_tracklet_cache_mutex);
    auto it = _processed_tracklet_caches.find(id);
    if(it != _processed_tracklet_caches.end()) {
        return it->second;
    }
    return {false, FrameRange{}};
}

std::shared_ptr<track::TrackletInformation> GUICache::tracklet_cache(Idx_t id) const
{
    std::shared_lock guard(_tracklet_cache_mutex);
    auto it = _tracklet_caches.find(id);
    if(it != _tracklet_caches.end()) {
        return it->second;
    }
    return nullptr;
}

void GUICache::update_graphs(const Frame_t frameIndex) {
    if(not SETTING(gui_show_graph).value<bool>()) {
        _displayed_graphs.clear();
        return;
    }
    
    auto output_frame_window = Frame_t(SETTING(output_frame_window).value<uint32_t>());
                    
    /*for(auto id : PD(cache).selected) {
        PD(fish_graphs)[i]->setup_graph(frameNr.get(), Rangel((frameNr.try_sub( window)).get(), (frameNr + window).get()), PD(cache).individuals.at(id), nullptr);
        PD(fish_graphs)[i]->graph().set_scale(base.scale().reciprocal());
        PD(fish_graphs)[i]->draw(base);
        
        if(++i >= PD(fish_graphs).size())
            break;
    }*/
    
    auto individuals = lock_individuals();
    auto prev_ids = extract_keys(_displayed_graphs);
    auto cached_output_fields = Output::Library::get_cached_fields();
    
    for(Idx_t id : selected) {
        if(not individuals.individuals.contains(id)) {
            FormatWarning("Individuals do not have ", id, ".");
        }
        auto fish = individuals.individuals.at(id);
        auto it = _displayed_graphs.find(id);
        if(it == _displayed_graphs.end()) {
            auto r = _displayed_graphs.emplace(id, std::make_unique<PropertiesGraph>());
            if(not std::get<1>(r)) {
                FormatWarning("Cannot insert ", id, " into the displayed_graphs");
                continue;
            }
            it = std::get<0>(r);
        }
        
        prev_ids.erase(id);
        
        it->second->setup_graph(cached_output_fields, frameIndex, Range<Frame_t>{
            frameIndex.try_sub(output_frame_window),
            frameIndex + output_frame_window
        }, fish, nullptr);
    }
    
    /// remove all the ones we dont have at the moment
    for(auto id : prev_ids) {
        _displayed_graphs.erase(id);
    }
}

}
