#include "GUICache.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <gui/DrawFish.h>
#include <tracking/Categorize.h>
#include <gui/Timeline.h>
#include <gui/DrawBase.h>
#include <tracking/IndividualManager.h>
#include <misc/default_config.h>
#include <grabber/misc/default_config.h>

namespace gui {
    using buffers = Buffers<std::unique_ptr<PPFrame>, decltype([](){ return std::make_unique<PPFrame>(); })>;

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

    GUICache::GUICache(DrawStructure* graph, pv::File* video)
        : _pool(saturate(cmn::hardware_concurrency(), 1u, 5u), "GUICache::_pool"),
            _video(video), _graph(graph),
            _preloader([this](Frame_t frameIndex) -> FramePtr {
                FramePtr ptr;
                try {
                    if(frameIndex.valid()
                       && _video->is_read_mode())
                    {
                        if(frameIndex >= _video->length())
                            return nullptr; // past the end
                        
                        pv::Frame frame;
                        _video->read_frame(frame, frameIndex);
                        
                        ptr = buffers::get();
                        ptr->clear();
                        
                        Tracker::instance()->preprocess_frame(std::move(frame), *ptr, &_pool, PPFrame::NeedGrid::Need, _video->header().resolution);
                    }
                    
                } catch(...) {
                    FormatExcept("Cannot load frame ", frameIndex, " from file ", _video->filename());
                }
                
                return ptr;
            },
            [](FramePtr&& ptr) {
                buffers::move_back(std::move(ptr));
            })

    {
        cache() = this;
        globals::Cache::init();
    }

    GUICache::~GUICache() {
        clear_animators();
        _fish_map.clear();
        display_blobs.clear();
        raw_blobs.clear();
        available_blobs_list.clear();

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
        if (GUICache::instance()._equalize_histograms && !percentiles.empty()) {
            image_pos = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), threshold, percentiles.front(), percentiles.back(), ptr->unsafe_get_source(), 0);
        } else {
            image_pos = blob->luminance_alpha_image(*Tracker::instance()->background(), threshold, ptr->unsafe_get_source(), 0);
        }

        if(SETTING(meta_encoding).value<grab::default_config::meta_encoding_t::Class>() == grab::default_config::meta_encoding_t::r3g3b2) {
            if(not ptr->empty()) {
                auto mat = ptr->unsafe_get_source().get();
                cv::Mat output;
                convert_from_r3g3b2<4,2>(mat, output);
                //print("converted ", mat.channels(), " to ", output.channels());
                //tf::imshow("output", output);
                //ptr->unsafe_get_source().create(output);
                ptr->set_source(Image::Make(output));
            }
        }
        
        //ptr->set_pos(image_pos);
        //ptr->updated_source();
        
        ptr->add_custom_data("blob_id", (void*)(uint64_t)(uint32_t)blob->blob_id());
        ptr->add_custom_data("frame", (void*)(uint64_t)frame.get());
        if(ptr->name().empty())
            ptr->set_name("SimpleBlob_"+Meta::toStr(blob->blob_id()));
        
        //blob = nullptr;
    }
    
    bool GUICache::has_selection() const {
        return !selected.empty() && individuals.count(selected.front()) != 0;
    }
    
    Individual * GUICache::primary_selection() const {
        return has_selection() && individuals.count(selected.front())
                ? individuals.at(selected.front())
                : nullptr;
    }

    Idx_t GUICache::primary_selected_id() const {
        return has_selection() ? selected.front() : Idx_t();
    }

    void GUICache::clear_animators() {
        _animators.clear();

        if (_graph) {
            for (auto& [name, ptr] : _animator_map) {
                auto handler = _delete_handles.at(ptr);
                ptr->remove_delete_handler(handler);
                _delete_handles.erase(ptr);
            }
        }
        _animator_map.clear();
    }
    
    bool GUICache::is_animating(std::string_view animator) const {
        if(GUI_SETTINGS(gui_happy_mode) && mode() == mode_t::tracking) {
            return true;
        }
        
        //print(" ");
        auto is_relevant = [this](const std::string_view& animator) {
            auto it = _animator_map.find(animator);
            if(it != _animator_map.end()) {
                //print(" * animator ", animator, " is ", it->second->rendered());
                return it->second->rendered();
			}
            return true;
        };

        if (animator.empty()) {
            for(auto& a : _animators)
                if(is_relevant(a))
					return true;
            //return !_animators.empty();
            return false;
        }
        auto it = _animators.find(animator);
        if(it != _animators.end())
            return true;
        
       /* for (auto& o : _animators) {
            if(o->is_child_of(obj)) {
                return true;
            }
        }*/
        
        return false;
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
        if(raw_blobs_dirty() || _dirty || (_mode == mode_t::tracking && _tracking_dirty) || (_mode == mode_t::blobs && _blobs_dirty) || is_animating())
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
    auto& _gui = *_graph;
    
    return  not processed_frame().index().valid()
            || frameIndex != processed_frame().index()
            || last_threshold != threshold
            || selected != previous_active_fish
            || active_blobs != previous_active_blobs
            || _gui.mouse_position() != previous_mouse_position
            || (is_tracking_dirty() && mode() == mode_t::tracking)
            || _tracking_dirty
            || raw_blobs_dirty()
            || _blobs_dirty
            || _frame_contained != tracked_frames.contains(frameIndex);
}
    
    Frame_t GUICache::update_data(Frame_t frameIndex) {
        const auto threshold = FAST_SETTING(track_threshold);
        const auto posture_threshold = FAST_SETTING(track_posture_threshold);
        auto& _gui = *_graph;
        _equalize_histograms = GUI_SETTINGS(gui_equalize_blob_histograms);
        
        /*std::vector<std::string> reasons;
        if(last_threshold != threshold)
            reasons.emplace_back("threshold");
        if(raw_blobs_dirty())
            reasons.emplace_back("raw_blobs_dirty");*/
        
        _fish_dirty = false;
        
        bool current_frame_matches = _current_processed_frame
             && _current_processed_frame->index() == frameIndex;
        bool next_frame_matches = _next_processed_frame
             && _next_processed_frame->index() == frameIndex;
        
        /*if(not current_frame_matches)
        {
            reasons.emplace_back("No matching next_frame found current="+Meta::toStr(current_frame_matches)+" next="+Meta::toStr(next_frame_matches)+". frame="+Meta::toStr(frameIndex)+"  current="+Meta::toStr(_current_processed_frame)+" next="+Meta::toStr(_next_processed_frame));
        }*/
        
        bool reload_blobs = not current_frame_matches
                            || last_threshold != threshold
                            || raw_blobs_dirty();
        
        if(not current_frame_matches) {
            if(_next_processed_frame
               && _last_success.elapsed() < 0.1
               && not next_frame_matches)
            {
                /// we dont have a timeout, and we have a next frame
                /// but it doesnt match the requested frame:
                /// discard it.
                buffers::move_back(std::move(_next_processed_frame));
            }
            
            if(not next_frame_matches) {
                /// the next frame does *not* match - at least should
                /// nudge the preloader:
                auto maybe_frame = _preloader.get_frame(frameIndex);
                
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
                        buffers::move_back(std::move(_next_processed_frame));
                    
                    if(maybe_frame.value()->index() != frameIndex
                        && _last_success.elapsed() < 0.1)
                    {
                        /// we dont have a timeout, and the indexes
                        /// dont match. discard and return...
                        buffers::move_back(std::move(maybe_frame.value()));
                        return {};
                        
                    } else {
                        /// we have a timeout, se just use what we have:
                        //print("Using maybe_frame anyway for ", maybe_frame.value()->index(), " != ", frameIndex, " since we waited ", _last_success.elapsed());
                        frameIndex = maybe_frame.value()->index();
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
           || _gui.mouse_position() != previous_mouse_position
           || is_tracking_dirty()
           || _blobs_dirty)
        {
            
        } else 
            return {};
        
        LockGuard guard(ro_t{}, "update_cache", 10);
        if(not guard.locked())
            return {};
        
        auto& _tracker = *Tracker::instance();
        frame_idx = frameIndex;
        _video_resolution = _video->size();
        
        if(not GUI_SETTINGS(nowindow)) {
            //! Calculate average pixel values. This is not a high-priority action, especially if the GUI is disabled. Only used for `gui_equalize_blob_histograms`.
            std::call_once(_percentile_once, [this](){
                percentile_ptr = std::make_unique<std::thread>([this](){
                    cmn::set_thread_name("percentile_thread");
                    print("open for writing: ", _video->is_write_mode());
                    if(_video->is_read_mode()) {
                        auto percentiles = _video->calculate_percentiles({0.05f, 0.95f});
                        
                        if(_graph) {
                            auto guard = GUI_LOCK(_graph->lock());
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
        active_blobs.clear();
        active.clear();
        selected_blobs.clear();
        inactive_ids.clear();
        active_ids.clear();
        fish_selected_blobs.clear();
        inactive_estimates.clear();
        
        if(properties) {
            active = _tracker.active_individuals(frameIndex);
            individuals = IndividualManager::copy();
            selected = SETTING(gui_focus_group).value<std::vector<Idx_t>>();
            tracked_frames = Range<Frame_t>(_tracker.start_frame(), _tracker.end_frame());
            
            auto delete_callback = [this](Individual* fish) {
                if(!cache() || !_graph)
                    return;
                
                auto guard = GUI_LOCK(_graph->lock());
                
                auto id = fish->identity().ID();
                auto it = individuals.find(id);
                if(it != individuals.end())
                    individuals.erase(it);
                
                active.clear();
                
                auto kit = active_ids.find(id);
                if(kit != active_ids.end())
                    active_ids.erase(kit);
                
                kit = inactive_ids.find(id);
                if(kit != inactive_ids.end())
                    inactive_ids.erase(kit);
                
                auto bit = fish_selected_blobs.find(id);
                if(bit != fish_selected_blobs.end())
                    fish_selected_blobs.erase(bit);
                
                auto cit = _registered_callback.find(fish);
                if(cit != _registered_callback.end())
                    _registered_callback.erase(cit);
            };
            
            IndividualManager::transform_all([&](auto, auto fish){
                if(!contains(_registered_callback, fish)) {
                    fish->register_delete_callback((void*)12341337, delete_callback);
                    _registered_callback.insert(fish);
                }
            });
            
            auto connectivity_map = SETTING(gui_connectivity_matrix).value<std::map<long_t, std::vector<float>>>();
            if(connectivity_map.count(frameIndex.get()))
                connectivity_matrix = connectivity_map.at(frameIndex.get());
            else
                connectivity_matrix.clear();
            
            double time = properties ? properties->time : 0;
            
            for(auto fish : active) {
                auto blob = fish->compressed_blob(frameIndex);
                if(blob) {
                    active_ids.insert(fish->identity().ID());
                    fish_selected_blobs[fish->identity().ID()] = blob->blob_id();
                } else {
                    inactive_ids.insert(fish->identity().ID());
                }
            }
            
            if(has_selection()) {
                auto previous = Tracker::properties(frameIndex - 1_f);
                for(auto id : selected) {
                    if(individuals.count(id)) {
                        auto fish = individuals.at(id);
                        if(!fish->has(frameIndex) && !fish->empty() && frameIndex >= fish->start_frame()) {
                            auto c = fish->cache_for_frame(previous, frameIndex, time);
                            if(c) {
                                inactive_estimates.push_back(c.value().estimated_px);
                                inactive_ids.insert(fish->identity().ID());
                            } else {
                                FormatError("Cannot create cache_for_frame of ", fish->identity(), " in frame ", frameIndex," because: ", c.error());
                            }
                        }
                    }
                }
            }
            
            // display all blobs that are assigned to an individual
            for(auto fish : active) {
                auto blob = fish->compressed_blob(frameIndex);
                if(blob)
                    active_blobs.insert(blob->blob_id());
            }
            
            if(!has_selection() || !SETTING(gui_auto_scale_focus_one)) {
                selected_blobs = active_blobs;
            } else {
                // display blobs that are selected
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
            print("frameIndex ", frameIndex, " contained=",contained);
        }
        
        if(_current_processed_frame && _current_processed_frame->index() != frameIndex) {
            buffers::move_back(std::move(_current_processed_frame));
            //print("current_processed_frame moved out for ", frameIndex);
        } else if(_current_processed_frame) {
            reload_blobs = false;
            //reasons.emplace_back("-");
            //print("current_processed_frame is fine for ", frameIndex, " = ", _current_processed_frame->index());
            if(_next_processed_frame)
                buffers::move_back(std::move(_next_processed_frame));
        }
        
        if(_next_processed_frame) {
            if(not reload_blobs) {
                reload_blobs = true;
                //reasons.emplace_back("next_processed_frame was != nullptr");
            }
            //print("current_processed_frame moved out for ", frameIndex, " = ", _next_processed_frame->index());
            _current_processed_frame = std::move(_next_processed_frame);
            assert(_current_processed_frame->index() == frameIndex);
        }
        
        //print("reload_blobs = ", reload_blobs);
        
        if(reload_blobs
           || selected != previous_active_fish
           || active_blobs != previous_active_blobs) 
        {
            set_tracking_dirty();
        }
        
        _global_segment_order = _tracker.unsafe_global_segment_order();
        previous_active_fish = selected;
        previous_active_blobs = active_blobs;
        previous_mouse_position = _gui.mouse_position();
        
        set_blobs_dirty();
        
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
            
            //print("Reloading blobs ", frameIndex);
            if(L < raw_blobs.size()) {
                std::move(raw_blobs.begin() + L, raw_blobs.end(), std::back_inserter(available_blobs_list));
                raw_blobs.erase(raw_blobs.begin() + L, raw_blobs.end());
                
            } else if(L != raw_blobs.size()) {
                raw_blobs.reserve(L);
            }
        } //else
            //print("Not reloading blobs ", frameIndex);
        
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
            blob.calculate_moments();
            
            if((nothing_to_zoom_on && blob.recount(-1) >= FAST_SETTING(blob_size_ranges).max_range().start)
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
            
#if !COMMONS_NO_PYTHON
            std::shared_lock guard(Categorize::DataStore::range_mutex());
            if(frameIndex.valid() && !Categorize::DataStore::_ranges_empty_unsafe()) {
                Frame_t f(frameIndex);
                
                if(raw_blobs.size() > 50) {
                    std::vector<int> labels(raw_blobs.size());
                    
                    distribute_indexes([&](auto i, auto start, auto end, auto){
                        for(auto it = start; it != end; ++it, ++i) {
                            labels[i] = Categorize::DataStore::_ranged_label_unsafe(f, (*it)->blob->blob_id());
                        }
                    }, _pool, raw_blobs.begin(), raw_blobs.end());
                    
                    for(size_t i=0; i<raw_blobs.size(); ++i) {
                        auto &b = raw_blobs[i];
                        _ranged_blob_labels[b->blob->blob_id()] = labels[i];
                    }
                    
                } else {
                    for(auto &b: raw_blobs) {
                        auto label = Categorize::DataStore::_ranged_label_unsafe(f, b->blob->blob_id());
                        _ranged_blob_labels[b->blob->blob_id()] = label;
                    }
                }
            }
#endif
        }
        
        boundary = Bounds(min_vec, max_vec - min_vec);
        last_threshold = threshold;
        
        if(reload_blobs) {
            size_t gpixels = 0;
            double gaverage_pixels = 0, gsamples = 0;
            display_blobs.clear();
            
            distribute_indexes([&](auto, auto start, auto end, auto){
                std::unordered_map<pv::bid, SimpleBlob*> map;
                //std::vector<std::unique_ptr<gui::ExternalImage>> vector;
                
                const bool gui_show_only_unassigned = SETTING(gui_show_only_unassigned).value<bool>();
                const bool tags_dont_track = SETTING(tags_dont_track).value<bool>();
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
            
            for(auto &b : display_blobs) {
                b.second->ptr->set_pos(b.second->image_pos);
                b.second->ptr->updated_source();
            }
            _current_pixels = gpixels;
            _average_pixels = gsamples > 0 ? gaverage_pixels / gsamples : 0;
            updated_raw_blobs();
        }
        
        _last_success.reset();
        
        if(properties && (reload_blobs || _fish_dirty))
        {
            set_of_individuals_t source;
            if(Tracker::has_identities() && GUI_SETTINGS(gui_show_inactive_individuals))
            {
                for(auto [id, fish] : individuals)
                    source.insert(fish);
                //! TODO: Tracker::identities().count(id) ?
                
            } else {
                for(auto fish : active)
                    source.insert(fish);
            }
            
            for (auto& fish : (source.empty() ? active : source)) {
                if (fish->empty()
                    || fish->start_frame() > frameIndex)
                    continue;

                auto segment = fish->segment_for(frameIndex);
                if (!GUI_SETTINGS(gui_show_inactive_individuals)
                    && (!segment || (segment->end() != Tracker::end_frame()
                        && segment->length().get() < sign_cast<uint32_t>(GUI_SETTINGS(output_min_frames)))))
                {
                    continue;
                }

                /*auto it = container->map().find(fish);
                if (it != container->map().end())
                    empty_map = &it->second;
                else
                    empty_map = NULL;*/

                std::unique_lock guard(_fish_map_mutex);
                auto id = fish->identity().ID();
                if (not _fish_map.contains(id)) {
                    _fish_map[id] = std::make_unique<gui::Fish>(*fish);
                    fish->register_delete_callback(_fish_map[id].get(), [this](Individual* f) {
                            std::unique_lock guard(_fish_map_mutex);
                            auto it = _fish_map.find(f->identity().ID());
                            if (it != _fish_map.end()) {
                                _fish_map.erase(it);
                            }
                            
                            set_tracking_dirty();
                        });
                }
                
                _fish_map[id]->set_data(*fish, frameIndex, properties->time, nullptr);
                //base.wrap_object(*PD(cache)._fish_map[fish]);
                //PD(cache)._fish_map[fish]->label(ptr, e);
            }
            
            updated_blobs();
            updated_raw_blobs();
        }
        
        //if(reload_blobs)
        //    print("reloading: ", reasons);
        return reload_blobs ? processed_frame().index() : Frame_t{};
    }
    
    void GUICache::set_animating(std::string_view animation, bool v, Drawable* parent) {
        if (animation.empty())
            throw std::invalid_argument("Empty animation.");

        if(v) {
            auto it = _animators.find(animation);
            if(it == _animators.end()) {
                _animators.insert(animation);
                if (parent) {
                    _animator_map[animation] = parent;
                    _delete_handles[parent] = parent->on_delete([this, animation]() {
                        if (!_graph)
                            return;
                        this->set_animating(animation, false);
                        //print("Animating object deleted (", animation, "). ", _animators);
                    });
                }
                else if (_animator_map.contains(animation)) {
                    _animator_map.erase(animation);
                }
                //print("Animating object added: ", animation," (",parent,"). ", _animators);
            }
        } else {
            auto it = _animators.find(animation);
            if(it != _animators.end()) {
                _animators.erase(it);
                //print("Animating object deleted (", animation, ") ", _animators);
                if(_animator_map.contains(animation)) {
                    auto ptr = _animator_map.at(animation);
                    if (_delete_handles.count(ptr)) {
                        auto handle = _delete_handles.at(ptr);
                        _delete_handles.erase(ptr);
                        ptr->remove_delete_handler(handle);
                    }
                    else {
                        FormatError("Cannot find delete handler in GUICache. Something went wrong?");
                    }
                    _animator_map.erase(animation);
				}
            }
        }
    }

    bool GUICache::has_probs(Idx_t fdx) {
        if(checked_probs.find(fdx) != checked_probs.end()) {
            return probabilities.find(fdx) != probabilities.end();
        }
        
        return probs(fdx) != nullptr;
    }

    const ska::bytell_hash_map<pv::bid, Individual::Probability>* GUICache::probs(Idx_t fdx) {
        if(checked_probs.find(fdx) != checked_probs.end()) {
            auto it = probabilities.find(fdx);
            if(it  != probabilities.end())
                return &it->second;
            return nullptr;
        }
        
        checked_probs.insert(fdx);
        
        {
            LockGuard guard(ro_t{}, "GUICache::probs");
            auto c = processed_frame().cached(fdx);
            if(c) {
                processed_frame().transform_blobs([&](const pv::Blob& blob) {
                    auto it = individuals.find(fdx);
                    if(it == individuals.end() || it->second->empty() || frame_idx < it->second->start_frame())
                        return;
                    
                    auto p = individuals.at(fdx)->probability(processed_frame().label(blob.blob_id()), *c, frame_idx, blob);
                    if(p/*.p*/ >= FAST_SETTING(matching_probability_threshold))
                        probabilities[c->_idx][blob.blob_id()] = p;
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

    std::tuple<Vec2, Vec2> GUICache::scale_with_boundary(Bounds& boundary, bool recording, Base* base, DrawStructure& graph, Section* section, bool singular_boundary)
    {
        constexpr const char* animation_name { "scale-boundaries-animation" };
        static Vec2 target_scale(1);
        static Vec2 target_pos(0, 0);
        static Size2 target_size(Tracker::average().dimensions());
        static bool lost = true;
        static float time_lost = 0;

        auto&& [offset, max_w] = Timeline::timeline_offsets(base);

        Size2 screen_dim = screen_dimensions(base, graph);
        Size2 screen_center = screen_dim * 0.5;

        if (screen_dim.max() <= 0)
            return { Vec2(), Vec2() };
        //if(_base)
        //    offset = Vec2((_base->window_dimensions().width / PD(gui).scale().x * gui::interface_scale() - _average_image.cols) * 0.5, 0);


        /**
         * Automatically zoom in on the group.
         */
        if (singular_boundary) {//SETTING(gui_auto_scale) && (singular_boundary || !SETTING(gui_auto_scale_focus_one))) {
            if (lost) {
                GUICache::instance().set_animating(animation_name, false);
            }

            if (boundary.x != FLT_MAX) {
                Size2 minimal_size = SETTING(gui_zoom_limit).value<Size2>();
                //Size2(_average_image) * 0.15;

                if (boundary.width < minimal_size.width) {
                    boundary.x -= (minimal_size.width - boundary.width) * 0.5;
                    boundary.width = minimal_size.width;
                }
                if (boundary.height < minimal_size.height) {
                    boundary.y -= (minimal_size.height - boundary.height) * 0.5;
                    boundary.height = minimal_size.height;
                }

                Vec2 scales(boundary.width / max_w,
                    boundary.height / screen_dim.height);

                float scale = 1.f / scales.max() * 0.8;

                //Vec2 topleft(Size2(max_w / PD(gui).scale().x, _average_image.rows) * 0.5 - offset / PD(gui).scale().x - boundary.size() * scale * 0.5);

                //boundary.pos() -= offset.div(scale);

                target_scale = Vec2(scale);
                Size2 image_center = boundary.pos() + boundary.size() * 0.5;

                offset = screen_center - image_center * scale;
                target_pos = offset;

                target_size = boundary.size();

                lost = false;
            }

        }
        else {
            static Timer lost_timer;
            if (!lost) {
                lost = true;
                time_lost = GUICache::instance().gui_time();
                lost_timer.reset();
                GUICache::instance().set_animating(animation_name, true);
            }

            if ((recording && GUICache::instance().gui_time() - time_lost >= 0.5)
                || (!recording && lost_timer.elapsed() >= 0.5))
            {
                target_scale = Vec2(1);
                //target_pos = offset;//Vec2(0, 0);
                target_size = Tracker::average().dimensions();
                target_pos = screen_center - target_size * 0.5;
                GUICache::instance().set_animating(animation_name, false);
            }
        }

        Float2_t mw = Tracker::average().cols;
        Float2_t mh = Tracker::average().rows;
        if (target_pos.x / target_scale.x < -mw * 0.95) {
#ifndef NDEBUG
            print("target_pos.x = ", target_pos.x, " target_scale.x = ", target_scale.x);
#endif
            target_pos.x = -mw * target_scale.x * 0.95f;
        }
        if (target_pos.y / target_scale.y < -mh * 0.95f)
            target_pos.y = -mh * target_scale.y * 0.95f;

        if (target_pos.x / target_scale.x > mw * 0.95f) {
#ifndef NDEBUG
            //print("target_pos.x = ", target_pos.x, " target_scale.x = ", target_scale.x, " screen_center.x = ", screen_center.width, " screen_dimensions.x = ", screen_dimensions.width, " window_dimensions.x = ", base()->window_dimensions().width);
#endif
            target_pos.x = mw * target_scale.x * 0.95f;
        }
        if (target_pos.y / target_scale.y > mh * 0.95f)
            target_pos.y = mh * target_scale.y * 0.95f;

        GUICache::instance().set_zoom_level(target_scale.x);

        static Timer timer;
        auto e = recording ? GUICache::instance().dt() : timer.elapsed(); //PD(recording) ? (1 / float(FAST_SETTING(frame_rate))) : timer.elapsed();
        //e = PD(cache).dt();

        e = min(0.1, e);
        e *= 3;

        auto check_target = [](const Vec2& start, const Vec2& target, Float2_t e) {
            Vec2 direction = (target - start) * e;
            Float2_t speed = direction.length();
            auto epsilon = max(target.abs().max(), start.abs().max()) * 0.000001;

            if (speed <= epsilon)
                return target;

            if (speed > 0)
                direction /= speed;

            auto scale = start + direction * speed;

            if ((direction.x > 0 && scale.x > target.x)
                || (direction.x < 0 && scale.x < target.x))
            {
                scale.x = target.x;
            }
            if ((direction.y > 0 && scale.y > target.y)
                || (direction.y < 0 && scale.y < target.y))
            {
                scale.y = target.y;
            }

            return scale;
        };


        target_pos.x = round(target_pos.x);
        target_pos.y = round(target_pos.y);

        if (!section->scale().Equals(target_scale)
            || !section->pos().Equals(target_pos))
        {
            GUICache::instance().set_animating(animation_name, true);

            auto playback_factor = max(1, sqrt(SETTING(gui_playback_speed).value<float>()));
            auto scale = check_target(section->scale(), target_scale, e * playback_factor);

            section->set_scale(scale);

            auto next_pos = check_target(section->pos(), target_pos, e * playback_factor);
            auto next_size = check_target(section->size(), target_size, e * playback_factor);

            section->set_bounds(Bounds(next_pos, next_size));

        }
        else {
            GUICache::instance().set_animating(animation_name, false);

            section->set_scale(target_scale);
            section->set_bounds(Bounds(target_pos, target_size));
        }

        timer.reset();

        return { Vec2(), Vec2() };
    }
}
