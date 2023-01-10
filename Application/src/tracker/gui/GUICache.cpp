#include "GUICache.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <gui/DrawFish.h>
#include <tracking/Categorize.h>
#include <gui/Timeline.h>
#include <gui/DrawBase.h>
#include <tracking/IndividualManager.h>

namespace gui {
    static std::unique_ptr<std::thread> percentile_ptr = nullptr;
    static std::mutex percentile_mutex;

    GUICache*& cache() {
        static GUICache* _cache{ nullptr };
        return _cache;
    }

    static GenericThreadPool _pool(cmn::hardware_concurrency(), "GUICache::_pool");

    GUICache& GUICache::instance() {
        if (!cache())
            throw U_EXCEPTION("No cache created yet.");
        return *cache();
    }

    bool GUICache::exists() {
        return cache() != nullptr;
    }

    SimpleBlob::SimpleBlob(std::unique_ptr<ExternalImage>&& available, pv::BlobWeakPtr b, int t)
        : blob(b), threshold(t), ptr(std::move(available))
    {
        assert(ptr);
        if (!ptr->source()) {
            ptr->set_source(Image::Make());
        }
    }

    GUICache::GUICache(DrawStructure* graph, pv::File* video)
        : _video(video), _graph(graph)
    {
        cache() = this;
        globals::Cache::init();
    }

    GUICache::~GUICache() {
        set_animating(nullptr, false);

        std::lock_guard guard(percentile_mutex);
        if(percentile_ptr) {
            percentile_ptr->join();
            percentile_ptr = nullptr;
        }

        cache() = nullptr;
    }
    
    void SimpleBlob::convert() {
        Vec2 image_pos;
        
        auto &percentiles = GUICache::instance().pixel_value_percentiles;
        if (GUICache::instance()._equalize_histograms && !percentiles.empty()) {
            image_pos = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), threshold, percentiles.front(), percentiles.back(), ptr->unsafe_get_source());
        } else {
            image_pos = blob->luminance_alpha_image(*Tracker::instance()->background(), threshold, ptr->unsafe_get_source());
        }

        ptr->set_pos(image_pos);
        ptr->updated_source();
        
        ptr->add_custom_data("blob_id", (void*)(uint64_t)(uint32_t)blob->blob_id());
        if(ptr->name().empty())
            ptr->set_name("SimpleBlob_"+Meta::toStr(blob->blob_id()));
    }
    
    bool GUICache::has_selection() const {
        return !selected.empty() && individuals.count(selected.front()) != 0;
    }
    
    Individual * GUICache::primary_selection() const {
        return has_selection() && individuals.count(selected.front())
                ? individuals.at(selected.front())
                : nullptr;
    }
    
    bool GUICache::is_animating(Drawable* obj) const {
        if(GUI_SETTINGS(gui_happy_mode) && mode() == mode_t::tracking) {
            return true;
        }
        
        if(!obj)
            return !_animators.empty();
        auto it = _animators.find(obj);
        if(it != _animators.end())
            return true;
        
        for(auto &o : _animators) {
            if(o->is_child_of(obj)) {
                return true;
            }
        }
        
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
            SETTING(heatmap_ids) = std::vector<uint32_t>();
        }
    }
    
    bool GUICache::is_selected(Idx_t id) const {
        return contains(selected, id);
    }
    
    void GUICache::do_select(Idx_t id) {
        if(!is_selected(id)) {
            selected.push_back(id);
            SETTING(gui_focus_group) = selected;
            SETTING(heatmap_ids) = std::vector<uint32_t>(selected.begin(), selected.end());
        }
    }
    
    void GUICache::deselect(Idx_t id) {
        auto it = std::find(selected.begin(), selected.end(), id);
        if(it != selected.end()) {
            selected.erase(it);
            SETTING(gui_focus_group) = selected;
            SETTING(heatmap_ids) = std::vector<uint32_t>(selected.begin(), selected.end());
        }
    }
    
    void GUICache::deselect_all_select(Idx_t id) {
        selected.clear();
        selected.push_back(id);
        
        SETTING(gui_focus_group) = selected;
        SETTING(heatmap_ids) = std::vector<uint32_t>(selected.begin(), selected.end());
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
        if(_raw_blobs_dirty || _dirty || (_mode == mode_t::tracking && _tracking_dirty) || (_mode == mode_t::blobs && _blobs_dirty) || is_animating())
            return true;
        return false;
    }
    
    void GUICache::update_data(Frame_t frameIndex) {
        const auto threshold = FAST_SETTING(track_threshold);
        auto& _tracker = *Tracker::instance();
        auto& _gui = *_graph;
        _equalize_histograms = GUI_SETTINGS(gui_equalize_blob_histograms);
        
        frame_idx = frameIndex;
        
        if(!GUI_SETTINGS(nowindow)) {
            //! Calculate average pixel values. This is not a high-priority action, especially if the GUI is disabled. Only used for `gui_equalize_blob_histograms`.
            static std::atomic<bool> done_calculating{false};
            static auto percentile_ptr = std::make_unique<std::thread>([this](){
                cmn::set_thread_name("percentile_thread");
                auto percentiles = _video->calculate_percentiles({0.05f, 0.95f});
                
                if(_graph) {
                    std::lock_guard guard(_graph->lock());
                    pixel_value_percentiles = percentiles;
                }
                
                done_calculating = true;
            });
            
            {
                std::lock_guard guard(percentile_mutex);
                if(percentile_ptr && done_calculating) {
                    percentile_ptr->join();
                    percentile_ptr = nullptr;
                }
            }
        }
        
        if(_statistics.size() < _tracker._statistics.size()) {
            auto start = _tracker._statistics.end();
            std::advance(start, (int64_t)_statistics.size() - (int64_t)_tracker._statistics.size());
            
            for (; start != _tracker._statistics.end(); ++start)
                _statistics[start->first] = start->second;
            
        } else if(_statistics.size() > _tracker._statistics.size()) {
            auto start = _statistics.begin();
            std::advance(start, (int64_t)_tracker._statistics.size());
            _statistics.erase(start, _statistics.end());
        }
        
        auto properties = _tracker.properties(frameIndex);
        if(properties) {
            active = _tracker.active_individuals(frameIndex);
            individuals = IndividualManager::copy();
            selected = SETTING(gui_focus_group).value<std::vector<Idx_t>>();
            active_blobs.clear();
            selected_blobs.clear();
            inactive_ids.clear();
            active_ids.clear();
            fish_selected_blobs.clear();
            inactive_estimates.clear();
            tracked_frames = Range<Frame_t>(_tracker.start_frame(), _tracker.end_frame());
            
            auto delete_callback = [this](Individual* fish) {
                if(!cache() || !_graph)
                    return;
                
                std::lock_guard<std::recursive_mutex> guard(_graph->lock());
                
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
                for(auto id : selected) {
                    if(individuals.count(id)) {
                        auto fish = individuals.at(id);
                        if(!fish->has(frameIndex) && !fish->empty() && frameIndex >= fish->start_frame()) {
                            auto c = fish->cache_for_frame(frameIndex, time);
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
            
        } else {
            active.clear();
            active_blobs.clear();
            selected_blobs.clear();
        }
        
        bool something_important_changed = (not last_frame.valid() or frameIndex != last_frame) || last_threshold != threshold || selected != previous_active_fish || active_blobs != previous_active_blobs || _gui.mouse_position() != previous_mouse_position;
        if(something_important_changed || (is_tracking_dirty() && mode() == mode_t::tracking)) {
            previous_active_fish = selected;
            previous_active_blobs = active_blobs;
            previous_mouse_position = _gui.mouse_position();
            if(mode() == mode_t::blobs && something_important_changed)
                set_blobs_dirty();
            //else
            if(something_important_changed && mode() == mode_t::tracking)
                set_tracking_dirty();
            
            bool reload_blobs = (not last_frame.valid() || frameIndex != last_frame) || last_threshold != threshold;
            if(reload_blobs) {
                processed_frame.clear();
                
                if(frameIndex.valid()) {
                    try {
                        pv::Frame frame;
                        _video->read_frame(frame, frameIndex);
                        Tracker::instance()->preprocess_frame(*_video, std::move(frame), processed_frame, &_pool);
                        
                    } catch(const UtilsException&) {
                        FormatExcept("Frame ", frameIndex," cannot be loaded from file.");
                    }
                }
                
                /*display_blobs.clear();
                std::move(raw_blobs.begin(), raw_blobs.end(), std::back_inserter(available_blobs_list));
                raw_blobs.clear();*/
                
                probabilities.clear();
                checked_probs.clear();
                
                set_raw_blobs_dirty();
            }
            
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
            auto L = processed_frame.number_objects();
            
            if(reload_blobs) {
                display_blobs.clear();
                if(L < raw_blobs.size()) {
                    std::move(raw_blobs.begin() + L, raw_blobs.end(), std::back_inserter(available_blobs_list));
                    raw_blobs.erase(raw_blobs.begin() + L, raw_blobs.end());
                    
                } else if(L != raw_blobs.size()) {
                    raw_blobs.reserve(L);
                }
            }
            
            size_t i = 0;
            processed_frame.transform_blobs([&](pv::Blob& blob) {
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
                        
                    } else {
                        std::unique_ptr<SimpleBlob> ptr;
                        if(!available_blobs_list.empty()) {
                            ptr = std::move(available_blobs_list.back());
                            available_blobs_list.pop_back();
                            
                            ptr->blob = &blob;
                            ptr->threshold = threshold;
                        } else
                            ptr = std::make_unique<SimpleBlob>(std::make_unique<ExternalImage>(), &blob, threshold);
                        
                        raw_blobs.emplace_back(std::move(ptr));
                    }
                }
                
                ++i;
            });
            
            processed_frame.transform_noise([&](pv::Blob& blob) {
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
                        
                    } else {
                        std::unique_ptr<SimpleBlob> ptr;
                        if(!available_blobs_list.empty()) {
                            ptr = std::move(available_blobs_list.back());
                            available_blobs_list.pop_back();
                            
                            ptr->blob = &blob;
                            ptr->threshold = threshold;
                        } else
                            ptr = std::make_unique<SimpleBlob>(std::make_unique<ExternalImage>(), &blob, threshold);
                        
                        raw_blobs.emplace_back(std::move(ptr));
                    }
                }
                
                ++i;
            });
            
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
            
            last_frame = frameIndex;
            last_threshold = threshold;
        }
    }
    
    void GUICache::set_animating(Drawable *obj, bool v) {
        if (!obj && !v) {
            for (auto& [_k, _v] : _delete_handles) {
                _k->remove_delete_handler(_v);
            }
            _animators.clear();
            _delete_handles.clear();
            return;
        }

        if(v) {
            auto it = _animators.find(obj);
            if(it == _animators.end()) {
                _animators.insert(obj);
                _delete_handles[obj] = obj->on_delete([this, obj](){
                    if(!_graph)
                        return;
                    this->set_animating(obj, false);
                });
            }
        } else {
            auto it = _animators.find(obj);
            if(it != _animators.end()) {
                if(_delete_handles.count(obj)) {
                    auto handle = _delete_handles.at(obj);
                    _delete_handles.erase(obj);
                    obj->remove_delete_handler(handle);
                    
                } else
                    FormatError("Cannot find delete handler in GUICache. Something went wrong?");
                _animators.erase(it);
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
            auto c = processed_frame.cached(fdx);
            if(c) {
                processed_frame.transform_blobs([&](const pv::Blob& blob) {
                    auto it = individuals.find(fdx);
                    if(it == individuals.end() || it->second->empty() || frame_idx < it->second->start_frame())
                        return;
                    
                    auto p = individuals.at(fdx)->probability(processed_frame.label(blob.blob_id()), *c, frame_idx, blob);
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
        //static Timer timer;
        static Rect temporary;
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
                GUICache::instance().set_animating(&temporary, false);
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
                GUICache::instance().set_animating(&temporary, true);
            }

            if ((recording && GUICache::instance().gui_time() - time_lost >= 0.5)
                || (!recording && lost_timer.elapsed() >= 0.5))
            {
                target_scale = Vec2(1);
                //target_pos = offset;//Vec2(0, 0);
                target_size = Tracker::average().dimensions();
                target_pos = screen_center - target_size * 0.5;
                GUICache::instance().set_animating(&temporary, false);
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
            GUICache::instance().set_animating(section, true);

            auto playback_factor = max(1, sqrt(SETTING(gui_playback_speed).value<float>()));
            auto scale = check_target(section->scale(), target_scale, e * playback_factor);

            section->set_scale(scale);

            auto next_pos = check_target(section->pos(), target_pos, e * playback_factor);
            auto next_size = check_target(section->size(), target_size, e * playback_factor);

            section->set_bounds(Bounds(next_pos, next_size));

        }
        else {
            GUICache::instance().set_animating(section, false);

            section->set_scale(target_scale);
            section->set_bounds(Bounds(target_pos, target_size));
        }

        timer.reset();

        return { Vec2(), Vec2() };
    }
}
