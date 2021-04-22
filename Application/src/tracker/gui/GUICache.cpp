#include "GUICache.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <tracking/Tracker.h>
#include <gui/DrawFish.h>
#include <gui/gui.h>

namespace gui {
    static std::unique_ptr<std::thread> percentile_ptr = nullptr;
    static std::mutex percentile_mutex;

    SimpleBlob::SimpleBlob(std::unique_ptr<ExternalImage>&& available, pv::BlobPtr b, int t)
        : blob(b), threshold(t), ptr(std::move(available))
    {
        
    }

    GUICache::~GUICache() {
        std::lock_guard guard(percentile_mutex);
        if(percentile_ptr) {
            percentile_ptr->join();
            percentile_ptr = nullptr;
        }
    }
    
    std::unique_ptr<ExternalImage> SimpleBlob::convert() {
        //static Timing timing("simpleblob", 10);
        //TakeTiming take(timing);
        Vec2 image_pos;
        Image::UPtr image;
        
        auto &percentiles = GUI::cache().pixel_value_percentiles;
        if(GUI::cache()._equalize_histograms && !percentiles.empty()) {
            auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), threshold, percentiles.front(), percentiles.back());
            image_pos = pos;
            image = std::move(img);
        } else {
            auto && [pos, img] = blob->luminance_alpha_image(*Tracker::instance()->background(), threshold);
            image_pos = pos;
            image = std::move(img);
        }
        
        //e->update_with(*image);
        if(!ptr)
            ptr = std::make_unique<ExternalImage>(std::move(image), image_pos);
        else {
            ptr->update_with(*image);
            ptr->set_pos(image_pos);
        }
        
        ptr->add_custom_data("blob_id", (void*)(uint64_t)blob->blob_id());
        if(ptr->name().empty())
            ptr->set_name("SimpleBlob_"+Meta::toStr(blob->blob_id()));
        return std::move(ptr);
    }
    
    GUICache::GUICache()
        : last_threshold(-1), last_frame(-1), _dirty(true), _equalize_histograms(true), _blobs_dirty(false), _raw_blobs_dirty(false), _mode(mode_t::tracking), _zoom_level(1),  _tracking_dirty(false), recognition_updated(false)
    {}
    
    bool GUICache::has_selection() const {
        return !selected.empty() && individuals.count(selected.front()) != 0;
    }
    
    Individual * GUICache::primary_selection() const {
        return has_selection() && individuals.count(selected.front()) ? individuals.at(selected.front()) : NULL;
    }
    
    bool GUICache::is_animating(Drawable* obj) const {
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
        selected.clear();
    }
    
    bool GUICache::is_selected(Idx_t id) const {
        return contains(selected, id);
    }
    
    void GUICache::do_select(Idx_t id) {
        if(!is_selected(id)) {
            selected.push_back(id);
            SETTING(gui_focus_group) = selected;
        }
    }
    
    void GUICache::deselect(Idx_t id) {
        auto it = std::find(selected.begin(), selected.end(), id);
        if(it != selected.end()) {
            selected.erase(it);
            SETTING(gui_focus_group) = selected;
        }
    }
    
    void GUICache::deselect_all_select(Idx_t id) {
        selected.clear();
        selected.push_back(id);
        
        SETTING(gui_focus_group) = selected;
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
        if(GUI::instance())
            GUI::instance()->gui().set_dirty(nullptr);
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
    
    void GUICache::update_data(long_t frameIndex) {
        const auto threshold = FAST_SETTINGS(track_threshold);
        auto& _tracker = *Tracker::instance();
        auto& _gui = GUI::instance()->gui();
        _equalize_histograms = GUI_SETTINGS(gui_equalize_blob_histograms);
        
        frame_idx = frameIndex;
        
        static std::atomic_bool done_calculating = false;
        {
            std::lock_guard guard(percentile_mutex);
            if(!done_calculating && !percentile_ptr) {
                percentile_ptr = std::make_unique<std::thread>([this](){
                    cmn::set_thread_name("percentile_thread");
                    auto percentiles = GUI::instance()->video_source()->calculate_percentiles({0.05f, 0.95f});
                    
                    if(GUI::instance()) {
                        std::lock_guard<std::recursive_mutex> guard(GUI::instance()->gui().lock());
                        pixel_value_percentiles = percentiles;
                    }
                    
                    done_calculating = true;
                });
            }
            
            if(percentile_ptr && done_calculating) {
                percentile_ptr->join();
                percentile_ptr = nullptr;
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
            individuals = _tracker.individuals();
            selected = SETTING(gui_focus_group).value<std::vector<Idx_t>>();
            active_blobs.clear();
            inactive_ids.clear();
            active_ids.clear();
            fish_selected_blobs.clear();
            inactive_estimates.clear();
            tracked_frames = Rangel(_tracker.start_frame(), _tracker.end_frame());
            
            auto delete_callback = [this](Individual* fish) {
                std::lock_guard<std::recursive_mutex> guard(GUI::instance()->gui().lock());
                
                auto id = fish->identity().ID();
                auto it = individuals.find(id);
                if(it != individuals.end())
                    individuals.erase(it);
                
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
            
            if(FAST_SETTINGS(manual_identities).empty()) {
                for(auto fish : active) {
                    if(!_registered_callback.count(fish)) {
                        fish->register_delete_callback((void*)12341337, delete_callback);
                        _registered_callback.insert(fish);
                    }
                }
            } else {
                
                for(auto id : FAST_SETTINGS(manual_identities)) {
                    auto it = individuals.find(id);
                    if(it != individuals.end()) {
                        it->second->register_delete_callback((void*)12341337, delete_callback);
                    }
                }
            }
            
            auto connectivity_map = SETTING(gui_connectivity_matrix).value<std::map<long_t, std::vector<float>>>();
            if(connectivity_map.count(frameIndex))
                connectivity_matrix = connectivity_map.at(frameIndex);
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
                
                fish->register_delete_callback((void*)133742, [this](auto){
                    std::lock_guard<std::recursive_mutex> guard(GUI::instance()->gui().lock());
                    active.clear();
                    individuals.clear();
                    active_blobs.clear();
                    active_ids.clear();
                    inactive_ids.clear();
                    last_frame = -1;
                    selected.clear();
                });
            }
            
            if(has_selection()) {
                for(auto id : selected) {
                    if(individuals.count(id)) {
                        auto fish = individuals.at(id);
                        if(!fish->has(frameIndex) && !fish->empty() && frameIndex >= fish->start_frame()) {
                            auto c = fish->cache_for_frame(frameIndex, time);
                            inactive_estimates.push_back(c.estimated_px);
                            inactive_ids.insert(fish->identity().ID());
                        }
                    }
                }
            }
            
            bool shift = _gui.is_key_pressed(gui::LShift) && (!_gui.selected_object() || !dynamic_cast<Textfield*>(_gui.selected_object()));
#if WITH_SFML
            shift = shift && (!_base || _base->window().hasFocus());
#endif
            
            if(!has_selection() || !SETTING(gui_auto_scale_focus_one) || shift) {
                // display all blobs that are assigned to an individual
                for(auto fish : active) {
                    auto blob = fish->compressed_blob(frameIndex);
                    if(blob)
                        active_blobs.insert(blob->blob_id());
                }
                
            } else {
                // display blobs that are selected
                for(auto id : selected) {
                    auto it = individuals.find(id);
                    if(it != individuals.end()) {
                        auto blob = it->second->compressed_blob(frameIndex);
                        if(blob)
                            active_blobs.insert(blob->blob_id());
                    }
                }
            }
            
        } else {
            active.clear();
            active_blobs.clear();
        }
        
        bool something_important_changed = frameIndex != last_frame || last_threshold != threshold || selected != previous_active_fish || active_blobs != previous_active_blobs || _gui.mouse_position() != previous_mouse_position;
        if(something_important_changed || (is_tracking_dirty() && mode() == mode_t::tracking)) {
            previous_active_fish = selected;
            previous_active_blobs = active_blobs;
            previous_mouse_position = _gui.mouse_position();
            if(mode() == mode_t::blobs && something_important_changed)
                set_blobs_dirty();
            //else
            if(something_important_changed && mode() == mode_t::tracking)
                set_tracking_dirty();
            
            //if(something_important_changed)
            //    set_raw_blobs_dirty();
            //set_blobs_dirty();
            //automatic_assignments = Tracker::blob_automatically_assigned(frameIndex);
            
            bool reload_blobs = frameIndex != last_frame || last_threshold != threshold;
            if(reload_blobs) {
                processed_frame.frame().clear();
                processed_frame.clear();
                
                if(frameIndex >= 0) {
                    Tracker::set_of_individuals_t prev_active;
                    if(_tracker.properties(frameIndex-1))
                        prev_active = _tracker.active_individuals(frameIndex-1);
                    
                    try {
                        auto file = static_cast<pv::File*>(GUI::instance()->video_source());
                        file->read_frame(processed_frame.frame(), (size_t)frameIndex);
                        
                        std::lock_guard<std::mutex> guard(GUI::instance()->blob_thread_pool_mutex());
                        Tracker::instance()->preprocess_frame(processed_frame, prev_active, &GUI::instance()->blob_thread_pool());
                        
                    } catch(const UtilsException& e) {
                        Except("Frame %d cannot be loaded from file.", frameIndex);
                    }
                }
                
                raw_blobs.clear();
                display_blobs.clear();
                
                std::move(display_blobs_list.begin(), display_blobs_list.end(), std::back_inserter(available_blobs_list));
                //std::reverse(available_blobs_list.begin(), available_blobs_list.end());
                
                //Debug("Size: %lu", available_blobs_list.size());
                
                display_blobs_list.clear();
                
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
            
            const bool nothing_to_zoom_on = !has_selection() || (inactive_estimates.empty() && active_blobs.empty());
            
            _num_pixels = 0;
            
            for (size_t i=0; i<processed_frame.blobs.size(); i++) {
                auto blob = processed_frame.blobs.at(i);
                
                if(nothing_to_zoom_on || active_blobs.find(blob->blob_id()) != active_blobs.end())
                {
                    min_vec = min(min_vec, blob->bounds().pos());
                    max_vec = max(max_vec, blob->bounds().pos() + blob->bounds().size());
                }
                
                _num_pixels += size_t(blob->bounds().width * blob->bounds().height);
                
                if(reload_blobs) {
                    std::unique_ptr<gui::ExternalImage> ptr;
                    if(!available_blobs_list.empty()) {
                        ptr = std::move(available_blobs_list.back());
                        available_blobs_list.pop_back();
                    }
                    
                    raw_blobs.push_back(std::make_shared<SimpleBlob>(std::move(ptr), blob, threshold));
                }
            }
            
            for(auto blob : processed_frame.filtered_out) {
                blob->calculate_moments();
                
                if((nothing_to_zoom_on && blob->recount(-1) >= FAST_SETTINGS(blob_size_ranges).max_range().start)
                   || active_blobs.find(blob->blob_id()) != active_blobs.end())
                {
                    min_vec = min(min_vec, blob->bounds().pos());
                    max_vec = max(max_vec, blob->bounds().pos() + blob->bounds().size());
                }
                
                if(reload_blobs) {
                    std::unique_ptr<gui::ExternalImage> ptr;
                    if(!available_blobs_list.empty()) {
                        ptr = std::move(available_blobs_list.back());
                        available_blobs_list.pop_back();
                    }
                    
                    raw_blobs.push_back(std::make_shared<SimpleBlob>(std::move(ptr), blob, threshold));
                }
            }
            
            if(reload_blobs) {
                display_blobs.clear();
                display_blobs_list.clear();
            }
            boundary = Bounds(min_vec, max_vec - min_vec);
            
            last_frame = frameIndex;
            last_threshold = threshold;
        }
    }
    
    void GUICache::set_animating(Drawable *obj, bool v) {
        if(v) {
            auto it = _animators.find(obj);
            if(it == _animators.end()) {
                _animators.insert(obj);
                _delete_handles[obj] = obj->on_delete([this, obj](){
                    if(!GUI::instance())
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
                    Error("Cannot find delete handler in GUICache. Something went wrong?");
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

    const std::map<uint32_t, Individual::Probability>* GUICache::probs(Idx_t fdx) {
        if(checked_probs.find(fdx) != checked_probs.end()) {
            auto it = probabilities.find(fdx);
            if(it  != probabilities.end())
                return &it->second;
            return nullptr;
        }
        
        checked_probs.insert(fdx);
        
        {
            Tracker::LockGuard guard("GUICache::probs");
            auto it = processed_frame.cached_individuals.find(fdx);
            if(it != processed_frame.cached_individuals.end()) {
                auto && [fdx, cache] = *it;
                for(auto blob : processed_frame.blobs) {
                    auto p = individuals.count(fdx) ? individuals.at(fdx)->probability(cache, frame_idx, blob) : Individual::Probability{0,0,0,0};
                    if(p.p >= FAST_SETTINGS(matching_probability_threshold))
                        probabilities[fdx][blob->blob_id()] = p;
                }
            }
        }
        
        auto it = probabilities.find(fdx);
        if(it != probabilities.end())
            return &it->second;
        return nullptr;
    }
}
