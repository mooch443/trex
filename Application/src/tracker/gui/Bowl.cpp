#include "Bowl.h"
#include <gui/GUICache.h>
#include <gui/DrawFish.h>
#include <gui/IdentityHeatmap.h>
#include <gui/VisualFieldWidget.h>
#include <gui/IdentityHeatmap.h>

using namespace track;

namespace gui {

Bowl::Bowl(GUICache* cache) : _cache(cache), _vf_widget(new VisualFieldWidget) {
    _current_scale = Vec2(1.0f, 1.0f);
    _target_scale = Vec2(1.0f, 1.0f);
    _current_pos = Vec2(0.0f, 0.0f);
    _target_pos = Vec2(0.0f, 0.0f);
    _aspect_ratio = Vec2(1.0f, 1.0f);
    _max_zoom = Vec2(300.0f, 300.0f);
    _screen_size = Vec2(0.0f, 0.0f);
    _center_of_screen = Vec2(0.0f, 0.0f);
    _timer.reset();
}

Bowl::~Bowl() {
    if(_vf_widget)
        delete _vf_widget;
    if(_heatmapController)
        delete _heatmapController;
}

bool Bowl::has_target_points_changed(const std::vector<Vec2>& new_target_points) const {
    return _target_points != new_target_points;
}

bool Bowl::has_screen_size_changed(const Vec2& new_screen_size) const {
    return _screen_size != new_screen_size;
}

void Bowl::set_video_aspect_ratio(float video_width, float video_height) {
    if (video_width == 0 or video_height == 0) {
        return;
    }
    _aspect_ratio = Vec2(video_width, video_height);
    _video_size = {video_width, video_height};
    FindCoord::set_video(_video_size);
}

void Bowl::fit_to_screen(const Vec2& screen_size) {
    if (not has_screen_size_changed(screen_size)) {
        return;
    }
    _screen_size = screen_size;
    _center_of_screen = _screen_size / 2;
    
    set_size(_video_size);
    update_goals();
}

void Bowl::set_target_focus(const std::vector<Vec2>& target_points) {
    if (not has_target_points_changed(target_points)) {
        return;
    }
    _target_points = target_points;
    update_goals();
}

void Bowl::update_goals() {
    if (_target_points.empty()) {
        float width_scale = _screen_size.x / _aspect_ratio.x;
        float height_scale = _screen_size.y / _aspect_ratio.y;
        float scale_factor = std::min(width_scale, height_scale);
        _target_scale = Vec2(scale_factor, scale_factor);
        _target_pos = _center_of_screen - _video_size.mul(_target_scale) / 2;
        
        return;
    }
    
    // Compute the bounding box and its center of mass
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::min();
    
    Vec2 sum_of_points(0.0f, 0.0f);
    for (const auto& point : _target_points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
        sum_of_points = sum_of_points + point;
    }
    
    if(_cache)
        _max_zoom = GUI_SETTINGS(gui_zoom_limit);
    else
        _max_zoom = SETTING(gui_zoom_limit).value<Size2>();
    
    Bounds bounding_box(Vec2(min_x, min_y), Size2(max_x - min_x + 1, max_y - min_y + 1));
    if(bounding_box.width < _max_zoom.x) {
        auto diff = _max_zoom.x - bounding_box.width;
        bounding_box.x -= diff / 2;
        bounding_box.width = _max_zoom.x;
    }
    if(bounding_box.height < _max_zoom.y) {
        auto diff = _max_zoom.y - bounding_box.height;
        bounding_box.y -= diff / 2;
        bounding_box.height = _max_zoom.y;
    }
    
    Vec2 theory_scale = _screen_size.div(bounding_box.size());
    auto o = _screen_size.div(theory_scale).max() * 0.25;
    bounding_box << bounding_box.pos() - o;
    bounding_box << bounding_box.size() + o * 2;
    
    // Find the required scaling factor, but don't exceed the max zoom
    Vec2 scale_required = _screen_size.div(bounding_box.size());
    Vec2 new_target_scale = Vec2( scale_required.min(), scale_required.min() );
    
    // Update the target scale
    _target_scale = new_target_scale;
    
    // Calculate the target position
    _target_pos = _center_of_screen - bounding_box.center().mul(_target_scale);
}

void Bowl::update_blobs(const Frame_t& frame) {
    //graph.section("blobs", [&, &_cache = _data->_cache, frame = GUI_SETTINGS(gui_frame)](DrawStructure &graph, Section* s)
    if(_cache) {
        //s->set_scale(_data->_bowl->scale());
        //s->set_pos(_data->_bowl->pos());
        
        const auto mode = GUI_SETTINGS(gui_mode);
        const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
        
        bool draw_blobs_separately = true;
        if(draw_blobs_separately)
        {
            if(GUI_SETTINGS(gui_mode) == gui::mode_t::tracking
               && _cache->tracked_frames.contains(frame))
            {
                std::unique_lock guard(_cache->_fish_map_mutex);
                for(auto &&[k,fish] : _cache->_fish_map) {
                    auto obj = fish->shadow();
                    if(obj)
                        advance_wrap(*obj);
                }
            }
            
            if (draw_blobs) {
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                const bool gui_macos_blur = GUI_SETTINGS(gui_macos_blur);
#endif
                if (GUI_SETTINGS(gui_mode) != gui::mode_t::blobs) {
                    for (auto& [b, ptr] : _cache->display_blobs) {
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                        if constexpr (std::is_same<MetalImpl, default_impl_t>::value) {
                            if (gui_macos_blur)
                                ptr->ptr->tag(Effects::blur);
                        }
#endif
                        advance_wrap(*(ptr->ptr));
                    }

                }
                else {
                    for (auto& [b, ptr] : _cache->display_blobs) {
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                        if constexpr (std::is_same<MetalImpl, default_impl_t>::value) {
                            if (gui_macos_blur)
                                ptr->ptr->untag(Effects::blur);
                        }
#endif
                        advance_wrap(*(ptr->ptr));
                    }
                }
            }
            
        } else if(draw_blobs
                  && GUI_SETTINGS(gui_mode) == gui::mode_t::tracking
                  && _cache->tracked_frames.contains(frame))
        {
            std::unique_lock guard(_cache->_fish_map_mutex);
            for(auto &&[k,fish] : _cache->_fish_map) {
                auto obj = fish->shadow();
                if(obj)
                    advance_wrap(*obj);
            }
        }
    }
}

void Bowl::set_data(Frame_t frame) {
}

void Bowl::update_scaling() {
    const auto dt = saturate(_timer.elapsed(), 0.001, 0.1);
    
    const float lerp_speed = 3.0f;
    float lerp_factor = 1.0f - std::exp(-lerp_speed * dt);
    
    _current_pos = _current_pos + (_target_pos - _current_pos) * lerp_factor;
    _current_scale = _current_scale + (_target_scale - _current_scale) * lerp_factor;
    
    _current_size = _current_size + (_video_size.mul(_current_scale) - _current_size) * lerp_factor;
    
    if(not scale().Equals(_current_scale)
       || not pos().Equals(_current_pos))
    {
        set_scale(_current_scale);
        set_pos(_current_pos);
        
        FindCoord::set_bowl_transform(global_transform());
    }
    
    _timer.reset();
}

void Bowl::update(Frame_t frame, DrawStructure &graph, const FindCoord& coord) {
    update([this, &frame, &graph, &coord](auto&) {
        update_blobs(frame);
        
        if(GUI_SETTINGS(gui_mode) != gui::mode_t::tracking)
            return;
        
        if(GUI_SETTINGS(gui_show_heatmap)) {
            if(!_heatmapController)
                _heatmapController = new gui::heatmap::HeatmapController;
            _heatmapController->set_frame(frame);
            advance_wrap(*_heatmapController);
        }
        
        if (_cache) {
            if (_cache->has_selection()
                && GUI_SETTINGS(gui_show_visualfield))
            {
                {
                    LockGuard guard(ro_t{}, "visual_field", 10);
                    set_of_individuals_t s;
                    for(auto idx : _cache->selected) {
                        if(auto it = _cache->individuals.find(idx);
                           it != _cache->individuals.end())
                        {
                            s.insert(it->second);
                        }
                    }
                    _vf_widget->update(frame, coord, s);
                }
                advance_wrap(*_vf_widget);
            }

            std::scoped_lock guard(_cache->_fish_map_mutex);
            for (auto& [id, fish] : _cache->_fish_map) {
                fish->update(coord, *this, graph);
            }
        }
    });
}

void Bowl::set_max_zoom_size(const Vec2& max_zoom) {
    _max_zoom = max_zoom;
}

}
