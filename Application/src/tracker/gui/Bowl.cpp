#include "Bowl.h"
#include <gui/GUICache.h>
#include <gui/DrawFish.h>
#include <gui/IdentityHeatmap.h>
#include <gui/VisualFieldWidget.h>
#include <gui/IdentityHeatmap.h>
#include <gui/DrawBase.h>
#include <tracking/LockGuard.h>

using namespace track;

namespace gui {

struct Shape {
    std::vector<Vec2> points;
    bool operator<(const Shape& rhs) const {
        return points < rhs.points;
    }

    std::string toStr() const {
        return "Shape<" + Meta::toStr(points) + ">";
    }
};

struct Bowl::Data {
    VisualFieldWidget _vf_widget;
    Frame_t _last_frame;
    
    //! The heatmap controller.
    gui::heatmap::HeatmapController* _heatmapController{nullptr};
    
    std::map<Shape, std::unique_ptr<Drawable>> _include_shapes, _ignore_shapes;
    
    ~Data() {
        if(_heatmapController)
            delete _heatmapController;
    }
};

Bowl::Bowl(GUICache* cache) :
    _data(new Data{ }), _cache(cache)
{
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

Bowl::~Bowl() {}

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

void Bowl::update_shapes() {
    if(!FAST_SETTING(track_include).empty())
    {
        auto keys = extract_keys(_data->_include_shapes);
        
        for(auto &rect : FAST_SETTING(track_include)) {
            Shape shape{rect};
            auto it = _data->_include_shapes.find(shape);
            if(it == _data->_include_shapes.end()) {
                if(rect.size() == 2) {
                    auto ptr = std::make_unique<Rect>(Box(rect[0], rect[1] - rect[0]), FillClr{Green.alpha(25)}, LineClr{Green.alpha(100)});
                    //ptr->set_clickable(true);
                    _data->_include_shapes[shape] = std::move(ptr);
                    
                } else if(rect.size() > 2) {
                    //auto r = std::make_shared<std::vector<Vec2>>(rect);
                    auto r = poly_convex_hull(&rect); // force a convex polygon for these shapes, as thats the only thing that the in/out polygon test works with
                    auto ptr = std::make_unique<gui::Polygon>(r);
                    ptr->set_fill_clr(Green.alpha(25));
                    ptr->set_border_clr(Green.alpha(100));
                    //ptr->set_clickable(true);
                    _data->_include_shapes[shape] = std::move(ptr);
                }
            }
            keys.erase(shape);
        }
        
        for(auto &key : keys) {
            _data->_include_shapes.erase(key);
        }
        
        _cache->set_raw_blobs_dirty();
        
    } else if(FAST_SETTING(track_include).empty()
              && !_data->_include_shapes.empty())
    {
        _data->_include_shapes.clear();
        _cache->set_raw_blobs_dirty();
    }
    
    if(!FAST_SETTING(track_ignore).empty())
    {
        auto keys = extract_keys(_data->_ignore_shapes);
        
        for(auto &rect : FAST_SETTING(track_ignore)) {
            Shape shape{rect};
            auto it = _data->_ignore_shapes.find(shape);
            if(it == _data->_ignore_shapes.end()) {
                if(rect.size() == 2) {
                    auto ptr = std::make_unique<Rect>(Box(rect[0], rect[1] - rect[0]), FillClr{Red.alpha(25)}, LineClr{Red.alpha(100)});
                    //ptr->set_clickable(true);
                    _data->_ignore_shapes[shape] = std::move(ptr);
                    
                } else if(rect.size() > 2) {
                    //auto r = std::make_shared<std::vector<Vec2>>(rect);
                    auto r = poly_convex_hull(&rect); // force convex polygon
                    auto ptr = std::make_unique<gui::Polygon>(r);
                    ptr->set_fill_clr(Red.alpha(25));
                    ptr->set_border_clr(Red.alpha(100));
                    //ptr->set_clickable(true);
                    _data->_ignore_shapes[shape] = std::move(ptr);
                }
            }
            keys.erase(shape);
        }
        
        for(auto &key : keys) {
            _data->_ignore_shapes.erase(key);
        }
        
        _cache->set_raw_blobs_dirty();
        
    } else if(FAST_SETTING(track_ignore).empty() && !_data->_ignore_shapes.empty()) {
        _data->_ignore_shapes.clear();
        _cache->set_raw_blobs_dirty();
    }
}

void Bowl::draw_shapes(DrawStructure &, const FindCoord &coord) {
    //! TODO: Thread-safety?
    //const auto size = coord.video_size();
    //const float max_w = size.width;
    //const float max_h = size.height;
    
    /*if((PD(tracking)._recognition_image.source()->cols != max_w || PD(tracking)._recognition_image.source()->rows != max_h) && Tracker::instance()->border().type() != Border::Type::none) {
        auto border_distance = Image::Make(max_h, max_w, 4);
        border_distance->set_to(0);
        
        auto worker = [&border_distance, max_h](ushort x) {
            for (ushort y = 0; y < max_h; ++y) {
                if(Tracker::instance()->border().in_recognition_bounds(Vec2(x, y)))
                    border_distance->set_pixel(x, y, DarkCyan.alpha(15));
            }
        };
        
        {
            print("Calculating border...");
            
            std::lock_guard<std::mutex> guard(blob_thread_pool_mutex());
            try {
                for(ushort x = 0; x < max_w; ++x) {
                    blob_thread_pool().enqueue(worker, x);
                }
            } catch(...) {
                FormatExcept("blob_thread_pool error when enqueuing worker to calculate border.");
            }
            blob_thread_pool().wait();
        }
        
        PD(tracking)._recognition_image.set_source(std::move(border_distance));
        PD(cache).set_tracking_dirty();
        PD(cache).set_blobs_dirty();
        PD(cache).set_raw_blobs_dirty();
        PD(cache).set_redraw();
    }*/
    update_shapes();
    
    Scale scale{coord.bowl_scale()};
    for(auto && [rect, ptr] : _data->_include_shapes) {
        advance_wrap(*ptr);
        
        if(ptr->hovered()) {
            const Font font(0.85 / (1 - ((1 - _cache->zoom_level()) * 0.5)), Align::VerticalCenter);
            add<Text>(Str("allowing "+Meta::toStr(rect)), Loc(ptr->pos() + Vec2(5, Base::default_line_spacing(font) + 5)), font, scale);
        }
    }
    
    for(auto && [rect, ptr] : _data->_ignore_shapes) {
        advance_wrap(*ptr);
        
        if(ptr->hovered()) {
            const Font font(0.85 / (1 - ((1 - _cache->zoom_level()) * 0.5)), Align::VerticalCenter);
            add<Text>(Str("excluding "+Meta::toStr(rect)), Loc(ptr->pos() + Vec2(5, Base::default_line_spacing(font) + 5)), font, scale);
        }
    }
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

void Bowl::set_data(Frame_t) {
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
        draw_shapes(graph, coord);
        
        update_blobs(frame);
        
        if(GUI_SETTINGS(gui_mode) != gui::mode_t::tracking)
            return;
        
        if(GUI_SETTINGS(gui_show_heatmap)) {
            if(!_data->_heatmapController)
                _data->_heatmapController = new gui::heatmap::HeatmapController;
            _data->_heatmapController->set_frame(frame);
            advance_wrap(*_data->_heatmapController);
        }
        
        if (_cache) {
            if (_cache->has_selection()
                && GUI_SETTINGS(gui_show_visualfield))
            {
                {
                    LockGuard guard(ro_t{}, "visual_field", 10);
                    set_of_individuals_t s;
                    auto lock = _cache->lock_individuals();
                    for(auto idx : _cache->selected) {
                        if(auto it = lock.individuals.find(idx);
                           it != lock.individuals.end())
                        {
                            s.insert(it->second);
                        }
                    }
                    _data->_vf_widget.update(frame, coord, s);
                }
                advance_wrap(_data->_vf_widget);
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
