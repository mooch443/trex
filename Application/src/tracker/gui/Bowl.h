#pragma once

#include <gui/types/Entangled.h>
#include <misc/Coordinates.h>
#include <misc/Timer.h>

namespace cmn::gui {

class GUICache;

class Bowl : public Entangled {
    struct Data;
    std::unique_ptr<Data> _data;
    GUICache* _cache;
    
public:
    Bowl(GUICache* cache);
    ~Bowl();
    void set_video_aspect_ratio(float video_width, float video_height);
    void fit_to_screen(const Vec2& screen_size);
    void set_target_focus(const std::vector<Vec2>& target_points);
    
    using Entangled::update;
    void update_scaling();
    void update(Frame_t, DrawStructure&, const FindCoord&);
    void set_max_zoom_size(const Vec2& max_zoom);
    
public:
    bool has_target_points_changed(const std::vector<Vec2>& new_target_points) const;
    bool has_screen_size_changed(const Vec2& new_screen_size) const;
    void update_goals();
    void update_blobs(const Frame_t& frame);
    void set_data(Frame_t frame);
    
    void draw_shapes(DrawStructure&, const FindCoord&);
    
    Vec2 _current_scale;
    Vec2 _target_scale;
    Vec2 _current_pos;
    Vec2 _target_pos;
    Vec2 _aspect_ratio;
    Vec2 _screen_size;
    Vec2 _center_of_screen;
    Vec2 _max_zoom;
    Vec2 _current_size;
    Vec2 _video_size;
    Timer _timer;
    std::vector<Vec2> _target_points;
};

}
