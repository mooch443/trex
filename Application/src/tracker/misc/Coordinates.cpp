#include "Coordinates.h"
#include <gui/DrawBase.h>
#include <gui/DrawStructure.h>

namespace gui {

FindCoord& FindCoord::instance() {
    static FindCoord instance;
    return instance;
}

FindCoord FindCoord::get() {
    auto lock = LOGGED_LOCK(mutex());
    return instance();
}

FindCoord::FindCoord() { }

BowlCoord FindCoord::convert(const HUDCoord& hud) const {
    auto point = hud_to_bowl.transformPoint(hud.x, hud.y);
    return BowlCoord(point.x, point.y);
}

HUDCoord FindCoord::convert(const BowlCoord& bowl) const {
    auto point = bowl_to_hud.transformPoint(bowl.x, bowl.y);
    return HUDCoord(point.x, point.y);
}

BowlRect FindCoord::convert(const HUDRect& hud) const {
    auto rect = hud_to_bowl.transformRect(hud);
    return BowlRect(rect);
}

HUDRect FindCoord::convert(const BowlRect& bowl) const {
    auto rect = bowl_to_hud.transformRect(bowl);
    return HUDRect(rect);
}

Size2 FindCoord::set_screen_size(const DrawStructure& graph, const Base &window, const Vec2& scale) {
    auto update = window.window_dimensions().div(graph.scale()).mul(scale) * gui::interface_scale();
    
    auto lock = LOGGED_LOCK(mutex());
    if(update != instance().window_size) {
        //print("Updating screen from ", instance().window_size, " to ", update, " with ", window.window_dimensions(), " scale=",graph.scale(), " and=",scale," interface=",gui::interface_scale());
        instance().window_size = update;
    }
    
    return update;
}

void FindCoord::set_bowl_transform(const Transform& t) {
    auto lock = LOGGED_LOCK(mutex());
    if(instance().bowl_to_hud != t) {
        // this is simply a copy:
        instance().bowl_to_hud = t;
        
        // get the inverse of the transform to map from
        // HUD coordinates to bowl coordinates:
        instance().hud_to_bowl = t.getInverse();
        
        //print("Coord::video_size = ", instance().video_size());
        //print("Coord::screen_size = ", instance().screen_size());
        //print("Coord::bowl_scale = ", instance().bowl_scale());
        //print("Coord::viewport = ", instance().viewport());
    }
}

void FindCoord::set_video(const Size2& r) {
    auto lock = LOGGED_LOCK(mutex());
    instance().resolution = r;
}

const Size2& FindCoord::video_size() const {
    return resolution;
}

const Size2& FindCoord::screen_size() const {
    return window_size;
}

attr::Scale FindCoord::bowl_scale() const {
    return attr::Scale(convert(BowlCoord(1)) - convert(BowlCoord(0)));
}

BowlRect FindCoord::viewport() const {
    return convert(HUDRect(Vec2(), screen_size()));
}

HUDRect FindCoord::hud_viewport() const {
    return convert(BowlRect(Vec2(), video_size()));
}

} // namespace gui
