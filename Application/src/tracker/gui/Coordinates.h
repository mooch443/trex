#pragma once
#include <commons.pc.h>
#include <misc/vec2.h>
#include <gui/Transform.h>
#include <gui/ControlsAttributes.h>

namespace gui {

class Base;
class DrawStructure;

struct HUDCoord : Vec2 {
    using Vec2::Vec2;

    explicit HUDCoord(const Vec2& v) : Vec2(v) { }
};

struct BowlCoord : Vec2 {
    using Vec2::Vec2;

    explicit BowlCoord(const Vec2& v) : Vec2(v) { }
};

struct HUDRect : Bounds {
    using Bounds::Bounds;

    explicit HUDRect(const Bounds& b) : Bounds(b) { }
};

struct BowlRect : Bounds {
    using Bounds::Bounds;

    explicit BowlRect(const Bounds& b) : Bounds(b) { }
};

class FindCoord {
    static auto& mutex() {
        static auto m = new LOGGED_MUTEX("FindCoord::mutex");
        return *m;
    }

    Size2 resolution, window_size;
    Transform hud_to_bowl;
    Transform bowl_to_hud;

    FindCoord();
    //~FindCoord();
    FindCoord(const FindCoord&) = default;
    FindCoord(FindCoord&&) = default;
    FindCoord& operator=(const FindCoord&) noexcept = default;
    FindCoord& operator=(FindCoord&&) noexcept = default;

public:

    BowlCoord convert(const HUDCoord& hud) const;
    HUDCoord convert(const BowlCoord& bowl) const;
    BowlRect convert(const HUDRect& hud) const;
    HUDRect convert(const BowlRect& bowl) const;

    const Size2& screen_size() const;
    const Size2& video_size() const;
    
    attr::Scale bowl_scale() const;
    BowlRect viewport() const;
    HUDRect hud_viewport() const;

    static void set_bowl_transform(const Transform& bowl_transform);
    static void set_video(const Size2& resolution);
    static Size2 set_screen_size(const DrawStructure&, const Base&, const Vec2& = Vec2(1));

    static FindCoord get();
    
private:
    static FindCoord& instance();
};

} // namespace gui
