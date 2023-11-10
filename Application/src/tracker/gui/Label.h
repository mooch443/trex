#pragma once

#include <types.h>
#include <misc/vec2.h>
#include <gui/types/StaticText.h>
#include <gui/DrawStructure.h>
#include <misc/Timer.h>
#include <gui/Coordinates.h>

namespace gui {

class Label {
    GETTER(Color, color)
    GETTER(derived_ptr<StaticText>, text)
    Line _line;
    GETTER(Bounds, source)
    GETTER(Vec2, center)
    Timer animation_timer;
    const std::string animator;
    bool _registered{false};
    GETTER(Frame_t, frame)
    
public:
    Label(const std::string& txt, const Bounds& source, const Vec2& center);
    ~Label();
    void update(const FindCoord&, Entangled& e, float alpha, float distance, bool disabled, double dt);
    void set_data(Frame_t frame, const std::string& text, const Bounds& source, const Vec2& center);
    float update_positions(Entangled& e, Vec2 text_pos, bool animate, double dt);

    std::string toStr() const {
        return "Label<"+Meta::toStr(_source) + ", '" + text()->text() + "'>";
    }
    static std::string class_name() {
        return "Label";
    }
};

}
