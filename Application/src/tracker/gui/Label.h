#pragma once

#include <types.h>
#include <misc/vec2.h>
#include <gui/types/StaticText.h>
#include <gui/DrawStructure.h>
#include <misc/Timer.h>
#include <gui/Coordinates.h>

namespace gui {

class Label : public Entangled {
    GETTER_SETTER_I(float, line_length, 60)
    GETTER_I(Color, line_color, Cyan)
    GETTER_I(Color, fill_color, Transparent)
    GETTER(Color, color);
    GETTER(derived_ptr<StaticText>, text);
    Line _line;
    GETTER(Bounds, source);
    GETTER(Vec2, center);
    Timer animation_timer;
    const std::string animator;
    bool _registered{false};
    GETTER(Frame_t, frame);
    GETTER_I(bool, position_override, false)
    GETTER_SETTER(Vec2, override_position)
    
public:
    Label(const std::string& txt = "", const Bounds& source = {}, const Vec2& center = {});
    ~Label();
    void update() override;
    float update(const FindCoord&, float alpha, float distance, bool disabled, double dt, Scale = {});
    void set_data(Frame_t frame, const std::string& text, const Bounds& source, const Vec2& center);
    float update_positions(Vec2 text_pos, bool animate, double dt);
    void set_line_color(const Color& c) {
		if(c == line_color())
			return;
		_line_color = c;
        set_content_changed(true);
	}
    void set_fill_color(const Color& c) {
        if (c == fill_color())
            return;
        _fill_color = c;
        set_content_changed(true);
    }

    using Entangled::set;
    void set(attr::Loc) override;
    void set(attr::FillClr) override;
    void set(attr::LineClr) override;

    std::string toStr() const override {
        return "Label<"+Meta::toStr(_source) + ", " + Meta::toStr(text()->text()) + ">";
    }
    static std::string class_name() {
        return "Label";
    }

    void set_position_override(bool v) {
		if(v == _position_override)
			return;
		_position_override = v;
        set_dirty();
        set_content_changed(true);
	}
};

}
