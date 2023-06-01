#pragma once

#include <types.h>
#include <misc/vec2.h>
#include <gui/Section.h>
#include <gui/types/StaticText.h>
#include <gui/DrawStructure.h>

namespace gui {

class Label {
    GETTER(Color, color)
    GETTER(derived_ptr<StaticText>, text)
    GETTER(Bounds, source)
    GETTER(Vec2, center)
    
public:
    Label(const std::string& txt, const Bounds& source, const Vec2& center);
    ~Label();
    void update(Base* base, Drawable* fishbowl, Entangled& e, float alpha, bool disabled);
    void set_data(const std::string& text, const Bounds& source, const Vec2& center);
    void update_positions(Entangled& e, Vec2 text_pos);

    std::string toStr() const {
        return "Label<"+Meta::toStr(_source) + ", '" + text()->text() + "'>";
    }
    static std::string class_name() {
        return "Label";
    }
};

}
