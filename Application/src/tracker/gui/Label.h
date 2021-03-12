#pragma once

#include <types.h>
#include <misc/vec2.h>
#include <gui/Section.h>
#include <gui/types/StaticText.h>
#include <gui/DrawStructure.h>

namespace gui {

class Label {
    derived_ptr<StaticText> _text;
    Bounds _source;
    Vec2 _center;
    
public:
    Label(const std::string& txt, const Bounds& source, const Vec2& center);
    void update(DrawStructure& base, Section*, float alpha, bool disabled);
    void set_data(const std::string& text, const Bounds& source, const Vec2& center);
};

}
