#include "Label.h"
#include <gui/gui.h>
#include <gui/IMGUIBase.h>

namespace gui {

Label::Label(const std::string& text, const Bounds& source, const Vec2& center)
    : _text(std::make_shared<StaticText>(text, Vec2())), _source(source), _center(center)
{
    _text->set_background(Transparent, Transparent);
    _text->set_origin(Vec2(0.5, 1));
    _text->set_clickable(false);
}

void Label::set_data(const std::string &text, const Bounds &source, const Vec2 &center) {
    if(text != _text->txt()) {
        _text->set_txt(text);
    }
    _source = source;
    _center = center;
}

void Label::update(DrawStructure& base, Section* s, float alpha, bool disabled) {
    alpha = max(0.5, alpha);
    
    Vec2 offset;
    Vec2 scale(1);
    Bounds screen;
    Size2 background(base.width() * 0.5, base.height() * 0.5);

    auto ptr = base.find("fishbowl");
    if (ptr) {
        screen = ptr->global_transform().getInverse().transformRect(Bounds(Vec2(), Size2(base.width(), base.height())));
        if(!base.scale().empty())
            scale = base.scale().reciprocal().mul(ptr->scale().reciprocal());
        
        screen._size = GUI::instance() && GUI::instance()->base() ? GUI::instance()->base()->window_dimensions().mul(scale * gui::interface_scale()) : Size2(Tracker::average().bounds().size());
        offset = -(_center - (screen.pos() + Size2(screen.width * 0.5, screen.height * 0.95))) / screen.width;
    }

    _text->set_scale(scale);
    _text->set_alpha(alpha);

    base.wrap_object(*_text);
    
    float distance = (_text->global_bounds().height + _source.height * scale.y);
    auto text_pos = _center - offset * (distance + 5 * scale.y);

    if(ptr && screen.contains(_center)) {
        auto o = -Vec2(0, _text->local_bounds().height);
        if(GUI::instance()
           && GUI::instance()->timeline().visible())
        {
            o -= Vec2(0, 40);
        }

        Bounds bds(text_pos + o, Size2(10));
        bds.restrict_to(screen);
        text_pos = bds.pos() - o;
    }
    
    _text->set_pos(text_pos);
    base.line(_center, text_pos, 1, (disabled ? Gray : Cyan).alpha(255 * alpha));
}

}
