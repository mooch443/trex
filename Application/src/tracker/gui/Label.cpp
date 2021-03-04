#include "Label.h"
#include <gui.h>

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
    
    Vec2 offset(_center);
    offset = offset.normalize();
    alpha = max(0.5, alpha);
    
    Size2 background(base.width() * 0.5, base.height() * 0.5);
    
    auto ptr = base.find("fishbowl");
    auto transform = ptr ? ptr->global_transform() : gui::Transform();
    auto screen = transform.getInverse().transformRect(Bounds(Vec2(), Size2(base.width(), base.height())));
    
    //const Font font(0.9 * 0.75 + 0.25 * 0.9 / interface_scale);
    //const float add_scale = 0.85f / (1 - ((1 - GUI::instance()->cache().zoom_level()) * 0.5f)) + 0.1;
    
    if(ptr)
        _text->set_scale(base.scale().reciprocal().mul(ptr->scale().reciprocal()));
    _text->set_alpha(alpha);
    base.wrap_object(*_text);
    
    float distance = (_text->global_bounds().height + _source.height * _text->scale().y); //+ base.width() * 0.006;
    offset = offset.mul(Vec2(0.5 * (background.width - _center.x) / background.width, 1));
    //Debug("Width: %f Height: %f offset: %f,%f distance: %f", background.width, background.height, offset.x, offset.y, distance);
    /*offset = offset.div(background).map([](Float2_t x) -> Float2_t {
        return min(1.0, x);
    });*/
    
    auto text_pos = _center - offset * (distance + 5);
    if(screen.contains(_center)) {
        auto o = -Vec2(0, _text->local_bounds().height);
        if(GUI::instance()
           && GUI::instance()->timeline().visible())
        {
            o -= Vec2(0, 50);
        }
        Bounds bds(text_pos + o, Size2(10));
        bds.restrict_to(screen);
        text_pos = bds.pos() - o;
    }
    
    _text->set_pos(text_pos);
    base.line(_center - offset * 15, text_pos, 1, (disabled ? Gray : Cyan).alpha(255 * alpha));
}

}
