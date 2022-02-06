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
    //_text->set_default_font(Font(0.6));
}

void Label::set_data(const std::string &text, const Bounds &source, const Vec2 &center) {
    if(text != _text->txt()) {
        _text->set_txt(text);
    }
    _source = source;
    _center = center;
}

void Label::update(Drawable*ptr, Entangled& base, float alpha, bool disabled) {
    alpha = max(0.5, alpha);
    
    Vec2 offset;
    Vec2 scale(1);
    Bounds screen;
    Size2 background(GUI::instance()->gui().width() * 0.5, GUI::instance()->gui().height() * 0.5);

    //auto ptr = base.find("fishbowl");
    if (ptr) {
        screen = base.global_transform().getInverse().transformRect(Bounds(0, 0, GUI::instance()->gui().width(), GUI::instance()->gui().height()));
        if(!GUI::instance()->gui().scale().empty())
            scale = GUI::instance()->gui().scale().reciprocal().mul(ptr->scale().reciprocal());
        
        auto size = GUI::instance() && GUI::instance()->base()
            ? GUI::instance()->base()->window_dimensions().mul(scale * gui::interface_scale())
            : GUI::average().bounds().size();
        screen << (Size2)size;
        offset = -(_center - (screen.pos() + Size2(screen.width * 0.5, screen.height * 0.95))) / screen.width;
    }

    //scale = scale.mul(0.75);
    _text->set_scale(scale);
    _text->set_alpha(alpha);

    base.advance_wrap(*_text);
    
    float distance = (_text->global_bounds().height + _source.height * scale.y); // scale.y;
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
    base.advance(new Line(_center, text_pos, (disabled ? Gray : Cyan).alpha(255 * alpha), 1));
}

}
