#include "Label.h"
//#include <gui/gui.h>
#include <gui/IMGUIBase.h>
#include <tracking/Tracker.h>
#include <gui/Timeline.h>
#include <gui/MouseDock.h>

namespace gui {

Label::Label(const std::string& text, const Bounds& source, const Vec2& center)
    : _text(std::make_shared<StaticText>(text)), _source(source), _center(center)
{
    _text->set_background(Transparent, Transparent);
    _text->set_origin(Vec2(0.5, 1));
    _text->set_clickable(false);
    //_text->set_default_font(Font(0.6));
}

Label::~Label() {
    MouseDock::unregister_label(this);
    print("Label destroyed ", this);
}

void Label::set_data(const std::string &text, const Bounds &source, const Vec2 &center) {
    if(text != _text->text()) {
        _text->set_txt(text);
    }
    _source = source;
    _center = center;
}

void Label::update(Base* base, Drawable*ptr, Entangled& e, float alpha, bool disabled) {
    auto distance_to_mouse = alpha;
    alpha = max(0.5, alpha);
    
    Vec2 offset;
    Vec2 scale(1);
    Bounds screen;

    auto stage = e.stage();
    if (!stage)
        return;

    Size2 background(stage->width() * 0.5, stage->height() * 0.5);

    //auto ptr = base.find("fishbowl");
    if (ptr) {
        screen = e.global_transform().getInverse().transformRect(Bounds(0, 0, stage->width(), stage->height()));
        if(!stage->scale().empty())
            scale = stage->scale().reciprocal().mul(ptr->scale().reciprocal());
        
        auto size = base ? base->window_dimensions().mul(scale * gui::interface_scale())
            : track::Tracker::average().bounds().size();
        screen << (Size2)size;
        offset = -(_center - (screen.pos() + Size2(screen.width * 0.5, screen.height * 0.95))) / screen.width;
    }

    //scale = scale.mul(0.75);
    _text->set_scale(scale);
    _text->set_alpha(alpha);

    auto mp = e.stage()->mouse_position();
    mp = (mp - ptr->pos()).div(ptr->scale());

    auto d = euclidean_distance(mp, _center);
    const bool is_in_mouse_dock = MouseDock::is_registered(this);

    if(not is_in_mouse_dock)
        e.advance_wrap(*_text);
    
    float distance = (_text->global_bounds().height + _source.height * scale.y); // scale.y;
    auto text_pos = _center - offset * (distance + 5 * scale.y);
    _color = (disabled ? Gray : Cyan).alpha(255 * alpha);

    if (is_in_mouse_dock) 
    {
        //text_pos = MouseDock::label_pos(this) + mp;
        _text->set_origin(Vec2(0, 0.5));

        /*if (ptr && screen.contains(_center)) {
            //auto o = -Vec2(0, _text->local_bounds().height);
            if (Timeline::visible()) {
                //o -= Vec2(0, 40);
            }

            Bounds bds(text_pos, Size2(10));
            bds.restrict_to(screen);
            text_pos = bds.pos(); //- o;
        }*/
    }
    else {
        _text->set_origin(Vec2(0.5, 1));

        if (ptr && screen.contains(_center)) {
            auto o = -Vec2(0, _text->local_bounds().height);
            if (Timeline::visible()) {
                o -= Vec2(0, 40);
            }

            Bounds bds(text_pos + o, Size2(10));
            bds.restrict_to(screen);
            text_pos = bds.pos() - o;
        }

        update_positions(e, text_pos);
    }
}

void Label::update_positions(Entangled& e, Vec2 text_pos) {
	_text->set_pos(text_pos);
    e.add<Line>(_center, text_pos, _color, 1);
}

}
