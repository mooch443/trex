#include "Label.h"
//#include <gui/gui.h>
#include <gui/IMGUIBase.h>
#include <tracking/Tracker.h>
#include <gui/Timeline.h>
#include <gui/MouseDock.h>
#include <gui/GUICache.h>

namespace gui {

Label::Label(const std::string& text, const Bounds& source, const Vec2& center)
    : _text(std::make_shared<StaticText>(Str(text), Font(0.5))), _source(source), _center(center), animator("label-animator-" + Meta::toStr((uint64_t)_text.get())), _line({}, 1)
{
    _text->set_background(Transparent, Transparent);
    _text->set_origin(Vec2(0.5, 1));
    _text->set_clickable(false);
}

Label::~Label() {
    MouseDock::unregister_label(this);
    //print("Label destroyed ", this);
    if(not animator.empty())
        GUICache::instance().set_animating(animator, false);
}

void Label::set_data(Frame_t frame, const std::string &text, const Bounds &source, const Vec2 &center) {
    if(text != _text->text()) {
        if(_registered) {
            _registered = false;
            GUICache::instance().set_animating(animator, false);
        }
        _text->set_txt(text);
        //+"-"+_text->text();
    }
    _source = source;
    _center = center;
    _frame = frame;
}

void Label::update(const FindCoord& coord, Entangled& e, float alpha, float _d, bool disabled, double dt) {
    alpha = saturate(alpha, 0.5, 1.0);
    
    if(disabled)
        alpha *= 0.5;
    
    //Bounds screen(Vec2(), coord.screen_size());

    auto stage = e.stage();
    if (!stage)
        return;

    Size2 background(stage->width() * 0.5, stage->height() * 0.5);

    //auto ptr = base.find("fishbowl");
    //if (ptr)
    auto scale = coord.bowl_scale().reciprocal();
    auto screen = coord.viewport();
    //auto screen = coord.convert(HUDRect(Vec2(), coord.screen_size()));
    
        //auto inverse = e.global_transform().getInverse();
        //scale = inverse.transformPoint(Vec2(1)) - inverse.transformPoint(Vec2(0));
        //screen = inverse.transformRect(Bounds(Vec2(), screen_size));
    auto offset = -(_center - (screen.pos() + Size2(screen.width * 0.5, screen.height * 0.95))).div(Vec2(screen.width * 0.5, screen.height * 0.95));

    //scale = scale.mul(0.75);
    _text->set_scale(scale);

    //auto mp = e.stage()->mouse_position();
    //mp = (mp - ptr->pos()).div(ptr->scale());

    //auto d = euclidean_distance(mp, _center);
    const bool is_in_mouse_dock = MouseDock::is_registered(this);

    if(not is_in_mouse_dock)
        e.advance_wrap(*_text);
    
    auto video_size = coord.video_size();
    auto center = screen.pos() + screen.size().mul(0.5, 1.05);
    //auto center = video_size.mul(0.5, 0.95);
    
    auto vec = center - _center;
    //e.add<Circle>(Loc(center), Radius{20}, FillClr{Red.alpha(100)}, LineClr{Red.alpha(200)});
    //e.add<Line>(Loc{_center}, Loc{_center + vec}, Red.alpha(200));
    
    if(not is_in_mouse_dock) {
        _text->set_alpha(Alpha{alpha});
    } else
        _text->set_alpha(Alpha{1});
    
    float distance = (_text->height() + _source.height) * scale.y; // scale.y;
    float percent = vec.length() / sqrtf(screen.width * screen.height);
    
    // maybe something about max_zoom_limit and current scale vs. video size?
    distance = /*_text->local_bounds().height +*/ 60 * SQR(percent) + 10 * scale.x;
    //auto text_pos = _center - offset * distance + 10 * scale.y;
    auto text_pos = _center - vec.normalize() * distance;
    //print("offset = ", offset, " distance=", distance, " len=", vec.length(), " norm=",vec.length() / sqrtf(screen.width * screen.height));
    
    _color = (disabled ? (is_in_mouse_dock ? White : Gray) : Cyan).alpha(255 * alpha);

    if(disabled)
        _text->set_text_color(LightGray);
    else
        _text->set_text_color(White);

    auto screen_target = coord.convert(BowlCoord(text_pos));
    auto screen_source = coord.convert(BowlCoord(_text->pos()));
    auto screen_rect = coord.convert(BowlRect(_source));
    auto dis = euclidean_distance(screen_target, screen_source) / screen_rect.size().max();
    //if(dis > 0.25)
    //    print("sqdistance ", screen_source, " => ", screen_target, " = ", dis, " for ", screen_rect.size().max(), text()->text());
    
    if (is_in_mouse_dock)
    {
        _text->set_origin(Vec2(0, 0.5));
        if(_registered) {
            _registered = false;
            GUICache::instance().set_animating(animator, false);
        }
    }
    else {
        /*if (screen.overlaps(_source)) {
            if (GUI_SETTINGS(gui_show_timeline)) //TODO: timeline update
            {
                screen.y += 60 * scale.y;
                screen.height -= 60 * scale.y;
            }
            
            auto local = _text->local_bounds();
            screen.y += local.height * _text->origin().y;
            screen.x += local.width * _text->origin().x;
            screen.width -= local.width;
            screen.height -= local.height + 10 * scale.y;
            Bounds bds(text_pos, Vec2(1));
            bds.restrict_to(screen);
            text_pos = bds.pos();
        }*/

        update_positions(e, text_pos, dis <= 1, dt);
    }
}

float Label::update_positions(Entangled& e, Vec2 text_pos, bool do_animate, double dt) {
    if (not do_animate) {
        _text->set_pos(text_pos);
        _line.create(_center, _text->pos(), _color);
        e.advance_wrap(_line);
        //e.add<Line>(_center, _text->pos(), _color, 1);
        if(_registered) {
            //print("animator is off ", next, " == ", _text->pos(), " for animator ", animator);
            _registered = false;
            GUICache::instance().set_animating(animator, false);
        }
        
        return 0;
    }

    dt = min(dt, 0.5) * 2;
    //animation_timer.reset();
    auto next = animate_position(_text->pos(), text_pos, dt * 2, InterpolationType::EASE_OUT);
    //if(next.Equals(_text->pos()) && not text_pos.Equals(_text->pos()))
    //    FormatWarning("Next: ", next, " equals ", _text->pos(), " but not ", text_pos);
    float d = 0;
    if(not text_pos.Equals(_text->pos())) {
        d = euclidean_distance(_text->pos(), text_pos);
        if(not _registered) {
            GUICache::instance().set_animating(animator, true, _text.get());
            _registered = true;
        }
        _text->set_pos(next);
    } else {
        if(_registered) {
            //print("animator is off ", next, " == ", _text->pos(), " for animator ", animator);
            _registered = false;
            GUICache::instance().set_animating(animator, false);
        }
    }
    

    _line.create(_center, _text->pos(), _color);
    e.advance_wrap(_line);
    //e.add<Line>(_center, _text->pos(), _color, 1);
    return d;
}

}
