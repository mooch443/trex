#include "Label.h"
#include <gui/IMGUIBase.h>
#include <gui/MouseDock.h>

namespace cmn::gui {

Label::Label(const std::string& text, const Bounds& source, const Vec2& center)
: _text(std::make_shared<StaticText>(Str(text), Font(0.5))), _source(source), _center(center), animator("label-animator-" + Meta::toStr((uint64_t)_text.get()))
{
    _text->set_origin(Vec2(0.5, 1));
    _text->set_clickable(false);
    //set_z_index(1);
    _text->set(StaticText::Shadow_t{ 1 });
    _text->set(Loc{center});
}

Label::~Label() {
    MouseDock::unregister_label(this);
    //Print("Label destroyed ", this);
    set_animating(false);
}

void Label::update() {
    auto ctx = OpenContext();

    const bool is_in_mouse_dock = _position_override;//MouseDock::is_registered(this);
    if (not is_in_mouse_dock) {
        advance_wrap(*_text);
        _text->set(FillClr{ _fill_color });
    }
    else {
        _text->set_background(Transparent, Transparent);
    }

    advance_wrap(_line);
}

void Label::set_data(Frame_t frame, const std::string &text, const Bounds &source, const Vec2 &center) {
    if(text != _text->text()) {
        /*if (_registered) {
            _registered = false;
            set_animating(false);
        }*/
        _text->set_txt(text);
        //+"-"+_text->text();
        //_initialized = false;
    }
    _source = source;
    _center = center;
    _frame = frame;
}

float Label::update(const FindCoord& coord, float alpha, float, bool disabled, double dt, Scale text_scale) {
    alpha = saturate(alpha, 0.5, 1.0);
    
    if(disabled)
        alpha *= 0.75;

    Vec2 target_origin(0.5, 1);
    const bool is_in_mouse_dock = _position_override;//MouseDock::is_registered(this);
    if (not is_in_mouse_dock) {
        _text->set_alpha(Alpha{ alpha });
        _text->set(StaticText::Shadow_t{ SQR(alpha) * float(_text->text_color().a) / 255.f * 0.5 });
    }
    else {
        _text->set_alpha(Alpha{ 1 });
        _text->set(StaticText::Shadow_t{ float(_text->text_color().a) / 255.f * 0.5 });
        target_origin = Vec2(0, 0.5);
    }


    _text->set_origin(animate_position<InterpolationType::EASE_OUT>(_text->origin(), target_origin, dt, 1/2.0));

    auto screen_size = coord.screen_size();
    auto scale = coord.bowl_scale().reciprocal();
    if (text_scale.empty())
        text_scale = scale;
    _text->set_scale(text_scale);

    Vec2 text_pos;

    if (not _position_override) {
        //auto screen = coord.viewport();
        //auto screen1 = coord.hud_viewport();
        //auto screen = coord.convert(screen1);
        //auto other = coord.convert(BowlRect(Vec2(), coord.video_size()));

        auto center_screen = HUDCoord(Vec2(screen_size.width * 0.5, screen_size.height * 1.05));
        Vec2 alternative = coord.convert(center_screen);
        auto global = global_transform().getInverse();
        alternative = global.transformPoint(center_screen);
        //Print("viewport = ", coord.viewport(), " hud_viewport = ", coord.hud_viewport(), " => ", coord.convert(HUDRect(Vec2(), coord.hud_viewport().size())), " ",coord.screen_size(), " other=",other, " alternative=",alternative, " screen_center=",center_screen);

        //screen = BowlRect(Bounds(Vec2(), coord.screen_size()));
        //auto screen = Bounds(Vec2(), coord.video_size());
        //auto screen = coord.convert(HUDRect(Vec2(), coord.screen_size()));

        //Print("screen = ", screen, " scale = ", scale, " gscale = ", gscale, " video = ", coord.video_size());
            //auto inverse = e.global_transform().getInverse();
            //scale = inverse.transformPoint(Vec2(1)) - inverse.transformPoint(Vec2(0));
            //screen = inverse.transformRect(Bounds(Vec2(), screen_size));
        //auto offset = -(_center - (screen.pos() + Size2(screen.width * 0.5, screen.height * 0.95))).div(Vec2(screen.width * 0.5, screen.height * 0.95));

        //scale = scale.mul(0.75);

        //auto mp = e.stage()->mouse_position();
        //mp = (mp - ptr->pos()).div(ptr->scale());

        //auto d = euclidean_distance(mp, _center);

        //auto video_size = coord.video_size();
        auto center = alternative;//(screen.pos() + screen.size().mul(0.5, 0.5));//.mul(scale);
        //auto center = video_size.mul(0.5, 0.95);

        auto vec = center - _center;
        //e.add<Circle>(Loc(center), Radius{20}, FillClr{Red.alpha(100)}, LineClr{Red.alpha(200)});
        //e.add<Line>(Loc{_center}, Loc{_center + vec}, Red.alpha(200));


        float distance = (_text->height() + _source.height) * scale.y; // scale.y;
        float percent = screen_size.empty()
            ? 1
            : vec.length() / sqrtf(SQR(screen_size.width) + SQR(screen_size.height));
        //Print("percent = ", percent, " center=", _center, " alternative=",alternative, " screen=", screen.size());

        // maybe something about max_zoom_limit and current scale vs. video size?
        distance = /*_text->local_bounds().height +*/ (_line_length * (percent) + 10);
        //auto text_pos = _center - offset * distance + 10 * scale.y;
        text_pos = _center - vec.normalize() * distance;
        //Print("offset = ", offset, " distance=", distance, " len=", vec.length(), " norm=",vec.length() / sqrtf(screen.width * screen.height));



        /*if (is_in_mouse_dock)
        {
            _text->set_origin(Vec2(0, 0.5));
            if(_registered) {
                _registered = false;
                set_animating(false);
            }
        }
        else {
            if (screen.overlaps(_source)) {
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
            }
            //text_pos = center;
        }*/
    }
    else
        text_pos = _override_position;

    auto screen_target = coord.convert(BowlCoord(text_pos));
    auto screen_source = coord.convert(BowlCoord(_text->pos()));
    auto screen_rect = coord.convert(BowlRect(_source));
    float max_w = screen_size.width *= 0.1;
    float max_h = screen_size.height *= 0.1;
    if (screen_rect.width * 0.5 > max_w) {
        screen_rect.x += screen_rect.width * 0.5 - max_w;
        screen_rect.width = max_w * 2;
    }
    if (screen_rect.height * 0.5 > max_h) {
        screen_rect.y += screen_rect.height * 0.5 - max_h;
        screen_rect.height = max_h * 2;
    }
    //auto dis = euclidean_distance(screen_target, screen_source) / screen_rect.size().max();
    auto dis = euclidean_distance(screen_target, screen_source) / (2 * sqrtf(SQR(screen_rect.width) + SQR(screen_rect.height)));
    //if (dis > 0.25)
        //Print("sqdistance ", screen_source, " => ", screen_target, " = ", dis, " for ", screen_rect.size().max(), " dock=", is_in_mouse_dock, " ", text()->text());

    _color = (disabled ? (is_in_mouse_dock ? White : Gray) : _line_color).alpha(255 * alpha);

    if (disabled)
        _text->set_text_color(LightGray);
    else
        _text->set_text_color(White);

    return update_positions(text_pos, _initialized && dis <= 1, dt);
}

void Label::set_uninitialized() {
    _initialized = false;
}

float Label::update_positions(Vec2 text_pos, bool do_animate, double dt) {
    _initialized = true;
    
    if (not do_animate) {
        _text->set_pos(text_pos);
        _line.create(Line::Point_t{ _center }, Line::Point_t{ _text->pos() }, LineClr{ _color }, Line::Thickness_t{ 2 });
        //e.add<Line>(_center, _text->pos(), _color, 1);
        if(_registered) {
            //Print("animator is off ", next, " == ", _text->pos(), " for animator ", animator);
            _registered = false;
            set_animating(false);
        }
        
        return 0;
    }

    dt = min(dt, 0.5) * 2;
    //animation_timer.reset();
    auto next = animate_position<InterpolationType::EASE_OUT>(_text->pos(), text_pos, dt, 1/2.0);
    //if(next.Equals(_text->pos()) && not text_pos.Equals(_text->pos()))
    //    FormatWarning("Next: ", next, " equals ", _text->pos(), " but not ", text_pos);
    float d = 0;
    if(not text_pos.Equals(_text->pos())) {
        d = euclidean_distance(_text->pos(), text_pos);
        if(not _registered) {
            set_animating(true);
            _registered = true;
        }
        _text->set_pos(next);
    } else {
        if(_registered) {
            //Print("animator is off ", next, " == ", _text->pos(), " for animator ", animator);
            _registered = false; 
            set_animating(false);
        }
    }
    

    _line.create(Line::Point_t{ _center }, LineClr{ _color.exposure(0.5).alpha(_color.a * 0.75) }, Line::Point_t{ _text->pos() }, LineClr{ _color }, Line::Thickness_t{ 2 });
    //e.add<Line>(_center, _text->pos(), _color, 1);
    return d;
}

void Label::set(attr::Loc loc)
{
    Print("Label::set ", loc);
    //Entangled::set(loc);
    //update(FindCoord::get(), )
}

void Label::set(attr::FillClr clr)
{
    if (_fill_color == clr)
        return;
    _fill_color = clr;
    set_content_changed(true);
}
void Label::set(attr::LineClr clr)
{
    if (_line_color == clr)
        return;
    _line_color = clr;
    set_content_changed(true);
}
}
