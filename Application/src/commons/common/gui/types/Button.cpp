#include "Button.h"
#include <gui/DrawSFBase.h>

namespace gui {
    Button::Button(const std::string& txt, const Bounds& size)
        : Button(txt, size, Drawable::accent_color)
    {}
    
    Button::Button(const std::string& txt, const Bounds& size, const Color& fill, const Color& text_clr, const Color& line)
        :   _txt(txt),
            _text_clr(text_clr),
            _fill_clr(fill),
            _line_clr(line),
            _toggled(false),
            _toggleable(false),
            _text(txt, Vec2(), text_clr, Font(0.75, Style::Regular, Align::Center))
    {
        set_clickable(true);
        set_background(fill, line);
        set_bounds(size);
        set_scroll_enabled(true);
        set_scroll_limits(Rangef(), Rangef());
        
        add_event_handler(MBUTTON, [this](Event e) {
            if(!e.mbutton.pressed && toggleable()) {
                _toggled = !_toggled;
                this->set_dirty();
            }
        });
    }
    
    void Button::set_font(Font font) {
        _text.set_font(font);
    }

    const Font& Button::font() const {
        return _text.font();
    }
    
    void Button::update() {
        Color clr(fill_clr());

        if(pressed()) {
            clr = clr.exposure(0.3);
            
        } else {
            if(toggleable() && toggled()) {
                if(hovered()) {
                    clr = clr.exposure(0.7);
                } else
                    clr = clr.exposure(0.3);
                
            } else if(hovered()) {
                clr = clr.exposure(1.5);
                clr.a = saturate(clr.a * 1.5);
            }
        }
        
        set_background(clr, line_clr());
        
        if(pressed()) {
            _text.set_pos(size() * 0.5 + Vec2(0.5, 0.5));
        } else
            _text.set_pos(size() * 0.5);
        
        if(content_changed()) {
            begin();
            advance_wrap(_text);
            end();
        }
    }
    
    void Button::set_fill_clr(const gui::Color &fill_clr) {
        if(_fill_clr == fill_clr)
            return;
        
        _fill_clr = fill_clr;
        set_dirty();
    }
    
    void Button::set_line_clr(const gui::Color &line_clr) {
        if(_line_clr == line_clr)
            return;
        
        _line_clr = line_clr;
        set_dirty();
    }
    
    void Button::set_txt(const std::string &txt) {
        if(_txt == txt)
            return;
        
        _txt = txt;
        _text.set_txt(txt);
        set_content_changed(true);
    }
    
    void Button::set_toggleable(bool v) {
        if(_toggleable == v)
            return;
        
        _toggleable = v;
        set_content_changed(true);
    }
    
    void Button::set_toggle(bool v) {
        if(_toggled == v)
            return;
        
        _toggled = v;
        set_content_changed(true);
    }
}
