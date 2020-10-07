#include "Checkbox.h"

namespace gui {
    IMPLEMENT(Checkbox::box_size) = Size2(15, 15);
    IMPLEMENT(Checkbox::margin)= float(5);

    Checkbox::Checkbox(const Vec2& pos, const std::string& text, bool checked, const Font& font)
        : _text(text),
          _checked(checked),
          _font(font),
          _description(text, Vec2(), Black, _font),
          _callback([](){})
    {
        set_pos(pos);
        set_background(White);
        set_scroll_enabled(true);
        set_scroll_limits(Rangef(0,0), Rangef(0,0));
        
        _box.set_fillclr(White);
        _box.set_lineclr(Black);
        _box.set_bounds(Bounds(Vec2(margin, (Base::default_line_spacing(font) - box_size.height) * 0.5), box_size));
        set_clickable(true);
        
        _description.set_pos(Vec2(box_size.width + _box.pos().x + 4, 0));
        if(!_text.empty()) {
            _description.set_txt(_text);
            set_size(Size2(_description.width() + _description.pos().x + margin, Base::default_line_spacing(_font)));
        } else {
            set_size(Size2(margin*2 + box_size.width, Base::default_line_spacing(_font)));
        }
        
        add_event_handler(HOVER, [this](auto) { this->set_dirty(); });
        add_event_handler(MBUTTON, [this](Event e) {
            if(e.mbutton.pressed || e.mbutton.button != 0)
                return;
            
            _checked = !_checked;
            this->set_content_changed(true);
            
            _callback();
        });
    }
    
    void Checkbox::update() {
        set_background(background()->fillclr().alpha(hovered() ? 150 : 100));
        
        if(_content_changed) {
            begin();
            advance_wrap(_box);
            if(_checked)
                advance(new Rect(Bounds(_box.pos() + Vec2(1, 1), _box.size() - Vec2(2, 2)), Black));
            if(!_text.empty())
                advance_wrap(_description);
            end();
            
            if(!_text.empty())
                set_size(Size2(_description.width() + _description.pos().x + margin, Base::default_line_spacing(_font)));
            else
                set_size(Size2(margin*2 + box_size.width, Base::default_line_spacing(_font)));
        }
    }
}
