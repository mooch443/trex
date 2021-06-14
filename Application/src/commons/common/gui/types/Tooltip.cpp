#include "Tooltip.h"
#include <gui/DrawStructure.h>

namespace gui {
    Tooltip::Tooltip(Drawable* other, float max_width)
        : _other(other), _text("", Vec2(), Size2(max_width > 0 || !_other ? max_width : _other->width(), -1), Font(0.7)), _max_width(max_width)
    {
        //set_background(Black.alpha(150));
        set_text("");
        set_origin(Vec2(0, 1));
        _text.set_clickable(false);
        set_z_index(2);
    }

    void Tooltip::set_other(Drawable* other) {
        if(other == _other)
            return;
        
        _other = other;
        _text.set_size(Size2(_max_width > 0 || !_other ? _max_width : _other->width(), -1));
        set_content_changed(true);
    }
    
    void Tooltip::update() {
        auto mp = stage()->mouse_position();
        if(parent()) {
            auto tf = parent()->global_transform().getInverse();
            mp = tf.transformPoint(mp) + Vec2(5, 0);
        }
        
        if(mp.y - _text.height() < 0)
            set_origin(Vec2(0, 0));
        else
            set_origin(Vec2(0, 1));
        
        if(!content_changed()) {
            set_pos(mp);
            return;
        }
        
        begin();
        advance_wrap(_text);
        end();
        
        _text.set_content_changed(true);
        _text.update();
        _text.set_bounds_changed();
        
        set_bounds(Bounds(mp, _text.size() + Vec2(5, 2) * 2));
    }
    
    void Tooltip::set_text(const std::string& text) {
        if(_text.txt() == text)
            return;
        
        _text.set_txt(text);
        set_content_changed(true);
    }
}
