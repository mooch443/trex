#pragma once

#include <gui/types/Entangled.h>
#include <gui/DrawSFBase.h>

namespace gui {
class Checkbox : public Entangled {
protected:
    GETTER(std::string, text)
    GETTER(bool, checked)
    
    Font _font;
    static const Size2 box_size;
    static const float margin;
    
    Rect _box;
    Text _description;
    
    std::function<void()> _callback;
    
public:
    Checkbox(const Vec2& pos, const std::string& text = "", bool checked = false, const Font& font = Font(0.75));
    
    void on_change(const decltype(_callback)& callback) {
        _callback = callback;
    }
    
    void set_checked(bool checked) {
        if(checked == _checked)
            return;
        
        _checked = checked;
        set_content_changed(true);
    }
    
    void set_font(const Font& font) {
        if(_font == font)
            return;
        
        _font = font;
        set_content_changed(true);
    }
    
    void set_text(const std::string& text) {
        if(_text == text)
            return;
        
        _text = text;
        if(!_text.empty()) {
            _description.set_txt(_text);
            set_size(Size2(_description.width() + _description.pos().x + margin, Base::default_line_spacing(_font)));
        } else {
            set_size(Size2(margin*2 + box_size.width, Base::default_line_spacing(_font)));
        }
        
        set_content_changed(true);
    }
    
protected:
    void update() override;
};
}
