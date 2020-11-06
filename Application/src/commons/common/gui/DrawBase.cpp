#include "DrawBase.h"
#include <misc/checked_casts.h>

namespace gui {
    Base *_latest_base = nullptr;
    std::function<Bounds(const std::string&, Drawable*, const Font&)> _restore_line_bounds;
    std::function<uint32_t(const Font&)> _restore_line_spacing;

    std::function<uint32_t(const Font&)> line_spacing_fn = [](const Font& font) -> uint32_t
    {
        return narrow_cast<uint32_t>(roundf(25 * font.size));
    };
    void Base::set_default_line_spacing(std::function<uint32_t(const Font&)> fn) {
        line_spacing_fn = fn;
    }
    uint32_t Base::default_line_spacing(const Font &font) {
        return line_spacing_fn(font);
    }

    std::function<Bounds(const std::string&, Drawable*, const Font&)> text_bounds_fn = [](const std::string& text, Drawable*, const Font& font) -> Bounds {
        return Bounds(0, 0, text.length() * 11.3f * font.size, line_spacing_fn(font));
    };
    Bounds Base::default_text_bounds(const std::string &text, Drawable* obj, const Font& font) {
        return text_bounds_fn(text, obj, font);
    }
    void Base::set_default_text_bounds(std::function<Bounds (const std::string &, Drawable *, const Font &)> fn) {
        text_bounds_fn = fn;
    }

    uint32_t Base::line_spacing(const Font& font) {
        return narrow_cast<uint32_t>(roundf(25 * font.size));
    }
    float Base::text_width(const Text &text) const {
        return text.txt().length() * 8.5f * text.font().size;
    }
    float Base::text_height(const Text &text) const {
        return 18.f * text.font().size;
    }
    Bounds Base::text_bounds(const std::string& text, Drawable*, const Font& font) {
        return Bounds(0, 0, text.length() * 11.3f * font.size, 26.f * font.size);
    }

    Base::Base() {
        _previous_base = _latest_base;
        _previous_line_spacing = line_spacing_fn;
        _previous_line_bounds = text_bounds_fn;
        
        _restore_line_bounds = _previous_line_bounds;
        _restore_line_spacing = _previous_line_spacing;
        
        _latest_base = this;
        
        set_default_line_spacing([this](const Font& font) -> uint32_t {
            return this->line_spacing(font);
        });
        set_default_text_bounds([this](const std::string & text, Drawable *obj, const Font &font) -> Bounds {
            return this->text_bounds(text, obj, font);
        });
    }
    Base::~Base() {
        if(_latest_base == this) {
            set_default_text_bounds(_previous_line_bounds);
            set_default_line_spacing(_previous_line_spacing);
            
            _restore_line_spacing = [](const Font& font) -> uint32_t
                {
                    return narrow_cast<uint32_t>(roundf(25 * font.size));
                };
            _restore_line_bounds = [](const std::string& text, Drawable*, const Font& font) -> Bounds
                {
                    return Bounds(0, 0, text.length() * 11.3f * font.size, line_spacing_fn(font));
                };
            
            _latest_base = _previous_base;
            
        } else {
            _restore_line_spacing = _previous_line_spacing;
            _restore_line_bounds = _previous_line_bounds;
        }
    }
}
