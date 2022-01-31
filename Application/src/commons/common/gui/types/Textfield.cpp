#include "Textfield.h"
#include <gui/DrawSFBase.h>
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>

#if __has_include ( <GLFW/glfw3.h> )
    #include <GLFW/glfw3.h>
    #define CMN_CLIPBOARD_GLFW
#elif __has_include ( <clip.h> )
    #include <clip.h>
    #define CMN_CLIPBOARD_CLIP
#endif

namespace gui {
    constexpr static const float margin = 2;

    void set_clipboard(const std::string& text) {
#ifdef CMN_CLIPBOARD_GLFW
        glfwSetClipboardString(nullptr, text.c_str());
#elif CMN_CLIPBOARD_CLIP
        clip::set_text(text);
#else
        Except("Cannot copy text '%S'. Please enable CLIP or GLFW in CMake.", &text);
#endif
    }
    
    std::string get_clipboard() {
#ifdef CMN_CLIPBOARD_GLFW
        auto ptr = glfwGetClipboardString(nullptr);
        if(ptr) {
            return std::string(ptr);
        }
        return std::string();
#elif CMN_CLIPBOARD_CLIP
        std::string paste;
        if(clip::get_text(paste)) {
            return paste;
        }
        
        return std::string();
#else
        Except("Cannot paste from clipboard. Please enable CLIP or GLFW in CMake.");
        return std::string();
#endif
    }

    std::string::size_type find_first_of_reverse(
                                                     std::string const& str,
                                                     std::string::size_type const pos,
                                                     std::string const& chars)
    {
        if(pos < 1)
            return std::string::npos;
        
        assert(pos >= 1);
        assert(pos <= str.size());
        
        std::string::size_type const res = str.find_last_not_of(chars, pos - 1) + 1;
        return res == pos ? find_first_of_reverse(str, pos - 1, chars)
            : res ? res
            : std::string::npos;
    }
    
    std::tuple<bool, bool> Textfield::system_alt() const {
        bool system = false, alt = false;
        const bool nowindow = GlobalSettings::has("nowindow") && SETTING(nowindow);
        
        // TODO: Must also accept this from Httpd!
        if(!nowindow) {
#if __APPLE__
            system = stage() && (stage()->is_key_pressed(LSystem) || stage()->is_key_pressed(RSystem));
#else
            system = stage() && (stage()->is_key_pressed(LControl) || stage()->is_key_pressed(LControl));
#endif
            alt = stage() && (stage()->is_key_pressed(RAlt) || stage()->is_key_pressed(LAlt));
        }
        
        return {system, alt};
    }
    
    Textfield::Textfield(const std::string& text, const Bounds& bounds)
        : gui::Entangled(),
            _cursor_position(text.length()),
            _cursor(Bounds(0, 0, 2,30), Black),
            _placeholder(NULL),
            _text_offset(0),
            _display_text_len(0),
            _selection(-1, -1),
            _valid(true),
            _check_text([](auto, auto, auto){return true;}),
            _on_enter([](){}),
            _on_text_changed([](){}),
            _text(text),
            _font(0.75),
            _text_color(Black),
            _fill_color(White.alpha(210)),
            _read_only(false),
            _text_display(text, Vec2(), Black, _font)
    {
        _selection_rect.set_fillclr(DarkCyan.alpha(100));
        
        set_bounds(bounds);
        set_clickable(true);
        set_scroll_enabled(true);
        set_scroll_limits(Rangef(0, 0), Rangef(0, 0));
        
        add_event_handler(HOVER, [this](Event e) {
            if(pressed())
                this->move_cursor(e.hover.x);
            else
                this->set_dirty();
        });
        add_event_handler(MBUTTON, [this](Event e) {
            if(e.mbutton.button != 0)
                return;
            
            if(e.mbutton.pressed) {
                this->move_cursor(e.mbutton.x);
                _selection_start = _cursor_position;
                _selection = lrange(_cursor_position, _cursor_position);
            } else
                this->move_cursor(e.mbutton.x);
        });
        
        add_event_handler(TEXT_ENTERED, [this](Event e) { if(_read_only) return; this->onEnter(e); });
        add_event_handler(SELECT, [this](auto) { this->set_dirty(); });
        
        add_event_handler(KEY, [this](Event e) {
            auto && [system, alt] = this->system_alt();
            if(_read_only && (e.key.code != Keyboard::C || !system)) return;
            if(e.key.pressed)
                this->onControlKey(e);
        });
    }
    
    void Textfield::set_text(const std::string &text) {
        if(text == _text)
            return;
        
        _text = text;
        _cursor_position = _text.length();
        _text_offset = 0;
        _selection = lrange(-1, -1);
        
        set_content_changed(true);
    }
    
    void Textfield::enter() {
        if(_valid) {
            if(stage())
                stage()->select(NULL);
            _on_enter();
        }
    }
    
    void Textfield::onControlKey(gui::Event e) {
        constexpr const char* alphanumeric = "abcdefghijklmnopqrstuvwxyz0123456789";
        
        auto && [system, alt] = system_alt();
        
        switch (e.key.code) {
            case Keyboard::Left:
                if(_cursor_position > 0) {
                    size_t before = _cursor_position;
                    
                    if(system)
                        _cursor_position = 0;
                    else if(alt) {
                        // find the first word
                        auto k = find_first_of_reverse(utils::lowercase(_text), before, alphanumeric);
                        if(k == std::string::npos)
                            k = 0;
                        
                        _cursor_position = k;
                    }
                    else
                        _cursor_position--;
                    
                    if(e.key.shift) {
                        if(_selection.empty())
                            _selection_start = before;
                        if(_selection_start < _cursor_position)
                            _selection = lrange(_selection_start, _cursor_position);
                        else
                            _selection = lrange(_cursor_position, _selection_start);
                    } else
                        _selection = lrange(-1, -1);
                    
                    set_content_changed(true);
                    
                } else if(!e.key.shift && !_selection.empty()) {
                    _selection = lrange(-1, -1);
                    set_content_changed(true);
                }
                break;
                
            case Keyboard::Right:
                if(_cursor_position < _text.length()) {
                    size_t before = _cursor_position;
                    
                    if(system)
                        _cursor_position = _text.length();
                    else if(alt) {
                        // find the first word
                        auto k = utils::lowercase(_text).find_first_of(alphanumeric, before);
                        if(k != std::string::npos) {
                            // find the end of the word
                            k = utils::lowercase(_text).find_first_not_of(alphanumeric, k);
                            if(k == std::string::npos) // not found? jumpt to eof
                                k = _text.length();
                            
                        } else
                            k = _text.length();
                        
                        _cursor_position = k;
                    }
                    else
                        _cursor_position++;
                    
                    if(e.key.shift) {
                        if(_selection.empty())
                            _selection_start = before;
                        if(_selection_start < _cursor_position)
                            _selection = lrange(_selection_start, _cursor_position);
                        else
                            _selection = lrange(_cursor_position, _selection_start);
                    } else
                        _selection = lrange(-1, -1);
                    
                    set_content_changed(true);
                    
                } else if(!e.key.shift && !_selection.empty()) {
                    _selection = lrange(-1, -1);
                    set_content_changed(true);
                }
                break;
                
            case Keyboard::A:
                if(system) {
                    _selection = lrange(0, _text.length());
                    _cursor_position = _text.length();
                    set_content_changed(true);
                }
                break;
                
            case Keyboard::C:
                if(system) {
                    if(!_selection.empty()) {
                        auto sub = _text.substr(_selection.first, _selection.last - _selection.first);
                        Debug("Copying %S", &sub);
                        set_clipboard(sub);
                    } else {
                        Debug("Copying %S", &_text);
                        set_clipboard(_text);
                    }
                }
                break;
                
            case Keyboard::V:
                if(system) {
                    std::string paste = get_clipboard();
                    if(!paste.empty()) {
                        Debug("Pasting %S", &paste);
                        
                        std::string copy = _text;
                        size_t before = _cursor_position;
                        
                        if(!_selection.empty()) {
                            copy.erase(copy.begin() + _selection.first, copy.begin() + _selection.last);
                            _cursor_position = _selection.first;
                        }
                        
                        copy.insert(copy.begin() + _cursor_position, paste.begin(), paste.end());
                        _cursor_position += paste.length();
                        
                        if(isTextValid(copy, 8, _cursor_position)) {
                            _selection = lrange(-1, -1);
                            if(_text != copy) {
                                _text = copy;
                                _on_text_changed();
                            }
                            
                        } else {
                            _cursor_position = before;
                        }
                        
                        set_content_changed(true);
                    }
                }
                break;
                
            case Keyboard::BackSpace: {
                std::string copy = _text;
                size_t before = _cursor_position;
                
                if(!_selection.empty()) {
                    copy.erase(copy.begin() + _selection.first, copy.begin() + _selection.last);
                    _cursor_position = _selection.first;
                }
                else if(_cursor_position>0) {
                    _cursor_position--;
                    copy.erase(copy.begin() + _cursor_position);
                }
                
                if(isTextValid(copy, 8, _cursor_position)) {
                    _selection = lrange(-1, -1);
                    if(_text != copy) {
                        _text = copy;
                        _on_text_changed();
                    }
                    
                } else {
                    _cursor_position = before;
                }
                
                set_content_changed(true);
                
                break;
            }
                
            case Keyboard::Return:
                enter();
                break;
                
            default:
                break;
        }
    }
    
    void Textfield::onEnter(gui::Event e) {
        std::string k = std::string()+e.text.c;
        if(e.text.c < 10)
            return;
        
        switch (e.text.c) {
            case '\n':
            case '\r':
            case 8:
                break;
            case 27:
                if(parent() && parent()->stage())
                    parent()->stage()->select(NULL);
                break;
                
            default: {
                std::string copy = _text;
                size_t before = _cursor_position;
                
                if(!_selection.empty()) {
                    copy.erase(copy.begin() + _selection.first, copy.begin() + _selection.last);
                    _cursor_position = _selection.first;
                }
                copy.insert(_cursor_position, k);
                _cursor_position++;
                _display_text_len++;
                
                if(isTextValid(copy, e.text.c, _cursor_position-1)) {
                    _text = copy;
                    _selection = lrange(_cursor_position, _cursor_position);
                    
                } else {
                    _cursor_position = before;
                }
                
                _on_text_changed();
                set_content_changed(true);
                
                break;
            }
        }
    }

void Textfield::set_text_color(const Color &c) {
    if(c == _text_color)
        return;
    
    _text_color = c;
    _cursor.set_fillclr(c);
    
    set_content_changed(true);
}

void Textfield::set_fill_color(const Color &c) {
    if(c == _fill_color)
        return;
    
    _fill_color = c;
    
    set_content_changed(true);
}

void Textfield::set_postfix(const std::string &p) {
    if(p == _postfix)
        return;
    
    _postfix = p;
    set_content_changed(true);
}
    
    void Textfield::update() {
        begin();
        
        static constexpr const Color BrightRed(255,150,150,255);
        Color base_color   = _fill_color,
              border_color = _text_color.alpha(255);
        
        if(!_valid)
            base_color = BrightRed.alpha(210);
        if(_read_only) {
            base_color = base_color.exposure(0.9);
            _text_display.set_color(DarkGray);
        } else
            _text_display.set_color(_text_color);
        
        if(hovered())
            base_color = base_color.alpha(255);
        else if(selected())
            base_color = base_color.alpha(230);
        
        set_background(base_color, border_color);
        
        if(content_changed()) {
            // assumes test_text is only gonna be used in one thread at a time
            Timer timer;
            const float max_w = width() - margin * 2; // maximal displayed text width
            
            //Vec2 scale = stage_scale();
            //Vec2 real_scale = Drawable::real_scale();
            auto real_scale = this;
            
            if(_cursor_position > _text.length())
                _cursor_position = _text.length();
            if(_text_offset > _text.length())
                _text_offset = _text.length();
            
            auto r = Base::default_text_bounds(_text, real_scale, _font);
            const float cursor_y = (height() - Base::default_line_spacing(_font))*0.5;
            
            if(_text_offset >= _cursor_position)
                _text_offset = (size_t)max(0, long(_cursor_position)-1);
            std::string before = _text.substr(_text_offset, _cursor_position - _text_offset);
            
            r = Base::default_text_bounds(before, real_scale, _font);
            
            if(_display_text_len < _cursor_position)
                _display_text_len = _cursor_position;
            
            while(r.width >= max_w && before.length() > 0 && _text_offset < _cursor_position) {
                _text_offset++;
                before = before.substr(1);
                r = Base::default_text_bounds(before, real_scale, _font);
            }
            
            // check whether after string is too short
            std::string after = _text.substr(_cursor_position, _display_text_len - _cursor_position);
            if(after.length() < 2
               && _cursor_position < _text.length()
               && after.length() < _text.length() - _cursor_position)
            {
                _text_offset++; _display_text_len++;
                before = before.substr(1);
                after = _text.substr(_cursor_position, _display_text_len - _cursor_position);
            }
            
            r = Base::default_text_bounds(before + after, real_scale, _font);
            
            // maximize after string
            while(r.width < max_w && _display_text_len < _text.length()) {
                _display_text_len++;
                after = _text.substr(_cursor_position, _display_text_len - _cursor_position);
                r = Base::default_text_bounds(before + after, real_scale, _font);
            }
            
            // limit after string
            /*while(r.width >= max_w && after.length() > 0) {
                after = after.substr(0, after.length()-1);
                r = SFBase::Base::default_text_bounds(before + after, scale, _font);
            }*/
            
            // ensure that the string displayed is long enough (otherwise the first character will be hidden, if another one is inserted right after it)
            while(r.width < max_w * 0.75 && _text_offset > 0) {
                _text_offset--;
                before = _text.substr(_cursor_position - before.length() - 1, before.length() + 1);
                r = Base::default_text_bounds(before + after, real_scale, _font);
            }
            
            while(r.width >= max_w + 5 && after.length() > 0) {
                after = after.substr(0, after.length()-1);
                r = Base::default_text_bounds(before + after, real_scale, _font);
            }
            
            r = Base::default_text_bounds(before, real_scale, _font);
            _cursor.set_bounds(Bounds(
                  Vec2(r.width + r.x + margin, cursor_y),
                  Size2(2, Base::default_line_spacing(_font))
                ));
            
            _display_text_len = after.length() + _cursor_position;
            
            _text_display.set_txt(before + after);
            _text_display.set_pos(Vec2(margin, cursor_y));
            
            /*
             * Determine selection size / position and set the selection rectangle.
             */
            if(!_selection.empty() && ((long)_text_offset < _selection.last))
            {
                // determine visible starting position
                float sx0;
                
                if((long)_cursor_position == _selection.first) {
                    sx0 = _cursor.pos().x;
                    
                } else {
                    sx0 = _text_display.pos().x;
                    
                    if((long)_text_offset < _selection.first) {
                        size_t visible_index = max((size_t)_selection.first, _text_offset);
                        std::string visible_not_selected = _text.substr(_text_offset, visible_index - _text_offset);
                        r = Base::default_text_bounds(visible_not_selected, real_scale, _font);
                        sx0 += r.width + r.x;
                    }
                }
                
                std::string visible_selected_text = _text.substr(_text_offset, min(min(_text.length(), _display_text_len + 1), (size_t)_selection.last) - _text_offset);
                
                // see how long the visible text is
                r = Base::default_text_bounds(visible_selected_text, real_scale, _font);
                
                float sx1 = r.width + margin + 1;
                if((long)_cursor_position == _selection.last) {
                    sx1 = _cursor.pos().x;
                }
                
                // set boundaries
                _selection_rect.set_bounds(Bounds(sx0, _cursor.pos().y, min(width() - sx0 - margin, sx1 - sx0), _cursor.height()));
            }
            
            //Debug("Parsing: %.2fms", timer.elapsed()*1000);
            //set_dirty();
        }
        
        advance_wrap(_text_display);
        
        if(_text.empty() && _placeholder && !selected()) {
            _placeholder->set_pos(_text_display.pos());
            advance_wrap(*_placeholder);
           
        } else if(!_postfix.empty()) {
            auto tmp = new Text(_postfix, Vec2(width() - 5, height() * 0.5), _text_color.exposure(0.5), Font(max(0.1, _text_display.font().size * 0.9)));
            tmp->set_origin(Vec2(1, 0.5));
            advance(tmp);
        }
            
        if(read_only()) {
            auto tmp = new Text("(read-only)", Vec2(width() - 5, height() * 0.5), Gray, Font(max(0.1, _text_display.font().size * 0.9)));
            tmp->set_origin(Vec2(1, 0.5));
            advance(tmp);
        }
        
        if(!_selection.empty()) {
            advance_wrap(_selection_rect);
        }
        
        if(selected() && !read_only())
            advance_wrap(_cursor);
        
        end();
    }
    
    void Textfield::move_cursor(float mx) {
        std::string display = _text.substr(_text_offset, _display_text_len - _text_offset);
        float x = 0;
        long idx = 0;
        
        const float character_size = roundf(25 * _font.size);
        while (x + character_size*0.5 < mx
               && idx <= long(display.length()))
        {
            auto r = Base::default_text_bounds(display.substr(0, (size_t)idx++), this, _font);
            x = r.width + r.x;
        }
        
        _cursor_position = _text_offset + (size_t)max(0, idx - 1);
        
        if(pressed()) {
            if(_cursor_position < _selection_start) {
                _selection = lrange(_cursor_position, _selection_start);
            } else
                _selection = lrange(_selection_start, _cursor_position);
        }
        
        set_content_changed(true);
    }
    
    void Textfield::set_placeholder(const std::string &text) {
        if(!text.empty()) {
            if(_placeholder && _placeholder->txt() == text)
                return;
            
            if(!_placeholder)
                _placeholder = new Text(text, Vec2(), Gray, _text_display.font());
            else
                _placeholder->set_txt(text);
            
        } else {
            if(text.empty() && !_placeholder)
                return;
            
            if(_placeholder) {
                delete _placeholder;
                _placeholder = NULL;
            }
        }
        
        set_content_changed(true);
    }
}
