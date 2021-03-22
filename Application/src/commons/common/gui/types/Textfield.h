#pragma once

#include <gui/GuiTypes.h>
#include <misc/metastring.h>
#include <gui/types/Entangled.h>

namespace gui {
    template<typename T>
    class NumericTextfield;
    
    template<typename T>
    class MetaTextfield;
    
    class Textfield : public Entangled {
    private:
        size_t _cursor_position;
        Rect _cursor;
        Rect _selection_rect;
        Text *_placeholder;
        
        size_t _text_offset;
        size_t _display_text_len;
        
        size_t _selection_start;
        lrange _selection;
        
        bool _valid;
        
    protected:
        std::function<bool(std::string& text, char inserted, size_t at)> _check_text;
        std::function<void()> _on_enter, _on_text_changed;
        
        GETTER(std::string, text)
        GETTER(std::string, postfix)
        GETTER(Font, font)
        GETTER(Color, text_color)
        GETTER(Color, fill_color)
        GETTER(bool, read_only)
        
    private:
        Text _text_display;
        
    public:
        Textfield(const std::string& text, const Bounds& bounds);
        void set_text_color(const Color& c);
        void set_fill_color(const Color& c);
        void set_postfix(const std::string&);
        
        virtual std::string name() const override { return "Textfield"; }
        
        void set_placeholder(const std::string& text);
        void set_text(const std::string& text);
        void set_font(const Font& font) {
            if(_font == font)
                return;
            
            _font = font;
            _text_display.set_font(font);
            if(_placeholder)
                _placeholder->set_font(font);
            
            set_content_changed(true);
        }
        void set_read_only(bool ro) {
            if(_read_only == ro)
                return;
            
            _read_only = ro;
            set_content_changed(true);
        }
        
        virtual void set_filter(const std::function<bool(std::string& text, char inserted, size_t at)>& fn) {
            _check_text = fn;
            _check_text(_text, 0, 0);
        }
        
        void on_enter(const std::function<void()>& fn) {
            _on_enter = fn;
        }
        
        void enter();
        
        void on_text_changed(const std::function<void()>& fn) {
            _on_text_changed = fn;
        }
        
        template<typename T>
        operator const NumericTextfield<T>&() const {
            if(dynamic_cast<const NumericTextfield<T>*>(this)) {
                return *static_cast<const NumericTextfield<T>*>(this);
            }
            
            U_EXCEPTION("Cannot cast.");
        }
        
        template<typename T>
        operator const MetaTextfield<T>&() const {
            if(dynamic_cast<const MetaTextfield<T>*>(this)) {
                return *static_cast<const MetaTextfield<T>*>(this);
            }
            
            U_EXCEPTION("Cannot cast.");
        }
        
    protected:
        std::tuple<bool, bool> system_alt() const;
        
        void set_valid(bool valid) {
            if(_valid == valid)
                return;
            
            _valid = valid;
            set_dirty();
        }
        
        virtual bool isTextValid(std::string& text, char inserted, size_t at) {
            return _check_text(text, inserted, at);
        }
        virtual std::string correctText(const std::string& text) {return text;}
        
    private:
        void update() override;
        void move_cursor(float mx);
        
        void onEnter(Event e);
        void onControlKey(Event e);
        
        bool swap_with(Drawable* d) override {
            if(d->type() == Type::ENTANGLED) {
                auto ptr = dynamic_cast<Textfield*>(d);
                if(!ptr)
                    return false;
                
                if(ptr->_text != _text) {
                    set_text(ptr->_text);
                    set_content_changed(true);
                }
                
                return true;
            }
            
            return false;
        }
    };
    
    template<typename T>
    class NumericTextfield : public Textfield {
        arange<T> _limits;
        
    public:
        NumericTextfield(const T& number, const Bounds& bounds, arange<T> limits = {0, 0})
            : Textfield(Meta::toStr(number), bounds), _limits(limits)
        { }
        
        T get_value() const {
            try {
                return Meta::fromStr<T>(_text);
            } catch(std::logic_error) {}
            
            if(_text == "-")
                return T(-0);
            return T(0);
        }
        
    protected:
        virtual void set_filter(const std::function<bool(std::string& text, char inserted, size_t at)>&) override {
            // do nothing
            U_EXCEPTION("This function is disabled for numeric textfields.");
        }
        
        virtual bool isTextValid(std::string& text, char inserted, size_t at) override
        {
            if(text.empty() || text == "-")
                return true;
            
            if(std::is_integral<T>::value && inserted == '.')
                return false;
            
            if(inserted == 8 // erased
               || (inserted == '.' && !utils::contains(_text, '.'))
               || (inserted == '-' && at == 0 && !utils::contains(_text, '-'))
               || irange('0', '9').contains(inserted))
            {
                if(text != "-") {
                    try {
                        T v = Meta::fromStr<T>(text);
                        if(_limits.first != _limits.last && !_limits.contains(v)) {
                            if(v < _limits.first)
                                v = _limits.first;
                            else if(v > _limits.last)
                                v = _limits.last;
                            else
                                return false;
                            
                            // correct value
                            text = Meta::toStr<T>(v);
                            
                            return true; // flash limits error
                        }
                    }
                    catch(std::logic_error e) { return false; }
                }
                
                return true;
            }
            
            return false;
        }
    };
    
    template<typename T>
    class MetaTextfield : public Textfield {
    public:
        typedef T value_type;
        
    public:
        MetaTextfield(const T& value, const Bounds& bounds)
            : Textfield(Meta::toStr<T>(value), bounds)
        { }
        
        T get_value() const {
            try {
                return Meta::fromStr<T>(_text);
            } catch(std::logic_error) {}
            
            return T();
        }
        
    protected:
        virtual bool isTextValid(std::string& text, char inserted, size_t at) override
        {
            try {
                Meta::fromStr<T>(text);
                
                if(_check_text(text, inserted, at)) {
                    set_valid(true);
                    return true;
                }
                
                set_valid(false);
                return false;
            }
            catch(illegal_syntax e) {Except("illegal_syntax: %s", e.what()); set_valid(false); return false;}
            catch(std::logic_error e) {Except("logic_error: %s", e.what());}
            
            // syntax error
            set_valid(false);
            return true;
        }
    };
}
