#ifndef _BUTTON_H
#define _BUTTON_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>

namespace gui {
    class Button : public Entangled {
        GETTER(std::string, txt)
        GETTER(Color, text_clr)
        GETTER(Color, fill_clr)
        GETTER(Color, line_clr)
        GETTER(bool, toggled)
        GETTER(bool, toggleable)
        
        Text _text;
        
    public:
        Button(const std::string& txt,
               const Bounds& size);
        Button(const std::string& txt,
               const Bounds& size,
               const Color& fill,
               const Color& text_clr = White,
               const Color& line = Black.alpha(200));
        virtual ~Button() {}
        
        void set_txt(const std::string& txt);
        void set_font(Font font);
        const Font& font() const;
        
        void set_text_clr(const decltype( _text_clr ) & text_clr) {
            if ( _text_clr == text_clr )
                return;
            
            _text_clr = text_clr;
            _text.set_color(text_clr);
            set_dirty();
        }
        
        void set_fill_clr(const Color& fill_clr);
        void set_line_clr(const Color& line_clr);
        
        void update() override;
        void set_toggleable(bool v);
        void set_toggle(bool v);
    };
}

#endif
