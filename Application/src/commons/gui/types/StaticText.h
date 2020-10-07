#ifndef _STATIC_TEXT_H
#define _STATIC_TEXT_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>

namespace gui {
    class StaticText : public Entangled {
        GETTER(std::string, txt)
        
        std::vector<std::shared_ptr<Text>> texts;
        std::vector<Vec2> positions;
        
        GETTER(Vec2, max_size)
        Vec2 _org_position;
        Bounds _margins;
        
        Font _default_font;
        Color _base_text_color;
        
    public:
        struct RichString {
            std::string str;
            Font font;
            Vec2 pos;
            Color clr;
            
            RichString(const std::string& str = "", const Font& font = Font(), const Vec2& pos = Vec2(), const Color& clr = Color())
                : str(str), font(font), pos(pos), clr(clr)
            {}
            
            void convert(std::shared_ptr<Text> text) const {
                text->set_color(clr);
                text->set_font(font);
                text->set_txt(str);
            }
        };
        
    public:
        StaticText(const std::string& txt, const Vec2& pos, const Vec2& max_size = Vec2(-1, -1), const Font& font = Font(0.75));
        virtual ~StaticText() {
            texts.clear();
        }
        
        void set_txt(const std::string& txt) {
            auto t = utils::find_replace(txt, "<br/>", "\n");
            if(_txt == t)
                return;
            
            _txt = t;
            update_text();
        }
        void set_base_text_color(const Color& c) {
            if(c == _base_text_color)
                return;
            
            _base_text_color = c;
            update_text();
        }
        
        void set_margins(const Bounds& margin);
        
        void update() override;
        void add_string(std::shared_ptr<RichString> ptr, std::vector<std::shared_ptr<RichString>>& output, Vec2& offset);
        void structure_changed(bool downwards) override;
        virtual void set_size(const Size2& size) override;
        
        virtual const Size2& size() override;
        virtual const Bounds& bounds() override;
        void set_max_size(const Size2&);
        
    private:
        void update_text();
    };
}

#endif
