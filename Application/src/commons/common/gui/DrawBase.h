#ifndef _DRAW_BASE_H
#define _DRAW_BASE_H

#include <misc/defines.h>

#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>
#include <misc/Image.h>

namespace gui {
    typedef uint32_t uint;

    enum class LoopStatus {
        IDLE,
        UPDATED,
        END
    };
    
    class Base {
    protected:
        GETTER(bool, frame_recording)
        std::function<Bounds(const std::string&, Drawable*, const Font&)> _previous_line_bounds;
        std::function<uint32_t(const Font&)> _previous_line_spacing;
        Base *_previous_base;
        
    public:
        Base();
        virtual ~Base();
        
        virtual LoopStatus update_loop() { return LoopStatus::IDLE; }
        virtual void set_background_color(const Color&) {}
        
        virtual void set_frame_recording(bool v) {
            _frame_recording = v;
        }
        
        virtual const Image::UPtr& current_frame_buffer() {
            static Image::UPtr _empty(nullptr);
            return _empty;
        }
        
        virtual void paint(DrawStructure& s) = 0;
        virtual void set_title(std::string) = 0;
        virtual const std::string& title() const = 0;
        virtual Size2 window_dimensions() { return Size2(-1); }
        virtual Event toggle_fullscreen(DrawStructure&g) { Event e(WINDOW_RESIZED); e.size.width = g.width(); e.size.height = g.height(); return e; }
        
        virtual float text_width(const Text &text) const;
        virtual float text_height(const Text &text) const;
        
        static inline Size2 text_dimensions(const std::string& text, Drawable* obj = NULL, const Font& font = Font()) {
            auto size = default_text_bounds(text, obj, font);
            return Size2(size.pos() + size.size());
        }
        
        virtual Bounds text_bounds(const std::string& text, Drawable*, const Font& font);
        static Bounds default_text_bounds(const std::string& text, Drawable* obj = NULL, const Font& font = Font());
        static void set_default_text_bounds(std::function<Bounds(const std::string&, Drawable*, const Font&)>);
        
        virtual uint32_t line_spacing(const Font& font);
        
        static uint32_t default_line_spacing(const Font& font);
        static void set_default_line_spacing(std::function<uint32_t(const Font&)>);
    };
}

#endif
