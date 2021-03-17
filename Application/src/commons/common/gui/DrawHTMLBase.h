#ifndef _DRAW_HTML_BASE_H
#define _DRAW_HTML_BASE_H

#include "DrawBase.h"

namespace gui {
    
    class HTMLCache : public CacheObject {
    protected:
        GETTER_SETTER(std::string, text)
        
    public:
        HTMLCache() { set_changed(true); }
    };
    
    class HTMLBase : public Base {
        std::stringstream _ss;
        std::vector<uchar> _vec;
        cv::Mat tmp;
        bool _initial_draw;
        Size2 _size;
        std::string _title;
        
    public:
        HTMLBase();
        ~HTMLBase() {}
        
        void set_window_size(const Size2& size);
        virtual void paint(DrawStructure& s) override;
        virtual void set_title(std::string title) override { _title = title; }
        const std::string& title() const override { return _title; }
        virtual Size2 window_dimensions() override;
        
        const std::vector<uchar>& to_bytes() const {
            return _vec;
        }
        
        void reset();
    };
}

#endif
