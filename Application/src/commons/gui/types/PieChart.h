#pragma once

#include <gui/types/Entangled.h>
#include <gui/DrawSFBase.h>

namespace gui {
    class PieChart : public Entangled {
    public:
        class Slice
        {
        public:
            float size;
            float scale;
            float explode;
            
            std::string name;
            
            Color color;
            
        public:
            Slice(const std::string& name = "", Color color = White, float size = 0, float scale = 1, float explode = 0);
            
            friend bool operator==(const gui::PieChart::Slice&, const gui::PieChart::Slice&);
        };
        
    protected:
        GETTER(Font, font)
        GETTER(std::vector<Slice>, slices)
        GETTER(float, radius)
        GETTER(std::vector<Vertex>, vertices)
        GETTER(float, alpha)
        
        std::vector<Vertex> _lines;
        std::vector<Vec2> _corners;
        std::vector<Vec2> _centers;
        
        Vec2 _mouse;
        long _hovered_slice;
        
        std::function<void(size_t, const std::string&)> _on_slice_hovered;
        std::function<void(size_t, const std::string&)> _on_slice_clicked;
        
    public:
        PieChart(const Vec2& pos, float radius, const std::vector<Slice>& slices, const Font& font = Font(0.75), const decltype(_on_slice_clicked)& fn_click = [](auto,auto){});
        virtual ~PieChart() {}
        
        void set_font(const Font& font) {
            if(_font == font)
                return;
            
            _font = font;
            set_content_changed(true);
        }
        
        void set_radius(float radius) {
            if(_radius == radius)
                return;
            
            _radius = radius;
            set_content_changed(true);
        }
        
        void set_alpha(float alpha) {
            if(_alpha == alpha)
                return;
            
            _alpha = alpha;
            set_content_changed(true);
        }
        
        void set_slices(const std::vector<Slice>& slices);
        void update_bounds() override;
        
    protected:
        void update() override;
        virtual void draw_slices();
        void update_triangles();
    };
}
