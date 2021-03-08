#include "PieChart.h"

namespace gui {
    float sign (Vec2 p1, Vec2 p2, Vec2 p3)
    {
        return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y);
    }
    
    bool PointInTriangle (Vec2 pt, Vec2 v1, Vec2 v2, Vec2 v3)
    {
        bool b1, b2, b3;
        
        b1 = sign(pt, v1, v2) < 0.0f;
        b2 = sign(pt, v2, v3) < 0.0f;
        b3 = sign(pt, v3, v1) < 0.0f;
        
        return ((b1 == b2) && (b2 == b3));
    }
    
    bool operator==(const gui::PieChart::Slice& a, const gui::PieChart::Slice& b)  {
        return a.size == b.size && a.scale == b.scale && a.explode == b.explode && a.name == b.name && a.color == b.color;
    }
    
    PieChart::Slice::Slice(const std::string& name, Color color, float size, float scale, float explode)
        : size(size), scale(scale), explode(explode), name(name), color(color)
    { }
    
    PieChart::PieChart(const Vec2& pos, float radius, const std::vector<Slice>& slices, const Font& font, const decltype(_on_slice_clicked)& fn_clicked)
        : _font(font), _radius(radius), _alpha(1), _hovered_slice(-1), _on_slice_hovered([](auto,auto){}), _on_slice_clicked(fn_clicked)
    {
        if(!slices.empty())
            set_slices(slices);
        set_bounds(Bounds(pos, Size2(radius * 2)));
        set_origin(Vec2(0.5));
        set_clickable(true);
        
        on_hover([this](Event e) {
            _mouse = Vec2(e.hover.x, e.hover.y);
            
            auto previous = _hovered_slice;
            update_triangles();
            
            if(_hovered_slice != previous && _hovered_slice != -1)
                _on_slice_hovered(_hovered_slice, _slices.at(_hovered_slice).name);
            this->set_content_changed(true);
        });
        
        add_event_handler(EventType::MBUTTON, [this](Event e) {
            if(!e.mbutton.pressed && e.mbutton.button == 0) {
                if(_hovered_slice != -1)
                    _on_slice_clicked(_hovered_slice, _slices.at(_hovered_slice).name);
            }
        });
    }
    
    void PieChart::set_slices(const std::vector<Slice> &slices) {
        if(_slices == slices)
            return;
        
        _slices.clear();
        for(auto &s : slices)
            _slices.push_back(Slice(s));
        
        set_content_changed(true);
    }
    
    void PieChart::update_bounds() {
        if(!bounds_changed())
            return;
        
        Entangled::update_bounds();
        update_triangles();
    }
    
    void PieChart::update_triangles() {
        const Vec2 halfSize(_radius);
        unsigned int N_triangles{ 0u };
        std::vector<unsigned int> triangles_per_slice;
        
        float slice_size = 1.f / _slices.size();
        for (auto& slice : _slices) {
            triangles_per_slice.emplace_back(static_cast<unsigned int>(std::floor(1.f + (slice.size ? slice.size : slice_size) * 50.f)));
            N_triangles += triangles_per_slice.back();
        }
        
        _vertices.resize(N_triangles * 3);
        _lines.resize(_slices.size() * 2);
        _corners.resize(_slices.size() * 3);
        _centers.resize(_slices.size());
        
        unsigned int currentVertex{ 0u };
        float currentAngle{ 0.f };
        
        _hovered_slice = -1;
        
        for(auto&l :_lines)
            l.color() = Black.alpha(_alpha);
        
        for (unsigned int slice{ 0u }; slice < _slices.size(); ++slice)
        {
            const float startAngle{ currentAngle };
            const float halfAngleDifference{ 180.f * (_slices[slice].size ? _slices[slice].size : slice_size) };
            const Vec2 offset{ Vec2(sin(RADIANS(startAngle + halfAngleDifference)), -cos(RADIANS(startAngle + halfAngleDifference))) * _slices[slice].explode };
            
            bool triangle_hover = false;
            uint32_t start_vertex = currentVertex;
            
            for (unsigned int triangle{ 0u }; triangle < triangles_per_slice[slice]; ++triangle)
            {
                _vertices[currentVertex + 0].position() = halfSize + Vec2(offset.x * halfSize.x, offset.y * halfSize.y);
                _vertices[currentVertex + 1].position() = halfSize + Vec2((offset.x + sin(RADIANS(currentAngle))) * halfSize.x, (offset.y - cos(RADIANS(currentAngle))) * halfSize.y);
                currentAngle += halfAngleDifference * 2.f / triangles_per_slice[slice];
                _vertices[currentVertex + 2].position() = halfSize + Vec2((offset.x + sin(RADIANS(currentAngle))) * halfSize.x, (offset.y - cos(RADIANS(currentAngle))) * halfSize.y);
                
                if(PointInTriangle(_mouse, _vertices[currentVertex], _vertices[currentVertex+1], _vertices[currentVertex+2]))
                {
                    triangle_hover = true;
                }
                
                currentVertex += 3;
            }
            
            float scale = 1;
            Color clr = _slices[slice].color;
            clr = Drawable::accent_color;//Color(slice / float(_slices.size()) * 255, 100, 50, 200);
            //if(slice && slice % 2 == 0) {
            clr = clr.toHSV();
            Color hover = clr.blue(clr.b * (1.5)).HSV2RGB();
            Color press = clr.blue(clr.b * (1 + 0.5 / float(_slices.size()+1))).HSV2RGB();
            clr = clr.blue(clr.b * (1 + (slice+1) / float(_slices.size()+1) * 0.5)).HSV2RGB();
            //}
            
            if(triangle_hover) {
                if(!pressed())
                    scale *= 1.05;
                else
                    scale *= 0.95;
                clr = pressed() ? press.toHSV() : hover.toHSV();
                
                clr = clr.blue(clr.b * (pressed() ? 0.6 :1.1)).HSV2RGB();
                
                _hovered_slice = (long)slice;
            }
            
            clr = clr.alpha(saturate(_alpha * 255));
            
            _slices[slice].scale = scale;
            
            for (unsigned int triangle{ 0u }, vertex = start_vertex; triangle < triangles_per_slice[slice]; ++triangle, vertex += 3)
            {
                _vertices[vertex + 1].position() = halfSize + (_vertices[vertex + 1].position() - halfSize) * scale;
                _vertices[vertex + 2].position() = halfSize + (_vertices[vertex + 2].position() - halfSize) * scale;
                
                _vertices[vertex + 0].color() =
                _vertices[vertex + 1].color() =
                _vertices[vertex + 2].color() = clr;
            }
            
            _lines[slice * 2 + 0].position() =
                _vertices[start_vertex + 3 * 0 + 1].position();
            _lines[slice * 2 + 1].position() =
                _vertices[start_vertex + 3 * 0].position();
            
            _corners[slice * 3 + 0] = _vertices[start_vertex + 3 * 0 + 1].position();
            _corners[slice * 3 + 1] = _vertices[start_vertex + 3 * (triangles_per_slice[slice]-1) + 2].position();
            _corners[slice * 3 + 2] = _vertices[start_vertex + 3 * 0 + 0].position();
            
            _centers[slice] = (_corners[slice * 3 + 0] + _corners[slice * 3 + 1] + _corners[slice * 3 + 2]) / 3.f;
        }
    }
    
    void PieChart::update() {
        update_triangles();
        
        begin();
        draw_slices();
        end();
    }
    
    void PieChart::draw_slices() {
        if(!begun())
            U_EXCEPTION("begin() first");
        
        advance(new Vertices(_vertices, PrimitiveType::Triangles, Vertices::COPY));
        advance(new Vertices(_lines, PrimitiveType::Lines, Vertices::COPY));
        
        for(size_t i=0; i<_slices.size(); ++i) {
            if(!_slices[i].name.empty())
                advance(new Text(_slices[i].name, _centers[i], White, Font(0.8, Align::Center), Vec2(_slices[i].scale * max(0.1,_radius / 250.f))));
        }
    }
}
