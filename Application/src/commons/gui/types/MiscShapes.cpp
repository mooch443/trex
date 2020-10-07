#include "MiscShapes.h"

namespace gui {
    
    Triangle::Triangle(const Vec2& center,
                       const Size2& size,
                       float angle,
                       const Color& fill,
                       const Color& line)
        :  _fill(fill),
           _line(line)
    {
        if(fill != Transparent) {
            _points = simple_triangle(fill, size);
            entangle(new Vertices(_points, PrimitiveType::Triangles, Vertices::TRANSPORT));
        }
        
        if(line != Transparent) {
            auto array = simple_triangle(line, size);
            if(_points.empty())
                _points = array;
            array.push_back(array.front());
            
            entangle(new Vertices(array, PrimitiveType::LineStrip, Vertices::COPY));
        }
        
        set_origin(Vec2(0.5, 0.65));
        set_bounds(Bounds(center, size));
        
        if(angle != 0)
            set_rotation(angle);
    }

    Triangle::Triangle(const std::vector<Vertex>& vertices)
        : _fill(Transparent),
          _line(Transparent)
    {
        assert(vertices.size() % 3 == 0);
        _points = vertices;
        entangle(new Vertices(_points, PrimitiveType::Triangles, Vertices::TRANSPORT));
    }

    std::vector<Vertex> Triangle::simple_triangle(const gui::Color &color, const Size2& size) {
        return {
            Vertex(size.width * 0.5, 0, color),
            Vertex(size.width, size.height, color),
            Vertex(0, size.height, color)
        };
    }

    bool Triangle::swap_with(gui::Drawable *d) {
        auto ptr = dynamic_cast<Triangle*>(d);
        if(!ptr)
            return false;
        
        if(!Entangled::swap_with(d))
            return false;
        
        set_fill(ptr->_fill);
        set_line(ptr->_line);
        
        if(_points != ptr->_points) {
            // did_change will already be set because of Vertices::swap_with
            _points = ptr->_points;
        }
        
        return true;
    }

    /*void Triangle::prepare() {
        if(_transport)
            Vertices::prepare();
    }*/

    bool Triangle::in_bounds(float x, float y) {
        if(_points.empty())
            return Entangled::in_bounds(x, y);
        
        auto &gtransform = global_transform();
        auto p0 = gtransform.transformPoint(_points.at(0).position());
        auto p1 = gtransform.transformPoint(_points.at(1).position());
        auto p2 = gtransform.transformPoint(_points.at(2).position());
        
        float Area = 0.5 *(-p1.y*p2.x + p0.y*(-p1.x + p2.x) + p0.x*(p1.y - p2.y) + p1.x*p2.y);
        float s = 1/(2*Area)*(p0.y*p2.x - p0.x*p2.y + (p2.y - p0.y)*x + (p0.x - p2.x)*y);
        float t = 1/(2*Area)*(p0.x*p1.y - p0.y*p1.x + (p0.y - p1.y)*x + (p1.x - p0.x)*y);
        
        return 0 <= s && s <= 1 && 0 <= t && t <= 1 && s + t <= 1;
    }
}
