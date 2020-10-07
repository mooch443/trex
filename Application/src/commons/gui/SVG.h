#pragma once

#include <gui/types/Basic.h>
#include <misc/vec2.h>

namespace gui {

class SVG {
    Size2 _size;
    std::stringstream _ss;
    
public:
    SVG(const Size2& size) : _size(size) {}
    ~SVG() {}
    
    void begin();
    void end();
    void line(const Vec2& A, const Vec2&B, const Color& fill = White, const Color& stroke = Transparent);
    void circle(const Vec2& pos, float r, const Color& fill = White, const Color& stroke = Transparent);
    void rect(const Bounds& bounds, const Color& fill = White, const Color& stroke = Transparent);
    std::string string() const;
    
private:
    static std::string color2svg(const Color& c);
};

}
