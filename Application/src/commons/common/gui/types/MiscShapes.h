#pragma once

#include <gui/GuiTypes.h>
#include <gui/types/Entangled.h>

namespace gui {
    class Triangle final : public Entangled {
        GETTER(Color, fill)
        GETTER(Color, line)
        GETTER(std::vector<Vertex>, points)
        
    public:
        Triangle(const Vec2& center, const Size2& size, float angle = 0, const Color& fill = White, const Color& line = Transparent);
        Triangle(const std::vector<Vertex>& vertices);
        
        CHANGE_SETTER(fill)
        CHANGE_SETTER(line)
        
        bool in_bounds(float x, float y) override;
        
    protected:
        bool swap_with(Drawable *d) override;
        //void prepare() override;
        static std::vector<Vertex> simple_triangle(const Color& color, const Size2& size);
    };
}
