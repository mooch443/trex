#pragma once

#include <gui/types/Entangled.h>
#include <gui/types/StaticText.h>

namespace gui {
    class Tooltip : public Entangled {
        GETTER_PTR(Drawable*, other)
        GETTER_NCONST(StaticText, text)
        float _max_width;
        
    public:
        Tooltip(Drawable* other, float max_width = -1);
        void set_text(const std::string& text);
        void set_other(Drawable* other);
        
    protected:
        virtual void update() override;
    };
}
