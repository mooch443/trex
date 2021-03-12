#pragma once

#include <gui/Section.h>

namespace gui {
    class DrawableCollection : public Section {
    public:
        DrawableCollection(const std::string& name)
            : Section(nullptr, nullptr, name)
        {}
        DrawableCollection()
            : Section(nullptr, nullptr, std::to_string((uint64_t)this))
        {}
        virtual ~DrawableCollection();
        
        virtual void update(DrawStructure& s) = 0;
    };
}
