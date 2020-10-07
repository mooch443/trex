#pragma once

#include <gui/Section.h>

namespace gui {
    class DrawableCollection : public Section {
    public:
        DrawableCollection(const std::string& name)
            : Section(NULL, NULL, name)
        {}
        DrawableCollection()
            : Section(NULL, NULL, std::to_string((long)this))
        {}
        virtual ~DrawableCollection();
        
        virtual void update(DrawStructure& s) = 0;
    };
}
