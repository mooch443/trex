#pragma once

#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <gui/types/Layout.h>

namespace gui {

class GUICache;

class VisualFieldWidget : public Entangled {
    const GUICache* _cache;
    std::vector<derived_ptr<Polygon>> _polygons;
public:
    VisualFieldWidget(const GUICache* cache) : _cache(cache) {}
    void update() override;
    void set_parent(SectionInterface*) override;
};

}
