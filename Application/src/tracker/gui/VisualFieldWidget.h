#pragma once

#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <gui/types/Layout.h>
#include <tracking/TrackingSettings.h>

namespace track {
    class Individual;
}

namespace gui {
class FindCoord;

class VisualFieldWidget : public Entangled {
    std::vector<derived_ptr<Polygon>> _polygons;
public:
    void update(Frame_t, const FindCoord&, const track::set_of_individuals_t& selected);
    void set_parent(SectionInterface*) override;
};

}
