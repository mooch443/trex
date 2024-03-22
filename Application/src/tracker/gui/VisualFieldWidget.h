#pragma once

#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <gui/types/Layout.h>
#include <misc/TrackingSettings.h>

namespace track {
    class Individual;
    class VisualField;
}

namespace gui {
class FindCoord;

class VisualFieldWidget : public Entangled {
    std::vector<derived_ptr<Polygon>> _polygons;
    Frame_t _last_frame;
    std::unordered_map<track::Idx_t, std::unique_ptr<track::VisualField>> _fields;
public:
    VisualFieldWidget();
    ~VisualFieldWidget();
    void update(Frame_t, const FindCoord&, const track::set_of_individuals_t& selected);
    void set_parent(SectionInterface*) override;
private:
    using Entangled::update;
};

}
