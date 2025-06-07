#include "DrawBorder.h"
#include <gui/DrawStructure.h>
#include <gui/GuiTypes.h>

namespace cmn::gui {

DrawBorder::DrawBorder(const track::Border& border) {
    for(auto& convex : border.polygons()) {
        auto ptr = std::make_shared<gui::Polygon>(*convex);
        ptr->set_fill_clr(gui::Transparent);
        ptr->set_border_clr(gui::Cyan);
        _polygons.push_back((std::shared_ptr<gui::Drawable>)ptr);
    }
}

}
