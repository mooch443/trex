#pragma once

#include <commons.pc.h>
#include <misc/Border.h>
#include <gui/types/Entangled.h>

namespace cmn::gui {
class DrawStructure;
class Drawable;

class DrawBorder : public Entangled {
    std::vector<std::shared_ptr<gui::Drawable>> _polygons;
public:
    DrawBorder(const track::Border&);
};

}
