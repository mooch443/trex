#pragma once

#include <commons.pc.h>
#include <gui/Transform.h>
#include <misc/bid.h>
#include <misc/frame_t.h>
#include <misc/Coordinates.h>

namespace cmn::gui {
class DrawStructure;
class GUICache;
class SectionInterface;
class Dropdown;
class Base;
class Textfield;
}

namespace cmn::gui::tracker {

struct DisplayParameters {
    DrawStructure& graph;
    GUICache& cache;
    const FindCoord& coord;
};

void draw_blob_view(const DisplayParameters&);
void draw_boundary_selection(DrawStructure& base, Base* window, GUICache& cache, SectionInterface* bowl);

void set_clicked_blob_id(pv::bid v);
void set_clicked_blob_frame(Frame_t v);

void clicked_background(DrawStructure& base, GUICache& cache, const Vec2& pos, bool v, std::string key);

void blob_view_shutdown();

}
