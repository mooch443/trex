#pragma once

#include <commons.pc.h>
#include <gui/Transform.h>
#include <misc/bid.h>
#include <misc/frame_t.h>
#include <misc/Coordinates.h>

namespace gui {
class DrawStructure;
class GUICache;
class SectionInterface;
class Dropdown;
class Base;
class Textfield;
}

namespace tracker {
namespace gui {

struct DisplayParameters {
    ::gui::DrawStructure& graph;
    ::gui::GUICache& cache;
    const ::gui::FindCoord& coord;
};

void draw_blob_view(const DisplayParameters&);
void draw_boundary_selection(::gui::DrawStructure& base, ::gui::Base* window, ::gui::GUICache& cache, ::gui::SectionInterface* bowl);

void set_clicked_blob_id(pv::bid v);
void set_clicked_blob_frame(::gui::Frame_t v);

void clicked_background(::gui::DrawStructure& base, ::gui::GUICache& cache, const ::gui::Vec2& pos, bool v, std::string key);

void blob_view_shutdown();

}
}
