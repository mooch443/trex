#pragma once

#include <misc/vec2.h>
#include <gui/Transform.h>
#include <misc/bid.h>
#include <misc/frame_t.h>

namespace gui {
class DrawStructure;
class GUICache;
class Section;
class Dropdown;
class Base;
class Textfield;
}

namespace tracker {
namespace gui {

struct DisplayParameters {
    const cmn::Vec2 &offset, &scale;
    ::gui::DrawStructure& graph;
    ::gui::Section* ptr;
    ::gui::GUICache& cache;
    const ::gui::Transform& transform;
    const cmn::Size2 &screen;
    ::gui::Base* base{nullptr};
};

void draw_blob_view(const DisplayParameters&);
void draw_boundary_selection(::gui::DrawStructure& base, ::gui::Base* window, ::gui::GUICache& cache, ::gui::Section* bowl, ::gui::Dropdown& settings_dropdown, ::gui::Textfield& value_input);

void set_clicked_blob_id(pv::bid v);
void set_clicked_blob_frame(::gui::Frame_t v);

void clicked_background(::gui::DrawStructure& base, ::gui::GUICache& cache, const ::gui::Vec2& pos, bool v, std::string key, ::gui::Dropdown& settings_dropdown, ::gui::Textfield& value_input);

}
}
