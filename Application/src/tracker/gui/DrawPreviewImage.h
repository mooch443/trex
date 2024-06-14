#pragma once

#include <gui/DrawStructure.h>
#include <misc/frame_t.h>
#include <tracking/PPFrame.h>
#include <tracking/Outline.h>
#include <tracking/FilterCache.h>

/**
 * This command previews all selected individuals the way they would be shown,
 * given parameters like `individual_image_size`, `individual_image_scale`
 * and `individual_image_normalization`.
 * The window is draggable and displayed on top of the rest of the GUI
 * in tracking view, if enabled.
 */

namespace cmn::gui {
namespace DrawPreviewImage {

void draw(const track::Background* average, const track::PPFrame&, Frame_t, DrawStructure&);

std::tuple<Image::Ptr, Vec2>
make_image(pv::BlobWeakPtr blob,
           const track::Midline* midline,
           const track::constraints::FilterCache* filters,
           const track::Background*);

}
}
