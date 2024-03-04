#pragma once
#include <commons.pc.h>
#include <tracker/misc/default_config.h>
#include <gui/GUITaskQueue.h>
#include <misc/DetectionTypes.h>

namespace cmn::sprite {
    class Map;
}
namespace pv {
    class File;
}

namespace cmn::settings {

void load(file::PathArray source,
          file::Path filename,
          default_config::TRexTask task,
          track::detect::ObjectDetectionType_t type,
          ExtendableVector exclude_parameters,
          const cmn::sprite::Map&);

void write_config(bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix = "");
float infer_cm_per_pixel(const cmn::sprite::Map* = nullptr);
float infer_meta_real_width_from(const pv::File &file, const sprite::Map* map = nullptr);

file::Path find_output_name(const sprite::Map&, file::PathArray source = {}, file::Path filename = {}, bool respect_user_choice = true);

}
