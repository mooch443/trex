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

void initialize_filename_for_tracking();

void load(file::PathArray source,
          file::Path filename,
          default_config::TRexTask task,
          track::detect::ObjectDetectionType_t type,
          ExtendableVector exclude_parameters,
          const cmn::sprite::Map&,
          bool quiet);

std::unordered_set<std::string_view>
set_defaults_for( track::detect::ObjectDetectionType_t detect_type,
                  cmn::sprite::Map& output,
                  ExtendableVector exclude = {});

SettingsMaps reset(const cmn::sprite::Map& extra_map = {}, cmn::sprite::Map* output = nullptr);

void write_config(const pv::File*, bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix = "");
Float2_t infer_cm_per_pixel(const cmn::sprite::Map* = nullptr);
Float2_t infer_meta_real_width_from(const pv::File &file, const sprite::Map* map = nullptr);

file::Path find_output_name(const sprite::Map&, file::PathArray source = {}, file::Path filename = {}, bool respect_user_choice = true);

}
