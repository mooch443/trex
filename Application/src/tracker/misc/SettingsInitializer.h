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

struct LoadContext {
    file::PathArray source;
    file::Path filename;
    default_config::TRexTask task{default_config::TRexTask_t::none};
    track::detect::ObjectDetectionType_t type{track::detect::ObjectDetectionType::none};
    ExtendableVector exclude_parameters;
    cmn::sprite::Map source_map;
    bool quiet{false};

    SettingsMaps combined;
    sprite::Map current_defaults;
    bool did_set_calculate_posture_to_false{false};

    ExtendableVector exclude, exclude_from_default;
    std::vector<std::string> system_variables = [](){
        std::vector<std::string> system_variables;
        for (auto& key : GlobalSettings::map().keys()) {
            if (GlobalSettings::access_level(key) >= AccessLevelType::SYSTEM) {
                system_variables.emplace_back(key);
            }
        }
        return system_variables;
    }();
    
    static constexpr auto default_excludes = std::array{
        //"meta_cmd", // is SYSTEM
        //"app_name", // is SYSTEM
        "nowindow",
        "gui_interface_scale",
        //"auto_quit",
        "load",
        "task",
        "filename",
        "source"
    };

    static constexpr auto exclude_from_external = std::array{
        "detect_model",
        "region_model",
        "detect_resolution",
        "region_resolution"
    };
    
    bool changed_model_manually{false};
    
    void init();
    bool set_config_if_different(const std::string_view& key, const sprite::Map& from, bool do_print = false);
    void init_filename();
    void fix_empty_source();
    void fix_empty_filename();
    void reset_default_filenames();
    
    void load_settings_from_source();
    void load_task_defaults();
    void load_settings_file();
    void load_gui_settings();
    
    void estimate_meta_variables();
    
    void finalize();
};

void load(LoadContext);

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
