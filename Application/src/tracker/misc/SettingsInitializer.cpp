#include "SettingsInitializer.h"
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>
#include <misc/CommandLine.h>
#include <misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <file/DataLocation.h>
#include <tracking/TrackingSettings.h>

using namespace cmn;
using namespace track;

namespace settings {

void load(default_config::TRexTask, std::vector<std::string> exclude_parameters) {
    DebugHeader("Reloading settings");
    
    SettingsMaps combined;
    const auto set_combined_access_level =
    [&combined](auto& name, AccessLevel level) {
        combined.access_levels[name] = level;
    };
    
    combined.map.set_print_by_default(false);
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    default_config::get(combined.map, combined.docs, set_combined_access_level);
    
    std::vector<std::string> save = combined.map.has("meta_write_these") ? combined.map.at("meta_write_these").value<std::vector<std::string>>() : std::vector<std::string>{};
    print("Have these keys:", combined.map.keys());
    std::set<std::string> deleted_keys;
    for (auto key : combined.map.keys()) {
        if (not contains(save, key)) {
            deleted_keys.insert(key);
            combined.map.erase(key);
        }
        
        if(GlobalSettings::has(key)
           && combined.map.has(key)
           && GlobalSettings::map().at(key) != combined.map.at(key))
        {
            auto A = GlobalSettings::map().at(key);
            auto B = combined.map.at(key);
            print(key, " differs from default: ", A.get().valueString()," != ", B.get().valueString());
            exclude_parameters.push_back(key);
        }
    }
    print("Deleted keys:", deleted_keys);
    print("Remaining:", combined.map.keys());
    
    thread_print("source = ", SETTING(source).value<file::PathArray>(), " ", (uint64_t)&GlobalSettings::map());
    GlobalSettings::map().set_print_by_default(true);
    //default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    //default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_focus_group"].get().set_do_print(false);
    GlobalSettings::map()["gui_source_video_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_displayed_frame"].get().set_do_print(false);
    
    // get cmd arguments
    auto& cmd = CommandLine::instance();
    
    // preload options to be excluded and get output prefix, since
    // that is important for finding the right settings file
    for (auto& option : cmd.settings()) {
        if (utils::lowercase(option.name) == "output_prefix") {
            SETTING(output_prefix) = option.value;
        }
        exclude_parameters.push_back(option.name);
    }
    
    auto default_path = file::DataLocation::parse("default.settings");
    if (default_path.exists()) {
        try {
            default_config::warn_deprecated(default_path, GlobalSettings::load_from_file(default_config::deprecations(), default_path.str(), AccessLevelType::STARTUP, exclude_parameters));
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",default_path,": ", ex.what() );
        }
    }
    
    thread_print("source = ", SETTING(source).value<file::PathArray>(), " ", (uint64_t)&GlobalSettings::map());
    
    auto settings_file = file::DataLocation::parse("settings");
    if(settings_file.exists()) {
        try {
            default_config::warn_deprecated(settings_file, GlobalSettings::load_from_file(default_config::deprecations(), settings_file.str(), AccessLevelType::STARTUP, exclude_parameters));
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",settings_file,": ", ex.what());
        }
        
    } else if(not settings_file.empty()) {
        FormatError("Settings file ", settings_file, " was not found.");
    }
    
    cmd.load_settings(&combined);
    
    file::Path path = file::find_basename(SETTING(source).value<file::PathArray>());
    if(path.has_extension()
       && path.extension() != "pv")
    {
        // did we mean .mp4.pv?
        auto prefixed = file::DataLocation::parse("output", path.add_extension("pv"));
        if(not prefixed.exists()) {
            path = path.remove_extension();
            
            //! do we remove the full path, or do we put the .pv file next
            //! to the original video file?
            //path = path.filename();
        } // else we can open it, so prefer it
    }

    file::Path filename = file::DataLocation::parse("output", path);
    SETTING(filename) = filename.remove_extension();
    
    if(SETTING(cm_per_pixel).value<Settings::cm_per_pixel_t>() == 0) {
        if(SETTING(source).value<file::PathArray>() == file::PathArray("webcam")) {
            SETTING(cm_per_pixel) = Settings::cm_per_pixel_t(0.01);
        }
    }
}

}
