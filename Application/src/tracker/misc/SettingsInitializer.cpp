#include "SettingsInitializer.h"
#include <misc/GlobalSettings.h>
#include <misc/SpriteMap.h>
#include <misc/CommandLine.h>
#include <misc/default_config.h>
#include <grabber/misc/default_config.h>
#include <file/DataLocation.h>
#include <misc/TrackingSettings.h>
#include <gui/DrawStructure.h>
#include <pv.h>

using namespace cmn;
using namespace track;
using namespace default_config;

namespace cmn::settings {

/**
 * When calling "load" or "open", what we expect is:
 *  1. Settings are reset to default
 *  2. **exclude** SYSTEM variables
 *  3. load **default.settings**
 *  4. load **command-line** (except exclude) => exclude?
 *     >> exclude `source` and `filename` since we get that
 *        from the function parameters if applicable
 *  5. **exclude** `STARTUP` variables
 *  6. Overwrite `filename` + `source` with parameters if not empty
 *  7. Overwrite `output_dir` + `output_prefix` from map parameter
 *  8. if `source` or `filename` are empty, load them from provided
 *     parameters (e.g. `task`::track + `filename`)
 *  9. **exclude** `output_dir` + `output_prefix` since we have now
 *     locked in the `filename` + `source` parameters
 * 10. If `task` is track, load settings from the .pv file
 * 11. (*optional*) load default values based on task, but exclude
 *                  all options from previous steps (cmd + *default*)
 * 12. Load `video.settings` if they exist
 * 13. load sprite::Map passed to function
 *
 * <=> User-relevant order <=>
 * 1. defaults
 * 2. default.settings
 * 3. task-specific defaults *(done by excluding cmd)*
 * 4. *(tracking)* pv-file
 * 5. *(optional)* <video>.settings
 * 6. command-line
 *
 * <=> IMPORTANT <=>
 *  1. the `source` + `filename` .settings files are not allowed
 *     to contain any of the following parameters:
 *       `{ "source", "filename", "output dir", "output prefix" }`
 *     as to not confuse the entire process of loading parameters.
 *  2. the same goes for the sprite::Map, since that would be
 *     confusing as well.
 *  3. stuff like `app_name` (SYSTEM) variables should not be read
 *
 * So for opening directly via command-line we need to:
 *  1. pass `source` + `filename` + `task` + `detect_type`
 *  2. call
 *
 * For opening tracking after switching from ConvertScene:
 *  /// happening in segmenter.cpp
 *  1. pass `source` + `filename` + `task` + `detect_type`
 *      (since these will be excluded from sprite map)
 *  2. fill everything into sprite::Map in order to maintain settings
 *  3. call
 *
 * For opening converting after settings:
 *  1. maintain settings
 *  2. call
 */

void load(file::PathArray source, 
          file::Path filename,
          TRexTask task,
          track::detect::ObjectDetectionType::Class type,
          ExtendableVector exclude_parameters,
          const cmn::sprite::Map& source_map)
{
    DebugHeader("Reloading settings"); 

    struct G {
        std::string s;
        G(const std::string& name) : s(name) {
            DebugHeader("// LOADING FROM ", s);
        }
        ~G() {
            DebugHeader("// LOADED ", s);
        }
    };
    
    SettingsMaps combined;
    const auto set_combined_access_level =
    [&combined](auto& name, AccessLevel level) {
        combined.access_levels[name] = level;
    };

    combined.map.set_print_by_default(false);
    
    /// ---------------------------------------------
    /// 1. setting default values, saved in combined:
    /// ---------------------------------------------
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    ::default_config::get(combined.map, combined.docs, set_combined_access_level);
    
    /// ---------------------------
    /// 2. exclude SYSTEM variables
    /// ---------------------------
    /// as well as other defaults
    auto default_excludes = std::array{
        //"meta_cmd", // is SYSTEM
        //"app_name", // is SYSTEM
        "nowindow",
        "gui_interface_scale",
        //"auto_quit",
        "task",
        "filename",
        "source"
    };

    std::vector<std::string> system_variables;
    for (auto& key : GlobalSettings::map().keys()) {
        if (GlobalSettings::access_level(key) >= AccessLevelType::SYSTEM) {
            system_variables.emplace_back(key);
        }
    }
    
    auto exclude = exclude_parameters + default_excludes + system_variables;
    print("Excluding from command-line and default.settings: ", exclude);
    
    /// -----------------------------------------
    /// 3. load default.settings from app folder:
    /// -----------------------------------------
    auto default_path = file::DataLocation::parse("default.settings", {}, &combined.map);
    if (default_path.exists()) {
        try {
            warn_deprecated(default_path, GlobalSettings::load_from_file(deprecations(), default_path.str(), AccessLevelType::STARTUP, exclude, &combined.map));
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",default_path,": ", ex.what() );
        }
    }
    
    /// ---------------------------------------------------
    /// 4. get cmd arguments and overwrite stuff with them:
    /// ---------------------------------------------------
    /// excluding filename and source + other defaults
    auto& cmd = CommandLine::instance();
    combined.map.set_print_by_default(true);
    
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_focus_group"].get().set_do_print(false);
    GlobalSettings::map()["gui_source_video_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_displayed_frame"].get().set_do_print(false);
    
    cmd.load_settings(nullptr, &combined.map, exclude.toVector());
    exclude += extract_keys( cmd.settings_keys() );

    /// ----------------------------
    /// 5. exclude STARTUP variables
    /// ----------------------------
    std::vector<std::string> startup_variables;
    for (auto& key : GlobalSettings::map().keys()) {
        if (GlobalSettings::access_level(key) >= AccessLevelType::STARTUP) {
            startup_variables.emplace_back(key);
        }
    }

    /// append cmd parameters so they wont be overwritten
    exclude += startup_variables;
    
    /// --------------------------------------------------------
    /// 6. set the source / filename properties from parameters:
    /// --------------------------------------------------------
    if(not filename.empty())
    {
        combined.map["filename"] = filename;
    }
    
    if(not source.empty())
    {
        combined.map["source"] = source;
        
        //if(combined.map.has("meta_source_path")
        //   && combined.map.at("meta_source_path").value<std::string>().empty())
        //{
        if(not contains(exclude.toVector(), "meta_source_path"))
            combined.map["meta_source_path"] = source.source();
        //}
    }

    /// ---------------------------------------------------------------------
    /// 7. set the `output_dir` / `output_prefix` properties from parameters:
    /// ---------------------------------------------------------------------
    if(source_map.has("output_dir"))
        source_map.at("output_dir").get().copy_to(&combined.map);
    if(source_map.has("output_prefix"))
        source_map.at("output_prefix").get().copy_to(&combined.map);
    
    /// -----------------------------------------------------
    /// 8. if `source` or `filename` are empty, generate them
    /// -----------------------------------------------------
    if(source.empty()
       && task == TRexTask_t::convert)
    {
        const auto source = combined.map.at("source").value<file::PathArray>();
        
        file::Path path = file::find_basename(source);
        if(path.has_extension()
           && path.extension() != "pv")
        {
            // did we mean .mp4.pv?
            auto prefixed = file::DataLocation::parse("output", path.add_extension("pv"), &combined.map);
            if(not prefixed.exists()) {
                path = path.remove_extension();
                
                //! do we remove the full path, or do we put the .pv file next
                //! to the original video file?
                //path = path.filename();
            } // else we can open it, so prefer it
        }
        
        if(CommandLine::instance().settings_keys().contains("filename")) {
            // automatic filename overwritten
            auto name = CommandLine::instance().settings_keys().at("filename");
            file::Path filename = file::DataLocation::parse("output", name, &combined.map);
            combined.map["filename"] = filename.remove_extension();
            
        } else {
            file::Path filename = file::DataLocation::parse("output", path, &combined.map);
            combined.map["filename"] = filename.remove_extension();
        }
        
    }
        
    if(filename.empty()
              //&& task == TRexTask_t::track
       )
    {
        const auto _source = source.empty()
            ? combined.map.at("source").value<file::PathArray>()
            : source;
        
        auto name = combined.map.at("filename").value<file::Path>();
        auto filename = name.empty() ? file::Path() : file::DataLocation::parse("output", name, &combined.map);
        if(not filename.empty() && (filename.is_regular() || filename.add_extension("pv").is_regular()))
        {
            combined.map["filename"] = filename;
        } else {
            file::Path path = file::find_basename(_source);
            print("found basename = ", path);
            if(task == TRexTask_t::track) {
                filename = file::DataLocation::parse("input", path, &combined.map);
                if(filename.is_regular() || filename.add_extension("pv").is_regular()) 
                {
                    
                } else {
                    filename = file::DataLocation::parse("output", path, &combined.map);
                    
                }
            } else {
                filename = file::DataLocation::parse("output", path, &combined.map);
            }
            
            if(filename.has_extension() && filename.extension() != "pv")
                filename = filename.remove_extension();
            
            combined.map["filename"] = filename;
        }
    }
    
    /// ----------------------------------------------------------------
    ///  9. **exclude** `output_dir` + `output_prefix` since we have now
    ///     locked in the `filename` + `source` parameters
    /// ----------------------------------------------------------------
    exclude = exclude + std::array{
        "output_dir",
        "output_prefix"
    };
    
    /// -----------------------------------------------
    /// 10. load settings from the .pv if tracking mode
    /// -----------------------------------------------
    auto exclude_from_default = exclude;
    if(task == TRexTask_t::track) {
        auto path = SETTING(filename).value<file::Path>();
        if(not path.has_extension() || path.extension() != "pv")
            path = path.add_extension("pv");
        if(path.is_regular()) {
            try {
                G g(path.str());
                sprite::Map tmp;
                pv::File f(path, pv::FileMode::READ);
                if(f.header().version < pv::Version::V_10) {
                    tmp["detect_type"] = detect::ObjectDetectionType::background_subtraction;
                    type = detect::ObjectDetectionType::background_subtraction;
                }
                
                const auto& meta = f.header().metadata;
                sprite::parse_values(sprite::MapSource{ path }, tmp, meta, & combined.map, exclude.toVector());
                
                exclude_from_default += tmp.keys();
                
                for(auto &key : tmp.keys())
                    tmp.at(key).get().copy_to(&combined.map);
                
                if(not tmp.has("detect_type")
                   && not tmp.has("detection_type"))
                {
                    combined.map["detect_type"] = detect::ObjectDetectionType::background_subtraction;
                    type = detect::ObjectDetectionType::background_subtraction;
                }
                
                if (not combined.map.has("meta_real_width")
                    || combined.map.at("meta_real_width").value<float>() == 0)
                {
                    combined.map["meta_real_width"] = infer_meta_real_width_from(f, &combined.map);
                }

            } catch(const std::exception& ex) {
                FormatWarning("Failed to execute settings stored inside ", path,": ",ex.what());
            }
        }
    }
    
    /// ---------------------------
    /// 11. defaults based on task
    /// ---------------------------
    if(type != track::detect::ObjectDetectionType::background_subtraction)
    {
        static const sprite::Map values {
            "track_threshold", 0,
            "track_posture_threshold", 0,
            "track_background_subtraction", false,
            "calculate_posture", false,
            "meta_encoding", grab::default_config::meta_encoding_t::r3g3b2,
            "track_do_history_split", false
        };
        
        for(auto &key : values.keys()) {
            if(contains(exclude_from_default.toVector(), key)) {
                print("Not setting default value ", key);
                continue;
            }
            values.at(key).get().copy_to(&combined.map);
            //all.emplace_back(key); // < not technically "custom"
        }
    } else {
        static const sprite::Map values {
            "track_threshold", 9,
            "track_posture_threshold", 9,
            "track_background_subtraction", true,
            "calculate_posture", true,
            "meta_encoding", grab::default_config::meta_encoding_t::gray,
            "track_do_history_split", true
        };
        
        for(auto &key : values.keys()) {
            if(contains(exclude_from_default.toVector(), key)) {
                print("// Not setting default value ", key);
                continue;
            }
            values.at(key).get().copy_to(&combined.map);
            //all.emplace_back(key); // < not technically "custom"
        }
    }
    
    /// --------------------------------------------
    /// 12. load the video settings (if they exist):
    /// --------------------------------------------
    auto settings_file = file::DataLocation::parse("settings", {},  &combined.map);
    if(settings_file.exists()) {
        try {
            sprite::Map map;
            map.set_print_by_default(false);

            auto rejected = GlobalSettings::load_from_file(deprecations(), settings_file.str(), AccessLevelType::STARTUP, exclude.toVector(), &map, &combined.map);
            warn_deprecated(settings_file, rejected);
            
            //auto before = combined.map.print_by_default();
            //combined.map.set_print_by_default(false);

            G g(settings_file.str());
            for(auto &key : map.keys()) {
                map.at(key).get().copy_to(&combined.map);
            }
            //combined.map.set_print_by_default(before);
            //exclude_from_pv = exclude_from_pv + map.keys();
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",settings_file,": ", ex.what());
        }
        
    } else if(not settings_file.empty()) {
        FormatError("Settings file ", settings_file, " was not found.");
    }
    
    /// -------------------------------------
    /// 13. optionally load the map parameter
    /// -------------------------------------
    print(source_map.at("track_background_subtraction"));
    print(source_map.at("track_threshold"));
    
    for(auto& key : source_map.keys()) {
        if(contains(exclude.toVector(), key))
        {
            if (combined.map.has(key)
                && combined.map.at(key) == source_map.at(key))
            {
                /// can be ignored / no print-out since it would
                /// not change anything
                continue;
            }
            print("// Not allowed to copy ", key, " from source map.");
            continue;
        }
        source_map.at(key).get().copy_to(&combined.map);
    }

    if (not combined.map.has("meta_real_width")
        || combined.map.at("meta_real_width").value<float>() == 0)
    {
        combined.map["meta_real_width"] = 1000.f;
    }
    
    if (combined.map.has("cm_per_pixel")
        && combined.map.at("cm_per_pixel").value<Settings::cm_per_pixel_t>() == 0)
    {
        if (combined.map.has("source")
            && combined.map.at("source").value<file::PathArray>() == file::PathArray("webcam"))
        {
            combined.map["cm_per_pixel"] = Settings::cm_per_pixel_t(0.01);
        } else
            combined.map["cm_per_pixel"] = infer_cm_per_pixel(&combined.map);
    }
    
    combined.map["detect_type"] = type;
    
    /// --------------------------------------
    DebugHeader("// FINAL CONFIG");

    for(auto &key : combined.map.keys()) {
        try {
            if(GlobalSettings::access_level(key) < AccessLevelType::SYSTEM
               && (not GlobalSettings::has(key)
                   || GlobalSettings::map().at(key).get() != combined.map.at(key).get())
               )
            {
                //if(not contains(copy.toVector(), key))
                {
                    //print("Updating ",combined.map.at(key));
                    combined.map.at(key).get().copy_to(&GlobalSettings::map());
                }
                /*else {
                 print("Would be updating ",combined.map.at(key), " but is forbidden.");
                 }*/
            }
        } catch(const std::exception& ex) {
            FormatExcept("Cannot parse setting ", key, " and copy it to GlobalSettings: ", ex.what());
        }
    }
    DebugHeader("// -------------");
    
    print(SETTING(filename));
    print(SETTING(output_dir));
    print(SETTING(output_prefix));
    print(SETTING(source));
    print(SETTING(detect_model));
    print(SETTING(region_model));
    print(SETTING(meta_source_path));
    print(SETTING(track_background_subtraction));
    print(SETTING(cm_per_pixel));
    print(SETTING(settings_file));
    print(SETTING(track_threshold));
    print(SETTING(calculate_posture));
    print(SETTING(meta_encoding));
    print(SETTING(track_do_history_split));
    print(SETTING(gpu_torch_device));
    print("TRexTask = ", task);
}

void write_config(bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix) {
    auto filename = file::DataLocation::parse(suffix == "backup" ? "backup_settings" : "output_settings");
    auto text = default_config::generate_delta_config().to_settings();
    
    if(filename.exists() && !overwrite) {
        if(queue) {
            queue->enqueue([text, filename](auto, gui::DrawStructure& graph){
                graph.dialog([str = text, filename](gui::Dialog::Result r) {
                    if(r == gui::Dialog::OKAY) {
                        if(!filename.remove_filename().exists())
                            filename.remove_filename().create_folder();
                        
                        FILE *f = fopen(filename.str().c_str(), "wb");
                        if(f) {
                            print("Overwriting file ",filename.str(),".");
                            fwrite(str.data(), 1, str.length(), f);
                            fclose(f);
                        } else {
                            FormatExcept("Dont have write permissions for file ",filename.str(),".");
                        }
                    }
                    
                }, "Overwrite file <i>"+filename/*.filename()*/.str()+"</i> ?", "Write configuration", "Yes", "No");
            });
            
        } else
            print("Settings file ",filename.str()," already exists. To overwrite, please add the keyword 'force'.");
        
    } else {
        if(!filename.remove_filename().exists())
            filename.remove_filename().create_folder();
        
        FILE *f = fopen(filename.str().c_str(), "wb");
        if(f) {
            fwrite(text.data(), 1, text.length(), f);
            fclose(f);
            DebugCallback("Saved ", filename, ".");
        } else {
            FormatExcept("Cannot write file ",filename,".");
        }
    }
}

float infer_cm_per_pixel(const sprite::Map* map) {
    if(map == nullptr)
        map = &GlobalSettings::map();

    // setting cm_per_pixel after average has been generated (and offsets have been set)
    if(not map->has("cm_per_pixel")
       || map->at("cm_per_pixel").value<Settings::cm_per_pixel_t>() == 0)
    {
        auto w = map->at("meta_real_width").value<float>();
        if(w <= 0) {
            return 1;
        }
        
        return 1 / max(1.0, w * 0.05);
        //return w / float(average().cols);
    }

    return map->at("cm_per_pixel").value<Settings::cm_per_pixel_t>();
}

float infer_meta_real_width_from(const pv::File &file, const sprite::Map* map) {
    if(not map)
        map = &GlobalSettings::map();
    
    if(not map->has("meta_real_width")
        || map->at("meta_real_width").value<float>() == 0)
    {
        if(file.header().meta_real_width <= 0) {
            FormatWarning("This video does not set `meta_real_width`. Please set this value during conversion (see https://trex.run/docs/parameters_trex.html#meta_real_width for details). Defaulting to 30cm.");
            return float(30.0);
        } else {
            if(not map->has("meta_real_width")
                || map->at("meta_real_width").value<float>() == 0)
            {
                return file.header().meta_real_width;
            }
        }
    }
    
    return map->at("meta_real_width").value<float>();
}

}
