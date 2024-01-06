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

namespace settings {

/**
 * When calling "load" or "open", what we expect is:
 *  1. Settings are reset to default
 *  2. load default.settings
 *  3. load command-line (except exclude) => exclude?
 *      3.1. if task == convert, load specific settings for the model type
 *           but exclude everything from the command-line
 *  4. IF `source` / `filename` is empty, generate them based on task
 *  5. load `source` + `filename` .settings
 *  6. load sprite::Map passed to function
 *
 * <=> IMPORTANT <=>
 *  1. the `source` + `filename` .settings files are not allowed
 *     to contain any of the following parameters:
 *       `{ "source", "filename", "output dir", "output prefix" }`
 *     as to not confuse the entire process of loading parameters.
 *  2. the same goes for the sprite::Map, since that would be
 *     confusing as well.
 *
 * So for opening directly via command-line we need to:
 *  1. command-line in sprite::Map
 *  2. call
 *
 * For opening tracking after switching from ConvertScene:
 *  1. pass `source` and `filename`
 *  2. call
 *
 * For opening converting after settings:
 *  1. put all settings in sprite::Map
 *  2. pass source + filename
 *  3. call
 *
 * => output is: `source` set to MP4 file, `filename` set
 *    to (prospective) pv file. all settings loaded as far
 *    as they are available.
 */

void load(file::PathArray source, 
          file::Path filename,
          TRexTask task,
          ExtendableVector exclude_parameters,
          const cmn::sprite::Map& source_map)
{
    DebugHeader("Reloading settings");
    
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
    
    /// -----------------------------------------
    /// 2. load default.settings from app folder:
    /// -----------------------------------------
    auto default_path = file::DataLocation::parse("default.settings", {}, &combined.map);
    if (default_path.exists()) {
        try {
            warn_deprecated(default_path, GlobalSettings::load_from_file(deprecations(), default_path.str(), AccessLevelType::STARTUP, exclude_parameters, &combined.map));
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",default_path,": ", ex.what() );
        }
    }
    
    /// ---------------------------------------------------
    /// 3. get cmd arguments and overwrite stuff with them:
    /// ---------------------------------------------------
    auto& cmd = CommandLine::instance();
    combined.map.set_print_by_default(true);
    
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_focus_group"].get().set_do_print(false);
    GlobalSettings::map()["gui_source_video_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_displayed_frame"].get().set_do_print(false);
    
    auto copy = exclude_parameters + std::array{ "filename", "source" };
    auto all = copy + extract_keys(cmd.settings_keys());
    cmd.load_settings(nullptr, &combined.map, copy.toVector());
    
    /// ---------------------------
    /// 3.1. defaults based on task
    /// ---------------------------
    if(task == TRexTask_t::convert) {
        static const sprite::Map values {
            "track_threshold", 0,
            "track_posture_threshold", 0,
            "track_background_subtraction", false,
            "calculate_posture", false,
            "meta_encoding", grab::default_config::meta_encoding_t::r3g3b2,
            "track_do_history_split", false
        };
        
        auto exclude = exclude_parameters + extract_keys( cmd.settings_keys() );
        combined.map.set_print_by_default(true);
        for(auto &key : values.keys()) {
            if(contains(exclude.toVector(), key))
                continue;
            values.at(key).get().copy_to(&combined.map);
            //all.emplace_back(key); // < not technically "custom"
        }
    }
    
    /// ----------------------------------------
    /// 4. set the source / filename properties:
    /// ----------------------------------------
    if(not filename.empty())
    {
        combined.map["filename"] = filename;
    }
    
    if(not source.empty())
    {
        combined.map["source"] = source;
        combined.map["meta_source_path"] = source.source();
    }
    
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
    
    /// add additional prefixes to exclude, only used for name resolution
    copy = copy + std::array{
        "output_dir",
        "output_prefix"
    };
    
    /// 5. load the video settings (if they exist):
    auto settings_file = file::DataLocation::parse("settings", {},  &combined.map);
    if(settings_file.exists()) {
        try {
            sprite::Map map;
            auto rejected = GlobalSettings::load_from_file(deprecations(), settings_file.str(), AccessLevelType::STARTUP, copy.toVector(), &map, &combined.map);
            warn_deprecated(settings_file, rejected);
            
            for(auto &key : map.keys()) {
                map.at(key).get().copy_to(&combined.map);
            }
            all = all + map.keys();
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",settings_file,": ", ex.what());
        }
        
    } else if(not settings_file.empty()) {
        FormatError("Settings file ", settings_file, " was not found.");
    }
    
    /// --------------------------------------
    /// 6. copy potential sprite map contents:
    /// --------------------------------------
    print(source_map.at("track_background_subtraction"));
    print(source_map.at("track_threshold"));
    
    if(task == TRexTask_t::track) {
        auto path = SETTING(filename).value<file::Path>();
        if(not path.has_extension() || path.extension() != "pv")
            path = path.add_extension("pv");
        if(path.is_regular()) {
            try {
                pv::File f(path, pv::FileMode::READ);
                const auto& meta = f.header().metadata;
                combined.map.set_print_by_default(true);
                
                sprite::parse_values(sprite::MapSource{ path }, combined.map, meta, & combined, all.toVector());

            } catch(const std::exception& ex) {
                FormatWarning("Failed to execute settings stored inside ", path,": ",ex.what());
            }
        }
    }
    
    for(auto& key : source_map.keys()) {
        if(contains(copy.toVector(), key))
        {
            print("Not allowed to copy ", key, " from source map.");
            continue;
        }
        source_map.at(key).get().copy_to(&combined.map);
    }
    
    if(combined.map.at("cm_per_pixel").value<Settings::cm_per_pixel_t>() == 0) {
        if(combined.map.has("source")
           && combined.map.at("source").value<file::PathArray>() == file::PathArray("webcam"))
        {
            combined.map["cm_per_pixel"] = Settings::cm_per_pixel_t(0.01);
        }
    }
    
    for(auto &key : combined.map.keys()) {
        try {
            if(GlobalSettings::access_level(key) < AccessLevelType::SYSTEM
               && (not GlobalSettings::has(key)
                   || GlobalSettings::map().at(key).get() != combined.map.at(key).get())
               )
            {
                //if(not contains(copy.toVector(), key))
                {
                    print("Updating ",combined.map.at(key));
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
    
    print(SETTING(filename));
    print(SETTING(output_dir));
    print(SETTING(output_prefix));
    print(SETTING(source));
    print(SETTING(model));
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

}
