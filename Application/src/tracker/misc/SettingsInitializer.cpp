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
#include <video/VideoSource.h>

using namespace track;
using namespace default_config;



namespace cmn::settings {

void initialize_filename_for_tracking() {
    file::Path path;
    
    if(not SETTING(filename).value<file::Path>().empty()) {
        path = SETTING(filename).value<file::Path>();
    } else {
        path = file::Path(settings::find_output_name(GlobalSettings::map()));
    }
    
    if(not path.has_extension()
       || path.extension() != "pv")
    {
        path = path.add_extension("pv");
    }
    
    if(not path.is_absolute())
        path = file::DataLocation::parse("output", path);
    
    if(path.is_regular()) {
        SETTING(filename) = path.remove_extension();
        
    } else if(auto source = SETTING(source).value<file::PathArray>();
              source.size() == 1
              && ((source.get_paths().front().is_regular()
                   && source.get_paths().front().has_extension("pv"))
                || source.get_paths().front().add_extension("pv").is_regular())
              )
    {
        auto path = source.get_paths().front();
        if(path.has_extension("pv"))
            path = path.remove_extension();
        
        SETTING(filename) = file::Path(path);
        
    } else {
        throw U_EXCEPTION("Cannot find the file ", path, " and nothing in ", SETTING(source).value<file::PathArray>()," seems to be a .pv file.");
    }
}

std::unordered_set<std::string_view>
set_defaults_for(detect::ObjectDetectionType_t detect_type,
                 cmn::sprite::Map& output,
                 ExtendableVector exclude)
{
    std::unordered_set<std::string_view> changed_keys;
        
    if(detect_type == detect::ObjectDetectionType::none)
        detect_type = detect::ObjectDetectionType::yolo;
    
    /// set or default the *detect_type*
    output["detect_type"] = detect_type;
    
    /// copy the values from a defaults map to the
    /// final output map:
    const auto apply_values = [&](const sprite::Map& values) {
        for(auto &key : values.keys()) {
            if(not contains(exclude.toVector(), key)) {
                if(not output.has(key)
                   || values.at(key).get() != output.at(key).get())
                {
                    changed_keys.insert(key);
                    values.at(key).get().copy_to(output);
                }
            }
        }
    };
    
    if(detect_type == track::detect::ObjectDetectionType::background_subtraction) {
        auto cm_per_pixel = SETTING(cm_per_pixel).value<Settings::cm_per_pixel_t>();
        if(cm_per_pixel <= 0) {
            cm_per_pixel = 1;
        }
        
        const sprite::Map values {
            "track_threshold", 15,
            "track_posture_threshold", 15,
            "track_background_subtraction", true,
            "calculate_posture", true,
            //"track_size_filter", SizeFilters(),
            "detect_size_filter", SizeFilters({Ranged(10 * SQR(cm_per_pixel),
                                                      100000 * SQR(cm_per_pixel))}),
            //"meta_encoding", meta_encoding_t::rgb8,
            //"track_do_history_split", true,
            "detect_classes", cmn::blob::MaybeObjectClass_t{},
            "individual_image_normalization", individual_image_normalization_t::posture,
            "blob_split_algorithm", blob_split_algorithm_t::threshold,
            "track_max_reassign_time", 0.5f,
            "detect_skeleton", blob::Pose::Skeleton(),
            "detect_format", track::detect::ObjectDetectionFormat::none
        };
        
        apply_values(values);
        
    } else {
        static const sprite::Map values {
            "track_threshold", 0,
            "track_posture_threshold", 0,
            "track_background_subtraction", false,
            "detect_size_filter", SizeFilters(),
            //"track_size_filter", SizeFilters(),
            "calculate_posture", true,
            "outline_resample", 1.f,
            "outline_approximate", uchar(3),
            //"track_do_history_split", true,
            "individual_image_normalization", individual_image_normalization_t::posture,
            "detect_model", file::Path(detect::yolo::default_model()),
            "blob_split_algorithm", blob_split_algorithm_t::none,
            "track_max_reassign_time", 1.f,
            "detect_format", track::detect::ObjectDetectionFormat::none,
            "detect_skeleton", blob::Pose::Skeleton("human", {
                {0, 1, "Nose to Left Eye"},
                {0, 2, "Nose to Right Eye"},
                {1, 3, "Left Eye to Ear"},
                {2, 4, "Right Eye to Ear"},
                {5, 6, "Left to Right Shoulder"},
                {5, 7, "Left Upper Arm"},
                {7, 9, "Left Forearm"},
                {6, 8, "Right Upper Arm"},
                {8, 10, "Right Forearm"},
                {5, 11, "Left Shoulder to Hip"},
                {6, 12, "Right Shoulder to Hip"},
                {11, 12, "Left to Right Hip"},
                {11, 13, "Left Thigh"},
                {13, 15, "Left Shin"},
                {12, 14, "Right Thigh"},
                {14, 16, "Right Shin"}
            })
        };
        
        apply_values(values);
    }
    
    return changed_keys;
}

SettingsMaps reset(const cmn::sprite::Map& extra_map, cmn::sprite::Map* output) {
    if(not output)
        output = &GlobalSettings::map();
    
    SettingsMaps combined;
    const auto set_combined_access_level =
        [&combined](auto& name, AccessLevel level) {
            combined.access_levels[name] = level;
        };
    
    combined.map.set_print_by_default(false);
    
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    ::default_config::get(combined.map, combined.docs, set_combined_access_level);
    
    if(not combined.map.has("detect_type")
       || combined.map.at("detect_type").value<detect::ObjectDetectionType_t>() == detect::ObjectDetectionType::none)
    {
        combined.map["detect_type"] = detect::ObjectDetectionType::yolo;
    }
    
    set_defaults_for(combined.map.at("detect_type"), combined.map, {});
    
    for(auto &key : extra_map.keys()) {
        try {
            /// don't allow modification of system variables
            if(GlobalSettings::access_level(key) >= AccessLevelType::SYSTEM) {
                continue;
            }
            
            extra_map.at(key).get().copy_to(combined.map);
            
        } catch(const std::exception& ex) {
            FormatExcept("Exception while copying ", key, " to combined map: ", ex.what());
        }
    }
    
    if(output != &combined.map) {
        for(auto &key : combined.map.keys()) {
            try {
                if(GlobalSettings::access_level(key) < AccessLevelType::SYSTEM
                   && (not output->has(key)
                       || output->at(key).get() != combined.map.at(key).get())
                   )
                {
                    /// special case convenience function for filename
                    /// since we dont need to set it if its just the *default*
                    if(key == "filename"
                       && (combined.map.at(key).value<file::Path>() == find_output_name(combined.map)
                           || (not combined.map.at(key).value<file::Path>().is_absolute()
                               && combined.map.at(key).value<file::Path>() == file::find_basename(combined.map.at("source").value<file::PathArray>()))))
                    {
                        SETTING(filename) = file::Path();
                        continue;
                    }
                    
                    /// same goes for *output_dir*
                    if(key == "output_dir"
                       && combined.map.at(key).value<file::Path>() == file::find_parent( combined.map.at("source").value<file::PathArray>()))
                    {
                        SETTING(output_dir) = file::Path();
                        continue;
                    }
                    
                    /// copy to destination map
                    if(not is_in(key, "gui_interface_scale"))
                        combined.map.at(key).get().copy_to(*output);
                }
            } catch(const std::exception& ex) {
                FormatExcept("Cannot parse setting ", key, " and copy it to output map: ", ex.what());
            }
        }
    }
    
    return combined;
}


struct G {
    std::string s;
    bool quiet;
    G(const std::string& name, bool quiet) : s(name), quiet(quiet) {
        if(not quiet)
            DebugHeader("// LOADING FROM ", s);
    }
    ~G() {
        //DebugHeader("// LOADED ", s);
        if(not quiet)
            Print("");
    }
};

void LoadContext::init() {
    combined.map.set_print_by_default(false);
    
    auto set_combined_access_level = [&](auto& name, AccessLevel level) {
        combined.access_levels[name] = level;
    };
    
    /// ---------------------------------------------
    /// 1. setting default values, saved in combined:
    /// ---------------------------------------------
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    ::default_config::get(combined.map, combined.docs, set_combined_access_level);
    
    /// ---------------------------
    /// 2. exclude SYSTEM variables
    /// ---------------------------
    /// as well as other defaults
    
    auto exclude = exclude_parameters + default_excludes + system_variables;
    if(not quiet)
        Print("Excluding from command-line and default.settings: ", exclude);
    
    /// -----------------------------------------
    /// 3. load default.settings from app folder:
    /// -----------------------------------------
    if (auto default_path = file::DataLocation::parse("default.settings", {}, &combined.map);
        default_path.exists())
    {
        try {
            auto str = utils::read_file(default_path.str());
            if(not str.empty()) {
                G g(default_path.str(), quiet);
                auto rejected = GlobalSettings::load_from_string(sprite::MapSource{default_path.str()}, deprecations(), combined.map, str, AccessLevelType::STARTUP, false, exclude, nullptr);
                warn_deprecated(default_path, rejected);
            }
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",default_path,": ", ex.what() );
        }
    }
    
    GlobalSettings::set_current_defaults(combined.map);
    
    /// ---------------------------------------------------
    /// 4. get cmd arguments and overwrite stuff with them:
    /// ---------------------------------------------------
    /// excluding filename and source + other defaults
    auto& cmd = CommandLine::instance();
    
    cmd.load_settings(nullptr, &combined.map, exclude.toVector());
    if(cmd.settings_keys().contains("wd")) {
        combined.map["wd"] = file::Path(cmd.settings_keys().at("wd"));
        set_config_if_different("wd", combined.map);
    }
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
}

void LoadContext::init_filename() {
    if(filename.has_extension("pv"))
        filename = filename.remove_extension();
    
    if(not filename.empty()) {
        /// we have gotten a filename... so set
        /// it in the target map
        if (filename.remove_filename().exists()
            && filename.is_absolute())
        {
            auto output_dir = filename.remove_filename();
            
            /// check whether the directory contains the *output_prefix*
            /// if so then we need to remove it:
            if(auto output_prefix = SETTING(output_prefix).value<std::string>();
               not output_prefix.empty()
               && output_dir.filename() == output_prefix)
            {
                output_dir = output_dir.remove_filename();
            }
            
            combined.map["output_dir"] = output_dir;
            set_config_if_different("output_dir", combined.map);
        }
        
        combined.map["filename"] = filename;
        set_config_if_different("filename", combined.map);
    }
    
    if(not source.empty()) {
        /// we did get a source as function parameter,
        /// which means this is to be prioritized. set
        /// it in the target map
        if(source.get_paths().size() == 1) {
            /// we only have one path given - could be
            /// webcam or a .pv file?
            if(source.get_paths().front() != "webcam"
               && not source.get_paths().front().exists())
            {
                auto path = source.get_paths().front().add_extension("pv");
                if(path.exists()) {
                    source = file::PathArray(path);
                    
                } else if(path = file::DataLocation::parse("output", path, &combined.map);
                        path.exists())
                {
                    source = file::PathArray(path);
                }
            }
        }
        
        combined.map["source"] = source;
        set_config_if_different("source", combined.map);
        
        if(not contains(exclude.toVector(), "meta_source_path")) {
            combined.map["meta_source_path"] = source.source();
            set_config_if_different("meta_source_path", combined.map);
        }
    }
    
    /// ---------------------------------------------------------------------
    /// 7. set the `output_dir` / `output_prefix` properties from parameters:
    /// ---------------------------------------------------------------------
    if(source_map.has("output_dir")) {
        set_config_if_different("output_dir", source_map);
    }
    if(source_map.has("output_prefix")) {
        set_config_if_different("output_prefix", source_map);
    }
    
    if(type != track::detect::ObjectDetectionType::none) {
        combined.map["detect_type"] = type;
        set_config_if_different("detect_type", combined.map);
    }
    
    if(not quiet)
        combined.map.set_print_by_default(true);
}

void LoadContext::fix_empty_source() {
    /// -----------------------------------------------------
    /// 8. if `source` or `filename` are empty, generate them
    /// -----------------------------------------------------
    if(source.empty()
       && task == TRexTask_t::convert)
    {
        /// ------------------
        G g{"Source is empty", quiet};
        /// ------------------
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
            if(not name.empty()) {
                file::Path filename = file::DataLocation::parse("output", name, &combined.map);
                if(filename.has_extension("pv"))
                    filename = filename.remove_extension();
                combined.map["filename"] = filename;
                set_config_if_different("filename", combined.map);
            }
            
        } else if(not path.empty()) {
            file::Path filename = file::DataLocation::parse("output", path, &combined.map);
            if(filename.has_extension("pv"))
                filename = filename.remove_extension();
            combined.map["filename"] = filename;
            set_config_if_different("filename", combined.map);
        }
    }
}

void LoadContext::fix_empty_filename() {
    if(filename.empty()) {
        /// -------------------------
        G g{"Fixing empty filename", quiet};
        /// -------------------------
        {
            auto name = combined.map.at("filename").value<file::Path>();
            filename = name.empty()
                            ? file::Path()
                            : file::DataLocation::parse("output", name, &combined.map);
        }
        
        if(filename.has_extension("pv"))
            filename = filename.remove_extension();
        
        if(not filename.empty()
           && filename.add_extension("pv").is_regular())
        {
            /// A PV file of that name exists (with .pv added)
            combined.map["filename"] = filename;
            set_config_if_different("filename", combined.map);
            
        } else {
            const auto _source = source.empty()
                ? combined.map.at("source").value<file::PathArray>()
                : source;
            
            file::Path path = file::find_basename(_source);
            if(task == TRexTask_t::track) {
                if(not path.empty()) {
                    filename = file::DataLocation::parse("input", path, &combined.map);
                    
                    if(filename.is_regular() || filename.add_extension("pv").is_regular())
                    { } else {
                        filename = file::DataLocation::parse("output", path, &combined.map);
                    }
                    
                } else
                    filename = {};
                
            } else if(not path.empty()) {
                filename = file::DataLocation::parse("output", path, &combined.map);
            } else {
                filename = {};
            }
            
            if(filename.has_extension("pv"))
                filename = filename.remove_extension();
            
            combined.map["filename"] = filename;
            set_config_if_different("filename", combined.map);
        }
    }
    
}

void LoadContext::reset_default_filenames() {
    if(not combined.map["filename"].value<file::Path>().empty()) {
        /// -----------------------
        /// In case the filename has been set, we could be in
        /// a situation where it needs to be emptied, since its
        /// just the default for a given video anyway and we
        /// dont want to later save it again as a fixed filename.
        /// always keep defaults as empty
        //G g{source.source(), quiet};
        /// -----------------------
        const auto _source = source.empty()
            ? combined.map.at("source").value<file::PathArray>()
            : source;
        auto default_path = find_output_name(combined.map, {}, {}, false);
        
        auto path = combined.map["filename"].value<file::Path>();
        if(path == default_path) {
            combined.map["filename"] = file::Path();
            set_config_if_different("filename", combined.map);
        } else if(path.is_absolute()) {
            combined.map["filename"] = file::Path(path.filename());
            set_config_if_different("filename", combined.map);
        } else {
#ifndef NDEBUG
            if(not quiet)
                Print("Not absolute: ", path);
#endif
            combined.map["filename"] = file::Path(path);
            set_config_if_different("filename", combined.map);
        }
    }
    
    if(not combined.map["filename"].value<file::Path>().empty()) {
        auto path = combined.map["filename"].value<file::Path>();
        if(path.is_absolute())
            path = path.filename();
        combined.map["filename"] = path;
        set_config_if_different("filename", combined.map);
    }
}

void LoadContext::load_settings_from_source() {
    /// -----------------------------------------------
    /// 10. load settings from the .pv if tracking mode
    /// -----------------------------------------------
    exclude_from_default = exclude;
    //bool is_source_a_pv_file = false;
    //if(task == TRexTask_t::track)
    {
        file::Path path;
        if(source.size() == 1) {
            path = source.get_paths().front();
            if(not path.has_extension("pv")) {
                path = path.add_extension("pv");
            }
            
            if(path.is_regular()) {
                //is_source_a_pv_file = true;
            } else {
                path = "";
            }
        }
        
        if(path.empty())
            path = find_output_name(combined.map, source, filename);
        
        //auto path = combined.map.at("filename").value<file::Path>();
        if(not path.has_extension() || path.extension() != "pv")
            path = path.add_extension("pv");
        
        if(path.is_regular()) {
            auto settings_file = file::DataLocation::parse("settings", {},  &combined.map);
            if(not settings_file.exists()
               && source_map.empty())
            {
                G g(path.str(), quiet);
                try {
                    sprite::Map tmp;
                    auto f = pv::File::Read(path);
                    if(f.header().version < pv::Version::V_10) {
                        /// we need to have a `detect_type` in order to set the
                        /// correct task-defaults in the next step.
                        ///
                        /// since there was no other `detect_type` before
                        /// **V_10** and there also was no type parameter to
                        /// query, we set bg subtraction:
                        tmp["detect_type"] = detect::ObjectDetectionType::background_subtraction;
                        tmp["meta_encoding"] = f.header().encoding;
                    }
                    
                    if(f.header().metadata.has_value()) {
                        const auto& meta = f.header().metadata.value();
                        sprite::parse_values(sprite::MapSource{ path }, tmp, meta, & combined.map,
                                             changed_model_manually
                                             ? (exclude + exclude_from_external).toVector()
                                             : exclude.toVector(),
                                             default_config::deprecations());
                    }
                    
                    exclude_from_default += tmp.keys();
                    //Print("// pv file keys = ", tmp.keys());
                    
                    for(auto &key : tmp.keys())
                        set_config_if_different(key, tmp, true);
                    
                    /// if we are running a tracking task, we need to use the stuff from the pv file
                    /// if we are in fact running one. otherwise, just set it if we dont have a detect_type
                    /// set anywhere yet.
                    if(type == track::detect::ObjectDetectionType::none
                        || task == TRexTask_t::track)
                    {
                        combined.map["detect_type"] = type = tmp.at("detect_type").value<detect::ObjectDetectionType_t>();
                        set_config_if_different("detect_type", combined.map);
                    }
                    
                    if (not combined.map.has("meta_real_width")
                        || combined.map.at("meta_real_width").value<Float2_t>() == 0)
                    {
                        combined.map["meta_real_width"] = infer_meta_real_width_from(f, &combined.map);
                        set_config_if_different("meta_real_width", combined.map);
                    }
                    
                } catch(const std::exception& ex) {
                    FormatWarning("Failed to execute settings stored inside ", path,": ",ex.what());
                }
                
            } else {
                G g(path.str(), quiet);
                try {
                    sprite::Map tmp;
                    auto f = pv::File::Read(path);
                    if(f.header().version < pv::Version::V_10) {
                        /// we need to have a `detect_type` in order to set the
                        /// correct task-defaults in the next step.
                        ///
                        /// since there was no other `detect_type` before
                        /// **V_10** and there also was no type parameter to
                        /// query, we set bg subtraction:
                        tmp["detect_type"] = detect::ObjectDetectionType::background_subtraction;
                        tmp["meta_encoding"] = f.header().encoding;
                    }
                    
                    if(f.header().metadata.has_value()) {
                        const auto& meta = f.header().metadata.value();
                        sprite::parse_values(sprite::MapSource{ path },
                                             tmp, meta, &combined.map,
                                             changed_model_manually
                                                 ? (exclude + exclude_from_external).toVector()
                                                 : exclude.toVector(),
                                             default_config::deprecations());
                    }
                    
                    // List of fields to check and their default handlers
                    const std::vector<std::string> fields_to_check = {
                        "meta_encoding",
                        "meta_source_path",
                        "meta_video_size",
                        "meta_real_width",
                        "frame_rate",
                        "cm_per_pixel",
                        "detect_type"
                    };
                    
                    if(not quiet)
                        Print("// Not loading all settings from ", path, " because the settings file ", settings_file, " exists. Checking only ", fields_to_check);

                    // Functions to compute default values when not available in tmp
                    const std::unordered_map<std::string, std::function<const sprite::PropertyType*(sprite::Map&)>> compute_defaults = {
                        {"meta_video_size", [&](auto& map) {
                            map["meta_video_size"] = Size2(f.size());
                            return &map.at("meta_video_size").get();
                        }},
                        {"meta_real_width", [&](auto& map) {
                            map["meta_real_width"] = infer_meta_real_width_from(f, &combined.map);
                            return &map.at("meta_real_width").get();
                        }},
                        {"cm_per_pixel", [&](auto& map) -> const sprite::PropertyType* {
                            if(tmp.has("cm_per_pixel")) {
                                tmp.at("cm_per_pixel").get().copy_to(map);
                                return &map.at("cm_per_pixel").get();
                            } else {
                                if(not quiet)
                                    FormatWarning("Source ", path, " does not have `cm_per_pixel`.");
                            }
                            return nullptr;
                        }},
                        {"detect_type", [&](auto& map) -> const sprite::PropertyType* {
                            if (tmp.has("detect_type")
                                && (task == TRexTask_t::track || type == track::detect::ObjectDetectionType::none))
                            {
                                tmp.at("detect_type").get().copy_to(map);
                                return &map.at("detect_type").get();
                            }
                            return nullptr;
                        }}
                    };

                    for (const auto& key : fields_to_check) {
                        // Skip if the key is present in source_map
                        if (source_map.has(key)) {
                            //Print("* Skip checking ", key, " since it is in the source_map with ", source_map.at(key).get());
                            continue;
                        }

                        // Get the default value for comparison
                        const auto default_value =
                                    current_defaults.has(key)
                                                ? &current_defaults.at(key).get()
                                                : nullptr;

                        // Determine if we need to update the value
                        // we need to update it if the default value has been changed,
                        // or we dont have one.
                        const bool needs_update = not default_value
                            || (default_value
                                && (not combined.map.has(key)
                                    || combined.map.at(key).get() == *default_value));
                        
                        // we also need to check if we have a compute function for it
                        if (compute_defaults.contains(key)) {
                            sprite::Map t;
                            auto p = compute_defaults.at(key)(t);
                            
                            if(p) {
                                // Compute and set the default value if a handler exists
                                //Print("* Checking ", key, " with computed defaults: ", *p);
                                set_config_if_different(key, t);
                                
                            } else {
                                /// otherwise we dont set anything, since its obviously
                                /// not wanted.
                            }
                            
                        } else if (needs_update) {
                            if (tmp.has(key)) {
                                // Copy value from tmp if available
                                //Print("* Checking ", key, ": ", tmp.at(key).get(), " combined=", combined.map.at(key));
                                tmp.at(key).get().copy_to(combined.map);
                                set_config_if_different(key, combined.map);
                            } else {
                                //Print("* Key not checked ", key);
                            }
                            // No action needed if neither tmp has the key nor a compute function is defined
                        } else {
                            //Print("* Skip checking ", key, " with default value ", default_value ? default_value->valueString() : "<null>", " and combined ", combined.map.has(key) ? combined.map.at(key).get().valueString() : "<null>");
                        }
                    }
                    
                } catch(const std::exception& ex) {
                    if(not quiet)
                        FormatWarning("Failed to execute settings stored inside ", path,": ",ex.what());
                }
            }
        }
        
    } //else {
        //Print("// Not loading settings from a potentially existing .pv file in tracking mode.");
   // }
}

void LoadContext::load_task_defaults() {
    /// ---------------------------
    /// 11. defaults based on task
    /// ---------------------------
    if(combined.map.has("detect_type")) {
        type = combined.map.at("detect_type").value<detect::ObjectDetectionType_t>();
    }
    
    {
        G g(type.toStr() + "-defaults", quiet);
        const sprite::Map values {
            [this](){
                sprite::Map values;
                set_defaults_for(type, values);
                return values;
            }()
        };
        
        for(auto &key : values.keys()) {
            /// we don't need detect type right now
            if(key == "detect_type")
                continue;
            
            if(not contains(exclude.toVector(), key))
                GlobalSettings::current_defaults(key, values);
            
            if(contains(exclude_from_default.toVector(), key)) {
                if(not quiet)
                    Print("// Not setting default value ", key);
                continue;
            }
            set_config_if_different(key, values);
            //all.emplace_back(key); // < not technically "custom"
        }
    }
}

void LoadContext::load_settings_file() {
    /// --------------------------------------------
    /// 12. load the video settings (if they exist):
    /// --------------------------------------------
    auto settings_file = file::DataLocation::parse("settings", {},  &combined.map);
    if(settings_file.exists())
    {
        G g(settings_file.str(), quiet);
        try {
            sprite::Map map;
            map.set_print_by_default(false);
            
            auto manual_exclude = changed_model_manually
                    ? (exclude + exclude_from_external).toVector()
                    : exclude.toVector();
            if(not quiet)
                Print("// Excluding ", manual_exclude, " from settings file.");

            auto rejected = GlobalSettings::load_from_file(deprecations(), settings_file.str(), AccessLevelType::STARTUP, manual_exclude, &map, &combined.map);
            
            warn_deprecated(settings_file, rejected);
            
            if(rejected.contains("meta_source_path")) {
                sprite::Map tmp;
                tmp["meta_source_path"] = std::string(rejected.at("meta_source_path"));
                if(not set_config_if_different("meta_source_path", tmp)
                   && not quiet)
                {
                    Print("// meta_source_path = ",no_quotes(tmp.at("meta_source_path").value<std::string>())," not set");
                }
                tmp.at("meta_source_path").get().copy_to(GlobalSettings::current_defaults_with_config());
            }
            
            //auto before = combined.map.print_by_default();
            //combined.map.set_print_by_default(false);

            for(auto &key : map.keys()) {
                if(not set_config_if_different(key, map)) {
                    //Print("// ", key, " was already set to ", no_quotes(map.at(key).get().valueString()));
                }
                
                map.at(key).get().copy_to(GlobalSettings::current_defaults_with_config());
            }
            //combined.map.set_print_by_default(before);
            //exclude_from_pv = exclude_from_pv + map.keys();
            
            /*for(auto &[key, value] : rejected) {
                if(not map.has(key) || not combined.map.has(key) || map.at(key) != combined.map.at(key)) {
                    // has been ignored
                    Print("// not setting ", key);
                }
            }*/
            
        } catch(const std::exception& ex) {
            FormatError("Failed to execute settings file ",settings_file,": ", ex.what());
        }
        
    } else if(not settings_file.empty()) {
        FormatError("Settings file ", settings_file, " was not found.");
    }
}

void LoadContext::load_gui_settings() {
    /// -------------------------------------
    /// 13. optionally load the map parameter
    /// -------------------------------------
    if(not source_map.empty()) {
        G g("GUI settings", quiet);
        //Print("gui settings contains: ", source_map.keys());
        
        for(auto& key : source_map.keys()) {
            if(contains(exclude.toVector(), key))
            {
                if (combined.map.has(key)
                    && combined.map.at(key) == source_map.at(key))
                {
                    /// can be ignored / no print-out since it would
                    /// not change anything
                    //Print("// Can ignore ", key, " from source map.");
                    continue;
                }
                if(not quiet)
                    Print("// Not allowed to copy ", key, " from source map.");
                continue;
            }
            
            set_config_if_different(key, source_map);
            //source_map.at(key).get().copy_to(combined.map);
            //source_map.at(key).get().copy_to(current_defaults);
        }
    }
    
}

void LoadContext::estimate_meta_variables() {
    if(source.empty()
       && (not combined.map.has("meta_video_size")
           || combined.map.at("meta_video_size").value<Size2>().empty()))
    {
        if(not quiet)
            Print("// Defaulting to meta_video_size of 1920x1080 for empty source.");
        combined.map["meta_video_size"] = Size2(1920_F, 1080_F);
        
    } else if(not combined.map.has("meta_video_size")
       || combined.map.at("meta_video_size").value<Size2>().empty())
    {
        G g{source.source(), quiet};
        try {
            if(auto source = combined.map.at("source").value<file::PathArray>();
               source == file::PathArray("webcam"))
            {
                combined.map["meta_video_size"] = Size2(1920_F, 1080_F);
                
            } else if(source.get_paths().size() == 1
                      && source.get_paths().front().has_extension("pv"))
            {
                /// we are looking at a .pv file as input
                if(not quiet)
                    Print("Should have already loaded this?");
                
                /// if this errors out, we should skip... so we let it through
                pv::File video(source.get_paths().front());
                if(video.size().empty())
                    throw InvalidArgumentException("Invalid video size read from ", video.filename());
                combined.map["meta_video_size"] = Size2(video.size());
                ///
                
            } else {
                VideoSource video(source);
                auto size = video.size();
                combined.map["meta_video_size"] = Size2(size);
            }
            
        } catch(...) {
            combined.map["meta_video_size"] = Size2(1920_F, 1080_F);
            if(not quiet)
                FormatWarning("Cannot open video source ", source, ". Please check permissions, or whether the file provided is broken. Defaulting to 1920px.");
        }
    }
    
    if (not combined.map.has("meta_real_width")
        || combined.map.at("meta_real_width").value<Float2_t>() == 0)
    {
        auto meta_video_size = combined.map.at("meta_video_size");
        Print(meta_video_size);
        assert(meta_video_size.valid() && not meta_video_size.value<Size2>().empty());
        combined.map["meta_real_width"] = meta_video_size.value<Size2>().width;
    }
    
    if (combined.map.has("cm_per_pixel")
        && combined.map.at("cm_per_pixel").value<Settings::cm_per_pixel_t>() == 0)
    {
        if (combined.map.has("source")
            && combined.map.at("source").value<file::PathArray>() == file::PathArray("webcam"))
        {
            combined.map["cm_per_pixel"] = Settings::cm_per_pixel_t(1);
        } else
            combined.map["cm_per_pixel"] = infer_cm_per_pixel(&combined.map);
    }
    
    const Float2_t tmp_cm_per_pixel = combined.map.at("cm_per_pixel").value<Settings::cm_per_pixel_t>();
    current_defaults["track_max_speed"] = 0.25_F * combined.map.at("meta_video_size").value<Size2>().width * (tmp_cm_per_pixel == 0 ? 1_F : tmp_cm_per_pixel);
    if(not quiet)
        Print(" * default max speed for a video of resolution ", combined.map.at("meta_video_size").value<Size2>().width, " would be ", no_quotes(current_defaults["track_max_speed"].get().valueString()));
    
    if(not combined.map.has("track_max_speed")
       || combined.map.at("track_max_speed").value<Settings::track_max_speed_t>() == 0)
    {
        combined.map["track_max_speed"] = current_defaults["track_max_speed"].value<Settings::track_max_speed_t>();
    }
    
    if(not quiet) {
        Print("track_max_speed = ", combined.map.at("track_max_speed").value<Settings::track_max_speed_t>());
        Print("cm_per_pixel = ", combined.map.at("cm_per_pixel").value<Settings::cm_per_pixel_t>());
        Print("meta_real_width = ", no_quotes(combined.map.at("meta_real_width").get().valueString()));
        Print("meta_video_size = ", no_quotes(combined.map.at("meta_video_size").get().valueString()));
    }
}

void LoadContext::finalize() {
    /// --------------------------------------
    G g("FINAL CONFIG", quiet);

    for(auto &key : combined.map.keys()) {
        try {
            if(GlobalSettings::access_level(key) < AccessLevelType::SYSTEM
               && (not GlobalSettings::has(key)
                   || GlobalSettings::map().at(key).get() != combined.map.at(key).get())
               )
            {
                //if(not contains(copy.toVector(), key))
                {
                    //Print("Updating ",combined.map.at(key));
                    if(key == "filename"
                       && (combined.map.at(key).value<file::Path>() == find_output_name(combined.map)
                           || (not combined.map.at(key).value<file::Path>().is_absolute()
                               && combined.map.at(key).value<file::Path>() == file::find_basename(combined.map.at("source").value<file::PathArray>()))))
                    {
                        SETTING(filename) = file::Path();
                        continue;
                    }
                    
                    if(key == "output_dir"
                       && combined.map.at(key).value<file::Path>() == file::find_parent( combined.map.at("source").value<file::PathArray>()))
                    {
                        SETTING(output_dir) = file::Path();
                        continue;
                    }
                    
                    if(not is_in(key, "gui_interface_scale"))
                        combined.map.at(key).get().copy_to(GlobalSettings::map());
                }
                /*else {
                 Print("Would be updating ",combined.map.at(key), " but is forbidden.");
                 }*/
            }
        } catch(const std::exception& ex) {
            FormatExcept("Cannot parse setting ", key, " and copy it to GlobalSettings: ", ex.what());
        }
    }
    
}

bool LoadContext::set_config_if_different(
      const std::string_view &key,
      const sprite::Map &from,
      [[maybe_unused]] bool do_print)
{
    bool was_different{false};
    
    if(&combined.map != &from) {
        if((combined.map.has(key)
            && combined.map.at(key) != from.at(key))
           || not GlobalSettings::defaults().has(key)
           || GlobalSettings::defaults().at(key) != from.at(key))
        {
            //if(do_print)
            /*if(not GlobalSettings::defaults().has(key) || GlobalSettings::defaults().at(key) != from.at(key))
            {
                Print("setting current_defaults ", from.at(key), " != ", GlobalSettings::defaults().at(key));
            }*/
            if(not combined.map.has(key) || combined.map.at(key) != from.at(key)) {
                //Print("setting combined.map ", key, " to ", from.at(key).get().valueString());
                from.at(key).get().copy_to(combined.map);
                was_different = true;
            }
            
            if(key == "detect_type")
                type = from.at(key).value<decltype(type)>();
        }
        /*else {
            Print("/// ", key, " is already set to ", combined.map.at(key).get().valueString());
        }*/
    }
    
    if(not current_defaults.has(key)
       || current_defaults.at(key) != from.at(key))
    {
        //if(do_print)
        //    Print("setting current_defaults ", from.at(key), " != ", current_defaults.at(key));
        if (not GlobalSettings::defaults().has(key)
            || GlobalSettings::defaults().at(key) != from.at(key))
        {
            from.at(key).get().copy_to(current_defaults);
            //Print("/// [current_defaults] ", current_defaults.at(key).get());
        }
        else if (current_defaults.has(key))
        {
            //Print("/// [current_defaults] REMOVE ", current_defaults.at(key).get());
            current_defaults.erase(key);
        }
        else {
            /// we dont have it, but it is default
            //Print("/// [current_defaults] ", key, " is default = ", from.at(key).get().valueString());
        }
        
    } //else if(current_defaults.has(key) && current_defaults.at(key) == from.at(key))
    else if(current_defaults.has(key)) {
        //Print("/// [current_defaults] ", key, " is already set to ", current_defaults.at(key).get().valueString());
        //current_defaults.erase(key);
    }
    else {
        //Print("/// *** WEIRD [current_defaults] ", key, " is default = ", from.at(key).get().valueString());
    }
    
    return was_different;
}

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

void load(LoadContext ctx) {
    // Entry point to reload and apply all settings in LoadContext 'ctx'.
    // This function resets defaults, loads from embedded and external sources,
    // applies command-line and GUI overrides, computes metadata, and finalizes GlobalSettings.
    if(not ctx.quiet)
        DebugHeader("Reloading settings");

    ctx.init();
    // Step 1: Reset to defaults, apply default and startup settings, and set up tracing.
    
    // Step 2: Monitor changes to 'calculate_posture'. If manually disabled,
    // record this fact to avoid automatic re-enabling later.
    ctx.combined.map.register_callbacks({"calculate_posture"}, [&](auto key) {
        //if(was_different
        //   && key == "calculate_posture")
        {
            bool calculate_posture = ctx.combined.map.at("calculate_posture").value<bool>();
            ctx.did_set_calculate_posture_to_false = not calculate_posture;
        }
    });
    
    // Step 3: Initialize the output filename from parameters or derive from source/defaults.
    ctx.init_filename();
    
    /// ------------------
    /// initial settings
    //ctx.stage_guard = nullptr;
    /// ------------------

    // Step 4a: If 'source' is empty, derive it from filename or provided defaults.
    ctx.fix_empty_source();
    // Step 4b: Ensure 'filename' is set; derive it from source if missing.
    ctx.fix_empty_filename();
    // Step 4c: Clear default-generated filenames to prevent overriding explicit settings.
    ctx.reset_default_filenames();

    // Step 5: Exclude 'output_dir' and 'output_prefix' from further default operations,
    // as filename/source have been locked in.
    ctx.exclude = ctx.exclude + std::array{
        "output_dir",
        "output_prefix"
    };

    // Step 6: Check if the detection model was manually specified; preserve it if so.
    ctx.changed_model_manually = ctx.combined.map.has("detect_model")
                && not ctx.combined.map.at("detect_model").value<file::Path>().empty();

    // Step 7: Load settings embedded in the source .pv file (if present).
    ctx.load_settings_from_source();

    // Step 8: Apply task-specific defaults based on 'detect_type'.
    ctx.load_task_defaults();

    // Commit current defaults to GlobalSettings before loading external settings.
    GlobalSettings::set_current_defaults_with_config(ctx.current_defaults);

    // Step 9: Load external settings file (video.settings), applying overrides appropriately.
    ctx.load_settings_file();

    // Step 10: Apply GUI-provided overrides from 'source_map'.
    ctx.load_gui_settings();

    // Step 11: Estimate and set metadata (video size, real-world scale, max speed).
    ctx.estimate_meta_variables();

    // Step 12: Ensure a valid 'detect_type'; default to YOLO if still unset.
    if(ctx.type == detect::ObjectDetectionType::none)
    {
        /// we need to have some kind of default.
        /// use the new technology first:
        ctx.type = detect::ObjectDetectionType::yolo;
    }
    ctx.combined.map["detect_type"] = ctx.type;

    // Step 13: For pure detection format (boxes), disable posture estimation.
    if(auto detect_format = ctx.combined.map.at("detect_format").value<track::detect::ObjectDetectionFormat_t>();
       detect_format == track::detect::ObjectDetectionFormat::boxes)
    {
        if(not ctx.did_set_calculate_posture_to_false) {
            FormatWarning("Disabling posture for now, since pure detection models cannot produce useful posture (everything will be rectangles).");
            ctx.combined.map["calculate_posture"] = false;
        }
    }

    // Step 14: Finalize settings: copy combined map to GlobalSettings and preserve print state.
    bool before = GlobalSettings::map().print_by_default();
    if(ctx.quiet)
        GlobalSettings::map().set_print_by_default(false);

    ctx.finalize();
    //Print("current defaults = ", current_defaults.keys());

    //combined.map.set_print_by_default(true);
    GlobalSettings::map().set_print_by_default(before);
    GlobalSettings::set_current_defaults_with_config(ctx.current_defaults);

    // Suppress printing of transient GUI-related settings to reduce log noise.
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_mode"].get().set_do_print(false);
    GlobalSettings::map()["gui_focus_group"].get().set_do_print(false);
    GlobalSettings::map()["gui_source_video_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_displayed_frame"].get().set_do_print(false);
    GlobalSettings::map()["heatmap_ids"].get().set_do_print(false);
    GlobalSettings::map()["gui_run"].get().set_do_print(false);
    GlobalSettings::map()["track_pause"].get().set_do_print(false);
    GlobalSettings::map()["terminate"].get().set_do_print(false);
    GlobalSettings::map()["gui_interface_scale"].get().set_do_print(false);

    // Step 15: Reset selected command-line settings to avoid persisting across runs.
    CommandLine::instance().reset_settings({
        //"output_dir",
        "gpu_torch_device", "wd"
    });
}

file::Path find_output_name(const sprite::Map& map,
                            file::PathArray source,
                            file::Path filename,
                            bool respect_user_choice)
{
    const auto _source = source.empty()
        ? map.at("source").value<file::PathArray>()
        : source;
    
    auto name = respect_user_choice
        ? map.at("filename").value<file::Path>()
        : file::Path{};
    
    filename = name.empty()
        ? file::Path()
        : file::DataLocation::parse("output", name, &map);
    
    if(not filename.empty())
    {
        if(filename.has_extension("pv"))
            filename = filename.remove_extension();
        return filename;
        
    } else {
        if(_source.get_paths().size() == 1
           && _source.get_paths().front().has_extension("pv"))
        {
            file::Path path = _source.get_paths().front();
            if(not path.empty()) {
                filename = path.absolute();//file::DataLocation::parse("output", path, &map);
            } else {
                filename = {};
            }
            
        } else {
            filename = file::find_basename(_source);
            if(filename.has_extension() && filename.exists())
                filename = filename.remove_extension();
        }
        
        if(not filename.empty()
           && not filename.has_extension("pv"))
        {
            filename = file::DataLocation::parse("output", filename, &map);
            
        } else if(filename.empty()) {
            filename = {};
        }
        
        if(filename.has_extension("pv"))
            filename = filename.remove_extension();
        
        return filename;
    }
}

void write_config(const pv::File* video, bool overwrite, gui::GUITaskQueue_t* queue, const std::string& suffix) {
    auto filename = file::DataLocation::parse(suffix == "backup" ? "backup_settings" : "output_settings");
    auto text = default_config::generate_delta_config(AccessLevelType::INIT, video).to_settings();
    
    if(filename.exists() && !overwrite) {
        if(queue) {
            queue->enqueue([queue, text, filename](auto, gui::DrawStructure& graph){
                graph.dialog([queue, str = text, filename](gui::Dialog::Result r) {
                    if(r == gui::Dialog::OKAY) {
                        if(!filename.remove_filename().exists())
                            filename.remove_filename().create_folder();
                        
                        FILE *f = fopen(filename.str().c_str(), "wb");
                        if(f) {
                            Print("Overwriting file ",filename.str(),".");
                            fwrite(str.data(), 1, str.length(), f);
                            fclose(f);
                        } else {
                            FormatExcept("Dont have write permissions for file ",filename.str(),".");
                            queue->enqueue([filename](auto, auto& graph){
                                graph.dialog("Cannot write configuration to <cyan><c>" + filename.str()+"</c></cyan>. Please check file permissions.", "<sym>☣</sym> Error");
                            });
                        }
                    }
                    
                }, "Overwrite file <i>"+filename/*.filename()*/.str()+"</i> ?", "Write configuration", "Yes", "No");
            });
            
        } else
            Print("Settings file ",filename.str()," already exists. Will not overwrite.");
        
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

Float2_t infer_cm_per_pixel(const sprite::Map* map) {
    if(map == nullptr)
        map = &GlobalSettings::map();

    // setting cm_per_pixel after average has been generated (and offsets have been set)
    if(not map->has("cm_per_pixel")
       || map->at("cm_per_pixel").value<Settings::cm_per_pixel_t>() == 0)
    {
        /*auto w = map->at("meta_real_width").value<Float2_t>();
        if(w <= 0) {
            return 1;
        }
        
        return 1_F / max(1.0_F, w * 0.05_F);*/
        return 1_F;
        //return w / float(average().cols);
    }

    return map->at("cm_per_pixel").value<Settings::cm_per_pixel_t>();
}

Float2_t infer_meta_real_width_from(const pv::File &file, const sprite::Map* map) {
    if(not map)
        map = &GlobalSettings::map();
    
    if(not map->has("meta_real_width")
        || map->at("meta_real_width").value<Float2_t>() == 0)
    {
        if(file.header().meta_real_width <= 0) {
            FormatWarning("This video does not set `meta_real_width`. Please set this value during conversion (see https://trex.run/docs/parameters_trex.html#meta_real_width for details). Defaulting to 30cm.");
            return Float2_t(30.0);
        } else {
            if(not map->has("meta_real_width")
                || map->at("meta_real_width").value<Float2_t>() == 0)
            {
                return file.header().meta_real_width;
            }
        }
    }
    
    return map->at("meta_real_width").value<Float2_t>();
}

}
