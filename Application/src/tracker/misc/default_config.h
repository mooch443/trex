#pragma once

#include <misc/defines.h>
#include <misc/GlobalSettings.h>
#include <file/Path.h>
#include <misc/utilsexception.h>

namespace default_config {
    using namespace cmn;
    
    void get(sprite::Map& config, GlobalSettings::docs_map_t& docs, std::function<void(const std::string& name, AccessLevel w)> fn);

    void execute_settings_string(const std::string& content, const file::Path& source, AccessLevelType::Class level, const std::vector<std::string>& exclude = {});
    bool execute_settings_file(const file::Path& source, AccessLevelType::Class level, const std::vector<std::string>& exclude = {});

    void warn_deprecated(const file::Path& source, sprite::Map& map);
    void warn_deprecated(const file::Path& source, const std::map<std::string, std::string>& keys);
    bool is_deprecated(const std::string& key);
    const std::map<std::string, std::string>& deprecations();
    std::string replacement(const std::string& key);

    struct Config {
        std::map<std::string, const sprite::PropertyType*> map;
        std::string to_settings() const;
        void write_to(sprite::Map& other);
        const sprite::PropertyType*& operator[](const std::string& key);
    };
    Config generate_delta_config(bool include_build_number = false, std::vector<std::string> additional_exclusions = {});
    void register_default_locations();
    void load_string_with_deprecations(const file::Path& source, const std::string& content, sprite::Map& map, AccessLevel, const std::vector<std::string>& exclude = {}, bool quiet = false);

    file::Path conda_environment_path();


    /*template<typename T>
    struct HasDocsMethod
    {
        template<typename U, size_t (U::*)() const> struct SFINAE {};
        template<typename U> static char Test(SFINAE<U, &U::used_memory>*);
        template<typename U> static int Test(...);
        static const bool Has = sizeof(Test<T>(0)) == sizeof(char);
    };*/
    
    ENUM_CLASS(recognition_border_t, none, heatmap, outline, shapes, grid, circle)
    ENUM_CLASS_HAS_DOCS(recognition_border_t)

    ENUM_CLASS(heatmap_normalization_t, none, value, cell, variance)
    ENUM_CLASS_HAS_DOCS(heatmap_normalization_t)

    ENUM_CLASS(individual_image_normalization_t, none, moments, posture, legacy)
    ENUM_CLASS_HAS_DOCS(individual_image_normalization_t)

    ENUM_CLASS(gpu_verbosity_t, silent, full, oneline)
    ENUM_CLASS_HAS_DOCS(gpu_verbosity_t)
    
    ENUM_CLASS(gui_recording_format_t, avi, mp4, jpg, png)
    ENUM_CLASS_HAS_DOCS(gui_recording_format_t)
    
    ENUM_CLASS(peak_mode_t, pointy, broad)
    ENUM_CLASS_HAS_DOCS(peak_mode_t)

    ENUM_CLASS(matching_mode_t, tree, approximate, hungarian, benchmark, automatic, none)
    ENUM_CLASS_HAS_DOCS(matching_mode_t)

    ENUM_CLASS(output_format_t, csv, npz)
    ENUM_CLASS_HAS_DOCS(output_format_t)

    ENUM_CLASS(output_invalid_t, inf, nan)
    ENUM_CLASS_HAS_DOCS(output_invalid_t)

    ENUM_CLASS(app_update_check_t, none, manually, automatically)
    ENUM_CLASS_HAS_DOCS(app_update_check_t)

    ENUM_CLASS(blob_split_algorithm_t, threshold, threshold_approximate, fill)
    ENUM_CLASS_HAS_DOCS(blob_split_algorithm_t)

    ENUM_CLASS(visual_identification_version_t, current, v118_3, v110, v100)
    ENUM_CLASS_HAS_DOCS(visual_identification_version_t)

    ENUM_CLASS(TRexTask_t, none, track, convert, annotate)
    ENUM_CLASS_HAS_DOCS(TRexTask_t)

    ENUM_CLASS(gpu_torch_device_t, automatic, cuda, mps, cpu)
    ENUM_CLASS_HAS_DOCS(gpu_torch_device_t)

    using TRexTask = TRexTask_t::Class;
}

namespace gui {
ENUM_CLASS(mode_t, blobs, tracking, optical_flow)

}
