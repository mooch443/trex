#pragma once

#include <types.h>
#include <misc/GlobalSettings.h>
#include <file/Path.h>

namespace default_config {
    using namespace cmn;
    
    void get(sprite::Map& config, GlobalSettings::docs_map_t& docs, decltype(GlobalSettings::set_access_level)* fn);
    void warn_deprecated(const std::string& source, sprite::Map& map);
    void warn_deprecated(const std::string& source, const std::map<std::string, std::string>& keys);
    bool is_deprecated(const std::string& key);
    const std::map<std::string, std::string>& deprecations();
    std::string replacement(const std::string& key);
    std::string generate_delta_config(bool include_build_number = false, std::vector<std::string> additional_exclusions = {});
    void register_default_locations();
    void load_string_with_deprecations(const file::Path& source, const std::string& content, sprite::Map& map, AccessLevel, bool quiet = false);

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

    ENUM_CLASS(recognition_normalization_t, none, moments, posture, legacy)
    ENUM_CLASS_HAS_DOCS(recognition_normalization_t)

    ENUM_CLASS(gpu_verbosity_t, silent, full, oneline)
    ENUM_CLASS_HAS_DOCS(gpu_verbosity_t)
    
    ENUM_CLASS(gui_recording_format_t, avi, jpg, png)
    ENUM_CLASS_HAS_DOCS(gui_recording_format_t)
    
    ENUM_CLASS(peak_mode_t, pointy, broad)
    ENUM_CLASS_HAS_DOCS(peak_mode_t)

    ENUM_CLASS(matching_mode_t, accurate, approximate, hungarian, benchmark, automatic)
    ENUM_CLASS_HAS_DOCS(matching_mode_t)

    ENUM_CLASS(output_format_t, csv, npz)
    ENUM_CLASS_HAS_DOCS(output_format_t)

    ENUM_CLASS(output_invalid_t, inf, nan)
    ENUM_CLASS_HAS_DOCS(output_invalid_t)

    ENUM_CLASS(app_update_check_t, none, manually, automatically)
    ENUM_CLASS_HAS_DOCS(app_update_check_t)
}

namespace gui {
ENUM_CLASS(mode_t, blobs, tracking, optical_flow)

}
