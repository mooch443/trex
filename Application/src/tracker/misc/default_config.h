#pragma once

#include <commons.pc.h>
#include <misc/GlobalSettings.h>
#include <file/Path.h>

namespace pv {
class File;
}

namespace default_config {
    using namespace cmn;

    using graphs_type = std::vector<std::pair<std::string, std::vector<std::string>>>;
    using default_options_type = std::unordered_map<std::string, std::vector<std::string>>;
    
    const std::string& homedir();
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
        ExtendableVector excluded;
        std::string to_settings() const;
        void write_to(sprite::Map& other);
        const sprite::PropertyType*& operator[](const std::string& key);
    };
    Config generate_delta_config(AccessLevel level, const pv::File* = nullptr, bool include_build_number = false, std::vector<std::string> additional_exclusions = {});
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

    ENUM_CLASS(heatmap_normalization_t, none, value, cell, variance)
    ENUM_CLASS_HAS_DOCS(heatmap_normalization_t)

    ENUM_CLASS(individual_image_normalization_t, none, moments, posture, legacy)
    ENUM_CLASS_HAS_DOCS(individual_image_normalization_t)

    individual_image_normalization_t::Class valid_individual_image_normalization(individual_image_normalization_t::Class = individual_image_normalization_t::none);

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

    ENUM_CLASS(blob_split_algorithm_t, threshold, threshold_approximate, fill, none)
    ENUM_CLASS_HAS_DOCS(blob_split_algorithm_t)

    ENUM_CLASS(visual_identification_version_t, current, v200, v119, v118_3, v110, v100, convnext_base, vgg_16, vgg_19, mobilenet_v3_small, mobilenet_v3_large, inception_v3, resnet_50_v2, efficientnet_b0, resnet_18)
    ENUM_CLASS_HAS_DOCS(visual_identification_version_t)

    ENUM_CLASS(TRexTask_t, none, track, convert, annotate, rst)
    ENUM_CLASS_HAS_DOCS(TRexTask_t)

    ENUM_CLASS(gpu_torch_device_t, automatic, cuda, mps, cpu)
    ENUM_CLASS_HAS_DOCS(gpu_torch_device_t)

    using TRexTask = TRexTask_t::Class;

/**
 * Finds all numeric pose indexes from user-defined "poseX##" / "poseY##" fields in the given output fields.
 *
 * @param output_fields The list of existing fields, e.g. from SETTING(output_fields).
 * @return A set of numeric indexes that the user has added.
 */
std::set<uint8_t> find_user_defined_pose_fields(const std::vector<std::pair<std::string, std::vector<std::string>>>& output_fields);

/**
 * Generates all auto-detected pose fields using either the provided keypoint names (if any)
 * for the first N keypoints, and default naming ("poseX#/poseY#") for the remaining keypoints.
 *
 * @return A vector of all possible pose fields.
 */
std::tuple<std::vector<size_t>, std::vector<std::pair<std::string, std::vector<std::string>>>> list_auto_pose_fields();

/**
 * Given a list of user-defined pose indexes (e.g. from find_user_defined_pose_fields()),
 * returns only the "missing" fields that the user has NOT defined, from the full
 * list of automatically generated fields (from list_auto_pose_fields()).
 *
 * @return A vector of newly needed poseX/poseY fields.
 */
std::vector<std::pair<std::string, std::vector<std::string>>> add_missing_pose_fields();


}

namespace cmn::gui {
ENUM_CLASS(mode_t, blobs, tracking, optical_flow)

}
