#include "default_config.h"
#include <misc/SpriteMap.h>
#include <file/Path.h>
#include <misc/SizeFilters.h>
#include <misc/idx_t.h>
#include "GitSHA1.h"
#include <misc/bid.h>
#include <misc/colors.h>
#include <misc/DetectionTypes.h>
#include <misc/Border.h>
#include <misc/TrackingSettings.h>
#include <pv.h>

#ifndef WIN32
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

#include <misc/default_settings.h>
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <misc/zipper.h>


#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <string>
#include <limits.h>
#endif

const auto homedir = []() {
#ifndef WIN32
    struct passwd* pw = getpwuid(getuid());
    const char* homedir = pw->pw_dir;
    return std::string(homedir);
#else
    char* home;
    size_t size;
    if (_dupenv_s(&home, &size, "USERPROFILE"))
        return std::string();
    auto str = std::string(home);
    free(home);
    return str;
#endif
}();

#include <misc/default_settings.h>

using namespace cmn::file;
#define CONFIG adding.add<ParameterCategoryType::CONVERTING>

namespace default_config {

const std::string& homedir() {
    return ::homedir;
}

ENUM_CLASS_DOCS(individual_image_normalization_t,
                "No normalization. Images will only be cropped out and used as-is.",
                "Images will be cropped out and aligned as in idtracker.ai using the main axis calculated using `image moments`.",
                "Images will be cropped out and rotated so that the head will be fixed in one position and only the tail moves.",
                "Images will be aligned parallel to the x axis."
)

ENUM_CLASS_DOCS(heatmap_normalization_t,
                "No normalization at all. Values will only be averaged per cell.",
                "Normalization based in value-space. The average of each cell will be divided by the maximum value encountered.",
                "The cell sum will be divided by the maximum cell value encountered.",
                "Displays the variation within each cell."
)

ENUM_CLASS_DOCS(gui_recording_format_t,
    "AVI / video format (codec MJPG is used)",
    "MP4 / video format (codec H264 is used)",
    "individual images in JPEG format",
    "individual images in PNG format"
)

ENUM_CLASS_DOCS(peak_mode_t,
    "The head is broader than the tail.",
    "The tail is broader than the head."
)

ENUM_CLASS_DOCS(matching_mode_t,
    "Maximizes the probability sum by assigning (or potentially not assigning) individuals to objects in the frame. This returns the correct solution, but might take long for high quantities of individuals.",
    "Simply assigns the highest probability edges (blob to individual) to all individuals - first come, first serve. Parameters have to be set very strictly (especially speed) in order to have as few objects to choose from as possible and limit the error.",
    "The hungarian algorithm (as implemented in O(n^3) by Mattias Andrée `https://github.com/maandree/hungarian-algorithm-n3`).",
    "Runs all algorithms and pits them against each other, outputting statistics every few frames.",
    "Uses automatic selection based on density.",
    "No algorithm, direct assignment."
)

ENUM_CLASS_DOCS(output_format_t,
    "A standard data format, comma-separated columns for each data stream. Use `output_csv_decimals` to adjust the maximum precision for exported data.",
    "NPZ is basically a collection of binary arrays, readable by NumPy and other plugins (there are plugins available for Matlab and R)."
)

ENUM_CLASS_DOCS(output_invalid_t,
    "Infinity (e.g. np.inf)",
    "NaN (e.g. np.nan)"
)

ENUM_CLASS_DOCS(gpu_verbosity_t,
    "No output during training.",
    "An animated bar with detailed information about the training progress.",
    "One line per epoch."
)

ENUM_CLASS_DOCS(app_update_check_t,
    "No status has been set yet and the program will ask the user.",
    "Manually check for updates, do not automatically check for them online.",
    "Automatically check for updates periodically (once per week)."
)

ENUM_CLASS_DOCS(blob_split_algorithm_t,
    "Adaptively increase the threshold of closeby objects, until separation.",
    "Same as threshold, but use heuristics to produce results faster. These results might not be as deterministic as with threshold, but usually only differ by 1 or 2 in found threshold value. It is guaranteed, however, that a solution is found if one exists.",
    "Use the previously known positions of objects to place a seed within the overlapped objects and perform a watershed run.",
    "Do not actually attempt to split blobs. Just ignore blobs until they split by themselves."
)

// current, v200, v119, v118_3, v110, v100, convnext_base, vgg_16, vgg_19, mobilenet_v3_small, mobilenet_v3_large, inception_v3, resnet_50_v2, efficientnet_b0, resnet_18
ENUM_CLASS_DOCS(visual_identification_version_t,
    "This always points to the current version.",
    "The v200 model introduces a deeper architecture with five convolutional layers, compared to the four layers in v119. The convolutional layers in v200 have channel sizes of 64, 128, 256, 512, and 512, whereas v119 has channel sizes of 256, 128, 32, and 128. Additionally, v200 incorporates global average pooling and larger dropout rates, enhancing regularization. Both models use Batch Normalization, ReLU activations, and MaxPooling, but v200 features a more complex fully connected layer structure with an additional dropout layer before the final classification.",
    "The v119 model introduces an additional convolutional layer, increasing the total to four layers with channel sizes of 256, 128, 32, and 128, compared to the three layers in v118_3 with channel sizes of 16, 64, and 100. Additionally, v119 features larger fully connected layers with 1024 units in the first layer, whereas v118_3 has 100 units. Both models maintain the use of MaxPooling, Dropout, Batch Normalization, and a final Softmax activation for class probabilities.",
    "The order of Max-Pooling layers was changed, along with some other minor changes.",
    "Changed activation order, added BatchNormalization. No Flattening to maintain spatial context.",
    "The original layout.",
    "The ConvNeXtBase architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a modernized ResNet-inspired structure with large kernel sizes, efficient attention mechanisms, and an optimized design for both computational efficiency and performance, aimed at achieving state-of-the-art results on large-scale image datasets.",
    "The VGG16 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a simple and straightforward structure with small kernel sizes, a large number of layers, and a focus on simplicity and ease of use, aimed at achieving strong results on small-scale image datasets.",
    "The VGG19 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a simple and straightforward structure with small kernel sizes, a large number of layers, and a focus on simplicity and ease of use, aimed at achieving strong results on small-scale image datasets.",
    "The MobileNetV3Small architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a lightweight structure with small kernel sizes, efficient depthwise separable convolutions, and an emphasis on computational efficiency and performance, aimed at achieving strong results on mobile and edge devices with limited computational resources.",
    "The MobileNetV3Large architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a lightweight structure with small kernel sizes, efficient depthwise separable convolutions, and an emphasis on computational efficiency and performance, aimed at achieving strong results on mobile and edge devices with limited computational resources.",
    "The InceptionV3 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a complex structure with multiple parallel paths, efficient factorization methods, and an emphasis on computational efficiency and performance, aimed at achieving strong results on large-scale image datasets.",
    "The ResNet50V2 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a modernized ResNet-inspired structure with bottleneck blocks, efficient skip connections, and an optimized design for both computational efficiency and performance, aimed at achieving strong results on large-scale image datasets.",
    "The EfficientNetB0 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a lightweight structure with small kernel sizes, efficient depthwise separable convolutions, and an emphasis on computational efficiency and performance, aimed at achieving strong results on mobile and edge devices with limited computational resources.",
    "The ResNet18 architecture is a deep convolutional neural network (CNN) designed for image classification tasks, featuring a modernized ResNet-inspired structure with bottleneck blocks, efficient skip connections, and an optimized design for both computational efficiency and performance, aimed at achieving strong results on large-scale image datasets."
)

ENUM_CLASS_DOCS(TRexTask_t,
    "No task forced. Auto-select.",
    "Load an existing .pv file and track / edit individuals.",
    "Convert source material to .pv file.",
    "Annotate video or image source material.",
    "Save .rst parameter documentation files to the output folder."
)

ENUM_CLASS_DOCS(gpu_torch_device_t,
    "The device is automatically chosen by PyTorch.",
    "Use a CUDA device (requires an NVIDIA graphics card).",
    "Use a METAL device (requires an Apple Silicone Mac).",
    "Use the CPU (everybody should have this)."
)

static void apply_whitelist(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    auto values = Meta::fromStr<std::vector<float>>((std::string)val);
    auto r = obj.replacement.value();
    
    if(values.size() == 4) {
        map[r] = std::vector<std::vector<Vec2>>{
            {
                Vec2(values[0], values[1]),
                Vec2(values[0] + values[2], values[1]),
                Vec2(values[0] + values[2], values[1] + values[3]),
                Vec2(values[0], values[1] + values[3])
            }
        };
        
    } else
        throw InvalidArgumentException("Invalid number of values while trying to correct ",val," deprecated parameter from ",obj.name," to ",r,".");
}

static void apply_whitelist_rects(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    auto values = Meta::fromStr<std::vector<Bounds>>((std::string)val);
    std::vector<std::vector<Vec2>> value;
    
    for(auto v : values) {
        value.push_back({
            v.pos(), v.pos() + Vec2(v.width, 0),
            v.pos() + v.size(),
            v.pos() + Vec2(0, v.height)
        });
    }
    
    map[obj.replacement.value()] = value;
}

static void apply_output_npz(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    auto options = GlobalSettings::LoadOptions{
        .source = sprite::MapSource("deprecations"),
        .deprecations = deprecations(),
        .access = AccessLevelType::SYSTEM,
        .target = &map
    };
    
    auto value = Meta::fromStr<bool>((std::string)val);
    GlobalSettings::load_from_string(obj.replacement.value() + " = " + (value ? "npz" : "csv") + "\n", options);
}

static void apply_match_use_approximate(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    auto value = Meta::fromStr<bool>((std::string)val);
    auto r = obj.replacement.value();
    auto options = GlobalSettings::LoadOptions{
        .source = sprite::MapSource("deprecations"),
        .deprecations = deprecations(),
        /// this is a security issue in theory
        .access = AccessLevelType::SYSTEM,
        .target = &map
    };
    GlobalSettings::load_from_string(r+" = "+(value ? "approximate" : "accurate")+"\n", options);
}

static void apply_analysis_stop_after(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    auto r = obj.replacement.value();
    auto options = GlobalSettings::LoadOptions{
        .source = sprite::MapSource("deprecations"),
        .deprecations = deprecations(),
        /// this is a security issue in theory
        .access = AccessLevelType::SYSTEM,
        .target = &map
    };
    GlobalSettings::load_from_string(r+" = [-1,"+(std::string)val+"]\n", options);
}

static void apply_recognition_normalize_direction(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    bool value = utils::lowercase(val) != "false";
    auto options = GlobalSettings::LoadOptions{
        .source = sprite::MapSource("deprecations"),
        .deprecations = deprecations(),
        /// this is a security issue in theory
        .access = AccessLevelType::SYSTEM,
        .target = &map
    };
    auto r = obj.replacement.value();
    GlobalSettings::load_from_string(r+" = "+Meta::toStr(value ? individual_image_normalization_t::posture : individual_image_normalization_t::none)+"\n", options);
}

static void apply_tracklet_export_difference_images(const Deprecation& obj, std::string_view val, sprite::Map& map)
{
    bool value = utils::lowercase(val) != "true";
    auto options = GlobalSettings::LoadOptions{
        .source = sprite::MapSource("deprecations"),
        .deprecations = deprecations(),
        /// this is a security issue in theory
        .access = AccessLevelType::SYSTEM,
        .target = &map
    };
    auto r = obj.replacement.value();
    GlobalSettings::load_from_string(r+" = "+Meta::toStr(value)+"\n", options);
}

static inline const Deprecations deprecated = Deprecations({
        {"analysis_paused", "track_pause"},
        {"meta_classes", "detect_classes"},
        {"meta_skeleton", "detect_skeleton"},
        {"detection_type", "detect_type"},
        {"detection_resolution", "detect_resolution"},
        {"model", "detect_model"},
        {"outline_step", "outline_smooth_step"},
        {"outline_smooth_range", "outline_smooth_samples"},
        {"max_frame_distance", "track_max_reassign_time"},
        {"fish_max_reassign_time", "track_max_reassign_time"},
        {"outline_curvature_range", ""},
        {"load_identity_network", ""},
        {"try_network_training_internally", ""},
        {"recognition_enable", ""},
        {"recognition_image_scale", "individual_image_scale"},
        {"recognition_image_size", "individual_image_size"},
        {"network_training_output_size", "individual_image_size"},
        {"gui_save_npy_quit", "auto_quit"},
        {"gui_auto_quit", "auto_quit"},
        {"gui_stop_after", "analysis_range"},
        {"analysis_stop_after", "analysis_range", apply_analysis_stop_after},
        {"track_segment_max_length", "tracklet_max_length"},
        {"track_end_tracklet_for_speed", "tracklet_punish_speeding"},
        {"huge_timestamp_ends_segment", "tracklet_punish_timedelta"},
        {"recognition_segment_add_factor", "accumulation_tracklet_add_factor"},
        {"recognition_save_training_images", "visual_identification_save_images"},
        {"gpu_enable_accumulation", "accumulation_enable"},
        {"gpu_accepted_uniqueness", "accumulation_sufficient_uniqueness"},
        {"gpu_accumulation_max_segments", "accumulation_max_tracklets"},
        {"gpu_accumulation_enable_final_step", "accumulation_enable_final_step"},
        {"fixed_count", ""},
        {"gui_dpi_scale", ""},
        {"output_graphs", "output_fields"},
        {"fish_minmax_size", "track_size_filter"},
        {"blob_size_range", "detect_size_filter"},
        {"segment_size_filter", "detect_size_filter"},
        {"blob_size_ranges", "track_size_filter"},
        {"fish_max_speed", "track_max_speed"},
        {"max_speed", "track_max_speed"},
        {"fish_speed_decay", "track_speed_decay"},
        {"fish_enable_direction_smoothing", "posture_direction_smoothing"},
        {"fish_use_matching", ""},
        {"fish_time_probability_enabled", "track_time_probability_enabled"},
        {"number_fish", "track_max_individuals"},
        {"outline_remove_loops", ""},
        {"whitelist_rects", "track_include", apply_whitelist_rects},
        {"exclude_rects", "track_ignore", apply_whitelist_rects},
        {"whitelist_rect", "track_include", apply_whitelist},
        {"track_whitelist", "track_include"},
        {"exclude_rect", "track_ignore", apply_whitelist},
        {"track_blacklist", "track_ignore"},
        {"posture_threshold_constant", "track_posture_threshold"},
        {"threshold_constant", "track_threshold"},
        {"recognition_rect", "recognition_shapes", apply_whitelist},
        {"recognition_normalization", "individual_image_normalization"},
        {"recognition_normalize_direction", "individual_image_normalization", apply_recognition_normalize_direction},
        {"match_use_approximate", "match_mode", apply_match_use_approximate},
        {"output_npz", "output_format", apply_output_npz},
        {"gui_heatmap_value_range", "heatmap_value_range"},
        {"gui_heatmap_smooth", "heatmap_smooth"},
        {"gui_heatmap_frames", "heatmap_frames"},
        {"gui_heatmap_dynamic", "heatmap_dynamic"},
        {"gui_heatmap_resolution", "heatmap_resolution"},
        {"gui_heatmap_normalization", "heatmap_normalization"},
        {"gui_heatmap_source", "heatmap_source"},
        {"tracklet_normalize_orientation", "tracklet_normalize"},
        {"tracklet_export_difference_images", "tracklet_force_normal_color", apply_tracklet_export_difference_images},
        {"track_label_confidence_threshold", "track_conf_threshold"},
        {"matching_probability_threshold", "match_min_probability"},
        {"manual_ignore_bdx", "track_ignore_bdx"},
        {"track_absolute_difference", "track_threshold_is_absolute"},
        {"enable_absolute_difference", "detect_threshold_is_absolute"},
        {"categories_min_sample_images", "categories_apply_min_tracklet_length"},
        {"enable_live_tracking", ""},
        {"export_visual_fields", "output_visual_fields"},
        {"output_image_per_tracklet", "output_tracklet_images"}
});

/**
 * Finds all user-defined pose indexes from output fields.
 * This function accepts both the numeric style ("poseX##"/"poseY##") and,
 * if keypoint names are provided, the name style (e.g. "<keypoint_name>_X" / "<keypoint_name>_Y").
 * The numeric style is always checked for backward compatibility.
 *
 * @param output_fields The list of existing fields (from SETTING(output_fields)).
 * @return A set of keypoint indexes that the user has defined.
 */
std::set<uint8_t> find_user_defined_pose_fields(
    const std::vector<std::pair<std::string, std::vector<std::string>>>& output_fields)
{
    std::set<uint8_t> user_defined_indexes;
#ifndef NDEBUG
    Print("Debug: Entering find_user_defined_pose_fields, number of fields = ", output_fields.size());
#endif

    auto detect_keypoint_names = SETTING(detect_keypoint_names).value<track::detect::KeypointNames>();
    
    // Always check for numeric style fields.
    for (const auto& [field_name, transforms] : output_fields) {
        if (utils::beginsWith(field_name, "poseX") || utils::beginsWith(field_name, "poseY")) {
#ifndef NDEBUG
            Print("Debug: Processing numeric style field: ", field_name);
#endif
            try {
                uint8_t index = Meta::fromStr<uint8_t>(field_name.substr(5));
#ifndef NDEBUG
                Print("Debug: Parsed index ", static_cast<int>(index), " from field: ", field_name);
#endif
                user_defined_indexes.insert(index);
            }
            catch (...) {
#ifndef NDEBUG
                Print("Debug: Failed to parse numeric index from field: ", field_name);
#endif
            }
        }
    }
    
    // Additionally, if keypoint names are defined, check for name-style fields.
    if (detect_keypoint_names.valid() && detect_keypoint_names.names.has_value()) {
        const std::vector<std::string>& names = *detect_keypoint_names.names;
        for (const auto& [field_name, transforms] : output_fields) {
            for (size_t i = 0; i < names.size(); ++i) {
                std::string expectedX = names[i] + "_X";
                std::string expectedY = names[i] + "_Y";
                if (field_name == expectedX || field_name == expectedY) {
#ifndef NDEBUG
                    Print("Debug: Found user-defined field ", field_name, " corresponding to index ", i);
#endif
                    user_defined_indexes.insert(static_cast<uint8_t>(i));
                }
            }
        }
    }
#ifndef NDEBUG
    Print("Debug: Exiting find_user_defined_pose_fields, found indexes count: ", user_defined_indexes.size());
#endif
    return user_defined_indexes;
}

/**
 * Generates all auto-detected pose fields using either the provided keypoint names (if any)
 * for the first N keypoints, and default naming ("poseX#/poseY#") for the remaining keypoints.
 *
 * @return A vector of all possible pose fields.
 */
std::tuple<std::vector<size_t>, std::vector<std::pair<std::string, std::vector<std::string>>>> list_auto_pose_fields()
{
#ifndef NDEBUG
    Print("Debug: Entering list_auto_pose_fields");
#endif
    /// return empty array if automatically generating the fields is disabled.
    if (not BOOL_SETTING(output_auto_pose)) {
#ifndef NDEBUG
        Print("Debug: Auto pose field generation is disabled (output_auto_pose is false).");
#endif
        return {};
    }
    
    // Retrieve the YOLO classes from a global setting:
    auto detect_keypoint_format = SETTING(detect_keypoint_format).value<track::detect::KeypointFormat>();
    auto detect_keypoint_names  = SETTING(detect_keypoint_names).value<track::detect::KeypointNames>();
#ifndef NDEBUG
    Print("Debug: Retrieved detect_keypoint_format=", detect_keypoint_format,
          " detect_keypoint_names=", detect_keypoint_names);
#endif
    
    if(not detect_keypoint_format.valid()) {
        FormatWarning("No valid detect_keypoint_format set. Cannot automatically determine keypoint indexes.");
        return {};
    }
    
    std::vector<std::pair<std::string, std::vector<std::string>>> auto_pose_fields;
    std::vector<size_t> indexes;
    
    // For each keypoint index, if a name is provided for that index, use it; otherwise, use default naming.
    for (size_t i = 0; i < detect_keypoint_format.n_points; ++i) {
        std::string x_field, y_field;
        if (auto name = detect_keypoint_names.name(i);
            name.has_value())
        {
            x_field = *name + "_X";
            y_field = *name + "_Y";
#ifndef NDEBUG
            Print("Debug: Adding auto pose fields for keypoint: ", *name);
#endif
        } else {
            x_field = "poseX" + Meta::toStr(i);
            y_field = "poseY" + Meta::toStr(i);
#ifndef NDEBUG
            Print("Debug: Adding auto pose fields for index: ", i);
#endif
        }
        auto_pose_fields.emplace_back(x_field, std::vector<std::string>{"RAW"});
        indexes.push_back(i);
        auto_pose_fields.emplace_back(y_field, std::vector<std::string>{"RAW"});
        indexes.push_back(i);
    }

#ifndef NDEBUG
    Print("Debug: Exiting list_auto_pose_fields, total auto pose fields generated: ", auto_pose_fields.size());
#endif
    return {indexes, auto_pose_fields};
}

/**
 * Given the current output_fields setting, returns only the missing auto-generated pose fields
 * that the user has not explicitly defined.
 *
 * @return A vector of newly needed pose fields.
 */
std::vector<std::pair<std::string, std::vector<std::string>>> add_missing_pose_fields()
{
#ifndef NDEBUG
    Print("Debug: Entering add_missing_pose_fields");
#endif
    // 1) Gather all automatically generated pose fields.
    auto [auto_field_indexes, auto_fields] = list_auto_pose_fields();
#ifndef NDEBUG
    Print("Debug: Auto-generated pose fields count: ", auto_fields.size());
#endif

    // 2) Get the user-defined fields from settings.
    auto current_fields = SETTING(output_fields)
        .value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
#ifndef NDEBUG
    Print("Debug: Current output fields count: ", current_fields.size());
#endif
    auto user_defined_indexes = find_user_defined_pose_fields(current_fields);
#ifndef NDEBUG
    Print("Debug: User-defined pose indexes count: ", user_defined_indexes.size());
#endif
    
    // 3) Collect missing pose fields.
    std::vector<std::pair<std::string, std::vector<std::string>>> needed;
    needed.reserve(auto_fields.size());
    for (const auto& [field_index, field_props] : Zip::Zip( auto_field_indexes, auto_fields ))
    {
        const auto& [field_name, transforms] = field_props;
        
        try {
#ifndef NDEBUG
            Print("Debug: Checking auto field ", field_name, " with index ", static_cast<int>(field_index));
#endif
            if (user_defined_indexes.find(field_index) == user_defined_indexes.end()) {
#ifndef NDEBUG
                Print("Debug: Field ", field_name, " is missing from user-defined fields, adding to needed list");
#endif
                needed.push_back({field_name, transforms});
            }
        }
        catch (...) {
#ifndef NDEBUG
            Print("Debug: Failed to parse numeric index from field: ", field_name);
#endif
        }
    }
#ifndef NDEBUG
    Print("Debug: Exiting add_missing_pose_fields, missing fields count: ", needed.size());
#endif
    return needed;
}

individual_image_normalization_t::Class valid_individual_image_normalization(individual_image_normalization_t::Class base) {
    const auto n = base != individual_image_normalization_t::none ? base : SETTING(individual_image_normalization).value<individual_image_normalization_t::Class>();
    const auto normalize = n == individual_image_normalization_t::posture && not SETTING(calculate_posture).value<bool>() ? individual_image_normalization_t::moments :  n;
    return normalize;
}

file::Path conda_environment_path() {
#ifdef COMMONS_PYTHON_EXECUTABLE
    auto compiled_path = file::Path(COMMONS_PYTHON_EXECUTABLE).is_regular() ? file::Path(COMMONS_PYTHON_EXECUTABLE).remove_filename().str() : file::Path(COMMONS_PYTHON_EXECUTABLE).str();
    if(compiled_path == "CONDA_PREFIX")
        compiled_path = "";
#if defined(__linux__) || defined(__APPLE__)
    if(utils::endsWith(compiled_path, "/bin"))
        compiled_path = file::Path(compiled_path).remove_filename().str();
#endif
#else
    std::string compiled_path = "";
#endif
    
    auto home = SETTING(python_path).value<file::Path>().str();
    if(file::Path(home).is_regular())
        home = file::Path(home).remove_filename().str();
#if defined(__linux__) || defined(__APPLE__)
    if(utils::endsWith(home, "/bin"))
        home = file::Path(home).remove_filename().str();
#endif

    if(is_in(home, "CONDA_PREFIX", "", compiled_path)) {
#ifndef NDEBUG
        if(!GlobalSettings::is_runtime_quiet())
            Print("Reset conda prefix ",home," / ",compiled_path);
#endif
        auto conda_prefix = getenv("CONDA_PREFIX");
        
        if(conda_prefix) {
            // we are inside a conda environment
            home = conda_prefix;
        } else if(utils::contains(SETTING(wd).value<file::Path>().str(), "envs"+Meta::toStr(file::Path::os_sep()))) {
            auto folders = utils::split(SETTING(wd).value<file::Path>().str(), file::Path::os_sep());
            std::string previous = "";
            home = "";
            
            for(auto &folder : folders) {
                home += folder;
                
                if(previous == "envs") {
                    break;
                }
                
                home += file::Path::os_sep();
                previous = folder;
            }
        }
    } else
        home = compiled_path;
    
    //if(!SETTING(quiet))
    //    Print("Set conda environment path = ",home);
    return home;
}
    
    const Deprecations& deprecations() {
        return deprecated;
    }
    
    void warn_deprecated(const file::Path& source, sprite::Map& map) {
        std::map<std::string, std::string> found;
        
        for(auto &key : map.keys()) {
            if(is_deprecated(key)) //!TODO: check what this does (toStr)
                found.insert({key, map.operator[](key).toStr()});
        }
        
        warn_deprecated(source, found);
    }
    
    void warn_deprecated(const file::Path& source, const std::map<std::string, std::string>& keys) {
        bool found = false;
        for (auto && [key, val] : keys) {
            if(is_deprecated(key)) {
                found = true;
                
                auto r = replacement(key);
                if(r.empty()) {
                    FormatWarning("[",source.c_str(),"] Setting ",key," has been removed from the tracker (with no replacement) and will be ignored.");
                } else
                    FormatExcept("[",source.c_str(),"] Setting ",key," is deprecated. Please use its replacement parameter ",r," instead.");
            }
        }
        
        if(found)
            Print("Found invalid settings in source ",source," (see above).");
    }
    
    bool is_deprecated(const std::string& key) {
        return deprecated.is_deprecated(key);
    }
    
    std::string replacement(const std::string& key) {
        if (not is_deprecated(key)) {
            throw U_EXCEPTION("Key ",key," is not deprecated.");
        }
        
        return deprecated.deprecations.at(utils::lowercase(key)).replacement.value();
    }

#define PYTHON_TIPPS ""
#ifdef WIN32
#define PYTHON_TIPPS " (containing pythonXX.exe)"
#endif

void execute_settings_string(const std::string &content, const file::Path& source, AccessLevelType::Class level, const std::vector<std::string>& exclude) {
    try {
        GlobalSettings::write([&](Configuration& config){
            GlobalSettings::load_from_string(content, {
                .source = source,
                .access = level,
                .correct_deprecations = true,
                .exclude = exclude,
                .target = &config.values
            });
        });
        //default_config::load_string_with_deprecations(source, content, GlobalSettings::map(), level, exclude);
        
    } catch(const cmn::illegal_syntax& e) {
        FormatError("Illegal syntax in settings file.");
        return;
    }
}

bool execute_settings_file(const file::Path& source, AccessLevelType::Class level, const std::vector<std::string>& exclude) {
    if(source.exists()) {
        DebugHeader("LOADING ", source);
        try {
            auto content = source.read_file();
            execute_settings_string(content, source, level, exclude);
            
        } catch(const cmn::illegal_syntax& e) {
            FormatError("Illegal syntax in settings file.");
            return false;
        }
        DebugHeader("LOADED ", source);
        return true;
    }
    
    return false;
}
    
    void get(Configuration& config)
    {
        //auto old = config.print_by_default();
        //config.set_print_by_default(true);
        //constexpr auto PUBLIC = AccessLevelType::PUBLIC;
        constexpr auto STARTUP = AccessLevelType::STARTUP;
        constexpr auto SYSTEM = AccessLevelType::SYSTEM;
        constexpr auto LOAD = AccessLevelType::LOAD;
        constexpr auto INIT = AccessLevelType::INIT;
        constexpr auto PUBLIC = AccessLevelType::PUBLIC;
        
        using namespace settings;
        Adding adding(config);
        
        CONFIG("app_name", std::string("TRex"), "Name of the application.", SYSTEM);
        CONFIG("app_check_for_updates", app_update_check_t::none, "If enabled, the application will regularly check for updates online (`https://api.github.com/repos/mooch443/trex/releases`).");
        CONFIG("app_last_update_check", uint64_t(0), "Time-point of when the application has last checked for an update.", SYSTEM);
        CONFIG("app_last_update_version", std::string(), "");
        CONFIG("version", std::string(g_GIT_DESCRIBE_TAG)+(std::string(g_GIT_CURRENT_BRANCH) != "main" ? "_"+std::string(g_GIT_CURRENT_BRANCH) : ""), "Current application version.", SYSTEM);
        CONFIG("build_architecture", std::string(g_TREX_BUILD_ARCHITECTURE), "The architecture this executable was built for.", SYSTEM);
        CONFIG("build_type", std::string(g_TREX_BUILD_TYPE), "The mode the application was built in.", SYSTEM);
        CONFIG("build_is_debug", std::string(compile_mode_name()), "If built in debug mode, this will show 'debug'.", SYSTEM);
        CONFIG("build_cxx_options", std::string(g_TREX_BUILD_CXX_OPTIONS), "The mode the application was built in.", SYSTEM);
        CONFIG("build", std::string(), "Current build version", SYSTEM);
        CONFIG("cmd_line", std::string(), "An approximation of the command-line arguments passed to the program.", SYSTEM);
        CONFIG("wd", file::Path(), "Working directory that the software was started from (defaults to the user directory).", SYSTEM);
        CONFIG("ffmpeg_path", file::Path(), "Path to an ffmpeg executable file. This is used for converting videos after recording them (from the GUI). It is not a critical component of the software, but mostly for convenience.");
        CONFIG("blobs_per_thread", 150.f, "Number of blobs for which properties will be calculated per thread.");
        CONFIG("individuals_per_thread", 1.f, "Number of individuals for which positions will be estimated per thread.");
        CONFIG("postures_per_thread", 1.f, "Number of individuals for which postures will be estimated per thread.");
        CONFIG("history_matching_log", file::Path(), "If this is set to a valid html file path, a detailed matching history log will be written to the given file for each frame.");
        CONFIG("filename", Path(""), "The converted video file (.pv file) or target for video conversion. Typically it would have the same basename as the video source (i.e. an MP4 file), but a different extension: pv.", LOAD);
        CONFIG("source", file::PathArray(), "This is the (video) source for the current session. Typically this would point to the original video source of `filename`.", LOAD);
        CONFIG("output_dir", Path(""), "Default output-/input-directory. Change this in order to omit paths in front of filenames for open and save.", INIT);
        CONFIG("data_prefix", Path("data"), "Subfolder (below `output_dir`) where the exported NPZ or CSV files will be saved (see `output_fields`).");
        CONFIG("settings_file", Path(""), "Name of the settings file. By default, this will be set to `filename`.settings in the same folder as `filename`.", LOAD);
        CONFIG("python_path", Path(COMMONS_PYTHON_EXECUTABLE), "Path to the python home folder" PYTHON_TIPPS ". If left empty, the user is required to make sure that all necessary libraries are in-scope the PATH environment variable.", STARTUP);

        CONFIG("frame_rate", uint32_t(0), "Specifies the frame rate of the video. It is used e.g. for playback speed and certain parts of the matching algorithm. Will be set by the metadata of the video. If you want to set a custom frame rate, different from the video metadata, you should set it during conversion. This guarantees that the timestamps generated will match up with your custom framerate during tracking.");
        CONFIG("track_enforce_frame_rate", true, "Enforce the `frame_rate` and override the frame_rate provided by the video file for calculating kinematic properties and probabilities. If this is not enabled, `frame_rate` is only a cosmetic property that influences the GUI and not exported data (for example).");
        
        CONFIG("calculate_posture", true, "Enables or disables posture calculation. Can only be set before the video is analysed (e.g. in a settings file or as a startup parameter).");
        
        CONFIG("meta_encoding", meta_encoding_t::rgb8, "The encoding used for the given .pv video.");
        CONFIG("detect_classes", cmn::blob::MaybeObjectClass_t{}, "Class names for object classification in video during conversion.");
        CONFIG("detect_skeleton", std::optional<blob::Pose::Skeletons>{}, "Skeleton to be used when displaying pose data. This is an optional map from classnames to ", PUBLIC, std::optional<std::optional<blob::Pose::Skeletons>>{std::optional<blob::Pose::Skeletons>{
            blob::Pose::Skeletons{
                ._skeletons = {{"human", std::vector<blob::Pose::Skeleton::Connection>{
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
                }}}
            }
        }});
        CONFIG("meta_source_path", std::string(""), "Path of the original video file for conversions (saved as debug info).", LOAD);
        CONFIG("meta_species", std::string(""), "Name of the species used.");
        CONFIG("meta_age_days", long_t(-1), "Age of the individuals used in days.");
        CONFIG("meta_conditions", std::string(""), "Treatment name.");
        CONFIG("meta_misc", std::string(""), "Other information.");
        CONFIG("meta_cmd", std::string(""), "Command-line of the framegrabber when conversion was started.", SYSTEM);
        CONFIG("meta_build", std::string(""), "The current commit hash. The video is branded with this information for later inspection of errors that might have occured.", SYSTEM);
        CONFIG("meta_video_size", Size2(), "Resolution of the original video.", LOAD);
        CONFIG("meta_video_scale", float(1), "Scale applied to the original video / footage.", LOAD);
        CONFIG("meta_conversion_time", std::string(""), "This contains the time of when this video was converted / recorded as a string.", LOAD);
        CONFIG("meta_real_width", Float2_t(0), "Used to calculate the `cm_per_pixel` conversion factor, relevant for e.g. converting the speed of individuals from px/s to cm/s (to compare to `track_max_speed` which is given in cm/s). By default set to 30 if no other values are available (e.g. via command-line). This variable should reflect actual width (in cm) of what is seen in the video image. For example, if the video shows a tank that is 50cm in X-direction and 30cm in Y-direction, and the image is cropped exactly to the size of the tank, then this variable should be set to 50.", INIT);
        CONFIG("cm_per_pixel", Float2_t(0), "The ratio of `meta_real_width / video_width` that is used to convert pixels to centimeters. Will be automatically calculated based on a meta-parameter saved inside the video file (`meta_real_width`) and does not need to be set manually.");
        CONFIG("video_length", uint64_t(0), "The length of the video in frames", LOAD);
        CONFIG("video_size", Size2(-1), "The dimensions of the currently loaded video.", LOAD);
        CONFIG("video_info", std::string(), "Information on the current video as provided by PV.", SYSTEM);
        
        /*
         * According to @citation the average zebrafish larvae weight would be >200mg after 9-week trials.
         * So this is most likely over-estimated.
         *
         * Siccardi AJ, Garris HW, Jones WT, Moseley DB, D’Abramo LR, Watts SA. Growth and Survival of Zebrafish (Danio rerio) Fed Different Commercial and Laboratory Diets. Zebrafish. 2009;6(3):275-280. doi:10.1089/zeb.2008.0553.
         */
        CONFIG("meta_mass_mg", float(200), "Used for exporting event-energy levels.");
        CONFIG("nowindow", false, "If set to true, no GUI will be created on startup (e.g. when starting from SSH).", STARTUP);
        CONFIG("track_background_subtraction", false, "If enabled, objects in .pv videos will first be contrasted against the background before thresholding (background_colors - object_colors). `track_threshold_is_absolute` then decides whether this term is evaluated in an absolute or signed manner.");
        CONFIG("use_differences", false, "This should be set to false unless when using really old files.");
        //config["debug_probabilities"] = false;
        CONFIG("track_pause", false, "Halts the analysis.");
        CONFIG("limit", 0.09f, "Limit for tailbeat event detection.");
        CONFIG("event_min_peak_offset", 0.15f, "");
        CONFIG("exec", file::Path(), "This can be set to the path of an additional settings file that is executed after the normal settings file.", STARTUP);
        CONFIG("log_file", file::Path(), "Set this to a path you want to save the log file to.", STARTUP);
        CONFIG("error_terminate", false, "", SYSTEM);
        CONFIG("terminate", false, "If set to true, the application terminates.", SYSTEM);
        
        //CONFIG("gui_transparent_background", false, "If enabled, fonts might look weird but you can record movies (and images) with transparent background (if gui_background_color.alpha is < 255).");
        
        CONFIG("gui_interface_scale", Float2_t(1), "Scales the whole interface. A value greater than 1 will make it smaller.", SYSTEM);
        CONFIG("gui_max_path_time", float(3), "Length (in time) of the trails shown in GUI.");
        
        CONFIG("gui_draw_only_filtered_out", false, "Only show filtered out blob texts.");
        CONFIG("gui_show_timeline", true, "If enabled, the timeline (top of the screen) will be shown in the tracking view.");
        CONFIG("gui_show_fish", std::tuple<pv::bid, Frame_t>{pv::bid::invalid, Frame_t()}, "Show debug output for {blob_id, fish_id}.");
        CONFIG("gui_source_video_frame", Frame_t(0u), "Best information the system has on which frame index in the original video the given `gui_frame` corresponds to (integrated into the pv file starting from V_9).", SYSTEM);
        CONFIG("gui_frame", Frame_t(0u), "The currently selected frame. `gui_displayed_frame` might differ, if loading from file is currently slow.");
        CONFIG("gui_show_skeletons", true, "Shows / hides keypoint data being shown in the graphical interface.");
        CONFIG("gui_displayed_frame", Frame_t(0u), "The currently visible frame.");
//#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
        CONFIG("gui_macos_blur", false, "MacOS supports a blur filter that can be applied to make unselected individuals look more interesting. Purely a visual effect. Does nothing on other operating systems.");
//#endif
        CONFIG("gui_faded_brightness", uchar(255), "The alpha value of tracking-related elements when timeline is hidden (0-255).");
        CONFIG("gui_equalize_blob_histograms", false, "Equalize histograms of blobs wihtin videos (makes them more visible).");
        CONFIG("gui_show_video_background", true, "If available, show an animated background of the original video.");
        CONFIG("gui_show_heatmap", false, "Showing a heatmap per identity, normalized by maximum samples per grid-cell.");
        CONFIG("gui_show_individual_preview", true, "Shows preview images for all selected individuals as they would be processed during network training, based on settings like `individual_image_size`, `individual_image_scale` and `individual_image_normalization`.");
        CONFIG("gui_draw_blobs_separately", false, "Draw blobs separately. If false, blobs will be drawn on a single full-screen texture and displayed. The second option may be better on some computers (not supported if `gui_macos_blur` is set to true).");
        CONFIG("gui_blob_label", std::string("{if:{dock}:{name} :''}{if:{active}:<a>:''}{real_size}{if:{split}: <gray>split</gray>:''}{if:{tried_to_split}: <orange>split tried</orange>:''}{if:{prediction}: {prediction}:''}{if:{instance}: <gray>instance</gray>:''}{if:{dock}:{if:{filter_reason}: [<gray>{filter_reason}</gray>]:''}:''}{if:{active}:</a>:''}{if:{category}: {category}:''}"), "This is what the graphical user interfaces displays as a label for each blob in raw view. Replace this with {help} to see available variables.");
        CONFIG("gui_fish_label", std::string("{if:{not:{has_pred}}:{name}:{if:{equal:{at:0:{max_pred}}:{id}}:<green>{name}</green>:<red>{name}</red> <i>loc</i>[<c><nr>{at:0:{max_pred}}</nr>:<nr>{int:{*:100:{at:1:{max_pred}}}}</nr><i>%</i></c>]}}{if:{tag}:' <a>tag:{tag.id} ({dec:2:{tag.p}})</a>':''}{if:{average_category}:' <nr>{average_category}</nr>':''}{if:{&&:{category}:{not:{equal:{category}:{average_category}}}}:' <b><i>{category}</i></b>':''}"), "This is what the graphical user interface displays as a label for each individual. Replace this with {help} to see the available variables.");
        CONFIG("heatmap_ids", std::vector<track::Idx_t>(), "Add ID numbers to this array to exclusively display heatmap values for those individuals.");
        CONFIG("heatmap_value_range", Range<double>(-1, -1), "Give a custom value range that is used to normalize heatmap cell values.");
        CONFIG("heatmap_smooth", double(0.05), "Value between 0 and 1, think of as `heatmap_smooth` times video-width, indicating the maximum upscaled size of the heatmaps shown in the tracker. Makes them prettier, but maybe much slower.");
        CONFIG("heatmap_normalization", heatmap_normalization_t::cell, "Normalization used for the heatmaps. If `value` is selected, then the maximum of all values encountered will be used to normalize the average of each cell. If `cell` is selected, the sum of each cell will be divided by the maximum cell value encountered.");
        CONFIG("heatmap_frames", uint32_t(0), "If `heatmap_dynamic` is enabled, this variable determines the range of frames that are considered. If set to 0, all frames up to the current frame are considered. Otherwise, this number determines the number of frames previous to the current frame that are considered.");
        CONFIG("heatmap_dynamic", false, "If enabled the heatmap will only show frames before the frame currently displayed in the graphical user interface.");
        CONFIG("heatmap_resolution", uint32_t(64), "Square resolution of individual heatmaps displayed with `gui_show_heatmap`. Will generate a square grid, each cell with dimensions (video_width / N, video_height / N), and sort all positions of each identity into it.");
        CONFIG("heatmap_source", std::string(), "If empty, the source will simply be an individuals identity. Otherwise, information from export data sources will be used.");
        CONFIG("gui_mode", gui::mode_t::tracking, "The currently used display mode for the GUI.");
        CONFIG("panic_button", int(0), "42");
        CONFIG("gui_run", false, "When set to true, the GUI starts playing back the video and stops once it reaches the end, or is set to false.");
        CONFIG("gui_show_match_modes", false, "Shows the match mode used for every tracked object. Green is 'approximate', yellow is 'hungarian', and red is 'created/loaded'.");
        CONFIG("gui_show_only_unassigned", false, "Showing only unassigned objects.");
        CONFIG("gui_show_memory_stats", false, "Showing or hiding memory statistics.");
        CONFIG("gui_show_outline", true, "Showing or hiding individual outlines in tracking view.");
        CONFIG("gui_show_midline", true, "Showing or hiding individual midlines in tracking view.");
        CONFIG("gui_show_shadows", true, "Showing or hiding individual shadows in tracking view.");
        CONFIG("gui_outline_thickness", uint8_t(1), "The thickness of outline / midlines in the GUI.");
        CONFIG("gui_show_texts", true, "Showing or hiding individual identity (and related) texts in tracking view.");
        CONFIG("gui_show_matching_info", true, "Showing or hiding probabilities for relevant blobs in the info card if an individual is selected.");
        CONFIG("gui_show_misc_metrics", true, "Showing or hiding some metrics for a selected individual in the info card.");
        CONFIG("gui_show_autoident_controls", false, "Showing or hiding controls for removing forced auto-ident in the info card if an individual is selected.");
        CONFIG("gui_show_infocard", true, "Showing / hiding some facts about the currently selected individual on the top left of the window.");
        CONFIG("gui_show_timing_stats", false, "Showing / hiding rendering information.");
        CONFIG("gui_show_blobs", true, "Showing or hiding individual raw blobs in tracking view (are always shown in RAW mode).");
        CONFIG("gui_show_paths", true, "Equivalent to the checkbox visible in GUI on the bottom-left.");
        CONFIG("gui_show_pixel_grid", false, "Shows the proximity grid generated for all blobs, which is used for history splitting.");
        CONFIG("gui_show_selections", true, "Show/hide circles around selected individual.");
        CONFIG("gui_show_inactive_individuals", false, "Show/hide individuals that have not been seen for longer than `track_max_reassign_time`.");
        //config["gui_show_texts"] = true;
        CONFIG("gui_show_histograms", false, "Equivalent to the checkbox visible in GUI on the bottom-left.");
        CONFIG("gui_show_posture", false, "Show/hide the posture window on the top-right.");
        CONFIG("gui_show_export_options", false, "Show/hide the export options widget.");
        CONFIG("gui_show_visualfield_ts", false, "Show/hide the visual field time series.");
        CONFIG("gui_show_visualfield", false, "Show/hide the visual field rays.");
        CONFIG("gui_show_uniqueness", false, "Show/hide uniqueness overview after training.");
        CONFIG("gui_show_probabilities", false, "Show/hide probability visualisation when an individual is selected.");
        CONFIG("gui_show_cliques", false, "Show/hide cliques of potentially difficult tracking situations.");
        //CONFIG("gui_show_manual_matches", true, "Show/hide manual matches in path.");
        CONFIG("gui_show_graph", false, "Show/hide the data time-series graph.");
        CONFIG("gui_show_number_individuals", false, "Show/hide the #individuals time-series graph.");
        CONFIG("gui_show_processing_time", false, "Show/hide the ms/frame time-series graph.");
        CONFIG("gui_show_recognition_summary", false, "Show/hide confusion matrix (if network is loaded).");
        CONFIG("gui_show_dataset", false, "Show/hide detailed dataset information on-screen.");
        CONFIG("gui_show_recognition_bounds", true, "Shows what is contained within tht recognition boundary as a cyan background. (See `recognition_border` for details.)");
        CONFIG("gui_show_boundary_crossings", true, "If set to true (and the number of individuals is set to a number > 0), the tracker will show whenever an individual enters the recognition boundary. Indicated by an expanding cyan circle around it.");
        CONFIG("gui_show_detailed_probabilities", false, "Show/hide detailed probability stats when an individual is selected.");
        CONFIG("gui_playback_speed", float(1.f), "Playback speed when pressing SPACE.");
        CONFIG("gui_wait_for_background", true, "Sacrifice video playback speed to wait for the background video the load in. This only applies if the background is actually displayed (`gui_show_video_background`).");
        CONFIG("gui_wait_for_pv", true, "Sacrifice video playback speed to wait for the pv file the load in.");
        CONFIG("gui_show_midline_histogram", false, "Displays a histogram for midline lengths.");
        CONFIG("gui_auto_scale", false, "If set to true, the tracker will always try to zoom in on the whole group. This is useful for some individuals in a huge video (because if they are too tiny, you cant see them and their posture anymore).");
        CONFIG("gui_auto_scale_focus_one", true, "If set to true (and `gui_auto_scale` set to true, too), the tracker will zoom in on the selected individual, if one is selected.");
        CONFIG("gui_pose_smoothing", Frame_t(0), "Blending between the current and previous / future frames for displaying smoother poses in the graphical user-interface. This does not affect data output.");
        CONFIG("gui_timeline_alpha", uchar(200), "Determines the Alpha value for the timeline / tracklets display.");
        CONFIG("gui_background_color", gui::Color(0,0,0,255), "Values < 255 will make the background (or video background) more transparent in standard view. This might be useful with very bright backgrounds.");
        CONFIG("gui_fish_color", std::string("identity"), "");
        CONFIG("gui_single_identity_color", gui::Transparent, "If set to something else than transparent, all individuals will be displayed with this color.");
        CONFIG("gui_zoom_limit", Size2(300, 300), "");
        CONFIG("gui_zoom_polygon", std::vector<Vec2>{}, "If this is non-empty, the view will be zoomed in on the center of the polygon with approximately the dimensions of the polygon.");

#ifdef __APPLE__
        auto default_recording_t = gui_recording_format_t::mp4;
#else
        auto default_recording_t = gui_recording_format_t::mp4;
#endif

        CONFIG("gui_recording_format", default_recording_t, "Sets the format for recording mode (when R is pressed in the GUI). Supported formats are 'avi', 'jpg' and 'png'. JPEGs have 75%% compression, AVI is using MJPEG compression.");
        CONFIG("gui_is_recording", false, "Is set to true when recording is active.", SYSTEM);
        CONFIG("gui_happy_mode", false, "If `calculate_posture` is enabled, enabling this option likely improves your experience with TRex.");
        CONFIG("individual_names", std::map<track::Idx_t, std::string>{}, "A map of `{individual-id: \"individual-name\", ...}` that names individuals in the GUI and exported data.");
        CONFIG("individual_prefix", std::string("id"), "The prefix that is added to all the files containing certain IDs. So individual 0 will turn into '[prefix]0' for all the npz files and within the program.");
        CONFIG("outline_approximate", uint8_t(3), "If this is a number > 0, the outline detected from the image will be passed through an elliptical fourier transform with `outline_approximate` number of coefficients. When the given number is sufficiently low, the outline will be smoothed significantly (and more so for lower numbers of coefficients).");
        CONFIG("outline_smooth_step", uint8_t(1), "Jump over N outline points when smoothing (reducing accuracy).");
        CONFIG("outline_smooth_samples", uint8_t(4), "Use N samples for smoothing the outline. More samples will generate a smoother (less detailed) outline.");
        CONFIG("outline_curvature_range_ratio", float(0.03), "Determines the ratio between number of outline points and distance used to calculate its curvature. Program will look at index +- `ratio * size()` and calculate the distance between these points (see posture window red/green color).");
        CONFIG("midline_walk_offset", float(0.025), "This percentage of the number of outline points is the amount of points that the midline-algorithm is allowed to move left and right upon each step. Higher numbers will make midlines more straight, especially when extremities are present (that need to be skipped over), but higher numbers will also potentially decrease accuracy for less detailed objects.");
        CONFIG("midline_stiff_percentage", float(0.15), "Percentage of the midline that can be assumed to be stiff. If the head position seems poorly approximated (straighened out too much), then decrease this value.");
        CONFIG("midline_resolution", uint32_t(25), "Number of midline points that are saved. Higher number increases detail.");
        CONFIG("posture_head_percentage", float(0.1), "The percentage of the midline-length that the head is moved away from the front of the body.");
        CONFIG("posture_closing_steps", uint8_t(0), "When enabled (> 0), posture will be processed using a combination of erode / dilate in order to close holes in the shape and get rid of extremities. An increased number of steps will shrink the shape, but will also be more time intensive.");
        CONFIG("posture_closing_size", uint8_t(2), "The kernel size for erosion / dilation of the posture algorithm. Only has an effect with  `posture_closing_steps` > 0.");
        CONFIG("outline_resample", float(1), "Spacing between outline points in pixels (`0<value<255`), after resampling the outline. A lower value here can drastically increase the number of outline points being generated (and decrease analysis speed), while a higher value is going to do the opposite. By default this value is 1-pixel, meaning that there is no artificial interpolation or down-sampling.");
        CONFIG("outline_use_dft", true, "If enabled, the program tries to reduce outline noise by convolution of the curvature array with a low pass filter.");
        CONFIG("midline_start_with_head", false, "If enabled, the midline is going to be estimated starting at the head instead of the tail.");
        CONFIG("midline_invert", false, "If enabled, all midlines will be inverted (tail/head swapped).");
        CONFIG("peak_mode", peak_mode_t::pointy, "This determines whether the tail of an individual should be expected to be pointy or broad.");
        CONFIG("manual_matches", std::map<Frame_t, std::map<track::Idx_t, pv::bid>>{ }, "A map of manually defined matches (also updated by GUI menu for assigning manual identities). `{{frame: {fish0: blob2, fish1: blob0}}, ...}`");
        CONFIG("manual_splits", std::map<Frame_t, std::set<pv::bid>>{}, "This map contains `{frame: [blobid1,blobid2,...]}` where frame and blobid are integers. When this is read during tracking for a frame, the tracker will attempt to force-split the given blob ids.");
        CONFIG("track_ignore_bdx", std::map<Frame_t, std::set<pv::bid>>{}, "This is a map of frame -> [bdx0, bdx1, ...] of blob ids that are specifically set to be ignored in the given frame. Can be reached using the GUI by clicking on a blob in raw mode.");
        CONFIG("match_mode", matching_mode_t::automatic, "Changes the default algorithm to be used for matching blobs in one frame with blobs in the next frame. The accurate algorithm performs best, but also scales less well for more individuals than the approximate one. However, if it is too slow (temporarily) in a few frames, the program falls back to using the approximate one that doesnt slow down.");
        CONFIG("match_min_probability", float(0.1), "The probability below which a possible connection between blob and identity is considered too low. The probability depends largely upon settings like `track_max_speed`.");
        CONFIG("track_do_history_split", true, "If disabled, blobs will not be split automatically in order to separate overlapping individuals. This usually happens based on their history.");
        CONFIG("track_history_split_threshold", Frame_t(), "If this is greater than 0, then individuals with tracklets < this threshold will not be considered for the splitting algorithm. That means that objects have to be detected for at least `N` frames in a row to play a role in history splitting.");
        CONFIG("tracklet_punish_speeding", true, "Sometimes individuals might be assigned to blobs that are far away from the previous position. This could indicate wrong assignments, but not necessarily. If this variable is set to true, tracklets will end whenever high speeds are reached, just to be on the safe side. For scenarios with lots of individuals (and no recognition) this might spam yellow bars in the timeline and may be disabled.");
        CONFIG("track_consistent_categories", false, "Utilise categories (if present) when tracking. This may break trajectories in places with imperfect categorization, but only applies once categories have been applied.");
        CONFIG("track_max_individuals", uint32_t(1024), "The maximal number of individual that are assigned at the same time (infinite if set to zero). If the given number is below the actual number of individual, then only a (random) subset of individual are assigned and a warning is shown.");
        CONFIG("detect_size_filter", SizeFilters(), "During the detection phase, objects outside this size range will be filtered out. If empty, no objects will be filtered out.");
        CONFIG("track_size_filter", SizeFilters(), "Blobs below the lower bound are recognized as noise instead of individuals. Blobs bigger than the upper bound are considered to potentially contain more than one individual. You can look these values up by pressing `D` in TRex to get to the raw view (see `https://trex.run/docs/gui.html` for details). The unit is #pixels * (cm/px)^2. `cm_per_pixel` is used for this conversion.");
        CONFIG("blob_split_max_shrink", float(0.2), "The minimum percentage of the starting blob size (after thresholding), that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.");
        CONFIG("blob_split_global_shrink_limit", float(0.2), "The minimum percentage of the minimum in `track_size_filter`, that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.");
        CONFIG("blob_split_algorithm", blob_split_algorithm_t::threshold, "The default splitting algorithm used to split objects that are too close together.");
        
        CONFIG("output_visual_fields", false, "Export visual fields for all individuals upon saving.");
        CONFIG("visual_field_shapes", std::vector<std::vector<Vec2>>{}, "A list of shapes that should be handled as view-blocking in visual field calculations.");
        CONFIG("visual_field_eye_offset", float(0.15), "A percentage telling the program how much the eye positions are offset from the start of the midline.");
        CONFIG("visual_field_eye_separation", float(60), "Degrees of separation between the eye and looking straight ahead. Results in the eye looking towards head.angle +- `visual_field_eye_separation`.");
        CONFIG("visual_field_history_smoothing", uint8_t(0), "The maximum number of previous values (and look-back in frames) to take into account when smoothing visual field orientations. If greater than 0, visual fields will use smoothed previous eye positions to determine the optimal current eye position. This is usually only necessary when postures are somewhat noisy to a degree that makes visual fields unreliable.");
        
        CONFIG("auto_minmax_size", false, "Program will try to find minimum / maximum size of the individuals automatically for the current `cm_per_pixel` setting. Can only be passed as an argument upon startup. The calculation is based on the median blob size in the video and assumes a relatively low level of noise.", STARTUP);
        CONFIG("auto_number_individuals", false, "Program will automatically try to find the number of individuals (with sizes given in `track_size_filter`) and set `track_max_individuals` to that value.");
        
        CONFIG("track_speed_decay", float(1.0), "The amount the expected speed is reduced over time when an individual is lost. When individuals collide, depending on the expected behavior for the given species, one should choose different values for this variable. If the individuals usually stop when they collide, this should be set to 1. If the individuals are expected to move over one another, the value should be set to `0.7 > value > 0`.");
        CONFIG("track_max_speed", Float2_t(0), "The maximum speed an individual can have (=> the maximum distance an individual can travel within one second) in cm/s. Uses and is influenced by `meta_real_width` and `cm_per_pixel` as follows: `speed(px/s) * cm_per_pixel(cm/px) -> cm/s`.");
        CONFIG("posture_direction_smoothing", uint16_t(0), "Enables or disables smoothing of the posture orientation based on previous frames (not good for fast turns).");
        CONFIG("speed_extrapolation", float(3), "Used for matching when estimating the next position of an individual. Smaller values are appropriate for lower frame rates. The higher this value is, the more previous frames will have significant weight in estimating the next position (with an exponential decay).");
        CONFIG("track_intensity_range", Rangel(-1, -1), "When set to valid values, objects will be filtered to have an average pixel intensity within the given range.");
        CONFIG("track_threshold", int(0), "Constant used in background subtraction. Pixels with grey values above this threshold will be interpreted as potential individuals, while pixels below this threshold will be ignored.");
        CONFIG("threshold_ratio_range", Rangef(0.5, 1.0), "If `track_threshold_2` is not equal to zero, this ratio will be multiplied by the number of pixels present before the second threshold. If the resulting size falls within the given range, the blob is deemed okay.");
        CONFIG("track_threshold_2", int(0), "If not zero, a second threshold will be applied to all objects after they have been deemed do be theoretically large enough. Then they are compared to #before_pixels * `threshold_ratio_range` to see how much they have been shrunk).");
        CONFIG("track_posture_threshold", int(0), "Same as `track_threshold`, but for posture estimation.");
        CONFIG("track_threshold_is_absolute", true, "If enabled, uses absolute difference values and disregards any pixel |p| < `threshold` during conversion. Otherwise the equation is p < `threshold`, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as `detect_threshold_is_absolute`, but during tracking instead of converting.");
        CONFIG("track_time_probability_enabled", bool(true), "");
        CONFIG("track_max_reassign_time", float(0.5), "Distance in time (seconds) where the matcher will stop trying to reassign an individual based on previous position. After this time runs out, depending on the settings, the tracker will try to find it based on other criteria, or generate a new individual.");
        
        CONFIG("gui_highlight_categories", false, "If enabled, categories (if applied in the video) will be highlighted in the tracking view.");
        CONFIG("categories_ordered", std::vector<std::string>{}, "Ordered list of names of categories that are used in categorization (classification of types of individuals).");
        CONFIG("categories_train_min_tracklet_length", uint32_t(50), "Minimum number of images for a sample to be considered relevant for training categorization. Will default to 50, meaning all tracklets longer than that will be presented for training.");
        CONFIG("categories_apply_min_tracklet_length", uint32_t(0), "Minimum number of images for a sample to be considered relevant when applying the categorization. This defaults to 0, meaning all samples are valid. If set to anything higher, only tracklets with more than N frames will be processed.");
        CONFIG("tracklet_max_length", float(0), "If set to something bigger than zero, this represents the maximum number of seconds that a tracklet can be.");
        
        CONFIG("track_only_segmentations", false, "If this is enabled, only segmentation results will be tracked - this avoids double tracking of bounding boxes and segmentation masks.");
        CONFIG("track_only_categories", std::vector<std::string>{}, "If this is a non-empty list, only objects that have previously been assigned one of the correct categories will be tracked. Note that this also affects anything below `categories_apply_min_tracklet_length` length (e.g. noise particles or short tracklets).");
        CONFIG("track_only_classes", std::vector<std::string>{}, "If this is a non-empty list, only objects that have any of the given labels (assigned by a ML network during video conversion) will be tracked.");
        CONFIG("track_conf_threshold", 0.1_F, "During tracking, detections with confidence levels below the given fraction (0-1) for labels (assigned by an ML network during video conversion) will be discarded. These objects will not be assigned to any individual.");
        
        CONFIG("web_quality", int(75), "JPEG quality of images transferred over the web interface.");
        CONFIG("web_time_threshold", float(0.050), "Maximum refresh rate in seconds for the web interface.");
        
        CONFIG("correct_illegal_lines", false, "In older versions of the software, blobs can be constructed in 'illegal' ways, meaning the lines might be overlapping. If the software is printing warnings about it, this should probably be enabled (makes it slower).");
        CONFIG("evaluate_thresholds", false, "This option, if enabled, previews the effects of all possible thresholds when applied to the given video. These are shown as a graph in a separate window. Can be used to debug parameters instead of try-and-error. Might take a few minutes to finish calculating.", STARTUP);
        
        auto output_fields = std::vector<std::pair<std::string, std::vector<std::string>>>
        {
            {"X", {"RAW", "WCENTROID"}},
            {"Y", {"RAW", "WCENTROID"}},
            {"X", {"RAW", "HEAD"}},
            {"Y", {"RAW", "HEAD"}},
            {"VX", {"RAW", "HEAD"}},
            {"VY", {"RAW", "HEAD"}},
            {"AX", {"RAW", "HEAD"}},
            {"AY", {"RAW", "HEAD"}},
            {"ANGLE", {"RAW"}},
            {"ANGULAR_V", {"RAW"}},
            {"ANGULAR_A", {"RAW"}},
            {"MIDLINE_OFFSET", {"RAW"}},
            {"normalized_midline", {"RAW"}},
            {"midline_length", {"RAW"}},
            {"midline_x", {"RAW"}},
            {"midline_y", {"RAW"}},
            {"midline_segment_length", {"RAW"}},
            {"SPEED", {"RAW", "WCENTROID"}},
            //{"SPEED", {"SMOOTH", "WCENTROID"}},
            {"SPEED", {"RAW", "PCENTROID"}},
            //{"SPEED", {"SMOOTH", "PCENTROID"}},
            {"SPEED", {"RAW", "HEAD"}},
            //{"SPEED", {"SMOOTH", "HEAD"}},
            //{"NEIGHBOR_DISTANCE", {"RAW"}},
            {"BORDER_DISTANCE", {"PCENTROID"}},
            {"time", {}},{"timestamp", {}},
            {"frame", {}},
            {"missing", {}},
            {"num_pixels", {}},
            {"ACCELERATION", {"RAW", "PCENTROID"}},
            //{"ACCELERATION", {"SMOOTH", "PCENTROID"}},
            {"ACCELERATION", {"RAW", "WCENTROID"}}
        };
        
        auto output_annotations = std::map<std::string, std::string>
        {
            {"X", "cm"}, {"Y", "cm"},
            {"VX", "cm/s"},{"VY", "cm/s"},
            {"SPEED", "cm/s"},{"SPEED_SMOOTH", "cm/s"},{"SPEED_OLD", "cm/s"},
            {"ACCELERATION", "cm/s2"}, {"ACCELERATION_SMOOTH", "cm/s2"},
            {"ORIENTATION", "rad"},
            {"BORDER_DISTANCE", "cm"},
            {"NEIGHBOR_DISTANCE", "cm"},
            {"global", "px"}
        };
        
        auto output_default_options = default_options_type
        {
            {"NEIGHBOR_DISTANCE", {"/10"}},
            {"DOT_V", {"/10"}},
            {"L_V", {"/10"}},
            {"v_direction", {"/10"}},
            {"event_acceleration", {"/10"}},
            {"SPEED", {"/10"}},
            {"ANGULAR_V", {"/10", "CENTROID"}},
            {"ANGULAR_A", {"/1000", "CENTROID"}},
            {"ACCELERATION", {"/15", "CENTROID"}},
            {"NEIGHBOR_VECTOR_T", {"/1"}},
            {"X", {"/100"}},
            {"Y", {"/100"}},
            {"tailbeat_threshold", {"pm"}},
            {"tailbeat_peak", {"pm"}},
            {"threshold_reached", {"POINTS"}},
            {"midline_length", {"/15"}},
            {"amplitude", {"/100"}},
            {"outline_size", {"/100"}},
            {"global", {"/10"}}
        };
        
        CONFIG("task", TRexTask_t::none, "The task selected by the user upon startup. This is used to determine which GUI mode to start in.", STARTUP);
        //CONFIG("load", false, "If set to true, the application will attempt to load results for the given pv file. If it does not exist then the application will proceed as usual.", LOAD);
        CONFIG("auto_quit", false, "If set to true, the application will automatically save all results and export CSV files and quit, after the analysis is complete."); // save and quit after analysis is done
        CONFIG("auto_apply", false, "If set to true, the application will automatically apply the network with existing weights once the analysis is done. It will then automatically correct and reanalyse the video.");
        CONFIG("auto_categorize", false, "If set to true, the program will try to load <video>_categories.npz from the `output_dir`. If successful, then categories will be computed according to the current categories_ settings. Combine this with the `auto_quit` parameter to automatically save and quit afterwards. If weights cannot be loaded, the app crashes.");
        CONFIG("auto_tags", false, "If set to true, the application will automatically apply available tag information once the results file has been loaded. It will then automatically correct potential tracking mistakes based on this information.");
        CONFIG("auto_tags_on_startup", false, "Used internally by the software.", SYSTEM);
        CONFIG("auto_no_memory_stats", true, "If set to true, no memory statistics will be saved on auto_quit.");
        CONFIG("auto_no_results", false, "If set to true, the auto_quit option will NOT save a .results file along with the NPZ (or CSV) files. This saves time and space, but also means that the tracked portion cannot be loaded via -load afterwards. Useful, if you only want to analyse the resulting data and never look at the tracked video again.");
        CONFIG("auto_no_tracking_data", false, "If set to true, the auto_quit option will NOT save any `output_fields` tracking data - just the posture data (if enabled) and the results file (if not disabled). This saves time and space if that is a need.");
        CONFIG("auto_no_outputs", false, "If set to true, no data will be exported upon `auto_quit`. Not even a .settings file will be saved.");
        CONFIG("auto_train", false, "If set to true, the application will automatically train the recognition network with the best track tracklet and apply it to the video.");
        CONFIG("auto_train_on_startup", false, "This is a parameter that is used by the system to determine whether `auto_train` was set on startup, and thus also whether a failure of `auto_train` should result in a crash (return code != 0).", SYSTEM);
        CONFIG("analysis_range", Range<long_t>(-1, -1), "Sets start and end of the analysed frames.");
        CONFIG("output_min_frames", uint16_t(1), "Filters all individual with less than N frames when exporting. Individuals with fewer than N frames will also be hidden in the GUI unless `gui_show_inactive_individuals` is enabled (default).");
        CONFIG("output_interpolate_positions", bool(false), "If turned on this function will linearly interpolate X/Y, and SPEED values, for all frames in which an individual is missing.");
        CONFIG("output_prefix", std::string(), "If this is not empty, all output files will go into `output_dir` / `output_prefix` / ... instead of just into `output_dir`. The output directory is usually the folder where the video is, unless set to a different folder by you.");
        CONFIG("output_auto_pose", true, "If this is set to false, then no poseX[n] and poseY[n] fields will automatically be added to the `output_fields` based on what the keypoint model reports. You can still manually add them if you like.");
        CONFIG("output_auto_detection_fields", true, "If set to true then this will automatically add fields like `detection_p` to the output files saved by TRex. You can also set this to false and add them manually if you like.");
        CONFIG("output_fields", output_fields, "The functions that will be exported when saving to CSV, or shown in the graph. `[['X',[option], ...]]`");
        CONFIG("tracklet_force_normal_color", true, "If set to true (default) then all images are saved as they appear in the original video. Otherwise, all images are exported according to the individual image settings (as seen in the image settings when an individual is selected) - in which case the background may have been subtracted from the original image and a threshold may have been applied (if `track_threshold` > 0 and `track_background_subtraction` is true).");
        CONFIG("tracklet_max_images", uint16_t(0), "This limits the maximum number of images that are being exported per tracklet given that `output_tracklet_images` is true. If the number is 0 (default), then every image will be exported. Otherwise, only a uniformly sampled subset of N images will be exported.");
        CONFIG("tracklet_normalize", true, "If enabled, all exported tracklet images are normalized according to the `individual_image_normalization` and padded / shrunk to `individual_image_size` (they appear as they do in the image preview when selecting an individual in the GUI).");
        CONFIG("output_tracklet_images", false, "If set to true, the program will output one median image per tracklet (time-series segment) and save it alongside the npz/csv files (inside `<filename>_tracklet_images.npz`). It will also output (if `tracklet_max_images` is 0) all images of each tracklet in a separate npz files named `<filename>_tracklet_images_single_*.npz`.");
        CONFIG("output_csv_decimals", uint8_t(2), "Maximum number of decimal places that is written into CSV files (a text-based format for storing data). A value of 0 results in integer values.");
        CONFIG("output_invalid_value", output_invalid_t::inf, "Determines, what is exported in cases where the individual was not found (or a certain value could not be calculated). For example, if an individual is found but posture could not successfully be generated, then all posture-based values (e.g. `midline_length`) default to the value specified here. By default (and for historic reasons), any invalid value is marked by 'inf'.");
        CONFIG("output_format", output_format_t::npz, "When pressing the S(ave) button or using `auto_quit`, this setting allows to switch between CSV and NPZ output. NPZ files are recommended and will be used by default - some functionality (such as visual fields, posture data, etc.) will remain in NPZ format due to technical constraints.");
        CONFIG("output_heatmaps", false, "When set to true, heatmaps are going to be saved to a separate file, or set of files '_p*' - with all the settings in heatmap_* applied.");
        CONFIG("output_statistics", false, "Save an NPZ file containing an array with shape Nx16 and contents [`adding_seconds`, `combined_posture_seconds`, `number_fish`, `loading_seconds`, `posture_seconds`, `match_number_fish`, `match_number_blob`, `match_number_edges`, `match_stack_objects`, `match_max_edges_per_blob`, `match_max_edges_per_fish`, `match_mean_edges_per_blob`, `match_mean_edges_per_fish`, `match_improvements_made`, `match_leafs_visited`, `method_used`] and an 1D-array containing all frame numbers. If set to true, a file called '`output_dir`/`fish_data_dir`/`<filename>_statistics.npz`' will be created. This will not output anything interesting, if the data was loaded instead of analysed.");
        CONFIG("output_posture_data", false, "Save posture data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '`output_dir`/`fish_data_dir`/`<filename>_posture_fishXXX.npz`' will be created for each individual XXX.");
        CONFIG("output_recognition_data", false, "Save recognition / probability data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '`output_dir`/`fish_data_dir`/`<filename>_recognition_fishXXX.npz`' will be created for each individual XXX.");
        CONFIG("output_normalize_midline_data", false, "If enabled: save a normalized version of the midline data saved whenever `output_posture_data` is set to true. Normalized means that the position of the midline points is normalized across frames (or the distance between head and point n in the midline array).");
        CONFIG("output_centered", false, "If set to true, the origin of all X and Y coordinates is going to be set to the center of the video. Using this overrides `output_origin`.");
        CONFIG("output_origin", Vec2(0), "When exporting the data, positions will be relative to this point - unless `output_centered` is set, which takes precedence.");
        CONFIG("output_default_options", output_default_options, "Default scaling and smoothing options for output functions, which are applied to functions in `output_fields` during export.");
        CONFIG("output_annotations", output_annotations, "Units (as a string) of output functions to be annotated in various places like graphs.");
        CONFIG("output_frame_window", uint32_t(100), "If an individual is selected during CSV output, use these number of frames around it (or -1 for all frames).");
        CONFIG("smooth_window", uint32_t(2), "Smoothing window used for exported data with the #smooth tag.");
        
        CONFIG("tags_path", file::Path(""), "If this path is set, the program will try to find tags and save them at the specified location.");
        CONFIG("tags_image_size", Size2(32, 32), "The image size that tag images are normalized to.");
        CONFIG("tags_dont_track", true, "If true, disables the tracking of tags as objects in TRex. This means that tags are not displayed like other objects and are instead only used as additional 'information' to correct tracks. However, if you enabled `tags_saved_only` in TGrabs, setting this parameter to true will make your TRex look quite empty.");
        //CONFIG("correct_luminance", true, "", STARTUP);
        
        CONFIG("grid_points", std::vector<Vec2>{}, "Whenever there is an identification network loaded and this array contains more than one point `[[x0,y0],[x1,y1],...]`, then the network will only be applied to blobs within circles around these points. The size of these circles is half of the average distance between the points.");
        CONFIG("grid_points_scaling", float(0.8), "Scaling applied to the average distance between the points in order to shrink or increase the size of the circles for recognition (see `grid_points`).");
        CONFIG("accumulation_tracklet_add_factor", 1.5_F, "This factor will be multiplied with the probability that would be pure chance, during the decision whether a tracklet is to be added or not. The default value of 1.5 suggests that the minimum probability for each identity has to be 1.5 times chance (e.g. 0.5 in the case of two individuals).");
        CONFIG("recognition_save_progress_images", false, "If set to true, an image will be saved for all training epochs, documenting the uniqueness in each step.");
        CONFIG("recognition_shapes", std::vector<std::vector<Vec2>>(), "If `recognition_border` is set to 'shapes', then the identification network will only be applied to blobs within the convex shapes specified here.");
        CONFIG("recognition_border", track::recognition_border_t::none, "This defines the type of border that is used in all automatic recognition routines. Depending on the type set here, you might need to set other parameters as well (e.g. `recognition_shapes`). In general, this defines whether an image of an individual is usable for automatic recognition. If it is inside the defined border, then it will be passed on to the recognition network - if not, then it wont."
        );
        CONFIG("debug_recognition_output_all_methods", false, "If set to true, a complete training will attempt to output all images for each identity with all available normalization methods.");
        CONFIG("recognition_border_shrink_percent", float(0.3), "The amount by which the recognition border is shrunk after generating it (roughly and depends on the method).");
        CONFIG("recognition_border_size_rescale", float(0.5), "The amount that blob sizes for calculating the heatmap are allowed to go below or above values specified in `track_size_filter` (e.g. 0.5 means that the sizes can range between `track_size_filter.min * (1 - 0.5)` and `track_size_filter.max * (1 + 0.5)`).");
        CONFIG("recognition_smooth_amount", uint16_t(200), "If `recognition_border` is 'outline', this is the amount that the `recognition_border` is smoothed (similar to `outline_smooth_samples`), where larger numbers will smooth more.");
        CONFIG("recognition_coeff", uint16_t(50), "If `recognition_border` is 'outline', this is the number of coefficients to use when smoothing the `recognition_border`.");
        CONFIG("individual_image_normalization", individual_image_normalization_t::posture, "This enables or disable normalizing the images before training. If set to `none`, the images will be sent to the GPU raw - they will only be cropped out. Otherwise they will be normalized based on head orientation (posture) or the main axis calculated using `image moments`.");
        CONFIG("pose_midline_indexes", track::PoseMidlineIndexes{.indexes = {}}, "This is an array of joint indexes (in the order as predicted by a YOLO-pose model), which are used to determine the joints making up the midline of an object. The first index is the head, the last the tail. This is used to generate a posture when using YOLO-pose models with `calculate_posture` enabled.");
        CONFIG("individual_image_size", Size2(80, 80), "Size of each image generated for network training.");
        CONFIG("individual_image_scale", float(1), "Scaling applied to the images before passing them to the network.");
        CONFIG("visual_identification_model_path", std::optional<file::Path>{}, "If this is set to a path, visual identification 'load weights' or 'apply' will try to load this path first if it exists. This way you can facilitate transfer learning (taking a model file from one video and applying it to a different video of the same individuals).");
        CONFIG("visual_identification_save_images", false, "If set to true, the program will save the images used for a successful training of the visual identification to `output_dir`.");
        CONFIG("visual_identification_version", visual_identification_version_t::v118_3, "Newer versions of TRex sometimes change the network layout for (e.g.) visual identification, which will make them incompatible with older trained models. This parameter allows you to change the expected version back, to ensure backwards compatibility. It also features many public network layouts available from the Keras package. In case training results do not match expectations, please first check the quality of your trajectories before trying out different network layouts.");
        CONFIG("accumulation_enable", true, "Enables or disables the idtrackerai-esque accumulation protocol cascade. It is usually a good thing to enable this (especially in more complicated videos), but can be disabled as a fallback (e.g. if computation time is a major constraint).");
        CONFIG("accumulation_sufficient_uniqueness", float(0), "If changed (from 0), the ratio given here will be the acceptable uniqueness for the video - which will stop accumulation if reached.");
        CONFIG("auto_train_dont_apply", false, "If set to true, setting `auto_train` will only train and not apply the trained network.");
        CONFIG("accumulation_enable_final_step", true, "If enabled, the network will be trained on all the validation + training data accumulated, as a last step of the accumulation protocol cascade. This is intentional overfitting.");
        CONFIG("gpu_learning_rate", float(0.0001f), "Learning rate for training a recognition network.");
        CONFIG("gpu_max_epochs", uchar(150), "Maximum number of epochs for training a recognition network (0 means infinite).");
        CONFIG("gpu_verbosity", gpu_verbosity_t::full, "Determines the nature of the output on the command-line during training. This does not change any behaviour in the graphical interface.");
        CONFIG("gpu_torch_device", gpu_torch_device_t::automatic, "If specified, indicate something like 'cuda:0' to use the first cuda device when doing machine learning using pytorch (e.g. TRexA). Other options can be looked up at `https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device`.");
        CONFIG("gpu_torch_device_index", int(-1), "Index of the GPU used by torch (or -1 for automatic selection).");
        CONFIG("gpu_torch_no_fixes", true, "Disable the fix for PyTorch on MPS devices that will automatically switch to CPU specifically for Ultralytics segmentation models.");
        CONFIG("detect_type", track::detect::ObjectDetectionType::none, "The method used to separate background from foreground when converting videos.", AccessLevelType::INIT);
        CONFIG("outline_compression", float(0.f), "Applies a *lossy* compression to the outlines generated by segmentation models. Walking around the outline, it removes line segments that do not introduce any noticable change in direction. The factor specified here controls how much proportional difference in radians/angle is allowed. The value isnt in real radians, as the true downsampling depends on the size of the object (smaller objects = smaller differences allowed).");
        CONFIG("detect_format", track::detect::ObjectDetectionFormat::none, "The type of data returned by the `detect_model`, which can be an instance segmentation", AccessLevelType::INIT);
        CONFIG("detect_keypoint_format", track::detect::KeypointFormat{}, "When a keypoint (pose) type model is loaded, this variable will be set to [n_points,n_dims].", AccessLevelType::SYSTEM);
        CONFIG("detect_keypoint_names", track::detect::KeypointNames{}, "An array of names in the correct keypoint index order for the given model.");
        CONFIG("detect_batch_size", uchar(1), "The batching size for object detection.");
        CONFIG("detect_tile_image", uchar(0), "If > 1, this will tile the input image for Object detection (SAHI method) before passing it to the network. These tiles will be `detect_resolution` pixels high and wide (with zero padding).");
        CONFIG("yolo_tracking_enabled", false, "If set to true, the program will try to use yolov8s internal tracking routine to improve results. This can be significantly slower and disables batching.");
        CONFIG("yolo_region_tracking_enabled", false, "If set to true, the program will try to use yolov8s internal tracking routine to improve results for region tracking. This can be significantly slower and disables batching.");
        CONFIG("detect_model", file::Path(), "The path to a .pt file that contains a valid PyTorch object detection model (currently only YOLO networks are supported).");
        CONFIG("detect_precomputed_file", file::PathArray{}, "If `detect_type` is set to `precomputed`, this should point to a csv file (or npz files) containing the necessary tracking data for the given `source` video.");
        CONFIG("detect_only_classes", track::detect::PredictionFilter{}, "An array of class ids that you would like to detect (as returned from the model). If left empty, no class will be filtered out.");
        CONFIG("region_model", file::Path(), "The path to a .pt file that contains a valid PyTorch object detection model used for region proposal (currently only YOLO networks are supported).");
        CONFIG("region_resolution", track::detect::DetectResolution{}, "The resolution of the region proposal network (`region_model`).", SYSTEM);
        CONFIG("detect_resolution", track::detect::DetectResolution{}, "The input resolution of the object detection model (`detect_model`).", SYSTEM);
        CONFIG("detect_iou_threshold", Float2_t(0.5), "Higher (==1) indicates that all overlaps are allowed, while lower values (>0) will filter out more of the overlaps. This depends strongly on the situation, but values between 0.25 and 0.7 are common.");
        CONFIG("detect_conf_threshold", Float2_t(0.1), "Confidence threshold (`0<=value<1`) for object detection / segmentation networks. Confidence is higher if the network is more *sure* about the object. Anything with a confidence level below `detect_conf_threshold` will not be considered an object and not saved to the PV file during conversion.");
        CONFIG("gpu_min_iterations", uchar(100), "Minimum number of iterations per epoch for training a recognition network.");
        CONFIG("gpu_max_cache", float(2), "Size of the image cache (transferring to GPU) in GigaBytes when applying the network.");
        CONFIG("gpu_max_sample_gb", float(2), "Maximum size of per-individual sample images in GigaBytes. If the collected images are too many, they will be sub-sampled in regular intervals.");
        CONFIG("gpu_min_elements", uint32_t(25000), "Minimum number of images being collected, before sending them to the GPU.");
        CONFIG("accumulation_max_tracklets", uint32_t(15), "If there are more than `accumulation_max_tracklets` global tracklets to be trained on, they will be filtered according to their quality until said limit is reached.");
        CONFIG("terminate_training", bool(false), "Setting this to true aborts the training in progress.");
        
        CONFIG("manually_approved", std::map<long_t,long_t>(), "A list of ranges of manually approved frames that may be used for generating training datasets, e.g. `{232:233,5555:5560}` where each of the numbers is a frame number. Meaning that frames 232-233 and 5555-5560 are manually set to be manually checked for any identity switches, and individual identities can be assumed to be consistent throughout these frames.");
        CONFIG("gui_focus_group", std::vector<track::Idx_t>(), "Focus on this group of individuals.");
        
        CONFIG("track_ignore", std::vector<std::vector<Vec2>>(), "If this is not empty, objects within the given rectangles or polygons (>= 3 points) `[[x0,y0],[x1,y1](, ...)], ...]` will be ignored during tracking.");
        CONFIG("track_include", std::vector<std::vector<Vec2>>(), "If this is not empty, objects within the given rectangles or polygons (>= 3 points) `[[x0,y0],[x1,y1](, ...)], ...]` will be the only objects being tracked. (overwrites `track_ignore`)");
        
        CONFIG("tracklet_punish_timedelta", true, "If enabled, a huge timestamp difference will end the current trajectory tracklet and will be displayed as a reason in the tracklet overview at the top of the selected individual info card.");
        CONFIG("track_trusted_probability", 0.25_F, "If the (purely kinematic-based) probability that is used to assign an individual to an object is smaller than this value, the current tracklet ends and a new one starts. Even if the individual may still be assigned to the object, TRex will be *unsure* and no longer assume that it is definitely the same individual.");
        CONFIG("huge_timestamp_seconds", 0.2, "Defaults to 0.5s (500ms), can be set to any value that should be recognized as being huge.");
        CONFIG("gui_foi_name", std::string("correcting"), "If not empty, the gui will display the given FOI type in the timeline and allow to navigate between them via M/N.");
        CONFIG("gui_foi_types", std::vector<std::string>{"none"}, "A list of all the foi types registered.", SYSTEM);
        
        CONFIG("gui_connectivity_matrix_file", file::Path(), "Path to connectivity table. Expected structure is a csv table with columns [frame | #(track_max_individuals^2) values] and frames in y-direction.");
        CONFIG("gui_connectivity_matrix", std::map<long_t, std::vector<float>>(), "Internally used to store the connectivity matrix.");
        
        CONFIG("webcam_index", uint8_t(0), "cv::VideoCapture index of the current webcam. If the program chooses the wrong webcam (`source` = webcam), increase this index until it finds the correct one.");
        CONFIG("cam_scale", float(1.0), "Scales the image down or up by the given factor.");
        CONFIG("cam_circle_mask", false, "If set to true, a circle with a diameter of the width of the video image will mask the video. Anything outside that circle will be disregarded as background.");
        CONFIG("cam_undistort", false, "If set to true, the recorded video image will be undistorted using `cam_undistort_vector` (1x5) and `cam_matrix` (3x3).");
        CONFIG("image_invert", false, "Inverts the image greyscale values before thresholding.");
        
        CONFIG("adaptive_threshold_scale", float(2), "Threshold value to be used for adaptive thresholding, if enabled.");
        CONFIG("use_adaptive_threshold", false, "Enables or disables adaptive thresholding (slower than normal threshold). Deals better with weird backgrounds.");
        CONFIG("dilation_size", int32_t(0), "If set to a value greater than zero, detected shapes will be inflated (and potentially merged). When set to a value smaller than zero, detected shapes will be shrunk (and potentially split).");
        CONFIG("use_closing", false, "Toggles the attempt to close weird blobs using dilation/erosion with `closing_size` sized filters.");
        CONFIG("closing_size", int(3), "Size of the dilation/erosion filters for if `use_closing` is enabled.");
        
        CONFIG("track_threshold_is_absolute", true, "If enabled, uses absolute difference values and disregards any pixel |p| < `threshold` during conversion. Otherwise the equation is p < `threshold`, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as `detect_threshold_is_absolute`, but during tracking instead of converting.");
        CONFIG("detect_threshold_is_absolute", true, "If enabled, uses absolute difference values and disregards any pixel |p| < `threshold` during conversion. Otherwise the equation is p < `threshold`, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as `track_threshold_is_absolute`, but during conversion instead of tracking.");
        
#if !CMN_WITH_IMGUI_INSTALLED
        config["nowindow"] = true;
#endif
        
        //config.set_print_by_default(old);
    }

    std::string Config::to_settings() const {
        std::stringstream ss;
        for (auto& [key, value] : map) {
            ss << key << " = " << value->valueString() << "\n";
        }
        return ss.str();
    }
    void Config::write_to(sprite::Map& other) {
        for (auto& [key, value] : this->map) {
            value->copy_to(other);
        }
    }

    const sprite::PropertyType*& Config::operator[](const std::string& key) {
		return map[key];
    }
    
    Config generate_delta_config(AccessLevel access, const pv::File* video, bool include_build_number, std::vector<std::string> additional_exclusions) {
        std::vector<std::string> exclude_fields = {
            "track_pause",
            //"filename",
            "app_name",
            "app_check_for_updates",
            "app_last_update_version",
            "app_last_update_check",
            "video_size",
            "video_info",
            "video_mask",
            "video_length",
            "terminate",
            "cam_limit_exposure",
            "reset_average",
            //"accumulation_sufficient_uniqueness",
            //"output_fields",
            "auto_minmax_size",
            "auto_number_individuals",
            //"output_default_options",
            //"output_annotations",
            "log_file",
            "history_matching_log",
            "gui_foi_types",
            "gui_mode",
            "gui_frame",
            "gui_show_infocard",
            "gui_show_timeline",
            "gui_zoom_polygon",
            "gui_displayed_frame",
            "gui_source_video_frame",
            "gui_run",
            //"settings_file",
            "nowindow",
            "task",
            "wd",
            "gui_show_fish",
            "auto_quit",
            "auto_no_outputs",
            "auto_apply", "auto_train",
            "auto_no_results",
            "auto_no_tracking_data",
            "load",
            //"output_dir",
            "auto_categorize",
            "tags_path",
            "analysis_range",
            //"output_prefix",

            /*"detect_model",
            "region_model",
            "detect_format",
            "detect_type",
            "region_resolution",
            "detect_resolution",*/

            "cmd_line",
            "ffmpeg_path",
            "httpd_port",
            "cam_undistort1",
            "cam_undistort2",
            
            // from info utility
            "print_parameters",
            "replace_background",
            "display_average",
            "blob_detail",
            "quiet",
            "write_settings",
            "merge_videos",
            "merge_output_path",
            "merge_background",
            "merge_dir",
            "merge_overlapping_blobs",
            "merge_mode",
            "exec",
            
            "gpu_torch_no_fixes"
        };
        
        std::set<std::string_view> explicitly_include{
            "meta_source_path",
            "meta_real_width",
            "detect_type",
            "detect_size_filter",
            "meta_encoding",
            "cm_per_pixel",
            "calculate_posture",
            "track_background_subtraction"
        };
        
        if(auto type = SETTING(detect_type).value<track::detect::ObjectDetectionType_t>();
           type == track::detect::ObjectDetectionType::yolo)
        {
            explicitly_include.emplace("detect_classes");
            explicitly_include.emplace("detect_format");
            
            if(auto format = SETTING(detect_format).value<track::detect::ObjectDetectionFormat_t>();
               format != track::detect::ObjectDetectionFormat::poses)
            {
                exclude_fields.push_back("detect_skeleton");
                
            } else {
                explicitly_include.emplace("detect_skeleton");
            }
            
            if(SETTING(region_model).value<file::Path>().empty()) {
                exclude_fields.push_back("region_model");
            }
            
        } else {
            exclude_fields.push_back("detect_iou_threshold");
            exclude_fields.push_back("detect_skeleton");
            exclude_fields.push_back("detect_conf_threshold");
            exclude_fields.push_back("detect_model");
            exclude_fields.push_back("detect_format");
            exclude_fields.push_back("detect_classes");
            
            explicitly_include.emplace("detect_threshold");
        }
        
        /**
         * Exclude some settings based on what would automatically be assigned
         * if they weren't set at all.
         */
        if(SETTING(cm_per_pixel).value<Float2_t>() == SETTING(meta_real_width).value<Float2_t>() / SETTING(video_size).value<Size2>().width)
        {
            exclude_fields.push_back("cm_per_pixel");
        }
        
        //if(GUI::instance() && SETTING(frame_rate).value<int>() == GUI::instance()->video_source()->framerate())
        //    exclude_fields.push_back("frame_rate");
        
        /**
         * Write the remaining settings.
         */
        Config result;
        result.excluded += exclude_fields;
        
        if(video) {
            sprite::Map tmp;
            try {
                auto& metadata = video->header().metadata;
                if(metadata.has_value()) {
                    GlobalSettings::write([&](Configuration& config) {
                        sprite::parse_values(sprite::MapSource{ video->filename() }, tmp, metadata.value(), & config.values, {}, default_config::deprecations());
                    });
                }
                /*if(tmp.has("meta_source_path")
                   //&& tmp.at("meta_source_path").value<std::string>() != SETTING(meta_source_path).value<std::string>()
                   )
                {
                    explicitly_include.insert("meta_source_path");
                }*/
                
            } catch(...) {
                FormatExcept("Error while trying to inspect the metadata of ", video, ".");
            }
        }
        
        auto keys = GlobalSettings::keys();
        //sprite::Map config;
        //docs_map_t docs;
        
        //config = GlobalSettings::get_current_defaults();
        //grab::default_config::get(config, docs, nullptr);
        //default_config::get(config, docs, NULL);
        
        for(auto &key : keys) {
            // dont write meta variables. this could be confusing if those
            // are loaded from the video file as well
            if(utils::beginsWith(key, "meta_")
               && not explicitly_include.contains(key))
            {
                result.excluded.push_back(key);
                continue;
            }
            
            // UPDATE: write only keys with values that have changed compared
            // to the default options
            
            auto current = GlobalSettings::read_current_default<NoType>(key);
            auto value = GlobalSettings::read_value<NoType>(key);
            
            if(not current.valid()
               || current != value
               || explicitly_include.contains(key))
            {
                if((include_build_number && utils::beginsWith(key, "build"))
                   || explicitly_include.contains(key)
                   || (GlobalSettings::access_level(key) <= access
                       && !contains(exclude_fields, key)
                       && !contains(additional_exclusions, key)))
                {
                    result[key] = &value.get();
                } else {
                    //Print("// ",key," not part of delta");
                }
            } else {
                //Print("// ", key, " not part of delta (!=)");
            }
        }
        
        return result;
    }

#if defined(__APPLE__)
inline bool isRunningInAppBundle() {
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        std::string execPath(path);
        // Find the .app bundle marker
        auto pos = execPath.find(".app/Contents/MacOS/");
        return pos != std::string::npos;
    }
    return false;
}
#endif
    
    void register_default_locations() {
        file::DataLocation::register_path("app", [](const sprite::Map& map, file::Path path) -> file::Path {
            auto wd = map.at("wd").value<file::Path>();
#if defined(TREX_CONDA_PACKAGE_INSTALL)
            auto conda_prefix = ::default_config::conda_environment_path();
            if(!conda_prefix.empty()) {
                wd = conda_prefix / "usr" / "share" / "trex";
            }
#elif __APPLE__
            if(isRunningInAppBundle()) {
                wd = wd / ".." / "Resources";
            }
#endif
            if(path.empty())
                return wd;
            return wd / path;
        });
        
        file::DataLocation::register_path("default.settings", [](const sprite::Map& map, file::Path) -> file::Path {
            auto settings_file = file::DataLocation::parse("app", "default.settings", &map);
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file is an empty string.");
            
            return settings_file;
        });
        
        file::DataLocation::register_path("settings", [](const sprite::Map& map, file::Path path) -> file::Path {
            if(path.empty())
                path = map.at("settings_file").value<Path>();
            if(path.empty()) {
                path = map.at("filename").value<Path>();
                if(path.has_extension("pv"))
                    path = path.remove_extension();
            }
            if(path.empty()) {
                auto array = map.at("source").value<file::PathArray>();
                auto base = file::find_basename(array);
                path = base;
            }
            
            if(path.empty())
                return "";
            
            if(not path.has_extension("settings"))
                path = path.add_extension("settings");
            
            if(not path.is_absolute()) {
                auto settings_file = file::DataLocation::parse("output", path, &map);
                if(settings_file.empty())
                    throw U_EXCEPTION("settings_file is an empty string.");
                
                return settings_file;
            } else {
                return path;
            }
        });
        
        file::DataLocation::register_path("output_settings", [](const sprite::Map& map, file::Path) -> file::Path {
            file::Path settings_file = map.at("filename").value<Path>().filename();
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file is an empty string.");
            
            if(!settings_file.has_extension() || settings_file.extension() != "settings")
                settings_file = settings_file.add_extension("settings");
            
            return file::DataLocation::parse("output", settings_file, &map);
        });
        
        file::DataLocation::register_path("backup_settings", [](const sprite::Map& map, file::Path) -> file::Path {
            file::Path settings_file(map.at("filename").value<Path>().filename());
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file (and like filename) is an empty string.");
            
            if(!settings_file.has_extension() || settings_file.extension() != "settings")
                settings_file = settings_file.add_extension("settings");
            
            return file::DataLocation::parse("output", "backup", &map) / settings_file;
        });
        
        file::DataLocation::register_path("input", [](const sprite::Map& map, file::Path filename) -> file::Path {
            if(filename.empty())
                return {};
            if(not filename.empty() && filename.is_absolute()) {
#ifndef NDEBUG
                if(!GlobalSettings::is_runtime_quiet())
                    Print("Returning absolute path ",filename.str(),". We cannot be sure this is writable.");
#endif
                return filename;
            }
            
            auto path = map.at("wd").value<file::Path>();
            if(path.empty()) {
                auto d = map.at("output_dir").value<file::Path>();
                if(d.empty())
                    return filename;
                else
                    return (d / filename);
            } else
                return (path / filename);
        });
        
        file::DataLocation::register_path("output", [](const sprite::Map& map, file::Path filename) -> file::Path
        {
            if(filename.empty())
                return {};
            
            auto prefix = map.at("output_prefix").value<std::string>();
            auto output_path = map.at("output_dir").value<file::Path>();
            auto absolute = filename.is_absolute();
            
            if(output_path.empty()) {
                auto source = map.at("source").value<file::PathArray>();
                auto base = file::find_parent(source);
                if(not base) {
                    output_path = map.at("wd").value<file::Path>();
                } else {
                    output_path = base.value();
                }
            }
            
            //! an output file is specified, we want to change whatever folder
            //! the input comes from to whatever folder we want to write to:
            if(not absolute) {
                //! file is not an absolute path
                if(not output_path.empty()) {
                    filename = output_path / filename;
                } else {
                    /// QUESTIONABLE: we might want to include / and \ again, but
                    /// right now this is turning it into /webcam a lot of the time.
                    if(not is_in(map.at("wd").value<file::Path>(), "", "/", "\\"))
                        filename = map.at("wd").value<file::Path>() / filename;
                }
                
            } else if(not filename.has_extension("pv")) {
                if(not output_path.empty())
                    filename = output_path / filename.filename();
            }
            
            if(!prefix.empty()) {
                //! insert a prefix in between the filename and the path
                if(not filename.remove_filename().empty()
                   && not absolute)
                    filename = filename.remove_filename() / prefix / filename.filename();
                else if(not absolute)
                    filename = prefix / filename;
                else
                    filename = filename.remove_filename() / prefix / filename.filename();
            }
            
            /*if(!filename.empty() && filename.is_absolute()) {
#ifndef NDEBUG
                if(!GlobalSettings::is_runtime_quiet())
                    Print("Returning absolute path ",filename.str(),". We cannot be sure this is writable.");
#endif
                return filename;
            }*/
            
            return filename;
        });
    }

}
