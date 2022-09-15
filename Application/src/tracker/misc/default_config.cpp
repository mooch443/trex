#include "default_config.h"
#include <misc/SpriteMap.h>
#include <file/Path.h>
#include <misc/BlobSizeRange.h>
#include <misc/idx_t.h>
#include "GitSHA1.h"

#ifndef WIN32
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif

#include <misc/default_settings.h>
#include <file/DataLocation.h>

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

#include <tracking/Tracker.h>
#include <misc/default_settings.h>
#include <misc/OutputLibrary.h>

using namespace file;
#define CONFIG adding.add

namespace default_config {
    ENUM_CLASS_DOCS(recognition_border_t,
        "No border at all. All points are inside the recognition boundary. (default)", // none
        "Looks at a subset of frames from the video, trying to find out where individuals go and masking all the places they do not.", // "heatmap"
        "Similar to heatmap, but tries to build a convex border around the around (without holes in it).", // {"outline"
        "Any array of convex shapes. Set coordinates by changing `recognition_shapes`.", // {"shapes"
        "The points defined in `grid_points` are turned into N different circles inside the arena (with points in `grid_points` being the circle centers), which define in/out if inside/outside any of the circles.", // "grid"
        "The video-file provides a binary mask (e.g. when `cam_circle_mask` was set to true during recording), which is then used to determine in/out." // {"circle",
    )

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
        "Uses automatic selection based on density."
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
        "Use the previously known positions of objects to place a seed within the overlapped objects and perform a watershed run."
    )

ENUM_CLASS_DOCS(visual_identification_version_t,
    "This always points to the current version.",
    "The order of Max-Pooling layers was changed, along with some other minor changes.",
    "Changed activation order, added BatchNormalization. No Flattening to maintain spatial context.",
    "The original layout."
)
    
    static const std::map<std::string, std::string> deprecated = {
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
        {"analysis_stop_after", "analysis_range"},
        {"fixed_count", ""},
        {"fish_minmax_size", "blob_size_ranges"},
        {"blob_size_range", "blob_size_ranges"},
        {"fish_max_speed", "track_max_speed"},
        {"fish_speed_decay", "track_speed_decay"},
        {"fish_enable_direction_smoothing", "posture_direction_smoothing"},
        {"fish_use_matching", ""},
        {"fish_time_probability_enabled", "track_time_probability_enabled"},
        {"number_fish", "track_max_individuals"},
        {"outline_remove_loops", ""},
        {"whitelist_rect", "track_include"},
        {"track_whitelist", "track_include"},
        {"exclude_rect", "track_ignore"},
        {"track_blacklist", "track_ignore"},
        {"posture_threshold_constant", "track_posture_threshold"},
        {"threshold_constant", "track_threshold"},
        {"recognition_rect", "recognition_shapes"},
        {"recognition_normalization", "individual_image_normalization"},
        {"recognition_normalize_direction", "individual_image_normalization"},
        {"match_use_approximate", "match_mode"},
        {"output_npz", "output_format"},
        {"gui_heatmap_value_range", "heatmap_value_range"},
        {"gui_heatmap_smooth", "heatmap_smooth"},
        {"gui_heatmap_frames", "heatmap_frames"},
        {"gui_heatmap_dynamic", "heatmap_dynamic"},
        {"gui_heatmap_resolution", "heatmap_resolution"},
        {"gui_heatmap_normalization", "heatmap_normalization"},
        {"gui_heatmap_source", "heatmap_source"},
    };

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

    if(home == "CONDA_PREFIX" || home == "" || home == compiled_path) {
#ifndef NDEBUG
        if(!SETTING(quiet))
            print("Reset conda prefix ",home," / ",compiled_path);
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
    
    if(!SETTING(quiet))
        print("Set conda environment path = ",home);
    return home;
}
    
    const std::map<std::string, std::string>& deprecations() {
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
            print("Found invalid settings in source ",source," (see above).");
    }
    
    bool is_deprecated(const std::string& key) {
        return deprecated.find(utils::lowercase(key)) != deprecated.end();
    }
    
    std::string replacement(const std::string& key) {
        if (!is_deprecated(key)) {
            throw U_EXCEPTION("Key ",key," is not deprecated.");
        }
        
        return deprecated.at(utils::lowercase(key));
    }

#define PYTHON_TIPPS ""
#ifdef WIN32
#define PYTHON_TIPPS " (containing pythonXX.exe)"
#endif


    constexpr std::string_view is_ndebug_enabled() {
#ifndef NDEBUG
        return std::string_view("debug");
#else
        return std::string_view("release");
#endif
    }
    
    void get(sprite::Map& config, GlobalSettings::docs_map_t& docs, decltype(GlobalSettings::set_access_level)* fn)
    {
        auto old = config.do_print();
        config.set_do_print(false);
        //constexpr auto PUBLIC = AccessLevelType::PUBLIC;
        constexpr auto STARTUP = AccessLevelType::STARTUP;
        constexpr auto SYSTEM = AccessLevelType::SYSTEM;
        
        using namespace settings;
        Adding adding(config, docs, fn);
        
        CONFIG<std::string>("app_name", "TRex", "Name of the application.", SYSTEM);
        CONFIG("app_check_for_updates", app_update_check_t::none, "If enabled, the application will regularly check for updates online (`https://api.github.com/repos/mooch443/trex/releases`).");
        CONFIG("app_last_update_check", uint64_t(0), "Time-point of when the application has last checked for an update.", SYSTEM);
        CONFIG("app_last_update_version", std::string(), "");
        CONFIG("version", std::string(g_GIT_DESCRIBE_TAG), "Current application version.", SYSTEM);
        CONFIG("build_architecture", std::string(g_TREX_BUILD_ARCHITECTURE), "The architecture this executable was built for.", SYSTEM);
        CONFIG("build_type", std::string(g_TREX_BUILD_TYPE), "The mode the application was built in.", SYSTEM);
        CONFIG("build_is_debug", std::string(is_ndebug_enabled()), "If built in debug mode, this will show 'debug'.", SYSTEM);
        CONFIG("build_cxx_options", std::string(g_TREX_BUILD_CXX_OPTIONS), "The mode the application was built in.", SYSTEM);
        CONFIG("build", std::string(), "Current build version", SYSTEM);
        CONFIG("cmd_line", std::string(), "An approximation of the command-line arguments passed to the program.", SYSTEM);
        CONFIG("ffmpeg_path", file::Path(), "Path to an ffmpeg executable file. This is used for converting videos after recording them (from the GUI). It is not a critical component of the software, but mostly for convenience.");
        CONFIG("blobs_per_thread", 150.f, "Number of blobs for which properties will be calculated per thread.");
        CONFIG("individuals_per_thread", 1.f, "Number of individuals for which positions will be estimated per thread.");
        CONFIG("postures_per_thread", 1.f, "Number of individuals for which postures will be estimated per thread.");
        CONFIG("history_matching_log", file::Path(), "If this is set to a valid html file path, a detailed matching history log will be written to the given file for each frame.");
        CONFIG("filename", Path("").remove_extension(), "Opened filename (without .pv).", STARTUP);
        CONFIG("output_dir", Path(std::string(homedir)+"/Videos"), "Default output-/input-directory. Change this in order to omit paths in front of filenames for open and save.");
        CONFIG("fishdata_dir", Path("data"), "Subfolder (below `output_dir`) where the exported NPZ or CSV files will be saved (see `output_graphs`).");
        CONFIG("settings_file", Path(""), "Name of the settings file. By default, this will be set to `filename`.settings in the same folder as `filename`.", STARTUP);
        CONFIG("python_path", Path(COMMONS_PYTHON_EXECUTABLE), "Path to the python home folder" PYTHON_TIPPS ". If left empty, the user is required to make sure that all necessary libraries are in-scope the PATH environment variable.");

        CONFIG("frame_rate", int(0), "Specifies the frame rate of the video. It is used e.g. for playback speed and certain parts of the matching algorithm. Will be set by the .settings of a video (or by the video itself).", STARTUP);
        CONFIG("calculate_posture", true, "Enables or disables posture calculation. Can only be set before the video is analysed (e.g. in a settings file or as a startup parameter).", STARTUP);
        
        CONFIG("meta_source_path", Path(""), "Path of the original video file for conversions (saved as debug info).", STARTUP);
        CONFIG("meta_real_width", float(0), "Used to calculate the `cm_per_pixel` conversion factor, relevant for e.g. converting the speed of individuals from px/s to cm/s (to compare to `track_max_speed` which is given in cm/s). By default set to 30 if no other values are available (e.g. via command-line). This variable should reflect actual width (in cm) of what is seen in the video image. For example, if the video shows a tank that is 50cm in X-direction and 30cm in Y-direction, and the image is cropped exactly to the size of the tank, then this variable should be set to 50.", STARTUP);
        CONFIG("cm_per_pixel", float(0), "The ratio of `meta_real_width / video_width` that is used to convert pixels to centimeters. Will be automatically calculated based on a meta-parameter saved inside the video file (`meta_real_width`) and does not need to be set manually.", STARTUP);
        CONFIG("video_length", uint64_t(0), "The length of the video in frames", STARTUP);
        CONFIG("video_size", Size2(-1), "The dimensions of the currently loaded video.", SYSTEM);
        CONFIG("video_info", std::string(), "Information on the current video as provided by PV.", SYSTEM);
        
        /*
         * According to @citation the average zebrafish larvae weight would be >200mg after 9-week trials.
         * So this is most likely over-estimated.
         *
         * Siccardi AJ, Garris HW, Jones WT, Moseley DB, D’Abramo LR, Watts SA. Growth and Survival of Zebrafish (Danio rerio) Fed Different Commercial and Laboratory Diets. Zebrafish. 2009;6(3):275-280. doi:10.1089/zeb.2008.0553.
         */
        CONFIG("meta_mass_mg", float(200), "Used for exporting event-energy levels.");
        CONFIG("midline_samples", uint64_t(0), "The maximum number of samples taken for generating a `median midline length`. Setting this to 0 removes the limit all together. A limit may be set for very long videos, or videos with lots of individuals, for memory reasons.");
        
        CONFIG("nowindow", false, "If set to true, no GUI will be created on startup (e.g. when starting from SSH).", STARTUP);
        CONFIG("debug", false, "Enables some verbose debug print-outs.");
        CONFIG("use_differences", false, "This should be set to false unless when using really old files.");
        //config["debug_probabilities"] = false;
        CONFIG("analysis_paused", false, "Halts the analysis.");
        CONFIG("limit", 0.09f, "Limit for tailbeat event detection.");
        CONFIG("event_min_peak_offset", 0.15f, "");
        CONFIG("exec", file::Path(), "This can be set to the path of an additional settings file that is executed after the normal settings file.");
        CONFIG("log_file", file::Path(), "Set this to a path you want to save the log file to.", STARTUP);
        CONFIG("httpd_port", 8080, "This is where the webserver tries to establish a socket. If it fails, this will be set to the port that was chosen.", STARTUP);
        CONFIG("httpd_accepted_ip", std::string(), "Set this to an IP address that you want to accept exclusively.");
        CONFIG("error_terminate", false, "", SYSTEM);
        CONFIG("terminate", false, "If set to true, the application terminates.");
        
        CONFIG("gui_transparent_background", false, "If enabled, fonts might look weird but you can record movies (and images) with transparent background (if gui_background_color.alpha is < 255).");
        
        CONFIG("gui_interface_scale", float(1.25), "Scales the whole interface. A value greater than 1 will make it smaller.");
        CONFIG("gui_max_path_time", float(3), "Length (in time) of the trails shown in GUI.");
        
        CONFIG("gui_draw_only_filtered_out", false, "Only show filtered out blob texts.");
        CONFIG<std::pair<pv::bid, Frame_t>>("gui_show_fish", {pv::bid::invalid, Frame_t()}, "Show debug output for {blob_id, fish_id}.");
        CONFIG("gui_frame", Frame_t(0), "The currently visible frame.");
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
        CONFIG("gui_blur_enabled", false, "MacOS supports a blur filter that can be applied to make unselected individuals look interesting.");
#endif
        CONFIG("gui_faded_brightness", uchar(255), "The alpha value of tracking-related elements when timeline is hidden (0-255).");
        CONFIG("gui_equalize_blob_histograms", true, "Equalize histograms of blobs wihtin videos (makes them more visible).");
        CONFIG("gui_show_heatmap", false, "Showing a heatmap per identity, normalized by maximum samples per grid-cell.");
        CONFIG("gui_show_individual_preview", true, "Shows preview images for all selected individuals as they would be processed during network training, based on settings like `individual_image_size`, `individual_image_scale` and `individual_image_normalization`.");
        CONFIG("heatmap_ids", std::vector<uint32_t>(), "Add ID numbers to this array to exclusively display heatmap values for those individuals.");
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
        CONFIG("gui_show_blobs", true, "Showing or hiding individual raw blobs in tracking view (are always shown in RAW mode).");
        CONFIG("gui_show_paths", true, "Equivalent to the checkbox visible in GUI on the bottom-left.");
        CONFIG("gui_show_pixel_grid", false, "Shows the proximity grid generated for all blobs, which is used for history splitting.");
        CONFIG("gui_show_selections", true, "Show/hide circles around selected individual.");
        CONFIG("gui_show_inactive_individuals", true, "Show/hide individuals that have not been seen for longer than `track_max_reassign_time`.");
        //config["gui_show_texts"] = true;
        CONFIG("gui_show_histograms", false, "Equivalent to the checkbox visible in GUI on the bottom-left.");
        CONFIG("gui_show_posture", true, "Show/hide the posture window on the top-right.");
        CONFIG("gui_show_export_options", false, "Show/hide the export options widget.");
        CONFIG("gui_show_visualfield_ts", false, "Show/hide the visual field time series.");
        CONFIG("gui_show_visualfield", false, "Show/hide the visual field rays.");
        CONFIG("gui_show_uniqueness", false, "Show/hide uniqueness overview after training.");
        CONFIG("gui_show_probabilities", false, "Show/hide probability visualisation when an individual is selected.");
        CONFIG("gui_show_cliques", false, "Show/hide cliques of potentially difficult tracking situations.");
        //CONFIG("gui_show_manual_matches", true, "Show/hide manual matches in path.");
        CONFIG("gui_show_graph", false, "Show/hide the data time-series graph.");
        CONFIG("gui_show_number_individuals", false, "Show/hide the #individuals time-series graph.");
        CONFIG("gui_show_recognition_summary", false, "Show/hide confusion matrix (if network is loaded).");
        CONFIG("gui_show_dataset", false, "Show/hide detailed dataset information on-screen.");
        CONFIG("gui_show_recognition_bounds", true, "Shows what is contained within tht recognition boundary as a cyan background. (See `recognition_border` for details.)");
        CONFIG("gui_show_boundary_crossings", true, "If set to true (and the number of individuals is set to a number > 0), the tracker will show whenever an individual enters the recognition boundary. Indicated by an expanding cyan circle around it.");
        CONFIG("gui_show_detailed_probabilities", false, "Show/hide detailed probability stats when an individual is selected.");
        CONFIG("gui_playback_speed", float(1.f), "Playback speed when pressing SPACE.");
        CONFIG("gui_show_midline_histogram", false, "Displays a histogram for midline lengths.");
        CONFIG("gui_auto_scale", false, "If set to true, the tracker will always try to zoom in on the whole group. This is useful for some individuals in a huge video (because if they are too tiny, you cant see them and their posture anymore).");
        CONFIG("gui_auto_scale_focus_one", true, "If set to true (and `gui_auto_scale` set to true, too), the tracker will zoom in on the selected individual, if one is selected.");
        CONFIG("gui_timeline_alpha", uchar(200), "Determines the Alpha value for the timeline / consecutive segments display.");
        CONFIG("gui_background_color", gui::Color(0,0,0,150), "Values < 255 will make the background more transparent in standard view. This might be useful with very bright backgrounds.");
        CONFIG("gui_fish_color", std::string("identity"), "");
        CONFIG("gui_single_identity_color", gui::Transparent, "If set to something else than transparent, all individuals will be displayed with this color.");
        CONFIG("gui_zoom_limit", Size2(300, 300), "");

#ifdef __APPLE__
        auto default_recording_t = gui_recording_format_t::mp4;
#else
        auto default_recording_t = gui_recording_format_t::avi;
#endif

        CONFIG("gui_recording_format", default_recording_t, "Sets the format for recording mode (when R is pressed in the GUI). Supported formats are 'avi', 'jpg' and 'png'. JPEGs have 75%% compression, AVI is using MJPEG compression.");
        CONFIG("gui_happy_mode", false, "If `calculate_posture` is enabled, enabling this option likely improves your experience with TRex.");
        CONFIG("individual_names", std::map<uint32_t, std::string>{}, "A map of `{individual-id: \"individual-name\", ...}` that names individuals in the GUI and exported data.");
        CONFIG("individual_prefix", std::string("fish"), "The prefix that is added to all the files containing certain IDs. So individual 0 will turn into '[prefix]0' for all the npz files and within the program.");
        CONFIG("outline_approximate", uint8_t(3), "If this is a number > 0, the outline detected from the image will be passed through an elliptical fourier transform with `outline_approximate` number of coefficients. When the given number is sufficiently low, the outline will be smoothed significantly (and more so for lower numbers of coefficients).");
        CONFIG("outline_smooth_step", uint8_t(1), "Jump over N outline points when smoothing (reducing accuracy).");
        CONFIG("outline_smooth_samples", uint8_t(4), "Use N samples for smoothing the outline. More samples will generate a smoother (less detailed) outline.");
        CONFIG("outline_curvature_range_ratio", float(0.03), "Determines the ratio between number of outline points and distance used to calculate its curvature. Program will look at index +- `ratio * size()` and calculate the distance between these points (see posture window red/green color).");
        CONFIG("midline_walk_offset", float(0.025), "This percentage of the number of outline points is the amount of points that the midline-algorithm is allowed to move left and right upon each step. Higher numbers will make midlines more straight, especially when extremities are present (that need to be skipped over), but higher numbers will also potentially decrease accuracy for less detailed objects.");
        CONFIG("midline_stiff_percentage", float(0.15), "Percentage of the midline that can be assumed to be stiff. If the head position seems poorly approximated (straighened out too much), then decrease this value.");
        CONFIG("midline_resolution", uint32_t(25), "Number of midline points that are saved. Higher number increases detail.", STARTUP);
        CONFIG("posture_head_percentage", float(0.1), "The percentage of the midline-length that the head is moved away from the front of the body.");
        CONFIG("posture_closing_steps", uint8_t(0), "When enabled (> 0), posture will be processed using a combination of erode / dilate in order to close holes in the shape and get rid of extremities. An increased number of steps will shrink the shape, but will also be more time intensive.");
        CONFIG("posture_closing_size", uint8_t(2), "The kernel size for erosion / dilation of the posture algorithm. Only has an effect with  `posture_closing_steps` > 0.");
        CONFIG("outline_resample", float(0.5), "Spacing between outline points in pixels, after resampling (normalizing) the outline. A lower value here can drastically increase the number of outline points generated (and decrease speed).");
        CONFIG("outline_use_dft", true, "If enabled, the program tries to reduce outline noise by convolution of the curvature array with a low pass filter.");
        CONFIG("midline_start_with_head", false, "If enabled, the midline is going to be estimated starting at the head instead of the tail.");
        CONFIG("midline_invert", false, "If enabled, all midlines will be inverted (tail/head swapped).");
        CONFIG("peak_mode", peak_mode_t::pointy, "This determines whether the tail of an individual should be expected to be pointy or broad.");
        CONFIG("manual_matches", std::map<Frame_t, std::map<track::Idx_t, pv::bid>>{ }, "A map of manually defined matches (also updated by GUI menu for assigning manual identities). `{{frame: {fish0: blob2, fish1: blob0}}, ...}`");
        CONFIG("manual_splits", std::map<Frame_t, std::set<pv::bid>>{}, "This map contains `{frame: [blobid1,blobid2,...]}` where frame and blobid are integers. When this is read during tracking for a frame, the tracker will attempt to force-split the given blob ids.");
        CONFIG("match_mode", matching_mode_t::automatic, "Changes the default algorithm to be used for matching blobs in one frame with blobs in the next frame. The accurate algorithm performs best, but also scales less well for more individuals than the approximate one. However, if it is too slow (temporarily) in a few frames, the program falls back to using the approximate one that doesnt slow down.");
        CONFIG("matching_probability_threshold", float(0.1), "The probability below which a possible connection between blob and identity is considered too low. The probability depends largely upon settings like `track_max_speed`.");
        CONFIG("track_do_history_split", true, "If disabled, blobs will not be split automatically in order to separate overlapping individuals. This usually happens based on their history.");
        CONFIG("track_end_segment_for_speed", true, "Sometimes individuals might be assigned to blobs that are far away from the previous position. This could indicate wrong assignments, but not necessarily. If this variable is set to true, consecutive frame segments will end whenever high speeds are reached, just to be on the safe side. For scenarios with lots of individuals (and no recognition) this might spam yellow bars in the timeline and may be disabled.");
        CONFIG("track_consistent_categories", false, "Utilise categories (if present) when tracking. This may break trajectories in places with imperfect categorization, but only applies once categories have been applied.");
        CONFIG("track_max_individuals", uint32_t(0), "The maximal number of individual that are assigned at the same time (infinite if set to zero). If the given number is below the actual number of individual, then only a (random) subset of individual are assigned and a warning is shown.", STARTUP);
        CONFIG("blob_size_ranges", BlobSizeRange({Rangef(0.1f, 3)}), "Blobs below the lower bound are recognized as noise instead of individuals. Blobs bigger than the upper bound are considered to potentially contain more than one individual. You can look these values up by pressing `D` in TRex to get to the raw view (see `https://trex.run/docs/gui.html` for details). The unit is #pixels * (cm/px)^2. `cm_per_pixel` is used for this conversion.");
        CONFIG("blob_split_max_shrink", float(0.2), "The minimum percentage of the starting blob size (after thresholding), that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.");
        CONFIG("blob_split_global_shrink_limit", float(0.2), "The minimum percentage of the minimum in `blob_size_ranges`, that a blob is allowed to be reduced to during splitting. If this value is set too low, the program might start recognizing parts of individual as other individual too quickly.");
        CONFIG("blob_split_algorithm", blob_split_algorithm_t::threshold, "The default splitting algorithm used to split objects that are too close together.");
        
        CONFIG("visual_field_eye_offset", float(0.15), "A percentage telling the program how much the eye positions are offset from the start of the midline.");
        CONFIG("visual_field_eye_separation", float(60), "Degrees of separation between the eye and looking straight ahead. Results in the eye looking towards head.angle +- `visual_field_eye_separation`.");
        CONFIG("visual_field_history_smoothing", uint8_t(0), "The maximum number of previous values (and look-back in frames) to take into account when smoothing visual field orientations. If greater than 0, visual fields will use smoothed previous eye positions to determine the optimal current eye position. This is usually only necessary when postures are somewhat noisy to a degree that makes visual fields unreliable.");
        
        CONFIG("auto_minmax_size", false, "Program will try to find minimum / maximum size of the individuals automatically for the current `cm_per_pixel` setting. Can only be passed as an argument upon startup. The calculation is based on the median blob size in the video and assumes a relatively low level of noise.", STARTUP);
        CONFIG("auto_number_individuals", false, "Program will automatically try to find the number of individuals (with sizes given in `blob_size_ranges`) and set `track_max_individuals` to that value.");
        
        CONFIG("track_speed_decay", float(0.7), "The amount the expected speed is reduced over time when an individual is lost. When individuals collide, depending on the expected behavior for the given species, one should choose different values for this variable. If the individuals usually stop when they collide, this should be set to 1. If the individuals are expected to move over one another, the value should be set to `0.7 > value > 0`.");
        CONFIG("track_max_speed", float(10), "The maximum speed an individual can have (=> the maximum distance an individual can travel within one second) in cm/s. Uses and is influenced by `meta_real_width` and `cm_per_pixel` as follows: `speed(px/s) * cm_per_pixel(cm/px) -> cm/s`.");
        CONFIG("posture_direction_smoothing", uint16_t(0), "Enables or disables smoothing of the posture orientation based on previous frames (not good for fast turns).");
        CONFIG("speed_extrapolation", float(3), "Used for matching when estimating the next position of an individual. Smaller values are appropriate for lower frame rates. The higher this value is, the more previous frames will have significant weight in estimating the next position (with an exponential decay).");
        CONFIG("track_intensity_range", Rangel(-1, -1), "When set to valid values, objects will be filtered to have an average pixel intensity within the given range.");
        CONFIG("track_threshold", int(15), "Constant used in background subtraction. Pixels with grey values above this threshold will be interpreted as potential individuals, while pixels below this threshold will be ignored.");
        CONFIG("threshold_ratio_range", Rangef(0.5, 1.0), "If `track_threshold_2` is not equal to zero, this ratio will be multiplied by the number of pixels present before the second threshold. If the resulting size falls within the given range, the blob is deemed okay.");
        CONFIG("track_threshold_2", int(0), "If not zero, a second threshold will be applied to all objects after they have been deemed do be theoretically large enough. Then they are compared to #before_pixels * `threshold_ratio_range` to see how much they have been shrunk).");
        CONFIG("track_posture_threshold", int(15), "Same as `track_threshold`, but for posture estimation.");
        CONFIG("enable_absolute_difference", true, "If set to true, the threshold values will be applied to abs(image - background). Otherwise max(0, image - background).");
        CONFIG("track_time_probability_enabled", bool(true), "");
        CONFIG("track_max_reassign_time", float(0.5), "Distance in time (seconds) where the matcher will stop trying to reassign an individual based on previous position. After this time runs out, depending on the settings, the tracker will try to find it based on other criteria, or generate a new individual.");
        
        CONFIG("gui_highlight_categories", false, "If enabled, categories (if applied in the video) will be highlighted in the tracking view.");
        CONFIG("categories_ordered", std::vector<std::string>{}, "Ordered list of names of categories that are used in categorization (classification of types of individuals).");
        CONFIG("categories_min_sample_images", uint32_t(50), "Minimum number of images for a sample to be considered relevant. This will default to 50, or ten percent of `track_segment_max_length`, if that parameter is set. If `track_segment_max_length` is set, the value of this parameter will be ignored. If set to zero or one, then all samples are valid.");
        CONFIG("track_segment_max_length", float(0), "If set to something bigger than zero, this represents the maximum number of seconds that a consecutive segment can be.");
        
        CONFIG("track_only_categories", std::vector<std::string>{}, "If this is a non-empty list, only objects that have previously been assigned one of the correct categories will be tracked. Note that this also excludes noise particles or very short segments with no tracking.");
        
        CONFIG("web_quality", int(75), "JPEG quality of images transferred over the web interface.");
        CONFIG("web_time_threshold", float(0.050), "Maximum refresh rate in seconds for the web interface.");
        
        CONFIG("correct_illegal_lines", false, "In older versions of the software, blobs can be constructed in 'illegal' ways, meaning the lines might be overlapping. If the software is printing warnings about it, this should probably be enabled (makes it slower).");
        
        auto output_graphs = std::vector<std::pair<std::string, std::vector<std::string>>>
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
            {"segment_length", {"RAW"}},
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
        
        auto output_default_options = Output::Library::default_options_type
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
        
        CONFIG("auto_quit", false, "If set to true, the application will automatically save all results and export CSV files and quit, after the analysis is complete."); // save and quit after analysis is done
        CONFIG("auto_apply", false, "If set to true, the application will automatically apply the network with existing weights once the analysis is done. It will then automatically correct and reanalyse the video.");
        CONFIG("auto_categorize", false, "If set to true, the program will try to load <video>_categories.npz from the `output_dir`. If successful, then categories will be computed according to the current categories_ settings. Combine this with the `auto_quit` parameter to automatically save and quit afterwards. If weights cannot be loaded, the app crashes.");
        CONFIG("auto_tags", false, "If set to true, the application will automatically apply available tag information once the results file has been loaded. It will then automatically correct potential tracking mistakes based on this information.");
        CONFIG("auto_tags_on_startup", false, "Used internally by the software.", SYSTEM);
        CONFIG("auto_no_memory_stats", true, "If set to true, no memory statistics will be saved on auto_quit.");
        CONFIG("auto_no_results", false, "If set to true, the auto_quit option will NOT save a .results file along with the NPZ (or CSV) files. This saves time and space, but also means that the tracked portion cannot be loaded via -load afterwards. Useful, if you only want to analyse the resulting data and never look at the tracked video again.");
        CONFIG("auto_no_tracking_data", false, "If set to true, the auto_quit option will NOT save any `output_graphs` tracking data - just the posture data (if enabled) and the results file (if not disabled). This saves time and space if that is a need.");
        CONFIG("auto_train", false, "If set to true, the application will automatically train the recognition network with the best track segment and apply it to the video.");
        CONFIG("auto_train_on_startup", false, "This is a parameter that is used by the system to determine whether `auto_train` was set on startup, and thus also whether a failure of `auto_train` should result in a crash (return code != 0).", SYSTEM);
        CONFIG("analysis_range", std::pair<long_t,long_t>(-1, -1), "Sets start and end of the analysed frames.");
        CONFIG("output_min_frames", uint16_t(1), "Filters all individual with less than N frames when exporting. Individuals with fewer than N frames will also be hidden in the GUI unless `gui_show_inactive_individuals` is enabled (default).");
        CONFIG("output_interpolate_positions", bool(false), "If turned on this function will linearly interpolate X/Y, and SPEED values, for all frames in which an individual is missing.");
        CONFIG("output_prefix", std::string(), "A prefix that is prepended to all output files (csv/npz).");
        CONFIG("output_graphs", output_graphs, "The functions that will be exported when saving to CSV, or shown in the graph. `[['X',[option], ...]]`");
        CONFIG("tracklet_export_difference_images", true, "If set to true, then all exported tracklet images are difference images. If set to false, all exported tracklet images are normal-color images.");
        CONFIG("tracklet_max_images", uint16_t(0), "Maximum number of images that are being output per tracklet given that `output_image_per_tracklet` is true. If the number is 0, then every image will be exported that has been recognized as an individual.");
        CONFIG("tracklet_normalize_orientation", true, "If enabled, all exported tracklet images are normalized according to the calculated posture orientation, so that all heads are looking to the left and only the body moves.");
        CONFIG("output_image_per_tracklet", false, "If set to true, the program will output one median image per tracklet (time-series segment) and save it alongside the npz/csv files.");
        CONFIG("output_csv_decimals", uint8_t(2), "Maximum number of decimal places that is written into CSV files (a text-based format for storing data). A value of 0 results in integer values.");
        CONFIG("output_invalid_value", output_invalid_t::inf, "Determines, what is exported in cases where the individual was not found (or a certain value could not be calculated). For example, if an individual is found but posture could not successfully be generated, then all posture-based values (e.g. `midline_length`) default to the value specified here. By default (and for historic reasons), any invalid value is marked by 'inf'.");
        CONFIG("output_format", output_format_t::npz, "When pressing the S(ave) button or using `auto_quit`, this setting allows to switch between CSV and NPZ output. NPZ files are recommended and will be used by default - some functionality (such as visual fields, posture data, etc.) will remain in NPZ format due to technical constraints.");
        CONFIG("output_heatmaps", false, "When set to true, heatmaps are going to be saved to a separate file, or set of files '_p*' - with all the settings in heatmap_* applied.");
        CONFIG("output_statistics", false, "Save an NPZ file containing an array with shape Nx16 and contents [`adding_seconds`, `combined_posture_seconds`, `number_fish`, `loading_seconds`, `posture_seconds`, `match_number_fish`, `match_number_blob`, `match_number_edges`, `match_stack_objects`, `match_max_edges_per_blob`, `match_max_edges_per_fish`, `match_mean_edges_per_blob`, `match_mean_edges_per_fish`, `match_improvements_made`, `match_leafs_visited`, `method_used`] and an 1D-array containing all frame numbers. If set to true, a file called '`output_dir`/`fish_data_dir`/`<filename>_statistics.npz`' will be created. This will not output anything interesting, if the data was loaded instead of analysed.");
        CONFIG("output_posture_data", false, "Save posture data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '`output_dir`/`fish_data_dir`/`<filename>_posture_fishXXX.npz`' will be created for each individual XXX.");
        CONFIG("output_recognition_data", false, "Save recognition / probability data npz file along with the usual NPZ/CSV files containing positions and such. If set to true, a file called '`output_dir`/`fish_data_dir`/`<filename>_recognition_fishXXX.npz`' will be created for each individual XXX.");
        CONFIG("output_normalize_midline_data", false, "If enabled: save a normalized version of the midline data saved whenever `output_posture_data` is set to true. Normalized means that the position of the midline points is normalized across frames (or the distance between head and point n in the midline array).");
        CONFIG("output_centered", false, "If set to true, the origin of all X and Y coordinates is going to be set to the center of the video.");
        CONFIG("output_default_options", output_default_options, "Default scaling and smoothing options for output functions, which are applied to functions in `output_graphs` during export.");
        CONFIG("output_annotations", output_annotations, "Units (as a string) of output functions to be annotated in various places like graphs.");
        CONFIG("output_frame_window", long_t(100), "If an individual is selected during CSV output, use these number of frames around it (or -1 for all frames).");
        CONFIG("smooth_window", uint32_t(2), "Smoothing window used for exported data with the #smooth tag.");
        
        CONFIG("tags_path", file::Path(""), "If this path is set, the program will try to find tags and save them at the specified location.");
        CONFIG("tags_image_size", Size2(32, 32), "The image size that tag images are normalized to.");
        CONFIG("tags_dont_track", true, "If true, disables the tracking of tags as objects in TRex. This means that tags are not displayed like other objects and are instead only used as additional 'information' to correct tracks. However, if you enabled `tags_saved_only` in TGrabs, setting this parameter to true will make your TRex look quite empty.");
        //CONFIG("correct_luminance", true, "", STARTUP);
        
        CONFIG("grid_points", std::vector<Vec2>{}, "Whenever there is an identification network loaded and this array contains more than one point `[[x0,y0],[x1,y1],...]`, then the network will only be applied to blobs within circles around these points. The size of these circles is half of the average distance between the points.");
        CONFIG("grid_points_scaling", float(0.8), "Scaling applied to the average distance between the points in order to shrink or increase the size of the circles for recognition (see `grid_points`).");
        CONFIG("recognition_segment_add_factor", float(1.5), "This factor will be multiplied with the probability that would be pure chance, during the decision whether a segment is to be added or not. The default value of 1.5 suggests that the minimum probability for each identity has to be 1.5 times chance (e.g. 0.5 in the case of two individuals).");
        CONFIG("recognition_save_progress_images", false, "If set to true, an image will be saved for all training epochs, documenting the uniqueness in each step.");
        CONFIG("recognition_shapes", std::vector<std::vector<Vec2>>(), "If `recognition_border` is set to 'shapes', then the identification network will only be applied to blobs within the convex shapes specified here.");
        CONFIG("recognition_border", recognition_border_t::none, "This defines the type of border that is used in all automatic recognition routines. Depending on the type set here, you might need to set other parameters as well (e.g. `recognition_shapes`). In general, this defines whether an image of an individual is usable for automatic recognition. If it is inside the defined border, then it will be passed on to the recognition network - if not, then it wont."
        );
        CONFIG("debug_recognition_output_all_methods", false, "If set to true, a complete training will attempt to output all images for each identity with all available normalization methods.");
        CONFIG("recognition_border_shrink_percent", float(0.3), "The amount by which the recognition border is shrunk after generating it (roughly and depends on the method).");
        CONFIG("recognition_border_size_rescale", float(0.5), "The amount that blob sizes for calculating the heatmap are allowed to go below or above values specified in `blob_size_ranges` (e.g. 0.5 means that the sizes can range between `blob_size_ranges.min * (1 - 0.5)` and `blob_size_ranges.max * (1 + 0.5)`).");
        CONFIG("recognition_smooth_amount", uint16_t(200), "If `recognition_border` is 'outline', this is the amount that the `recognition_border` is smoothed (similar to `outline_smooth_samples`), where larger numbers will smooth more.");
        CONFIG("recognition_coeff", uint16_t(50), "If `recognition_border` is 'outline', this is the number of coefficients to use when smoothing the `recognition_border`.");
        CONFIG("individual_image_normalization", individual_image_normalization_t::posture, "This enables or disable normalizing the images before training. If set to `none`, the images will be sent to the GPU raw - they will only be cropped out. Otherwise they will be normalized based on head orientation (posture) or the main axis calculated using `image moments`.");
        CONFIG("individual_image_size", Size2(80, 80), "Size of each image generated for network training.");
        CONFIG("individual_image_scale", float(1), "Scaling applied to the images before passing them to the network.");
        CONFIG("recognition_save_training_images", false, "If set to true, the program will save the images used for a successful training of the recognition network to the output path.");
        CONFIG("visual_identification_version", visual_identification_version_t::current, "Newer versions of TRex sometimes change the network layout for (e.g.) visual identification, which will make them incompatible with older trained models. This parameter allows you to change the expected version back, to ensure backwards compatibility.");
        CONFIG("gpu_enable_accumulation", true, "Enables or disables the idtrackerai-esque accumulation protocol cascade. It is usually a good thing to enable this (especially in more complicated videos), but can be disabled as a fallback (e.g. if computation time is a major constraint).");
        CONFIG("gpu_accepted_uniqueness", float(0), "If changed (from 0), the ratio given here will be the acceptable uniqueness for the video - which will stop accumulation if reached.");
        CONFIG("auto_train_dont_apply", false, "If set to true, setting `auto_train` will only train and not apply the trained network.");
        CONFIG("gpu_accumulation_enable_final_step", true, "If enabled, the network will be trained on all the validation + training data accumulated, as a last step of the accumulation protocol cascade. This is intentional overfitting.");
        CONFIG("gpu_learning_rate", float(0.0001), "Learning rate for training a recognition network.");
        CONFIG("gpu_max_epochs", uchar(150), "Maximum number of epochs for training a recognition network (0 means infinite).");
        CONFIG("gpu_verbosity", gpu_verbosity_t::full, "Determines the nature of the output on the command-line during training. This does not change any behaviour in the graphical interface.");
        CONFIG("gpu_min_iterations", uchar(100), "Minimum number of iterations per epoch for training a recognition network.");
        CONFIG("gpu_max_cache", float(2), "Size of the image cache (transferring to GPU) in GigaBytes when applying the network.");
        CONFIG("gpu_max_sample_gb", float(2), "Maximum size of per-individual sample images in GigaBytes. If the collected images are too many, they will be sub-sampled in regular intervals.");
        CONFIG("gpu_min_elements", uint32_t(25000), "Minimum number of images being collected, before sending them to the GPU.");
        CONFIG("gpu_accumulation_max_segments", uint32_t(15), "If there are more than `gpu_accumulation_max_segments` global segments to be trained on, they will be filtered according to their quality until said limit is reached.");
        CONFIG("terminate_training", bool(false), "Setting this to true aborts the training in progress.");
        
        CONFIG("manually_approved", std::map<long_t,long_t>(), "A list of ranges of manually approved frames that may be used for generating training datasets, e.g. `{232:233,5555:5560}` where each of the numbers is a frame number. Meaning that frames 232-233 and 5555-5560 are manually set to be manually checked for any identity switches, and individual identities can be assumed to be consistent throughout these frames.");
        CONFIG("gui_focus_group", std::vector<track::Idx_t>(), "Focus on this group of individuals.");
        
        CONFIG("track_ignore", std::vector<std::vector<Vec2>>(), "If this is not empty, objects within the given rectangles or polygons (>= 3 points) `[[x0,y0],[x1,y1](, ...)], ...]` will be ignored during tracking.");
        CONFIG("track_include", std::vector<std::vector<Vec2>>(), "If this is not empty, objects within the given rectangles or polygons (>= 3 points) `[[x0,y0],[x1,y1](, ...)], ...]` will be the only objects being tracked. (overwrites `track_ignore`)");
        
        CONFIG("huge_timestamp_ends_segment", true, "");
        CONFIG("track_trusted_probability", float(0.5), "If the probability, that is used to assign an individual to an object, is smaller than this value, the current segment will be ended (thus this will also not be a consecutive segment anymore for this individual).");
        CONFIG("huge_timestamp_seconds", 0.2, "Defaults to 0.5s (500ms), can be set to any value that should be recognized as being huge.");
        CONFIG("gui_foi_name", std::string("correcting"), "If not empty, the gui will display the given FOI type in the timeline and allow to navigate between them via M/N.");
        CONFIG("gui_foi_types", std::vector<std::string>(), "A list of all the foi types registered.", STARTUP);
        
        CONFIG("gui_connectivity_matrix_file", file::Path(), "Path to connectivity table. Expected structure is a csv table with columns [frame | #(track_max_individuals^2) values] and frames in y-direction.");
        CONFIG("gui_connectivity_matrix", std::map<long_t, std::vector<float>>(), "Internally used to store the connectivity matrix.", STARTUP);
        
        std::vector<float> buffer {
            -0.2576632f , -0.19233586f,  0.00245493f,  0.00398822f,  0.35924019f
        };
        
        std::vector<float> matrix = {
            2.94508959e+03f,   0.00000000e+00f,   6.17255441e+02f,
            0.00000000e+00f,   2.94282514e+03f,   6.82473623e+02f,
            0.00000000e+00f,   0.00000000e+00f,   1.00000000e+00f
        };
        
        CONFIG("cam_undistort_vector", buffer, "");
        CONFIG("cam_matrix", matrix, "");
        CONFIG("cam_scale", float(1.0), "Scales the image down or up by the given factor.");
        CONFIG("cam_circle_mask", false, "If set to true, a circle with a diameter of the width of the video image will mask the video. Anything outside that circle will be disregarded as background.");
        CONFIG("cam_undistort", false, "If set to true, the recorded video image will be undistorted using `cam_undistort_vector` (1x5) and `cam_matrix` (3x3).");
        CONFIG("image_invert", false, "Inverts the image greyscale values before thresholding.");
        
#if !CMN_WITH_IMGUI_INSTALLED
        config["nowindow"] = true;
#endif
        
        config.set_do_print(old);
    }
    
    std::string generate_delta_config(bool include_build_number, std::vector<std::string> additional_exclusions) {
        auto keys = GlobalSettings::map().keys();
        std::stringstream ss;
        
        static sprite::Map config;
        static GlobalSettings::docs_map_t docs;
        config.set_do_print(false);
        
        if(config.empty())
            default_config::get(config, docs, NULL);
        
        std::vector<std::string> exclude_fields = {
            "analysis_paused",
            "filename",
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
            "cam_undistort_vector",
            "gpu_accepted_uniqueness",
            //"output_graphs",
            "auto_minmax_size",
            "auto_number_individuals",
            //"output_default_options",
            //"output_annotations",
            "log_file",
            "history_matching_log",
            "gui_foi_types",
            "gui_mode",
            "gui_frame",
            "gui_run",
            "settings_file",
            "nowindow",
            "wd",
            "gui_show_fish",
            "auto_quit",
            "auto_apply",
            "output_dir",
            "auto_categorize",
            "tags_path",
            "analysis_range",
            "output_prefix",
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
            "exec"
        };
        
        /**
         * Exclude some settings based on what would automatically be assigned
         * if they weren't set at all.
         */
        if(SETTING(cm_per_pixel).value<float>() == SETTING(meta_real_width).value<float>() / float(SETTING(video_size).value<Size2>().width))
        {
            exclude_fields.push_back("cm_per_pixel");
        }
        
        //if(GUI::instance() && SETTING(frame_rate).value<int>() == GUI::instance()->video_source()->framerate())
        //    exclude_fields.push_back("frame_rate");
        
        /**
         * Write the remaining settings.
         */
        for(auto &key : keys) {
            // dont write meta variables. this could be confusing if those
            // are loaded from the video file as well
            if(utils::beginsWith(key, "meta_")) {
                continue;
            }
            
            // UPDATE: write only keys with values that have changed compared
            // to the default options
            if(!config.has(key) || config[key] != GlobalSettings::get(key)) {
                if((include_build_number && utils::beginsWith(key, "build"))
                   || (GlobalSettings::access_level(key) <= AccessLevelType::STARTUP
                       && !contains(exclude_fields, key)
                       && !contains(additional_exclusions, key)))
                {
                    ss << key << " = " << GlobalSettings::get(key).get().valueString() << "\n";
                }
            }
        }
        
        return ss.str();
    }
    
    void register_default_locations() {
        file::DataLocation::register_path("app", [](file::Path path) -> file::Path {
            auto wd = SETTING(wd).value<file::Path>();
#if defined(TREX_CONDA_PACKAGE_INSTALL)
            auto conda_prefix = ::default_config::conda_environment_path();
            if(!conda_prefix.empty()) {
                wd = conda_prefix / "usr" / "share" / "trex";
            }
#elif __APPLE__
            wd = wd / ".." / "Resources";
#endif
            if(path.empty())
                return wd;
            return wd / path;
        });
        
        file::DataLocation::register_path("default.settings", [](file::Path) -> file::Path {
            auto settings_file = file::DataLocation::parse("app", "default.settings");
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file is an empty string.");
            
            return settings_file;
        });
        
        file::DataLocation::register_path("settings", [](file::Path path) -> file::Path {
            if(path.empty())
                path = SETTING(settings_file).value<Path>();
            if(path.empty()) {
                path = SETTING(filename).value<Path>();
                if(path.has_extension() && path.extension() == "pv")
                    path = path.remove_extension();
            }
            
            if(!path.has_extension() || path.extension() != "settings")
                path = path.add_extension("settings");
            
            auto settings_file = file::DataLocation::parse("input", path);
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file is an empty string.");
            
            return settings_file;
        });
        
        file::DataLocation::register_path("output_settings", [](file::Path) -> file::Path {
            file::Path settings_file = SETTING(filename).value<Path>().filename();
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file is an empty string.");
            
            if(!settings_file.has_extension() || settings_file.extension() != "settings")
                settings_file = settings_file.add_extension("settings");
            
            return file::DataLocation::parse("output", settings_file);
        });
        
        file::DataLocation::register_path("backup_settings", [](file::Path) -> file::Path {
            file::Path settings_file(SETTING(filename).value<Path>().filename());
            if(settings_file.empty())
                throw U_EXCEPTION("settings_file (and like filename) is an empty string.");
            
            if(!settings_file.has_extension() || settings_file.extension() != "settings")
                settings_file = settings_file.add_extension("settings");
            
            return file::DataLocation::parse("output", "backup") / settings_file;
        });
        
        file::DataLocation::register_path("input", [](file::Path filename) -> file::Path {
            if(!filename.empty() && filename.is_absolute()) {
#ifndef NDEBUG
                if(!SETTING(quiet))
                    print("Returning absolute path ",filename.str(),". We cannot be sure this is writable.");
#endif
                return filename;
            }
            
            auto path = SETTING(output_dir).value<file::Path>();
            if(path.empty())
                return filename;
            else
                return path / filename;
        });
        
        file::DataLocation::register_path("output", [](file::Path filename) -> file::Path {
            if(!filename.empty() && filename.is_absolute()) {
#ifndef NDEBUG
                if(!SETTING(quiet))
                    print("Returning absolute path ",filename.str(),". We cannot be sure this is writable.");
#endif
                return filename;
            }
            
            auto prefix = SETTING(output_prefix).value<std::string>();
            auto path = SETTING(output_dir).value<file::Path>();
            
            if(!prefix.empty()) {
                path = path / prefix;
            }
            
            if(path.empty())
                return filename;
            else
                return path / filename;
        });
    }


void load_string_with_deprecations(const file::Path& settings_file, const std::string& content, sprite::Map& map, AccessLevel accessLevel, bool quiet) {
    auto rejections = GlobalSettings::load_from_string(deprecations(), map, content, accessLevel);
    if(!rejections.empty()) {
        for (auto && [key, val] : rejections) {
            if (default_config::is_deprecated(key)) {
                auto r = default_config::replacement(key);
                if(r.empty()) {
                    if(!quiet)
                        FormatWarning("[", settings_file.c_str(),"] Deprecated setting ", key," = ",val," found. Ignoring, as there is no replacement.");
                } else {
                    if(!quiet)
                        print("[",settings_file.c_str(),"] Deprecated setting ",key," = ",val," found. Replacing with ",r," = ",val);
                    if(key == "whitelist_rect" || key == "exclude_rect" || key == "recognition_rect") {
                        auto values = Meta::fromStr<std::vector<float>>(val);
                        if(values.size() == 4) {
                            map[r] = std::vector<std::vector<Vec2>>{
                                { Vec2(values[0], values[1]), Vec2(values[0] + values[2], values[1] + values[3]) }
                            };
                            
                        } else if(!quiet)
                            FormatExcept("Invalid number of values while trying to correct ",val," deprecated parameter from ",key," to ",r,".");
                        
                    } else if(key == "whitelist_rects" || key == "exclude_rects") {
                        auto values = Meta::fromStr<std::vector<Bounds>>(val);
                        std::vector<std::vector<Vec2>> value;
                        
                        for(auto v : values) {
                            value.push_back({v.pos(), v.pos() + v.size()});
                        }
                        
                        map[r] = value;
                        
                    } else if(key == "output_npz") {
                        auto value = Meta::fromStr<bool>(val);
                        GlobalSettings::load_from_string(deprecations(), map, r + " = " + (value ? "npz" : "csv") + "\n", accessLevel);
                        
                    } else if(key == "match_use_approximate") {
                        auto value = Meta::fromStr<bool>(val);
                        GlobalSettings::load_from_string(deprecations(), map, r+" = "+(value ? "approximate" : "accurate")+"\n", accessLevel);
                    
                    } else if(key == "analysis_stop_after") {
                        GlobalSettings::load_from_string(deprecations(), map, r+" = [-1,"+val+"]\n", accessLevel);
                    } else if(key == "recognition_normalize_direction") {
                        bool value = utils::lowercase(val) != "false";
                        GlobalSettings::load_from_string(deprecations(), map, r+" = "+Meta::toStr(value ? individual_image_normalization_t::posture : individual_image_normalization_t::none)+"\n", accessLevel);
                        
                    } else GlobalSettings::load_from_string(deprecations(), map, r+" = "+val+"\n", accessLevel);
                }
            }
        }
    }
}

}
