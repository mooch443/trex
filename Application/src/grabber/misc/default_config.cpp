#include "default_config.h"
#include <misc/SpriteMap.h>
#include <file/Path.h>
#include <misc/CropOffsets.h>
#include <video/GenericVideo.h>
#include <video/AveragingAccumulator.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>
#include <processing/Background.h>

#ifndef WIN32
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif
#include <misc/default_settings.h>

#include "GitSHA1.h"

namespace cmn {

ENUM_CLASS_DOCS(meta_encoding_t,
                "Grayscale video, calculated by simply extracting one channel (default R) from the video.",
                "Encode all colors into a 256-colors unsigned 8-bit integer. The top 2 bits are blue (4 shades), the following 3 bits green (8 shades) and the last 3 bits red (8 shades).",
                "Encode all colors into a full color 8-bit R8G8B8 array.");

}

namespace grab {
#ifndef WIN32
struct passwd *pw = getpwuid(getuid());
const char *homedir = pw->pw_dir;
#else
const char *homedir = getenv("USERPROFILE");
#endif

using namespace cmn;
using namespace cmn::file;
#define CONFIG adding.add<ParameterCategoryType::TRACKING>
namespace default_config {

    static const std::map<std::string, std::string> deprecated = {
        {"fish_minmax_size", "blob_size_range"},
        {"use_dilation", "dilation_size"},
        {"threshold_constant", "threshold"}
    };

    const std::map<std::string, std::string>& deprecations() {
        return deprecated;
    }
        
    void warn_deprecated(sprite::Map& map) {
        for(auto &key : map.keys()) {
            if(deprecated.find(utils::lowercase(key)) != deprecated.end()) {
                if(deprecated.at(utils::lowercase(key)).empty()) {
                    Print("Setting ",key," has been removed from the tracker and will be ignored.");
                } else
                    throw U_EXCEPTION("Setting '",key,"' is deprecated. Please use '",deprecated.at(utils::lowercase(key)),"' instead.");
            }
        }
    }
    
    void get(sprite::Map& config, GlobalSettings::docs_map_t& docs, std::function<void(const std::string& name, AccessLevel w)> fn)
    {
        constexpr auto STARTUP = AccessLevelType::STARTUP;
        constexpr auto SYSTEM = AccessLevelType::SYSTEM;
        constexpr auto LOAD = AccessLevelType::LOAD;
        constexpr auto INIT = AccessLevelType::INIT;
        
        using namespace settings;
        Adding adding(config, docs, fn);
        
        std::vector<double> buffer {
        };
        
        std::vector<double> matrix = {
        };
        
        CONFIG("cam_undistort_vector", buffer, "");
        CONFIG("cam_matrix", matrix, "");
        CONFIG("cam_scale", float(1.0), "Scales the image down or up by the given factor.");
        
#if WITH_PYLON
        CONFIG("cam_serial_number", std::string(), "Serial number of a Basler camera you want to choose, if multiple are present.");
#endif
        
#if WITH_FFMPEG
        CONFIG("ffmpeg_path", file::Path(), "Path to an ffmpeg executable file. This is used for converting videos after recording them with the ffmpeg API. It is not a critical component of the software, but mostly for convenience.");
        CONFIG("ffmpeg_crf", uint32_t(20), "Quality for crf (see ffmpeg documentation) used when encoding as libx264.");
#endif
        
        CONFIG("app_name", std::string("TGrabs"), "Name of the application.", SYSTEM);
        CONFIG("version", std::string(g_GIT_DESCRIBE_TAG)+(std::string(g_GIT_CURRENT_BRANCH) != "main" ? "_"+std::string(g_GIT_CURRENT_BRANCH) : ""), "Version of the application.", SYSTEM);
        CONFIG("color_channel", uint8_t(1), "Index (0-2) of the color channel to be used during video conversion, if more than one channel is present in the video file.");
        CONFIG("system_memory_limit", uint64_t(0), "Custom override of how many bytes of system RAM the program is allowed to fill. If `approximate_length_minutes` or `stop_after_minutes` are set, this might help to increase the resulting RAW video footage frame_rate.");
        
        CONFIG("frame_rate", uint32_t(0), "Frame rate of the video will be set according to `cam_framerate` or, for video conversion, the metadata of a given video. If you want to modify your frame rate, please set either `cam_framerate` or `frame_rate` during conversion.", LOAD);
        CONFIG("blob_size_range", Rangef(0.01f, 500000.f), "Minimum or maximum size of the individuals on screen after thresholding. Anything smaller or bigger than these values will be disregarded as noise.");
        CONFIG("crop_offsets", CropOffsets(), "Percentage offsets [left, top, right, bottom] that will be cut off the input images (e.g. [0.1,0.1,0.5,0.5] will remove 10%% from the left and top and 50%% from the right and bottom and the video will be 60%% smaller in X and Y).");
        CONFIG("crop_window", false, "If set to true, the grabber will open a window before the analysis starts where the user can drag+drop points defining the crop_offsets.");
        
        CONFIG("approximate_length_minutes", uint32_t(0), "If available, please provide the approximate length of the video in minutes here, so that the encoding strategy can be chosen intelligently. If set to 0, infinity is assumed. This setting is overwritten by `stop_after_minutes`.");
        CONFIG("stop_after_minutes", uint32_t(0), "If set to a value above 0, the video will stop recording after X minutes of recording time.");
        
        CONFIG("threshold", int(15), "Threshold to be applied to the input image to find blobs.");
        CONFIG("threshold_maximum", int(255), "");
        
        CONFIG("web_quality", int(75), "Quality for images transferred over the web interface (0-100).");
        CONFIG("save_raw_movie", false, "Saves a RAW movie (.mov) with a similar name in the same folder, while also recording to a PV file. This might reduce the maximum framerate slightly, but it gives you the best of both worlds.", INIT);
        CONFIG("save_raw_movie_path", file::Path(), "The path to the raw movie file. If empty, the same path as the PV file will be used (but as a .mov).", INIT);
        
        CONFIG("video_conversion_range", Range<long_t>(-1, -1), "If set to a valid value (!= -1), start and end values determine the range converted.", INIT);
        
        CONFIG("output_dir", Path(""), "Default output-/input-directory. Change this in order to omit paths in front of filenames for open and save.", INIT);
        CONFIG("output_prefix", std::string(), "A prefix that is added as a folder between `output_dir` and any subsequent filenames (`output_dir`/`output_prefix`/[filename]) or omitted if empty (default).", INIT);
        CONFIG("video_source", std::string("webcam"), "Where the video is recorded from. Can be the name of a file, or one of the keywords ['basler', 'webcam', 'test_image'].", LOAD);
        CONFIG("test_image", std::string("checkerboard"), "Defines, which test image will be used if `video_source` is set to 'test_image'.", LOAD);
        CONFIG("filename", Path(""), "The output filename.", LOAD);
        CONFIG("settings_file", Path(), "The settings filename.", LOAD);
        CONFIG("recording", true, "If set to true, the program will record frames whenever individuals are found.");
        CONFIG("terminate", false, "Terminates the program gracefully.", SYSTEM);
        CONFIG("terminate_error", false, "Internal variable.", SYSTEM);
        
        CONFIG("web_time_threshold", float(0.125), "Time-threshold after which a new request can be answered (prevents DDoS).");
        CONFIG("tgrabs_use_threads", true, "Use threads to process images (specifically the blob detection).", STARTUP);
        CONFIG("video_reading_use_threads", true, "Use threads to read images from a video file.", STARTUP);
        CONFIG("adaptive_threshold_scale", float(2), "Threshold value to be used for adaptive thresholding, if enabled.");
        CONFIG("use_adaptive_threshold", false, "Enables or disables adaptive thresholding (slower than normal threshold). Deals better with weird backgrounds.");
        CONFIG("dilation_size", int32_t(0), "If set to a value greater than zero, detected shapes will be inflated (and potentially merged). When set to a value smaller than zero, detected shapes will be shrunk (and potentially split).", INIT);
        CONFIG("use_closing", false, "Toggles the attempt to close weird blobs using dilation/erosion with `closing_size` sized filters.", INIT);
        CONFIG("closing_size", int(3), "Size of the dilation/erosion filters for if `use_closing` is enabled.", INIT);
        CONFIG("image_adjust", false, "Converts the image to floating-point (temporarily) and performs f(x,y) * `image_contrast_increase` + `image_brightness_increase` plus, if enabled, squares the image (`image_square_brightness`).");
        CONFIG("image_square_brightness", false, "Squares the floating point input image after background subtraction. This brightens brighter parts of the image, and darkens darker regions.");
        CONFIG("image_contrast_increase", float(3), "Value that is multiplied to the preprocessed image before applying the threshold (see `image_adjust`). The neutral value is 1 here.");
        CONFIG("image_brightness_increase", float(0), "Value that is added to the preprocessed image before applying the threshold (see `image_adjust`). The neutral value is 0 here.");
        CONFIG("blur_difference", false, "Enables a special mode that will 1. truncate all difference values below threshold, 2. blur the remaining difference, 3. threshold again.");
        CONFIG("enable_difference", true, "Enables background subtraction. If disabled, `threshold` will be applied to the raw greyscale values instead of difference values.");
        CONFIG("track_absolute_difference", true, "If enabled, uses absolute difference values and disregards any pixel |p| < `threshold` during conversion. Otherwise the equation is p < `threshold`, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as `enable_absolute_difference`, but during tracking instead of converting.");
        CONFIG("enable_absolute_difference", true, "If enabled, uses absolute difference values and disregards any pixel |p| < `threshold` during conversion. Otherwise the equation is p < `threshold`, meaning that e.g. bright spots may not be considered trackable when dark spots would. Same as `track_absolute_difference`, but during conversion instead of tracking.");
        CONFIG("correct_luminance", false, "Attempts to correct for badly lit backgrounds by evening out luminance across the background.", INIT);
        CONFIG("equalize_histogram", false, "Equalizes the histogram of the image before thresholding and background subtraction.");
        CONFIG("quit_after_average", false, "If set to true, this will terminate the program directly after generating (or loading) a background average image.", STARTUP);
        CONFIG("averaging_method", averaging_method_t::mean, "Determines the way in which the background samples are combined. The background generated in the process will be used to subtract background from foreground objects during conversion.");
        CONFIG("average_samples", uint32_t(25), "Number of samples taken to generate an average image. Usually fewer are necessary for `averaging_method`'s max, and min.");
        CONFIG("reset_average", false, "If set to true, the average will be regenerated using the live stream of images (video or camera).");
        CONFIG("solid_background_color", uchar(255), "A greyscale value in case `enable_difference` is set to false - TGrabs will automatically generate a background image with the given color.");
        CONFIG("video_size", Size2(-1,-1), "Is set to the dimensions of the resulting image.", LOAD);
        CONFIG("cam_resolution", Size2(-1, -1), "Defines the dimensions of the camera image.", LOAD);
        CONFIG("cam_framerate", int(-1), "If set to anything else than 0, this will limit the basler camera framerate to the given fps value.", LOAD);
        CONFIG("cam_limit_exposure", int(5500), "Sets the cameras exposure time in micro seconds.");
        
        CONFIG("tags_model_path", file::Path("tag_recognition_network.h5"), "The pretrained model used to recognize QRcodes/tags according to `https://github.com/jgraving/pinpoint/blob/2d7f6803b38f52acb28facd12bd106754cad89bd/barcodes/old_barcodes_py2/4x4_4bit/master_list.pdf`. Path to a pretrained network .h5 file that takes 32x32px images of tags and returns a (N, 122) shaped tensor with 1-hot encoding.");
        
        CONFIG("tags_maximum_image_size", Size2(80,80), "Tags that are bigger than these pixel dimensions may be cropped off. All extracted tags are then pre-aligned to any of their sides, and normalized/scaled down or up to a 32x32 picture (to make life for the machine learning network easier).");
        CONFIG("tags_size_range", Range<double>(0.08,2), "The minimum and maximum area accepted as a (square) physical tag on the individuals.");
        CONFIG("tags_equalize_hist", false, "Apply a histogram equalization before applying a threshold. Mostly this should not be necessary due to using adaptive thresholds anyway.");
        CONFIG("tags_threshold", int(-5), "Threshold passed on to cv::adaptiveThreshold, lower numbers (below zero) are equivalent to higher thresholds / removing more of the pixels of objects and shrinking them. Positive numbers may invert the image/mask.");
        CONFIG("tags_save_predictions", false, "Save images of tags, sorted into folders labelled according to network predictions (i.e. 'tag 22') to '`output_dir` / `tags_` `filename` / `<individual>.<frame>` / `*`'. ");
        CONFIG("tags_num_sides", Range<int>(3,7), "The number of sides of the tag (e.g. should be 4 if it is a rectangle).");
        CONFIG("tags_approximation", 0.025f, "Higher values (up to 1.0) will lead to coarser approximation of the rectangle/tag shapes.");
        CONFIG("tags_enable", false, "(beta) If enabled, TGrabs will search for (black) square shapes with white insides (and other stuff inside them) - like QRCodes or similar tags. These can then be recognized using a pre-trained machine learning network (see `tags_recognize`), and/or exported to PNG files using `tags_save_predictions`.");
        CONFIG("tags_debug", false, "(beta) Enable debugging for tags.");
        CONFIG("tags_recognize", false, "(beta) Apply an existing machine learning network to turn images of tags into tag ids (numbers, e.g. 1-122). Be sure to set `tags_model_path` along-side this.");
        CONFIG("tags_saved_only", false, "(beta) If set to true, all objects other than the detected blobs are removed and will not be written to the output video file.");
         
         
        CONFIG("cam_circle_mask", false, "If set to true, a circle with a diameter of the width of the video image will mask the video. Anything outside that circle will be disregarded as background.");
        CONFIG("cam_undistort", false, "If set to true, the recorded video image will be undistorted using `cam_undistort_vector` (1x5) and `cam_matrix` (3x3).");
        CONFIG("image_invert", false, "Inverts the image greyscale values before thresholding.");
        
        CONFIG("gui_interface_scale", Float2_t(1), "A lower number will make the texts and GUI elements bigger.", SYSTEM);
        
        CONFIG("meta_encoding", meta_encoding_t::gray, "The encoding used for the given .pv video.");
        CONFIG("meta_species", std::string(""), "Name of the species used.");
        CONFIG("meta_age_days", long_t(-1), "Age of the individuals used in days.");
        CONFIG("meta_conditions", std::string(""), "Treatment name.");
        CONFIG("meta_misc", std::string(""), "Other information.");
        CONFIG("meta_real_width", float(30), "Width of whatever is visible in the camera frame from left to right. Used to calculate `cm_per_pixel` ratio.", INIT);
        CONFIG("meta_source_path", std::string(""), "Path of the original video file for conversions (saved as debug info).", LOAD);
        CONFIG("meta_cmd", std::string(""), "Command-line of the framegrabber when conversion was started.", SYSTEM);
        CONFIG("meta_build", std::string(""), "The current commit hash. The video is branded with this information for later inspection of errors that might have occured.", SYSTEM);
        CONFIG("meta_video_size", Size2(), "Resolution of the original video.", LOAD);
        CONFIG("meta_video_scale", float(1), "Scale applied to the original video / footage.", LOAD);
        
        CONFIG("meta_conversion_time", std::string(""), "This contains the time of when this video was converted / recorded as a string.", LOAD);
        
        CONFIG("mask_path", Path(""), "Path to a video file containing a mask to be applied to the video while recording. Only works for conversions.", STARTUP);
        CONFIG("log_file", Path(""), "If set to a filename, this will save output to a log file.");
        
        CONFIG("meta_write_these", std::vector<std::string>{
            "meta_species",
            "meta_age_days",
            "meta_conditions",
            "meta_misc",
            "cam_limit_exposure",
            "meta_real_width",
            "meta_source_path",
            "meta_cmd",
            "meta_build",
            "meta_conversion_time",
            "meta_video_scale",
            "meta_video_size",
            "detect_classes",
            "meta_encoding",
            "detect_skeleton",

            "frame_rate",
            "calculate_posture",
            "cam_undistort_vector",
            "cam_matrix",
            "cm_per_pixel",
            "track_size_filter",
            "track_threshold",
            "track_posture_threshold",
            "track_do_history_split",
            "track_max_individuals",
            "track_background_subtraction",
            "track_max_speed",
            "detect_model",
            "region_model",
            "detect_resolution",
            "region_resolution",
            "detect_batch_size",
            "detect_type",
            "detect_iou_threshold",
            "detect_conf_threshold",
            "video_conversion_range",
            "detect_batch_size",
            "threshold",
            "output_dir",
            "output_prefix",
            "filename"

        }, "The given settings values will be written to the video file.");

        CONFIG("nowindow", false, "Start without a window enabled (for terminal-only use).", STARTUP);
        CONFIG("closed_loop_enable", false, "When enabled, live tracking will be executed for every frame received. Frames will be sent to the 'closed_loop.py' script - see this script for more information. Sets `enable_live_tracking` to true. Allows the tracker to skip frames by default, in order to catch up to the video.");
        CONFIG("closed_loop_path", file::Path("closed_loop_beta.py"), "Set the path to a Python file to be used in closed_loop. Please also enable closed loop processing by setting `closed_loop_enable` to true.");
        CONFIG("enable_live_tracking", false, "When enabled, the program will save a .results file for the recorded video plus export the data (see `output_graphs` in the tracker documentation).");
        CONFIG("grabber_force_settings", false, "If set to true, live tracking will always overwrite a settings file with `filename`.settings in the output folder.");
        
#if !CMN_WITH_IMGUI_INSTALLED
        config["nowindow"] = true;
#endif
    }
}

}
