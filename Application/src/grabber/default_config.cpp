#include "default_config.h"
#include <misc/SpriteMap.h>
#include <file/Path.h>
#include <misc/CropOffsets.h>
#include <video/GenericVideo.h>

#ifndef WIN32
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#endif
#include <misc/default_settings.h>

namespace grab {
#ifndef WIN32
struct passwd *pw = getpwuid(getuid());
const char *homedir = pw->pw_dir;
#else
const char *homedir = getenv("USERPROFILE");
#endif

using namespace file;
#define CONFIG adding.add

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
                    Warning("Setting '%S' has been removed from the tracker and will be ignored.", &key);
                } else
                    U_EXCEPTION("Setting '%S' is deprecated. Please use '%S' instead.", &key, &deprecated.at(utils::lowercase(key)));
            }
        }
    }
    
    void get(sprite::Map& config, GlobalSettings::docs_map_t& docs, decltype(GlobalSettings::set_access_level)* fn)
    {
        constexpr auto STARTUP = AccessLevelType::STARTUP;
        constexpr auto SYSTEM = AccessLevelType::SYSTEM;
        
        using namespace settings;
        Adding adding(config, docs, fn);
        
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
        
#if WITH_FFMPEG
        CONFIG("ffmpeg_path", file::Path(), "Path to an ffmpeg executable file. This is used for converting videos after recording them with the ffmpeg API. It is not a critical component of the software, but mostly for convenience.");
        CONFIG("ffmpeg_crf", uint32_t(20), "Quality for crf (see ffmpeg documentation) used when encoding as libx264.");
#endif
        
        CONFIG("app_name", std::string("TGrabs"), "Name of the application.", SYSTEM);
        CONFIG("version", std::string("1.0.6"), "Version of the application.", SYSTEM);
        CONFIG("color_channel", size_t(1), "Index (0-2) of the color channel to be used during video conversion, if more than one channel is present in the video file.");
        CONFIG("system_memory_limit", uint64_t(0), "Custom override of how many bytes of system RAM the program is allowed to fill. If `approximate_length_minutes` or `stop_after_minutes` are set, this might help to increase the resulting RAW video footage frame_rate.");
        
        CONFIG("frame_rate", int(-1), "Frame rate of the video will be set according to `cam_framerate` or the framerate of a given video for conversion.");
        CONFIG("blob_size_range", Rangef(0.01f, 500000.f), "Minimum or maximum size of the individuals on screen after thresholding. Anything smaller or bigger than these values will be disregarded as noise.");
        CONFIG("crop_offsets", CropOffsets(), "Percentage offsets [left, top, right, bottom] that will be cut off the input images (e.g. [0.1,0.1,0.5,0.5] will remove 10%% from the left and top and 50%% from the right and bottom and the video will be 60%% smaller in X and Y).");
        CONFIG("crop_window", false, "If set to true, the grabber will open a window before the analysis starts where the user can drag+drop points defining the crop_offsets.");
        
        CONFIG("approximate_length_minutes", uint32_t(0), "If available, please provide the approximate length of the video in minutes here, so that the encoding strategy can be chosen intelligently. If set to 0, infinity is assumed. This setting is overwritten by `stop_after_minutes`.");
        CONFIG("stop_after_minutes", uint32_t(0), "If set to a value above 0, the video will stop recording after X minutes of recording time.");
        
        CONFIG("threshold", int(9), "Threshold to be applied to the input image to find blobs.");
        CONFIG("threshold_maximum", int(255), "");
        
        CONFIG("web_quality", int(75), "Quality for images transferred over the web interface (0-100).");
        CONFIG("save_raw_movie", false, "Saves a RAW movie (.mov) with a similar name in the same folder, while also recording to a PV file. This might reduce the maximum framerate slightly, but it gives you the best of both worlds.");
        
        CONFIG("video_conversion_range", std::pair<long_t, long_t>(-1, -1), "If set to a valid value (!= -1), start and end values determine the range converted.");
        
        CONFIG("output_dir", Path(std::string(homedir)+"/Videos"), "Default output-/input-directory. Change this in order to omit paths in front of filenames for open and save.");
        CONFIG("output_prefix", std::string(), "A prefix that is added as a folder between `output_dir` and any subsequent filenames (`output_dir`/`output_prefix`/[filename]) or omitted if empty (default).", STARTUP);
        CONFIG("video_source", std::string("basler"), "Where the video is recorded from. Can be the name of a file, or one of the keywords ['basler', 'webcam', 'test_image'].", STARTUP);
        CONFIG("test_image", std::string("checkerboard"), "Defines, which test image will be used if `video_source` is set to 'test_image'.", STARTUP);
        CONFIG("filename", Path(""), "The output filename.", STARTUP);
        CONFIG("settings_file", Path(), "The settings filename.", STARTUP);
        CONFIG("recording", true, "If set to true, the program will record frames whenever individuals are found.");
        CONFIG("terminate", false, "Terminates the program gracefully.");
        CONFIG("terminate_error", false, "Internal variable.", STARTUP);
        
        CONFIG("web_time_threshold", float(0.125), "Time-threshold after which a new request can be answered (prevents DDoS).");
        CONFIG("grabber_use_threads", true, "Use threads to process images (specifically the blob detection).");
        CONFIG("adaptive_threshold_scale", float(2), "Threshold value to be used for adaptive thresholding, if enabled.");
        CONFIG("use_adaptive_threshold", false, "Enables or disables adaptive thresholding (slower than normal threshold). Deals better with weird backgrounds.");
        CONFIG("dilation_size", int32_t(0), "If set to a value greater than zero, detected shapes will be inflated (and potentially merged). When set to a value smaller than zero, detected shapes will be shrunk (and potentially split).", STARTUP);
        CONFIG("use_closing", false, "Toggles the attempt to close weird blobs using dilation/erosion with `closing_size` sized filters.", STARTUP);
        CONFIG("closing_size", int(3), "Size of the dilation/erosion filters for if `use_closing` is enabled.", STARTUP);
        CONFIG("image_adjust", false, "Converts the image to floating-point (temporarily) and performs f(x,y) * `image_contrast_increase` + `image_brightness_increase` plus, if enabled, squares the image (`image_square_brightness`).");
        CONFIG("image_square_brightness", false, "Squares the floating point input image after background subtraction. This brightens brighter parts of the image, and darkens darker regions.");
        CONFIG("image_contrast_increase", float(3), "Value that is multiplied to the preprocessed image before applying the threshold (see `image_adjust`). The neutral value is 1 here.");
        CONFIG("image_brightness_increase", float(0), "Value that is added to the preprocessed image before applying the threshold (see `image_adjust`). The neutral value is 0 here.");
        CONFIG("enable_difference", true, "Enables background subtraction. If disabled, `threshold` will be applied to the raw greyscale values instead of difference values.");
        CONFIG("enable_absolute_difference", true, "Uses absolute difference values and disregards anything below `threshold` during conversion.");
        CONFIG("correct_luminance", false, "Attempts to correct for badly lit backgrounds by evening out luminance across the background.", STARTUP);
        CONFIG("equalize_histogram", false, "Equalizes the histogram of the image before thresholding and background subtraction.");
        CONFIG("quit_after_average", false, "If set to true, this will terminate the program directly after generating (or loading) a background average image.", STARTUP);
        CONFIG("averaging_method", averaging_method_t::mean, "Determines the way in which the background samples are combined. The background generated in the process will be used to subtract background from foreground objects during conversion.");
        CONFIG("average_samples", uint32_t(100), "Number of samples taken to generate an average image. Usually fewer are necessary for `average_method`s max, and min.");
        CONFIG("reset_average", false, "If set to true, the average will be regenerated using the live stream of images (video or camera).");
        
        CONFIG("video_size", Size2(-1,-1), "Is set to the dimensions of the resulting image.", SYSTEM);
        CONFIG("cam_resolution", cv::Size(2048, 2048), "[BASLER] Defines the dimensions of the camera image.", STARTUP);
        CONFIG("cam_framerate", int(30), "[BASLER] If set to anything else than 0, this will limit the basler camera framerate to the given fps value.", STARTUP);
        CONFIG("cam_limit_exposure", int(5500), "[BASLER] Sets the cameras exposure time in micro seconds.");
        
        CONFIG("cam_circle_mask", false, "If set to true, a circle with a diameter of the width of the video image will mask the video. Anything outside that circle will be disregarded as background.");
        CONFIG("cam_undistort", false, "If set to true, the recorded video image will be undistorted using `cam_undistort_vector` (1x5) and `cam_matrix` (3x3).");
        CONFIG("image_invert", false, "Inverts the image greyscale values before thresholding.");
        
        CONFIG("gui_interface_scale",
#if defined(__linux__)
              float(1.25)
#elif defined(WIN32) || defined(__WIN32__)
              float(1.25)
#else
              float(0.75)
#endif
               , "A lower number will make the texts and GUI elements bigger.");
        
        CONFIG("meta_species", std::string(""), "Name of the species used.");
        CONFIG("meta_age_days", long_t(-1), "Age of the individuals used in days.");
        CONFIG("meta_conditions", std::string(""), "Treatment name.");
        CONFIG("meta_misc", std::string(""), "Other information.");
        CONFIG("meta_real_width", float(30), "Width of whatever is visible in the camera frame from left to right. Used to calculate `cm_per_pixel` ratio.", STARTUP);
        CONFIG("meta_source_path", Path(""), "Path of the original video file for conversions (saved as debug info).", STARTUP);
        CONFIG("meta_cmd", std::string(""), "Command-line of the framegrabber when conversion was started.", STARTUP);
        CONFIG("meta_build", std::string(""), "The current commit hash. The video is branded with this information for later inspection of errors that might have occured.", STARTUP);
        CONFIG("meta_conversion_time", std::string(""), "This contains the time of when this video was converted / recorded as a string.", STARTUP);
        
        CONFIG("mask_path", Path(""), "Path to a video file containing a mask to be applied to the video while recording. Only works for conversions.");
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
            "frame_rate",
            //"cam_undistort",
            "cam_undistort_vector",
            "cam_matrix"
        }, "The given settings values will be written to the video file.");

        CONFIG("nowindow", false, "Start without a window enabled (for terminal-only use).");
        CONFIG("enable_closed_loop", false, "When enabled, live tracking will be executed for every frame received. Frames will be sent to the 'closed_loop.py' script - see this script for more information. Sets `enable_live_tracking` to true. Allows the tracker to skip frames by default, in order to catch up to the video.");
        CONFIG("enable_live_tracking", false, "When enabled, the program will save a .results file for the recorded video plus export the data (see `output_graphs` in the tracker documentation).");
        CONFIG("grabber_force_settings", false, "If set to true, live tracking will always overwrite a settings file with `filename`.settings in the output folder.");
        
#if !CMN_WITH_IMGUI_INSTALLED
        config["nowindow"] = true;
#endif
    }
}

}
