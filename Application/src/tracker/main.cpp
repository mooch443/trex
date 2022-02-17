//===========================================================================//
//                                                                           //
// Project: TRrex                                                            //
//                                                                           //
//===========================================================================//

//-Includes--------------------------------------------------------------------

#include <signal.h>
#ifndef WIN32
#include <execinfo.h>
#endif

#include <queue>
#include <thread>
#include <misc/stacktrace.h>

#include <time.h>
#include <iomanip>

#include <python/GPURecognition.h>

#include <misc/CommandLine.h>
#include <gui/SFLoop.h>
#include <tracking/Tracker.h>
#include <gui/DrawStructure.h>
#include <misc/default_config.h>
#include <misc/OutputLibrary.h>
#include <misc/Output.h>
#include <tracking/Recognition.h>
#include <gui/WorkProgress.h>
#include <gui/CheckUpdates.h>

#include <tracking/SplitBlob.h>

#include <misc/ConnectedTasks.h>
#include <processing/PadImage.h>
#include <processing/LuminanceGrid.h>
#include <processing/CPULabeling.h>
#include <tracking/HistorySplit.h>

#include <misc/Image.h>
#include <gui/gui.h>

#include <tracking/DetectTag.h>
#include <misc/default_settings.h>
#include <misc/PixelTree.h>

#include <misc/PVBlob.h>

#include <misc/pretty.h>
#include <misc/default_settings.h>

#if WITH_GITSHA1
#include "GitSHA1.h"
#endif

#include <gui/IMGUIBase.h>
#include <gui/FileChooser.h>
#include <gui/types/Checkbox.h>
#include <misc/MemoryStats.h>
#include <tracking/Categorize.h>
#include <gui/DrawCVBase.h>
#include "VideoOpener.h"

#if WIN32
#include <shellapi.h>
#endif

#include <opencv2/core/utils/logger.hpp>

#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

#if __linux__                                                                
#include <X11/Xlib.h>                                                        
#endif 

//-Functions-------------------------------------------------------------------

using namespace track;
using namespace file;

std::mutex data_mutex;
double data_sec = 0.0, data_kbytes = 0.0;
double frames_sec = 0, frames_count = 0;

ENUM_CLASS(Arguments,
           d,dir,i,input,s,settings,nowindow,load,h,fs,p,r,update,quiet)

#ifndef WIN32
struct sigaction sigact;
#endif
char *progname;

void panic(const char *fmt, ...) {
    CrashProgram::crash_pid = std::this_thread::get_id();
    
    printf("\033[%02d;%dmPanic ", 0, 0);

    char buf[50];
    va_list argptr;
    va_start(argptr, fmt);
    vsprintf(buf, fmt, argptr);
    va_end(argptr);
    fprintf(stderr, "%s", buf);
    exit(-1);
}

bool pause_stuff = false;

#ifndef WIN32
static void dumpstack(void) {
    void *array[20];
    int size;
#ifdef __unix__
    printf("\033[%02d;%dm", 0, 0);
#endif
    size = backtrace(array, 20);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    
    return;
}

static void signal_handler(int sig) {
    if (sig == SIGHUP) panic("FATAL: Program hanged up\n");
    if (sig == SIGSEGV || sig == SIGBUS){
        dumpstack();
        panic("FATAL: %s Fault. Logged StackTrace\n", (sig == SIGSEGV) ? "Segmentation" : ((sig == SIGBUS) ? "Bus" : "Unknown"));
    }
    if (sig == SIGQUIT) {
        pause_stuff = true;
    }
    if (sig == SIGKILL) panic("KILL signal ended program\n");
    if(sig == SIGINT) {
        if(!SETTING(error_terminate))
            SETTING(error_terminate) = true;
        if(!SETTING(terminate)) {
            SETTING(terminate) = true;
        }
        
        if(Tracker::instance()) {
            Tracker::emergency_finish();
        }
        
        signal(SIGINT, SIG_DFL); // don't catch the signal anymore
        kill(getpid(), SIGINT);
    }
}

static void at_exit() {
    sigemptyset(&sigact.sa_mask);
#ifdef __unix__
    printf("\033[%02d;%dm", 0, 0);
#endif
}
#endif

void init_signals() {
    CrashProgram::main_pid = std::this_thread::get_id();
    
#ifndef WIN32
    sigact.sa_handler = signal_handler;
    sigemptyset(&sigact.sa_mask);
    sigact.sa_flags = 0;
    sigaction(SIGINT, &sigact, (struct sigaction *)NULL);
    
    sigaddset(&sigact.sa_mask, SIGSEGV);
    sigaction(SIGSEGV, &sigact, (struct sigaction *)NULL);
    
    sigaddset(&sigact.sa_mask, SIGBUS);
    sigaction(SIGBUS, &sigact, (struct sigaction *)NULL);
    
    sigaddset(&sigact.sa_mask, SIGQUIT);
    sigaction(SIGQUIT, &sigact, (struct sigaction *)NULL);
    
    sigaddset(&sigact.sa_mask, SIGHUP);
    sigaction(SIGHUP, &sigact, (struct sigaction *)NULL);
    
    sigaddset(&sigact.sa_mask, SIGKILL);
    sigaction(SIGKILL, &sigact, (struct sigaction *)NULL);
#endif
}

#include <gui/GLImpl.h>

int main(int argc, char** argv)
{
#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
    
#if __APPLE__
    std::string PATH = (std::string)getenv("PATH");
    if(!utils::contains(PATH, "/usr/local/bin")) {
        PATH += ":/usr/local/bin";
        setenv("PATH", PATH.c_str(), 1);
    }
#endif
    
    default_config::register_default_locations();
    GlobalSettings::map().set_do_print(true);
    
    gui::init_errorlog();
    set_thread_name("main");
    
    progname = *(argv);
#ifndef WIN32
    std::atexit(at_exit);
    setenv("KMP_DUPLICATE_LIB_OK", "True", 1);
#endif
    init_signals();
    
#if TRACKER_GLOBAL_THREADS
    Warning("Using only %d threads (-DTRACKER_GLOBAL_THREADS).", TRACKER_GLOBAL_THREADS);
#endif
#if __linux__
    XInitThreads();
#endif
    
    srand ((uint)time(NULL));
    
    FILE *log_file = NULL;
    std::mutex log_mutex;
    
    Timer timer;
    
    /**
     * Set default values for global settings
     */
    using namespace Output;
    DebugHeader("LOADING DEFAULT SETTINGS");
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    GlobalSettings::map().dont_print("gui_frame");
    GlobalSettings::map().dont_print("gui_focus_group");
    
    if(argc == 2) {
        if(std::string(argv[1]) == "-options") {
            for(auto arg : Arguments::names) {
                printf("-%s ", arg);
            }
            exit(0);
            
        } else if(std::string(argv[1]) == "-path") {
            default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
            default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
            
            CommandLine cmd(argc, argv, true);
            cmd.cd_home();
            
            if(Path("default.settings").exists()) {
                default_config::warn_deprecated("default.settings", GlobalSettings::load_from_file(default_config::deprecations(), "default.settings", AccessLevelType::STARTUP));
            }
            
            printf("%s", SETTING(output_dir).value<file::Path>().str().c_str());
            
            exit(0);
            
        }
    }
    
    file::Path load_results_from;
#ifdef WIN32
    LPSTR lpFilePart;
    char filename[MAX_PATH];

    if(!SearchPath( NULL, "ffmpeg", ".exe", MAX_PATH, filename, &lpFilePart))
    {
        auto conda_prefix = ::default_config::conda_environment_path().str();
        Debug("Conda prefix: %S", &conda_prefix);
        if(!conda_prefix.empty()) {
            std::set<file::Path> files;
            if (file::Path(conda_prefix + "/bin").exists()) {
                files = file::Path(conda_prefix + "/bin").find_files();
            }
            else
                files = file::Path(conda_prefix + "/Library/bin").find_files();

            for(auto file : files) {
                if(file.filename() == "ffmpeg" || file.filename() == "ffmpeg.exe") {
                    Debug("Found ffmpeg in '%S'", &file.str());
                    SETTING(ffmpeg_path) = file;
                    break;
                }
            }
        }
        
        if(SETTING(ffmpeg_path).value<file::Path>().empty())
            Warning("Cannot find ffmpeg.exe in search paths.");
    } else
        SETTING(ffmpeg_path) = file::Path(std::string(filename));
#else
    {
        auto PATH = getenv("PATH");
        if(PATH) {
            auto parts = utils::split(std::string(PATH), ':');
            auto conda_prefix = ::default_config::conda_environment_path().str();
            if(!conda_prefix.empty()) {
                parts.insert(parts.begin(), conda_prefix+"/bin");
            }
            
            for(auto &part : parts) {
                if(file::Path(part).exists()) {
                    auto files = file::Path(part).find_files();
                    for(auto file : files) {
                        if(file.filename() == "ffmpeg") {
                            Debug("Found ffmpeg in '%S'", &file.str());
                            SETTING(ffmpeg_path) = file;
                            break;
                        }
                    }
                }
                
                if(!SETTING(ffmpeg_path).value<file::Path>().empty())
                    break;
            }
        }
    }
#endif

#if WITH_GITSHA1
    SETTING(build) = std::string(g_GIT_SHA1);
#else
    SETTING(build) = std::string("<undefined>");
#endif
    std::stringstream ss;
    for(int i=0; i<argc; ++i) {
        ss << " " << argv[i];
    }
    SETTING(cmd_line) = ss.str();
    
    DebugHeader("LOADING COMMANDLINE");
    CommandLine cmd(argc, argv, true);
    cmd.cd_home();
#if __APPLE__
    std::string _wd = "../Resources/";
    if (!chdir(_wd.c_str()))
        Debug("Changed directory to '%S'.", &_wd);
    else
        Error("Cannot change directory to '%S'.", &_wd);
#elif defined(TREX_CONDA_PACKAGE_INSTALL)
    auto conda_prefix = ::default_config::conda_environment_path().str();
    if(!conda_prefix.empty()) {
        file::Path _wd(conda_prefix);
        _wd = _wd / "usr" / "share" / "trex";
        //Debug("change directory to conda environment resource folder: '%S'", &_wd.str());
        
        if(chdir(_wd.c_str()))
            Except("Cannot change directory to '%S'", &_wd.str());
    }
#endif
    
    for(auto &option : cmd.settings()) {
        if(utils::lowercase(option.name) == "output_prefix") {
            SETTING(output_prefix) = option.value;
        }
    }

    if(Path("default.settings").exists()) {
        DebugHeader("LOADING FROM 'default.settings'");
        default_config::warn_deprecated("default.settings", GlobalSettings::load_from_file(default_config::deprecations(), "default.settings", AccessLevelType::STARTUP));
        DebugHeader("LOADED 'default.settings'");
    }

    gpuMat average;
    bool load_results = false, go_fullscreen = false;
    std::vector<std::string> load_settings_from_results;
    
    /**
     * Process command-line options.
     */
    for(auto &option : cmd) {
        if(Arguments::has(option.name)) {
            switch (Arguments::get(option.name)) {
                case Arguments::nowindow:
                    SETTING(nowindow) = true;
                    break;
                    
                case Arguments::load:
                    load_results = true;
                    break;
                    
                case Arguments::fs:
                    go_fullscreen = true;
                    break;
                    
                case Arguments::i:
                case Arguments::input: {
                    if(utils::contains(option.value, '*')) {
                        std::set<file::Path> found;
                        
                        auto parts = utils::split(option.value, '*');
                        Path folder = pv::DataLocation::parse("input", Path(option.value).remove_filename());
                        Debug("Scanning pattern '%S' in folder '%S'...", &option.value, &folder.str());
                        
                        for(auto &file: folder.find_files("pv")) {
                            if(!file.is_regular())
                                continue;
                            
                            auto filename = (std::string)file.filename();
                            
                            bool all_contained = true;
                            size_t offset = 0;
                            
                            for(size_t i=0; i<parts.size(); ++i) {
                                auto & part = parts.at(i);
                                if(part.empty()) {
                                    continue;
                                }
                                
                                auto index = filename.find(part, offset);
                                if(index == std::string::npos
                                   || (i == 0 && index > 0))
                                {
                                    all_contained = false;
                                    break;
                                }
                                
                                offset = index + part.length();
                            }
                            
                            if(all_contained) {
                                found.insert(file);
                            }
                        }
                        
                        if(found.size() == 1) {
                            Path path = pv::DataLocation::parse("input", *found.begin());
                            if(!path.exists())
                                U_EXCEPTION("Cannot find video file '%S'. (%d)", &path.str(), path.exists());
                            
                            Debug("Using file '%S'", &path.str());
                            SETTING(filename) = path.remove_extension();
                            break;
                            
                        } else if(found.size() > 1) {
                            auto str = Meta::toStr(found);
                            Debug("Found too many files matching the pattern '%S': %S.", &option.value, &str);
                        } else
                            Debug("No files found that match the pattern '%S'.", &option.value);
                    }
                    
                    Path path = pv::DataLocation::parse("input", Path(option.value).add_extension("pv"));
                    if(!path.exists())
                        U_EXCEPTION("Cannot find video file '%S'. (%d)", &path.str(), path.exists());
                    
                    SETTING(filename) = path.remove_extension();
                    break;
                }
                    
                case Arguments::r:
                    load_results_from = Path(option.value);
                    load_results = true;
                    Debug("Loading results from '%S'.", &load_results_from.str());
                    break;

                case Arguments::s:
                case Arguments::settings:
                    SETTING(settings_file) = Path(option.value).add_extension("settings");
                    break;
                    
                case Arguments::d:
                case Arguments::dir:
                    SETTING(output_dir) = Path(option.value);
                    break;

                case Arguments::p:
                    SETTING(output_prefix) = std::string(option.value);
                    break;
                    
                case Arguments::h:
                    if(option.value == "rst") {
                        cmd.load_settings();
                        
                        auto rst = cmn::settings::help_restructured_text("TRex parameters", GlobalSettings::defaults(), GlobalSettings::docs(), GlobalSettings::access_levels());
                        
                        file::Path path = pv::DataLocation::parse("output", "parameters_trex.rst");
                        auto f = path.fopen("wb");
                        if(!f)
                            U_EXCEPTION("Cannot open '%S'", &path.str());
                        fwrite(rst.data(), sizeof(char), rst.length(), f);
                        fclose(f);
                        
                        //printf("%s\n", rst.c_str());
                        Debug("Saved at '%S'.", &path.str());
                        
                        exit(0);
                    }
                    else if(option.value == "settings") {
                        auto html = cmn::settings::help_html(GlobalSettings::defaults(), GlobalSettings::docs(), GlobalSettings::access_levels());
                        
                        auto filename = cmd.wd().str();
#if __APPLE__
                        filename += "/../Resources";
#endif
                        filename += "/help.html";
                        FILE* f = NULL;
                        if(!GlobalSettings::has("nowindow") || !SETTING(nowindow))
                            f = fopen(filename.c_str(), "wb");
                        if(!f) {
                            cmn::settings::print_help(GlobalSettings::defaults(), GlobalSettings::docs(), &GlobalSettings::access_levels());
                        } else {
                            fwrite(html.data(), sizeof(char), html.length(), f);
                            fclose(f);
                            
                            Debug("Opening '%S' in browser...", &filename);
#if __linux__
                            auto pid = fork();
                            if (pid == 0) {
                                execl("/usr/bin/xdg-open", "xdg-open", filename.c_str(), (char *)0);
                                exit(0);
                            }
#elif __APPLE__
                            auto pid = fork();
                            if (pid == 0) {
                                execl("/usr/bin/open", "open", filename.c_str(), (char *)0);
                                exit(0);
                            }
#elif defined(WIN32)
                            ShellExecute(
                                NULL,
                                "open",
                                filename.c_str(),
                                NULL,
                                NULL,
                                SW_SHOWNORMAL
                            );
                            
                            exit(0);
#endif
                        }
                        
                    } else {
                        printf("\n");
                        DebugHeader("AVAILABLE OPTIONS");
                        Debug("-i <filename>       Input video source (basler/webcam/video path)");
                        Debug("-d <folder>         Set the default input/output folder");
                        Debug("-s <filename>       Set the .settings file to be used (default is <input>.settings)");
                        Debug("-load               Loads previously saved .results file (if it exists)");
                        Debug("-h                  Prints this message");
                        Debug("-h settings         Displays all default settings with description in the default browser");
                        exit(0);
                    }
                    
                    exit(0);
                    break;
                    
                case Arguments::update: {
                    auto status = CheckUpdates::perform(false).get();
                    if(status == CheckUpdates::VersionStatus::OLD || status == CheckUpdates::VersionStatus::ALREADY_ASKED)
                    {
                        CheckUpdates::display_update_dialog();
                    } else if(status == CheckUpdates::VersionStatus::NEWEST) {
                        Debug("You have the newest version (%S).", &CheckUpdates::newest_version());
                    } else
                         Error("Error checking for the newest version: '%S'. Please check your internet connection and try again.", &CheckUpdates::last_error());
                    
                    PythonIntegration::quit();
                    exit(0);
                    break;
                }
                    
                default:
                    Warning("Unknown option '%S' with value '%S'", &option.name, &option.value);
                    break;
            }
            
        } else if(option.name == "load_settings_from_results") {
            load_settings_from_results = Meta::fromStr<std::vector<std::string>>(option.value);
            
        } else {
            Warning("Unknown option '%S' with value '%S'", &option.name, &option.value);
        }
    }

    if (argc == 2) {
        Path path = pv::DataLocation::parse("input", Path(argv[1]));
        if (path.exists()) {
            SETTING(filename) = path.remove_extension();
            SETTING(output_dir) = path.remove_filename();
        }
    }
    
    // check whether a file exists
    gui::VideoOpener::Result opening_result;
    
    if(SETTING(filename).value<Path>().empty()) {
        cmd.load_settings();
        
        if((GlobalSettings::map().has("nowindow") ? SETTING(nowindow).value<bool>() : false) == false) {
            SETTING(settings_file) = file::Path();
            
            gui::VideoOpener opener;
            opening_result = opener._result;
        }

        if (!opening_result.selected_file.empty()) {
            if (opening_result.tab.extension == "pv") {
                if (opening_result.load_results)
                    load_results = true;
                else
                    load_results = false;
                
                if (!opening_result.load_results_from.empty())
                    load_results_from = opening_result.load_results_from;
            }
            else {
                auto wd = SETTING(wd).value<file::Path>();
                Debug("Opening a video file: '%S' (wd: '%S')", &opening_result.tab.name, &wd.str());
#if defined(__APPLE__)
                wd = wd / ".." / ".." / ".." / "TGrabs.app" / "Contents" / "MacOS" / "TGrabs";
#else
                if (wd.empty())
                    wd = "tgrabs";
                else
                    wd = wd / "tgrabs";
#endif
                auto exec = wd.str() + " " + opening_result.cmd;
                Debug("Executing '%S'", &exec);

#if defined(WIN32)
                //file::exec(exec.c_str());                    
                STARTUPINFO info = { sizeof(info) };
                PROCESS_INFORMATION processInfo;
                if (CreateProcess(NULL, exec.data(), NULL, NULL, TRUE, 0, NULL, NULL, &info, &processInfo))
                {
                    WaitForSingleObject(processInfo.hProcess, INFINITE);
                    CloseHandle(processInfo.hProcess);
                    CloseHandle(processInfo.hThread);
                }
#else
                auto pid = fork();
                if (pid == 0) {
                    file::exec(exec.c_str());
                    exit(0);
            }
#endif
                return 0;
            }
        }
        else
            SETTING(filename) = file::Path();
        
        if(SETTING(filename).value<Path>().empty()) {
            Debug("You can specify a file to be opened using ./tracker -i <filename>");
            return 0;
        }
    }

    /**
     * Load video file and additional settings from files.
     */
    bool executed_a_settings = false;
    
    DebugHeader("LOADING FILE");
    
    pv::File video(SETTING(filename).value<Path>());
    video.start_reading();
    
    if(video.header().version <= pv::Version::V_2) {
        SETTING(crop_offsets) = CropOffsets();
        
        Path settings_file = pv::DataLocation::parse("settings");
        if(GUI::execute_settings(settings_file, AccessLevelType::STARTUP))
            executed_a_settings = true;
        
        auto output_settings = pv::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            if(GUI::execute_settings(output_settings, AccessLevelType::STARTUP))
                executed_a_settings = true;
        }
        
        video.close();
        video.start_reading();
    }
    
    try {
        if(!video.header().metadata.empty())
            sprite::parse_values(GlobalSettings::map(), video.header().metadata);
    } catch(const UtilsException& e) {
        // dont do anything, has been printed already
    }
    
    /**
     * Load and process average image.
     */
    video.average().copyTo(average);
    if(average.cols == video.size().width && average.rows == video.size().height)
        video.processImage(average, average);
    
    SETTING(video_size) = Size2(average.cols, average.rows);
    SETTING(video_mask) = video.has_mask();
    SETTING(video_length) = uint64_t(video.length());
    SETTING(video_info) = std::string(video.get_info());
    
    if(SETTING(frame_rate).value<int>() <= 0) {
        Warning("frame_rate == 0, calculating from frame tdeltas.");
        video.generate_average_tdelta();
        SETTING(frame_rate) = max(1, int(video.framerate()));
    }
    
    Library::InitVariables();
    
    Path settings_file = pv::DataLocation::parse("settings");
    if(SETTING(settings_file).value<file::Path>().empty()) {
        if(GUI::execute_settings(settings_file, AccessLevelType::STARTUP))
            executed_a_settings = true;
        else {
            SETTING(settings_file) = file::Path();
            Warning("Settings file '%S' does not exist.", &settings_file.str());
        }
    }
    
    if(SETTING(meta_real_width).value<float>() == 0) {
        Warning("This video does not set `meta_real_width`. Please set this value during conversion (see https://trex.run/docs/parameters_trex.html#meta_real_width for details).");
        SETTING(meta_real_width) = float(30.0);
    }
    
    // setting cm_per_pixel after average has been generated (and offsets have been set)
    if(!GlobalSettings::map().has("cm_per_pixel") || SETTING(cm_per_pixel).value<float>() == 0)
        SETTING(cm_per_pixel) = SETTING(meta_real_width).value<float>() / float(average.cols);
    
    /**
     * Try to load Settings from the command-line that have been
     * ignored previously.
     */
    cmd.load_settings();
    
    if(SETTING(settings_file).value<file::Path>().empty()) {
        auto output_settings = pv::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            if(GUI::execute_settings(output_settings, AccessLevelType::STARTUP))
                executed_a_settings = true;
            else if(!executed_a_settings)
                Warning("Output settings '%S' does not exist.", &output_settings.str());
        }
        
    } else {
        if(GUI::execute_settings(settings_file, AccessLevelType::STARTUP))
            executed_a_settings = true;
        else
            Warning("Settings file '%S' does not exist.", &settings_file.str());
    }

    Tracker tracker;
    tracker.update_history_log();
    
    bool contains_illegal_options = false;
    for(auto &option : cmd.options()) {
        if(default_config::is_deprecated(option.name)) {
            if(utils::lowercase(option.name) == "match_use_approximate") {
                bool v = option.value.empty() || Meta::fromStr<bool>(option.value);
                SETTING(match_mode) = v ? default_config::matching_mode_t::approximate : default_config::matching_mode_t::automatic;
                continue;
            }
            contains_illegal_options = true;
            
            auto r = default_config::replacement(option.name);
            if(!r.empty()) {
                Except("You are using the deprecated command-line option '%S'. Please use '%S' instead.", &option.name, &r);
            } else
                Except("You are using the deprecated command-line option '%S'. There is no replacement. Please remove it from your start parameters.", &option.name);
        }
    }
    
    if(contains_illegal_options) {
        U_EXCEPTION("Cannot continue with the mentioned deprecated command-line options.");
    }
    
    cmd.load_settings();
    
    if(SETTING(output_graphs).value< std::vector<std::pair<std::string, std::vector<std::string>>>>().empty()) {
        sprite::Map config;
        GlobalSettings::docs_map_t docs;
        default_config::get(config, docs, NULL);
        
        SETTING(output_graphs) = config.get<std::vector<std::pair<std::string, std::vector<std::string>>>>("output_graphs").value();
    }
    
    if(SETTING(cam_undistort)) {
        cv::Mat map1, map2;
        cv::Size size = video.header().resolution;
        
        cv::Mat cam_matrix = cv::Mat(3, 3, CV_32FC1, SETTING(cam_matrix).value<std::vector<float>>().data());
        cv::Mat cam_undistort_vector = cv::Mat(1, 5, CV_32FC1, SETTING(cam_undistort_vector).value<std::vector<float>>().data());
        
        cv::Mat drawtransform = cv::getOptimalNewCameraMatrix(cam_matrix, cam_undistort_vector, size, 1.0, size);
        print_mat("draw_transform", drawtransform);
        print_mat("cam", cam_matrix);
        //drawtransform = SETTING(cam_matrix).value<cv::Mat>();
        cv::initUndistortRectifyMap(
                                    cam_matrix,
                                    cam_undistort_vector,
                                    cv::Mat(),
                                    drawtransform,
                                    size,
                                    CV_32FC1,
                                    map1, map2);
        
        GlobalSettings::get("cam_undistort1") = map1;
        GlobalSettings::get("cam_undistort2") = map2;
    }
    
    if(!SETTING(exec).value<file::Path>().empty()) {
        Path exec_settings = pv::DataLocation::parse("settings", SETTING(exec).value<file::Path>());
        if(!GUI::execute_settings(exec_settings, AccessLevelType::STARTUP))
            Except("Settings file '%S' cannot be found or execution failed.", &exec_settings.str());
        else
            executed_a_settings = true;
        
        SETTING(exec) = file::Path();
    }
    
    //! if we used the open file dialog and changed settings, use them
    if(!opening_result.extra_command_lines.empty()) {
        GlobalSettings::load_from_string(default_config::deprecations(), GlobalSettings::map(), opening_result.extra_command_lines, AccessLevelType::STARTUP);
    }
    
    cv::Mat local;
    average.copyTo(local);
    tracker.set_average(Image::Make(local));
    
    if(!SETTING(log_file).value<file::Path>().empty()) {
        auto path = pv::DataLocation::parse("output", SETTING(log_file).value<file::Path>());
        
        DEBUG::SetDebugCallback({
            DEBUG::DEBUG_TYPE::TYPE_ERROR,
            DEBUG::DEBUG_TYPE::TYPE_EXCEPTION,
            DEBUG::DEBUG_TYPE::TYPE_WARNING,
            DEBUG::DEBUG_TYPE::TYPE_INFO
        }, [&log_mutex, &log_file](auto, const std::string& msg)
            {
                std::lock_guard<std::mutex> guard(log_mutex);
                if(log_file) {
                    char nl = '\n';
                    fwrite(msg.c_str(), 1, msg.length(), log_file);
                    fwrite(&nl, 1, 1, log_file);
                    fflush(log_file);
                }
            });
        
        log_mutex.lock();
        log_file = fopen(path.str().c_str(), "wb");
        log_mutex.unlock();
        
        Debug("Logging to '%S'.", &path.str());
    }
    
    if(SETTING(evaluate_thresholds)) {
        std::vector<Vec2> values;
        std::vector<float> numbers;
        std::vector<float> samples;
        values.resize(100);
        numbers.resize(values.size());
        samples.resize(values.size());
        
        std::vector<Median<long_t>> medians;
        medians.resize(values.size());
        
        float start_threshold = 5;
        float end_threshold = 230;
        float threshold_step = (end_threshold - 20 - start_threshold) / narrow_cast<float>(values.size());
        
        GenericThreadPool pool(cmn::hardware_concurrency());
        std::mutex sync;
        
        size_t added_frames = 0, processed_frames = 0;
        
        auto step = (video.length() - video.length()%500) / 500;
        for (size_t i=0; i<video.length()-1; i+=step) {
            ++added_frames;
            
            pool.enqueue([&](size_t i) {
                pv::Frame frame;
                video.read_frame(frame, i);
            //video.read_frame(next_frame, i+1);
            
                size_t j = 0;
                for(float threshold = start_threshold; threshold <= end_threshold; threshold += threshold_step, ++j)
                {
                    
                    float pixel_average = 0, pixel_samples = 0;
                    float number = 0;
                    
                    for (uint16_t k=0; k<frame.n(); ++k) {
                        if(frame.pixels().at(k)->size() > 30) {
                            // consider blob
                            auto &l = frame.mask().at(k);
                            auto &p = frame.pixels().at(k);
                            auto blob = std::make_shared<pv::Blob>(std::move(l), std::move(p));
                            auto blobs = pixel::threshold_blob(blob, narrow_cast<int>(threshold), Tracker::instance()->background());
                            float pixels = 0, samps = 0;
                            
                            for(auto &b : blobs) {
                                if(b->pixels()->size() > 30) {
                                    pixels += b->pixels()->size();
                                    ++samps;
                                }
                            }
                            
                            if(samps > 0) {
                                pixels /= samps;
                                pixel_average += pixels;
                                ++pixel_samples;
                                number += samps;
                            }
                        }
                    }
                    
                    
                    if(pixel_samples > 0) {
                        std::lock_guard<std::mutex> guard(sync);
                        pixel_average /= pixel_samples;
                        ++samples.at(j);
                        values.at(j).y += pixel_average;
                        values.at(j).x = j;
                        numbers[j] += number / pixel_samples;
                        medians.at(j).addNumber(narrow_cast<int>(number));
                    }
                }
                
                std::lock_guard<std::mutex> guard(sync);
                ++processed_frames;
                if(processed_frames % 100 == 0) {
                    Debug("%d / %d (%.1f%%)", processed_frames, added_frames, processed_frames / float(added_frames) * 100);
                }
            
            }, i);
        }
        
        pool.wait();
        
        float max_value = 0;
        
        for (size_t i=0; i<values.size(); ++i) {
            if(samples.at(i) > 0)
                values[i] /= samples.at(i);
            if(values.at(i).y > max_value)
                max_value = values.at(i).y;
        }
        
        gui::Graph graph(Bounds(50,50,980,300), "thresholds");
        graph.set_zero(0);
        graph.set_ranges(Rangef(0, 255), Rangef(0, 1));
        graph.add_function(gui::Graph::Function("sizes", gui::Graph::Type::DISCRETE, [&, max_val = max_value](float x) -> float {
            auto threshold = (x - start_threshold) / threshold_step;
            if(size_t(threshold) < values.size()) {
                return values.at(size_t(threshold)).y / max_val;
            }
            return gui::Graph::invalid();
        }));
        
        max_value = 0;
        for (size_t i=0; i<numbers.size(); ++i) {
            if(samples.at(i) > 0) {
                numbers[i] /= samples.at(i);
            }
            if(numbers.at(i) > max_value)
                max_value = numbers.at(i);
        }
        
        graph.add_function(gui::Graph::Function("samples", gui::Graph::Type::DISCRETE, [&, max_val = max_value](float x) -> float {
            auto threshold = (x - start_threshold) / float(end_threshold - start_threshold) * numbers.size();
            if(size_t(threshold) < numbers.size()) {
                return numbers.at(size_t(threshold)) / max_val;
            }
            return gui::Graph::invalid();
        }));
        
        max_value = 0;
        for (size_t i=0; i<medians.size(); ++i) {
            if(medians.at(i).getValue() > max_value)
                max_value = medians.at(i).getValue();
        }
        
        graph.add_function(gui::Graph::Function("median_number", gui::Graph::Type::DISCRETE, [&, max_val = max_value](float x) -> float {
            auto threshold = (x - start_threshold) / float(end_threshold - start_threshold) * numbers.size();
            if(size_t(threshold) < numbers.size()) {
                return medians.at(size_t(threshold)).getValue() / max_val;
            }
            return gui::Graph::invalid();
        }));
        
        size_t j=0;
        for(auto threshold = start_threshold; threshold <= end_threshold && j < values.size(); threshold += threshold_step, ++j)
        {
            printf("%f : %f (%d), %f\n", threshold, numbers.at(j), medians.at(j).getValue(), values.at(j).y);
        }
        printf("\n");
        
        gui::DrawStructure gui(1024, 500);
        gui.wrap_object(graph);
        
        cv::Mat window(500, 1024, CV_8UC4);
        gui::CVBase base(window);
        base.paint(gui);
        base.display();
    }
    
    if(!load_results && !executed_a_settings) {
        Warning("No settings file can be loaded, so the program will try to automatically determine individual sizes and numbers.");
        sprite::Map default_map;
        GlobalSettings::docs_map_t default_docs;
        default_map.set_do_print(false);
        default_config::get(default_map, default_docs, NULL);
        
        if(SETTING(auto_number_individuals).value<bool>() == default_map.get<bool>("auto_number_individuals").value())
        {
            SETTING(auto_number_individuals) = SETTING(track_max_individuals).value<uint32_t>() == default_map.get<uint32_t>("track_max_individuals").value();
        }
        
        if(SETTING(auto_minmax_size).value<bool>() == default_map.get<bool>("auto_minmax_size").value())
        {
            SETTING(auto_minmax_size) = SETTING(blob_size_ranges).value<BlobSizeRange>() == default_map.get<BlobSizeRange>("blob_size_ranges").value();
        }
    }
    
    Tracker::auto_calculate_parameters(video);
    
    if(SETTING(manual_identities).value<std::set<track::Idx_t>>().empty() && SETTING(track_max_individuals).value<uint32_t>() != 0)
    {
        std::set<track::Idx_t> vector;
        for(uint32_t i=0; i<SETTING(track_max_individuals).value<uint32_t>(); ++i) {
            vector.insert(track::Idx_t(i));
        }
        SETTING(manual_identities) = vector;
    }
    
    default_config::warn_deprecated("global", GlobalSettings::map());
    
    if(FAST_SETTINGS(track_max_individuals) == 1
       && SETTING(auto_apply))
    {
        Error("Cannot use a network on a single individual. Disabling auto_apply.");
        SETTING(auto_apply) = false;
    }
    
    if(FAST_SETTINGS(track_max_individuals) == 1
       && SETTING(auto_train))
    {
        Error("Cannot train a network on a single individual. Disabling auto_train.");
        SETTING(auto_train) = false;
    }
    
    if(SETTING(auto_train) || SETTING(auto_apply)) {
        SETTING(auto_train_on_startup) = true;
    }
    
    if(!SETTING(auto_train_on_startup) && SETTING(auto_train_dont_apply)) {
        Warning("auto_train_dont_apply was set without auto_train enabled. This may lead to confusing behavior. Overwriting auto_train_dont_apply = false.");
        SETTING(auto_train_dont_apply) = false;
    }
    
    Library::Init();
    DebugHeader("STARTING PROGRAM");
    
    cmn::Blob blob;
    auto copy = blob.properties();
    Debug("BasicStuff<%lu> PostureStuff<%lu> Individual<%lu> Blob<%lu> MotionRecord<%lu> Image::Ptr<%lu> std::shared_ptr<std::vector<HorizontalLine>>:%lu Bounds:%lu bool:%lu cmn::Blob::properties:%lu", 
        sizeof(track::Individual::BasicStuff), 
        sizeof(track::Individual::PostureStuff),
        sizeof(track::Individual), 
        sizeof(pv::Blob), 
        sizeof(MotionRecord), 
        sizeof(Image::Ptr), 
        sizeof(std::shared_ptr<std::vector<HorizontalLine>>), 
        sizeof(Bounds), 
        sizeof(bool), 
        sizeof(decltype(copy)));
    Debug("localcache:%lu identity:%lu std::map<long_t, Vec2>:%lu", sizeof(Individual::LocalCache), sizeof(Identity), sizeof(std::map<long_t, Vec2>));
    Debug("BasicStuff:%lu pv::Blob:%lu Compressed:%lu", sizeof(Individual::BasicStuff), sizeof(pv::Blob), sizeof(pv::CompressedBlob));
    Debug("Midline:%lu MinimalOutline:%lu", sizeof(Midline), sizeof(MinimalOutline));
    
    GUI *tmp = new GUI(video, tracker.average(), tracker);
    std::unique_lock<std::recursive_mutex> gui_lock(tmp->gui().lock());
    
    //try {
    GUI &gui = *tmp;
    gui.frameinfo().video_length = video.length() > 0 ? (video.length() - 1) : 0;
    
    if(!SETTING(gui_connectivity_matrix_file).value<file::Path>().empty()) {
        try {
            gui.load_connectivity_matrix();
        } catch(const UtilsException&) { }
    }
    
    bool please_stop_analysis = false;
    
    std::atomic<Frame_t> currentID(Frame_t{});
    std::queue<std::shared_ptr<PPFrame>> unused;
    std::mutex mutex;
    const Frame_t cache_size{10};
    
    for (auto i=0_f; i<cache_size; ++i)
        unused.push(std::make_shared<PPFrame>());
        
    //std::mutex stage1_mutex;
    //double time_stage1 = 0, time_stage2 = 0, stage1_samples = 0, stage2_samples = 0;
    GenericThreadPool pool(hardware_concurrency(), [](std::exception_ptr e) {
        std::rethrow_exception(e);
    }, "preprocess_main");
    
    //! Stages
    std::vector<std::function<bool(ConnectedTasks::Type, const ConnectedTasks::Stage&)>> tasks =
    {
        [&](std::shared_ptr<PPFrame> ptr, auto&) -> bool {
            auto idx = ptr->index();
            auto range = Tracker::analysis_range();
            if(!range.contains(idx) && idx != range.end && idx > Tracker::end_frame()) {
                std::unique_lock<std::mutex> lock(mutex);
                unused.push(ptr);
                return false;
            }

            Timer timer;
            video.read_frame(ptr->frame(), (size_t)idx.get());
            ptr->frame().set_index(idx.get());
            Tracker::preprocess_frame(*ptr, {}, pool.num_threads() > 1 ? &pool : NULL, NULL, false);

            ptr->frame().set_loading_time(narrow_cast<float>(timer.elapsed()));

            return true;
        },

        [&](std::shared_ptr<PPFrame> ptr, auto&) -> bool {
            static Timer fps_timer;
            static Image empty(0, 0, 0);

            Timer timer;

            static Timing all_processing("Analysis::process()", 50);
            TakeTiming all(all_processing);

            Tracker::LockGuard guard("Analaysis::process()");
            if(GUI_SETTINGS(terminate))
                return false;
            
            auto range = Tracker::analysis_range();

            auto idx = ptr->index();
            if (idx >= range.start
                && max(range.start, tracker.end_frame() + 1_f) == idx
                && !tracker.properties(idx)
                && idx <= Tracker::analysis_range().end)
            {
                tracker.add(*ptr);

                static Timing after_track("Analysis::after_track", 10);
                TakeTiming after_trackt(after_track);
                
                if(size_t(idx.get() + 1) == video.length())
                    please_stop_analysis = true;

                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    data_kbytes += ptr->frame().size() / 1024.0;
                }

                double elapsed = fps_timer.elapsed();
                if (elapsed >= 1) {
                    std::lock_guard<std::mutex> lock(data_mutex);

                    frames_sec = frames_count / elapsed;
                    data_sec = data_kbytes / elapsed;

                    frames_count = 0;
                    data_kbytes = 0;
                    fps_timer.reset();

                    if(frames_sec > 0) {
                        static double frames_sec_average=0;
                        static double frames_sec_samples=0;
                        static Timer print_timer;

                        frames_sec_average += frames_sec;
                        ++frames_sec_samples;

                        float percent = min(1, (ptr->index() - range.start).get() / float(range.length().get() + 1)) * 100;
                        DurationUS us{ uint64_t(max(0, (double)(range.end - ptr->index()).get() / double(/*frames_sec*/ frames_sec_average / frames_sec_samples ) * 1000 * 1000)) };
                        auto duration = us.to_string();
                        std::string str;
                        
                        if(FAST_SETTINGS(analysis_range).first != -1 || FAST_SETTINGS(analysis_range).second != -1)
                            str = DEBUG::format("frame %lu/%lu(%lu) (%.2fMB/s @ %.2ffps eta %S)", ptr->index(), range.end, video.length(), data_sec / 1024.0, frames_sec, &duration);
                        else
                            str = DEBUG::format("frame %lu/%lu (%.2fMB/s @ %.2ffps eta %S)", ptr->index(), range.end, data_sec / 1024.0, frames_sec, &duration);

                        {
                            // synchronize with debug messages
                            std::lock_guard<std::mutex> debug_lock(DEBUG::debug_mutex());
                            size_t i;
                            printf("[");
                            for(i=0; i<percent * 0.5; ++i) {
                                printf("=");
                            }
                            for(; i<100 * 0.5; ++i) {
                                printf(" ");
                            }
                            printf("] %.2f%% %s\r", percent, str.c_str());
                            fflush(stdout);
                        }

                        // log occasionally
                        if(print_timer.elapsed() > 30) {
                            Debug("%.2f%% %S", percent, &str);
                            print_timer.reset();
                        }
                        //Debug("frame %lu/%lu (%.2fMB/s @ %.2ffps)", ptr->index(), video.length(), data_sec / 1024.0, frames_sec);
                    }

                    if(tmp)
                        gui.frameinfo().current_fps = narrow_cast<int>(frames_sec);
                }

                frames_count++;
            }

            static Timing procpush("Analysis::process::unused.push", 10);
            TakeTiming ppush(procpush);
            std::unique_lock<std::mutex> lock(mutex);
            unused.push(ptr);

            return true;
        }
    };
    
    std::shared_ptr<ConnectedTasks> analysis;
    analysis = std::make_shared<ConnectedTasks>(tasks);
    analysis->start(// main thread
        [&]() {
            auto endframe = tracker.end_frame();
            auto current = currentID.load();
            if(current > endframe + cache_size
               || !current.valid()
               || (analysis->stage_empty(0) && analysis->stage_empty(1))
               || current < endframe)
            {
                current = currentID = endframe; // update current as well
            }
        
            auto range = Tracker::analysis_range();
            if(current < range.start)
                currentID = range.start - 1_f;
            
            if(FAST_SETTINGS(analysis_range).second != -1
               && endframe >= Frame_t(FAST_SETTINGS(analysis_range).second)
               && !SETTING(terminate)
               && !please_stop_analysis)
            {
                please_stop_analysis = true;
            }
            
            while(currentID.load() < max(range.start, endframe) + cache_size
                  && size_t((currentID.load() + 1_f).get()) < video.length())
            {
                std::unique_lock<std::mutex> lock(mutex);
                if(unused.empty())
                    break;
                
                auto ptr = unused.front();
                unused.pop();
                
                currentID = currentID.load() + 1_f;
                ptr->set_index(currentID.load());
                
                analysis->add(ptr);
            }
        }
    );
        
    gui.set_analysis(analysis.get());
    gui_lock.unlock();
    
    CheckUpdates::init();
    
    auto callback = "TRex::main";
    GlobalSettings::map().register_callback(callback, [&analysis, &gui, callback](sprite::Map::Signal signal, sprite::Map& map, const std::string& key, const sprite::PropertyType& value)
    {
        if(signal == sprite::Map::Signal::EXIT) {
            map.unregister_callback(callback);
            return;
        }
        
        if (key == "analysis_paused") {
            analysis->bump();
            
            bool pause = value.value<bool>();
            if(analysis->paused() != pause) {
                Debug("Adding to queue...");
                
                gui.work().add_queue("pausing", [&analysis, pause](){
                    if(analysis->paused() != pause) {
                        analysis->set_paused(pause);
                        Debug("Paused.");
                    }
                });
                
                Debug("Added.");
            }
        }
    });
    
    auto get_settings_from_results = [](const Path& filename) -> std::string {
        Debug("Trying to open results '%S'", &filename.str());
        ResultsFormat file(filename, NULL);
        file.start_reading();
        
        if(file.header().version >= ResultsFormat::V_14) {
            return file.header().settings;
        } else
            Except("Cannot load settings from results file < V_14");
        return "{}";
    };
    
    if(FAST_SETTINGS(analysis_paused) || load_results) {
        analysis->set_paused(true).get();
        
        if(load_results) {
            if(!executed_a_settings) {
                auto path = TrackingResults::expected_filename();
                auto str = get_settings_from_results(load_results_from.empty() ? path : load_results_from);
                Debug("Loading settings from '%S'...", &path);
                try {
                    default_config::warn_deprecated(path.str(), GlobalSettings::load_from_string(default_config::deprecations(), GlobalSettings::map(), str, AccessLevelType::STARTUP));
                    executed_a_settings = true;
                } catch(const UtilsException& e) {
                    Except("Cannot load settings from results file. Skipping.");
                }
            }
            
            gui.load_state(GUI::GUIType::TEXT, load_results_from);
            
            // explicitly set gui_frame if present in command-line
            if(cmd.settings_keys().find("gui_frame") != cmd.settings_keys().end()) {
                gui.work().add_queue("", [&](){
                    SETTING(gui_frame) = Meta::fromStr<Frame_t>(cmd.settings_keys().at("gui_frame"));
                });
            }
        }
    }
    
    if(!load_results && !settings_file.exists()) {
        Error("Settings file '%S' cannot be found.", &settings_file.str());
    }
    
    if(!load_settings_from_results.empty()) {
        auto path = TrackingResults::expected_filename();
        auto str = get_settings_from_results(path);
        sprite::Map defaults;
        defaults.set_do_print(false);
        
        GlobalSettings::docs_map_t docs;
        default_config::get(defaults, docs, NULL);
        auto added = GlobalSettings::load_from_string(default_config::deprecations(), defaults, str, AccessLevelType::STARTUP, true);
        
        DebugHeader("LOADING SETTINGS FROM '%S'", &path.str());
        
        for(auto name : load_settings_from_results) {
            try {
                if(added.find(name) != added.end()) {
                    name = utils::lowercase(name);
                    auto use = name;
                    
                    // we found the requested setting in the results file
                    if(default_config::is_deprecated(name)) {
                        use = default_config::deprecations().at(name);
                    }
                    
                    auto &prop = GlobalSettings::get(use).get();
                    prop.set_value_from_string(defaults[use].get().valueString());
                } else
                    throw std::invalid_argument("Cannot find "+name+" in results file.");
                
            } catch(...) {
                Except("Cannot load '%S' from results file.", &name);
            }
        }
        
        DebugHeader("/ LOADED SETTINGS FROM '%S'", &path.str());
    }
    
    if(SETTING(auto_train)) {
        if(!Recognition::recognition_enabled())
            U_EXCEPTION("auto_train switch cant be used without recognition_enable = true.");
        Warning("The application is going to attempt to automatically train the network upon finding a suitable consecutive segment.");
    }
    if(SETTING(auto_apply)) {
        if(SETTING(auto_train) || !Recognition::network_weights_available()) {
            auto path = Recognition::network_path();
            path = path.add_extension("npz");
            
            SETTING(terminate_error) = true;
            SETTING(terminate) = true;
            U_EXCEPTION("Cannot apply a network without network_weights available. (searching at '%S')", &path.str());
        }
        
        Warning("The application is going to apply a trained network after finishing the analysis and auto_correct it afterwards.");
    }
    if(SETTING(auto_categorize)) {
        if(!Categorize::weights_available()) {
            auto file = (std::string)SETTING(filename).value<file::Path>().filename();
            auto output = (std::string)pv::DataLocation::parse("output").str();
            
            SETTING(terminate_error) = true;
            SETTING(terminate) = true;
            U_EXCEPTION("Make sure that a file called '%S_categories.npz' is located inside '%S'", &file, &output);
        }
        Warning("The application is going to load a pretrained categories network and apply it after finishing the analysis (or loading).");
    }
    if(SETTING(auto_quit))
        Warning("Application is going to quit after analysing and exporting data.");
    
    gui::IMGUIBase *imgui_base = nullptr;
    if((GlobalSettings::map().has("nowindow") ? SETTING(nowindow).value<bool>() : false) == false) {
        imgui_base = new gui::IMGUIBase(gui.window_title(), gui.gui(), [&](){
            //std::lock_guard<std::recursive_mutex> lock(gui.gui().lock());
            if(SETTING(terminate))
                return false;
            
            return true;
        }, GUI::event);
        
        gui.set_base(imgui_base);
        imgui_base->platform()->set_icons({
            "gfx/"+SETTING(app_name).value<std::string>()+"Icon16.png",
            "gfx/"+SETTING(app_name).value<std::string>()+"Icon32.png",
            "gfx/"+SETTING(app_name).value<std::string>()+"Icon64.png"
        });
    }
    
    if(go_fullscreen)
        gui.toggle_fullscreen();
    
    gui::SFLoop loop(gui.gui(), imgui_base, [&](gui::SFLoop&, gui::LoopStatus status){
        {
            std::unique_lock<std::recursive_mutex> guard(gui.gui().lock());
            GUI::run_loop(status);
        }
        
        if(pause_stuff) {
            pause_stuff = false;
            
            std::string cmd;
            bool before;
            {
                before = analysis->is_paused();
                analysis->set_paused(true).get();
                
                Tracker::LockGuard guard("pause_stuff");
            
                Debug("Console opened.");
                Debug("Please enter command below (type help for available commands):");
                printf(">> ");
                std::getline(std::cin, cmd);
                //Debug("Entered '%S'", &str);
            }
            
            gui.work().add_queue("", [&before, &analysis, cmd, &gui](){
                bool executed = false;
                
                if(!utils::contains(cmd, "=") || utils::beginsWith(cmd, "python")) {
                    auto command = utils::lowercase(cmd);
                    
                    executed = true;
                    if(command == "quit")
                        SETTING(terminate) = true;
                    else if(command == "load_results") {
                        gui.load_state(GUI::GUIType::TEXT);
                    }
                    else if(command == "help") {
                        Debug("You may type any of the following commands:");
                        Debug("\tinfo\t\t\t\tPrints information about the current file");
                        Debug("\tsave_results [force]\t\tSaves a .results file (if one already exists, force is required to overwrite).");
                        Debug("\texport_data\t\tExports the tracked data to CSV/NPZ files according to settings.");
                        Debug("\tsave_config [force]\t\tSaves the current settings (if settings exist, force to overwrite).");
                        Debug("\tauto_correct [force]\t\tGenerates auto_corrected manual_matches. If force is set, applies them.");
                        Debug("\ttrain_network [load]\t\tStarts network training with currently selected segment. If load is set, loads weights and applies them.");
                        Debug("\treanalyse\t\t\tReanalyses the whole video from frame 0.");
                    }
                    else if(command == "info") {
                        auto str = gui.info(false);
                        Debug("%S", &str);
                    }
                    else if(command == "retrieve_matches") {
                        GUI::work().add_queue("retrieving matches", [](){
                            Settings::manual_matches_t manual_matches;
                            {
                                Tracker::LockGuard guard("retrieving matches");
                                
                                for(auto && [id, fish] : Tracker::individuals()) {
                                    for(auto frame : fish->manually_matched()) {
                                        auto blob = fish->blob(frame);
                                        if(blob) {
                                            if(manual_matches[frame].find(id) != manual_matches[frame].end()
                                               && manual_matches[frame][id] != blob->blob_id())
                                            {
                                                Debug("Other blob (%d != %d) was assigned fish %d in frame %d", manual_matches[frame][id], blob->blob_id(), id, frame);
                                            }
                                            for(auto && [fdx, bdx] : manual_matches[frame]) {
                                                if(fdx != id && bdx == blob->blob_id()) {
                                                    Debug("Other fish (%d != %d) was assigned blob %d in frame %d", fdx, id, bdx, frame);
                                                    break;
                                                }
                                            }
                                            
                                            manual_matches[frame][id] = blob->blob_id();
                                        }
                                    }
                                }
                            }
                            
                            auto str = prettify_array(Meta::toStr(manual_matches));
                            Debug("%S", &str);
                            
                            SETTING(manual_matches) = manual_matches;
                        });
                    }
                    else if(utils::beginsWith(command, "save_results")) {
                        gui.save_state(GUI::GUIType::TEXT, utils::endsWith(command, " force"));
                    }
                    else if(utils::beginsWith(command, "export_data")) {
                        gui.export_tracks();
                    }
                    else if(utils::beginsWith(command, "python ")) {
                        auto copy = cmd;
                        for(size_t i=0; i<cmd.length(); ++i) {
                            if(cmd.at(i) == ' ') {
                                copy = cmd.substr(i+1);
                                break;
                            }
                        }
                    
                        copy = utils::find_replace(copy, "\\n", "\n");
                        copy = utils::find_replace(copy, "\\t", "\t");
                        PythonIntegration::async_python_function([copy]()->bool
                        {
                            Debug("Executing '%S'", &copy);
                            try {
                                PythonIntegration::execute(copy);
                            } catch(const SoftException& e) {
                                Debug("Runtime error: '%s'", e.what());
                            }
                            return true;
                        });
                    }
                    else if(utils::beginsWith(command, "continue")) {
                        before = true;
                        SETTING(analysis_paused) = false;
                    }
                    else if(utils::lowercase(command) == "print_memory") {
                        Tracker::LockGuard guard("print_memory");
                        mem::IndividualMemoryStats overall;
                        for(auto && [fdx, fish] : Tracker::individuals()) {
                            mem::IndividualMemoryStats stats(fish);
                            stats.print();
                            overall += stats;
                        }
                    
                        overall.print();
                        
                        mem::TrackerMemoryStats stats;
                        stats.print();
                        
                        mem::OutputLibraryMemoryStats ol;
                        ol.print();
                    }
                    else if(utils::beginsWith(command, "save_config")) {
                        gui.write_config(utils::endsWith(command, " force"), GUI::GUIType::TEXT);
                    }
                    else if(utils::beginsWith(command, "auto_correct")) {
                        gui.auto_correct(GUI::GUIType::TEXT, utils::endsWith(command, " force"));
                    }
                    else if(utils::beginsWith(command, "train_network")) {
                        gui.training_data_dialog(GUI::GUIType::TEXT, utils::endsWith(command, " load"));
                        
                    } else if(utils::beginsWith(command, "reanalyse")) {
                        GUI::reanalyse_from(0_f, false);
                        SETTING(analysis_paused) = false;
                        /*{
                            Tracker::LockGuard guard;
                            Tracker::instance()->remove_frames(0);
                        }
                        
                        if(SETTING(analysis_paused))
                            SETTING(analysis_paused) = false;*/
                        
                    } else if(GlobalSettings::map().has(command)) {
                        Debug("Object '%S'", &command);
                        GlobalSettings::get(command).print_object();
                    }
                    else {
                        std::set<std::string> matches;
                        for(auto key : GlobalSettings::map().keys()) {
                            if(utils::contains(utils::lowercase(key), utils::lowercase(command))) {
                                matches.insert(key);
                            }
                        }
                        
                        if(!matches.empty()) {
                            auto str = prettify_array(Meta::toStr(matches));
                            Debug("Did you mean any of these settings keys? %S", &str);
                        }
                        
                        executed = false;
                    }
                }
                
                if(!executed)
                    default_config::warn_deprecated("input", GlobalSettings::load_from_string(default_config::deprecations(), GlobalSettings::map(), cmd, AccessLevelType::PUBLIC));
                
                if(!before)
                    SETTING(analysis_paused) = false;
                analysis->bump();
            });
        }
        
        static bool already_pausing = false;
        if(please_stop_analysis && !already_pausing) {
            already_pausing = true;
            please_stop_analysis = false;
            gui.work().add_queue("pausing", [&](){
                analysis->set_paused(true).get();
                already_pausing = false;
                
                if(SETTING(auto_categorize)) {
                    GUI::auto_categorize();
                } else if(SETTING(auto_train)) {
                    GUI::auto_train();
                } else if(SETTING(auto_apply)) {
                    GUI::auto_apply();
                }
                
                // check if results should be saved and the app should quit
                // automatically after analysis is done.
                else if(SETTING(auto_quit)) {
                    GUI::auto_quit();
                }
            });
        }
        
        if(!imgui_base) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    },
    [&](auto&){
        // ---
        // IN CASE RECORDING IS ACTIVATED
        // ---
        std::unique_lock<std::recursive_mutex> guard(gui.gui().lock());
        try {
            gui.do_recording();
        } catch(const std::exception& ex) {
            Except("Exception while recording ('%s').", ex.what());
        }
    },
    [&](gui::SFLoop& loop){
        static int last_seconds = -1;
        int seconds = (int)loop.time_since_last_update().elapsed();
        if(seconds != last_seconds) {
            if(seconds > 1 && !CheckUpdates::user_has_been_asked()) {
                static bool currently_asking = false;
                if(!currently_asking) {
                    currently_asking = true;
                    GUI::instance()->gui().dialog([](gui::Dialog::Result r) {
                        if(r == gui::Dialog::OKAY) {
                            SETTING(app_check_for_updates) = default_config::app_update_check_t::automatically;
                        } else if(r == gui::Dialog::ABORT) {
                            SETTING(app_check_for_updates) = default_config::app_update_check_t::manually;
                            
                            auto website = "https://github.com/mooch443/trex/releases";
                #if __linux__
                            auto pid = fork();
                            if (pid == 0) {
                                execl("/usr/bin/xdg-open", "xdg-open", website, (char *)0);
                                exit(0);
                            }
                #elif __APPLE__
                            auto pid = fork();
                            if (pid == 0) {
                                execl("/usr/bin/open", "open", website, (char *)0);
                                exit(0);
                            }
                #elif defined(WIN32)
                            ShellExecute(
                                NULL,
                                "open",
                                website,
                                NULL,
                                NULL,
                                SW_SHOWNORMAL
                            );
                #endif
                        } else {
                            SETTING(app_check_for_updates) = default_config::app_update_check_t::manually;
                        }
                        
                        try {
                            CheckUpdates::write_version_file();
                            
                        } catch(...) { }
                        
                    }, "Do you want to check for updates automatically? Automatic checks are performed in the background weekly if you've been idle for a while. Otherwise you can still check manually by opening the top-right menu and choosing <b><str>check updates</str></b>, or you can super-manually go to <ref>https://github.com/mooch443/trex</ref> and check for the latest releases yourself.", "Check for updates", "Weekly", "Super Manually", "Manually");
                }
                
            } else if(seconds > 1) {
                CheckUpdates::this_is_a_good_time();
            }
        }
        last_seconds = seconds;
    });
    
    Debug("Preparing for shutdown...");
    CheckUpdates::cleanup();
    Categorize::terminate();
    Recognition::notify();
    
    {
        std::lock_guard<std::mutex> lock(data_mutex);
        delete tmp;
        tmp = nullptr;
    }
    if(imgui_base)
        delete imgui_base;
    analysis->terminate();
    
    tracker.prepare_shutdown();
    
    if(log_file)
        fclose(log_file);
    log_file = NULL;
    
    exit(SETTING(error_terminate) ? 1 : 0);
}
