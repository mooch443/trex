//===========================================================================//
//                                                                           //
// Project: TRrex                                                            //
//                                                                           //
//===========================================================================//

//-Includes--------------------------------------------------------------------

#include <regex>
#include <signal.h>
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
#include <execinfo.h>
#endif

#include <queue>
#include <thread>
#include <misc/stacktrace.h>

#include <time.h>
#include <iomanip>

#if !COMMONS_NO_PYTHON
#include <python/GPURecognition.h>
#include <tracking/VisualIdentification.h>
#endif

#include <misc/CommandLine.h>
#include <gui/SFLoop.h>
#include <tracking/Tracker.h>
#include <gui/DrawStructure.h>
#include <misc/default_config.h>
#include <tracking/OutputLibrary.h>
#include <tracking/Output.h>
#include <gui/WorkProgress.h>
#include <gui/CheckUpdates.h>
#include <tracking/IndividualManager.h>

#include <tracking/SplitBlob.h>

#include <tracking/ConnectedTasks.h>
#include <processing/PadImage.h>
#include <processing/LuminanceGrid.h>
#include <processing/CPULabeling.h>
#include <tracking/HistorySplit.h>

#include <misc/Image.h>
#include "gui.h"

#include <tracking/DetectTag.h>
#include <misc/PixelTree.h>

#include <misc/PVBlob.h>

#include <misc/pretty.h>
#include <grabber/misc/default_config.h>
#include <misc/default_settings.h>

#if WITH_GITSHA1
#include "GitSHA1.h"
#endif

#include <gui/IMGUIBase.h>
#include <gui/FileChooser.h>
#include <gui/types/Checkbox.h>
#include <tracking/MemoryStats.h>
#include <tracking/Categorize.h>
#include <gui/DrawCVBase.h>
#include <gui/GUICache.h>
#include "VideoOpener.h"
#include <misc/PythonWrapper.h>
#include <gui/DrawBlobView.h>

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

#include <file/DataLocation.h>

//-Functions-------------------------------------------------------------------

using namespace track;
using namespace file;
namespace py = Python;

//! Combining grab::default settings and track::default settings
SettingsMaps combined;

std::mutex data_mutex;
double data_sec = 0.0, data_kbytes = 0.0;
double frames_sec = 0, frames_count = 0;

ENUM_CLASS(Arguments,
           d,dir,i,input,s,settings,nowindow,load,h,fs,p,r,update,quiet,y)

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
    vsnprintf(buf, sizeof(buf), fmt, argptr);
    va_end(argptr);
    fprintf(stderr, "%s", buf);
    exit(-1);
}

bool pause_stuff = false;

#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
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
    
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
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

std::set<file::Path> parse_input(const cmn::CommandLine::Option& option) {
    file::Path path = file::DataLocation::parse("input", file::Path(option.value));

    if (utils::contains(option.value, '*')) {
        std::set<file::Path> found;

        std::regex pattern(utils::find_replace(option.value, "*", ".*"));
        file::Path folder = file::DataLocation::parse("input", file::Path(option.value).remove_filename());
        print("Scanning pattern ", option.value, " in folder ", folder.str(), "...");

        for (auto &file : folder.find_files("pv")) {
            if (!file.is_regular()) {
                continue;
            }

            auto filename = (std::string) file.filename();

            if (std::regex_match(filename, pattern)) {
                found.insert(file);
            }
        }

        if (found.size() == 1) {
            path = file::DataLocation::parse("input", *found.begin());

        } else if (found.size() > 1) {
            print("Found too many files matching the pattern ", option.value, ": ", found, ".");
            return found;
        } else {
            print("No files found that match the pattern ", option.value, ".");
        }
    }
    
    return {path};
}

int main(int argc, char** argv)
{
    // set environment variable KMP_DUPLICATE_LIB_OK to "TRUE"
    //SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE");

#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
    
    const char* locale = "C";
    std::locale::global(std::locale(locale));
    
#if __APPLE__
    std::string PATH = (std::string)getenv("PATH");
    if(!utils::contains(PATH, "/usr/local/bin")) {
        PATH += ":/usr/local/bin";
        setenv("PATH", PATH.c_str(), 1);
    }
#endif
    
    default_config::register_default_locations();
    GlobalSettings::map().set_print_by_default(true);
    
    gui::init_errorlog();
    set_thread_name("main");
    
    progname = *(argv);
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
    std::atexit(at_exit);
    setenv("KMP_DUPLICATE_LIB_OK", "True", 1);
#endif
    init_signals();
    
#if TRACKER_GLOBAL_THREADS
    FormatWarning("Using only",TRACKER_GLOBAL_THREADS,"threads (-DTRACKER_GLOBAL_THREADS).");
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
    
    const auto set_combined_access_level = [](auto& name, AccessLevel level) {
        combined.access_levels[name] = level;
    };
    
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    //default_config::get(combined.map, combined.docs, set_combined_access_level);
    
    std::vector<std::string> save = combined.map.has("meta_write_these") ? combined.map.at("meta_write_these").value<std::vector<std::string>>() : std::vector<std::string>{};
    print("Have these keys:", combined.map.keys());
    std::set<std::string> deleted_keys;
    for(auto key : combined.map.keys()) {
        if(not contains(save, key)) {
            deleted_keys.insert(key);
            combined.map.erase(key);
        }
    }
    print("Deleted keys:", deleted_keys);
    print("Remaining:", combined.map.keys());
    
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_focus_group"].get().set_do_print(false);
    
    CommandLine::init(argc, argv, true);
    
    if(argc == 2) {
        if(std::string(argv[1]) == "-options") {
            for(auto arg : Arguments::names) {
                printf("-%s ", arg);
            }
            exit(0);
            
        } else if(std::string(argv[1]) == "-path") {
            default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
            default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
            
            file::cd(file::DataLocation::parse("app"));
            
            auto default_path = file::DataLocation::parse("default.settings");
            if(default_path.exists()) {
                DebugHeader("LOADING FROM ",default_path);
                default_config::warn_deprecated(default_path, GlobalSettings::load_from_file(default_config::deprecations(), default_path.str(), AccessLevelType::STARTUP));
                DebugHeader("LOADED ",default_path);
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
        print("Conda prefix: ", conda_prefix);
        if(!conda_prefix.empty()) {
            std::set<file::Path> files;
            try {
                if (file::Path(conda_prefix + "/bin").exists()) {
                    files = file::Path(conda_prefix + "/bin").find_files();
                }
                else
                    files = file::Path(conda_prefix + "/Library/bin").find_files();
            }
            catch (const UtilsException&) {
            }

            for(auto file : files) {
                if(file.filename() == "ffmpeg" || file.filename() == "ffmpeg.exe") {
                    print("Found ffmpeg in ",file);
                    SETTING(ffmpeg_path) = file;
                    break;
                }
            }
        }
        
        if(SETTING(ffmpeg_path).value<file::Path>().empty())
            FormatWarning("Cannot find ffmpeg.exe in search paths.");
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
                            print("Found ffmpeg in ", file);
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
    
    print("CWD: ", file::cwd());
    DebugHeader("LOADING COMMANDLINE");
    file::cd(file::DataLocation::parse("app"));
    print("CWD: ", file::cwd());
    
    auto&cmd = CommandLine::instance();
    for(auto &option : cmd.settings()) {
        if(utils::lowercase(option.name) == "output_prefix") {
            SETTING(output_prefix) = option.value;
        }
    }

    auto default_path = file::DataLocation::parse("default.settings");
    if(default_path.exists()) {
        DebugHeader("LOADING FROM ",default_path);
        default_config::warn_deprecated(default_path, GlobalSettings::load_from_file(default_config::deprecations(), default_path.str(), AccessLevelType::STARTUP));
        DebugHeader("LOADED ",default_path);
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
                    auto files = parse_input(option);
                    if(files.size() == 1) {
                        Path path = file::DataLocation::parse("input", Path(option.value).add_extension("pv"));
                        if(!path.exists())
                            throw U_EXCEPTION("Cannot find video file ",path,". (",path.exists(),")");
                        
                        SETTING(filename) = path.remove_extension();
                    }
                    break;
                }
                    
                case Arguments::r:
                    load_results_from = Path(option.value);
                    load_results = true;
                    print("Loading results from ",load_results_from,"...");
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
                        
                        file::Path path = file::DataLocation::parse("output", "parameters_trex.rst");
                        auto f = path.fopen("wb");
                        if(!f)
                            throw U_EXCEPTION("Cannot open ",path.str());
                        fwrite(rst.data(), sizeof(char), rst.length(), f.get());
                        
                        //printf("%s\n", rst.c_str());
                        print("Saved at ",path,".");
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
                            
                            print("Opening ",filename," in browser...");
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
                        print("-i <filename>       Input video source (basler/webcam/video path)");
                        print("-d <folder>         Set the default input/output folder");
                        print("-s <filename>       Set the .settings file to be used (default is <input>.settings)");
                        print("-load               Loads previously saved .results file (if it exists)");
                        print("-h                  Prints this message");
                        print("-h settings         Displays all default settings with description in the default browser");
                        exit(0);
                    }
                    
                    exit(0);
                    break;
                    
#if !COMMONS_NO_PYTHON
                case Arguments::update: {
                    auto status = CheckUpdates::perform(false).get();
                    if(is_in(status, CheckUpdates::VersionStatus::OLD, CheckUpdates::VersionStatus::ALREADY_ASKED))
                    {
                        CheckUpdates::display_update_dialog(GUI::instance() ? &GUI::instance()->gui() : nullptr);
                    } else if(status == CheckUpdates::VersionStatus::NEWEST) {
                        print("You have the newest version (",CheckUpdates::newest_version(),").");
                    } else
                         FormatError("Error checking for the newest version: ",CheckUpdates::last_error(),". Please check your internet connection and try again.");
                    
                    py::deinit().get();
                    exit(0);
                    break;
                }
#endif
                    
                default:
                    FormatWarning("Unknown option ",option.name," with value ", option.value);
                    break;
            }
            
        } else if(option.name == "load_settings_from_results") {
            load_settings_from_results = Meta::fromStr<std::vector<std::string>>(option.value);
            
        } else {
            FormatWarning("Unknown option ", option.name, " with value ", option.value);
        }
    }

    if (argc == 2) {
        Path path = file::DataLocation::parse("input", Path(argv[1]));
        if (path.exists()) {
            SETTING(filename) = path.remove_extension();
            SETTING(output_dir) = path.remove_filename();
        }
    }
    
    // check whether a file exists
    gui::VideoOpener::Result opening_result;
    
    if(SETTING(filename).value<Path>().empty()) {
        cmd.load_settings(&combined.map);
        
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
                print("Opening a video file: ",opening_result.tab.name," (wd: ",wd,")");
#if defined(__APPLE__)
                wd = wd / ".." / ".." / ".." / "TGrabs.app" / "Contents" / "MacOS" / "TGrabs";
#else
                if (wd.empty())
                    wd = "tgrabs";
                else
                    wd = wd / "tgrabs";
#endif
                auto exec = wd.str() + " " + opening_result.cmd;
                print("Executing ", exec);

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
            print("You can specify a file to be opened using ./trex -i <filename>");
            return 0;
        }
    }

    /**
     * Load video file and additional settings from files.
     */
    bool executed_a_settings = false;
    
    DebugHeader("LOADING FILE");
    
    pv::File video(SETTING(filename).value<Path>(), pv::FileMode::READ);
    
    if(video.header().version <= pv::Version::V_2) {
        SETTING(crop_offsets) = CropOffsets();
        
        Path settings_file = file::DataLocation::parse("settings");
        if(default_config::execute_settings_file(settings_file, AccessLevelType::STARTUP))
            executed_a_settings = true;
        
        auto output_settings = file::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            if(default_config::execute_settings_file(output_settings, AccessLevelType::STARTUP))
                executed_a_settings = true;
        }
        
        video.close();
    }
    
    try {
        if(!video.header().metadata.empty())
            sprite::parse_values(sprite::MapSource{video.filename()}, GlobalSettings::map(), video.header().metadata, &combined.map);
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
    SETTING(video_length) = uint64_t(video.length().get());
    SETTING(video_info) = std::string(video.get_info());
    
    if(SETTING(frame_rate).value<uint32_t>() <= 0) {
        FormatWarning("frame_rate == 0, calculating from frame tdeltas.");
        video.generate_average_tdelta();
        SETTING(frame_rate) = (uint32_t)max(1, int(video.framerate()));
    }
    
    Library::InitVariables();
    
    Path settings_file = file::DataLocation::parse("settings");
    if(SETTING(settings_file).value<file::Path>().empty()) {
        if(default_config::execute_settings_file(settings_file, AccessLevelType::STARTUP))
            executed_a_settings = true;
        else {
            SETTING(settings_file) = file::Path();
            FormatWarning("Settings file ",settings_file," does not exist.");
        }
    }
    
    /**
     * Try to load Settings from the command-line that have been
     * ignored previously.
     */
    cmd.load_settings(&combined.map);
    
    if(SETTING(settings_file).value<file::Path>().empty()) {
        auto output_settings = file::DataLocation::parse("output_settings");
        if(output_settings.exists() && output_settings != settings_file) {
            if(default_config::execute_settings_file(output_settings, AccessLevelType::STARTUP))
                executed_a_settings = true;
            else if(!executed_a_settings)
                FormatWarning("Output settings ",output_settings," does not exist.");
        }
        
    } else {
        if(default_config::execute_settings_file(settings_file, AccessLevelType::STARTUP))
            executed_a_settings = true;
        else
            FormatWarning("Settings file ",settings_file," does not exist.");
    }
    
    file::Path first_time_setup("first_time_setup");
    if(!first_time_setup.exists()) {
        DebugHeader("TRex first time setup");
        print("Software is started the first time!");
        
        if(contains(cmd.options(), "y")) {
            print("-y is set, so you accept the default output_dir choice:");
            print("\toutput_dir = ", SETTING(output_dir).value<file::Path>());
            first_time_setup.fopen("wb"); // opening suffices to write it
        } else {
            
        }
    }
    
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
                FormatExcept("You are using the deprecated command-line option ",option.name,". Please use ",r," instead.");
            } else
                FormatExcept("You are using the deprecated command-line option ",option.name,". There is no replacement. Please remove it from your start parameters.");
        }
    }
    
    if(contains_illegal_options) {
        throw U_EXCEPTION("Cannot continue with the mentioned deprecated command-line options.");
    }
    
    cmd.load_settings();
    
    if(SETTING(output_fields).value< std::vector<std::pair<std::string, std::vector<std::string>>>>().empty()) {
        sprite::Map config;
        GlobalSettings::docs_map_t docs;
        default_config::get(config, docs, NULL);
        
        SETTING(output_fields) = config.at("output_fields").value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
    }
    
    if(SETTING(cam_undistort)) {
        cv::Mat map1, map2;
        cv::Size size = video.header().resolution;
        
        auto cam_data = SETTING(cam_matrix).value<std::vector<float>>();
        cv::Mat cam_matrix = cv::Mat(3, 3, CV_32FC1, cam_data.data());
        auto undistort_data = SETTING(cam_undistort_vector).value<std::vector<float>>();
        cv::Mat cam_undistort_vector = cv::Mat(1, 5, CV_32FC1, undistort_data.data());
        
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
        
        GlobalSettings::map()["cam_undistort1"].get().set_do_print(false);
        GlobalSettings::map()["cam_undistort2"].get().set_do_print(false);
        GlobalSettings::get("cam_undistort1") = map1;
        GlobalSettings::get("cam_undistort2") = map2;
    }
    
    if(!SETTING(exec).value<file::Path>().empty()) {
        Path exec_settings = file::DataLocation::parse("settings", SETTING(exec).value<file::Path>());
        if(!default_config::execute_settings_file(exec_settings, AccessLevelType::STARTUP))
            FormatExcept("Settings file ",exec_settings.str()," cannot be found or execution failed.");
        else
            executed_a_settings = true;
        
        SETTING(exec) = file::Path();
    }
    
    //! if we used the open file dialog and changed settings, use them
    if(!opening_result.extra_command_lines.empty()) {
        GlobalSettings::load_from_string(sprite::MapSource::CMD, default_config::deprecations(), GlobalSettings::map(), opening_result.extra_command_lines, AccessLevelType::STARTUP);
    }
    
    Tracker tracker(Image::Make(average), video);
    tracker.update_history_log();
    
    if(!SETTING(log_file).value<file::Path>().empty()) {
        auto path = SETTING(log_file).value<file::Path>();//file::DataLocation::parse("output", SETTING(log_file).value<file::Path>());
        set_log_file(path.str());
        print("Logging to ", path,".");
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
        float threshold_step = (end_threshold - start_threshold) / narrow_cast<float>(values.size()) + 0.5;
        
        GenericThreadPool pool(cmn::hardware_concurrency(), "evaluate_thresholds");
        std::mutex sync;
        
        size_t added_frames = 0, processed_frames = 0;
        
        auto range = arange<Frame_t>{
            0_f, video.length(),
            max(1_f, Frame_t(Frame_t::number_t(float(video.length().get()) / 1000.f)))
        };
        distribute_indexes([&](auto, auto start, auto end, auto){
            pv::Frame frame;
            CPULabeling::ListCache_t cache;
            
            for(auto it = start; it != end; ++it) {
                video.read_frame(frame, *it);
            //video.read_frame(next_frame, i+1);
            
                size_t j = 0;
                for(float threshold = start_threshold; threshold <= end_threshold; threshold += threshold_step, ++j)
                {
                    float pixel_average = 0, pixel_samples = 0;
                    float number = 0;
                    
                    for (uint16_t k=0; k<frame.n(); ++k) {
                        if(frame.pixels().at(k)->size() > 30)
                        {
                            // consider blob
                            auto blob = frame.blob_at(k);
                            auto blobs = pixel::threshold_blob(cache, blob.get(), narrow_cast<int>(threshold), Tracker::instance()->background());
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
                
                assert(j <= values.size());
            }
            
            std::lock_guard<std::mutex> guard(sync);
            processed_frames += (*end).get() - (*start).get() + 1;
            if(processed_frames % 10000 == 0) {
                print(processed_frames,"/",added_frames," (",processed_frames / float(added_frames) * 100,"%)");
            }
            
        }, pool, range.begin(), range.end());
        
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
            return gui::GlobalSettings::invalid();
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
            return gui::GlobalSettings::invalid();
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
            return gui::GlobalSettings::invalid();
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
        FormatWarning("No settings file can be loaded, so the program will try to automatically determine individual sizes and numbers.");
        sprite::Map default_map;
        GlobalSettings::docs_map_t default_docs;
        default_config::get(default_map, default_docs, NULL);
        
        if(SETTING(auto_number_individuals).value<bool>() == default_map.at("auto_number_individuals").value<bool>())
        {
            SETTING(auto_number_individuals) = SETTING(track_max_individuals).value<uint32_t>() == default_map.at("track_max_individuals").value<uint32_t>();
        }
        
        if(SETTING(auto_minmax_size).value<bool>() == default_map.at("auto_minmax_size").value<bool>())
        {
            SETTING(auto_minmax_size) = SETTING(track_size_filter).value<BlobSizeRange>() == default_map.at("track_size_filter").value<BlobSizeRange>();
        }
    }
    
    Tracker::auto_calculate_parameters(video);
    
    default_config::warn_deprecated("global", GlobalSettings::map());
    
    if(FAST_SETTING(track_max_individuals) == 1
       && SETTING(auto_apply))
    {
        FormatError("Cannot use a network on a single individual. Disabling auto_apply.");
        SETTING(auto_apply) = false;
    }
    
    if(FAST_SETTING(track_max_individuals) == 1
       && SETTING(auto_train))
    {
        FormatError("Cannot train a network on a single individual. Disabling auto_train.");
        SETTING(auto_train) = false;
    }
    
    if(SETTING(auto_train) || SETTING(auto_apply)) {
        SETTING(auto_train_on_startup) = true;
    }
    
    if(SETTING(auto_tags)) {
        SETTING(auto_tags_on_startup) = true;
    }
    
    if(!SETTING(auto_train_on_startup) && SETTING(auto_train_dont_apply)) {
        FormatWarning("auto_train_dont_apply was set without auto_train enabled. This may lead to confusing behavior. Overwriting auto_train_dont_apply = false.");
        SETTING(auto_train_dont_apply) = false;
    }
    
    Library::Init();
    DebugHeader("STARTING PROGRAM");
    
    pv::Blob blob;
    auto copy = blob.properties();
    print("BasicStuff<",sizeof(track::BasicStuff),"> "
          "PostureStuff<",sizeof(track::PostureStuff),"> "
          "Individual<",sizeof(track::Individual),"> "
          "Blob<",sizeof(pv::Blob),"> "
          "MotionRecord<",sizeof(MotionRecord),"> "
          "Image<",sizeof(Image::Ptr),"> "
          "std::shared_ptr<std::vector<HorizontalLine>><",sizeof(std::shared_ptr<std::vector<HorizontalLine>>),"> "
          "Bounds<",sizeof(Bounds),"> "
          "bool<",sizeof(bool),"> "
          "cmn::Blob::properties<",sizeof(decltype(copy)),">");
    print("localcache:",sizeof(Individual::LocalCache)," identity:",sizeof(Identity)," std::map<long_t, Vec2>:", sizeof(std::map<long_t, Vec2>));
    print("BasicStuff:",sizeof(BasicStuff)," pv::Blob:",sizeof(pv::Blob)," Compressed:", sizeof(pv::CompressedBlob), " pv::bid:", sizeof(pv::bid));
    print("Midline:",sizeof(Midline)," MinimalOutline:",sizeof(MinimalOutline));
    
    auto graph = std::make_unique<gui::DrawStructure>();
    GUI *gui_instance = new GUI(graph.get(), video, tracker.average(), tracker);
    auto gui_lock = GUI_LOCK(gui_instance->gui().lock());
    
    //try {
    GUI &gui = *gui_instance;
    gui.frameinfo().video_length = video.length() > 0_f ? (video.length().get() - 1) : 0;
    
    if(!SETTING(gui_connectivity_matrix_file).value<file::Path>().empty()) {
        try {
            gui.load_connectivity_matrix();
        } catch(const UtilsException&) { }
    }
    
    static bool please_stop_analysis = false;
    
    static std::atomic<Frame_t> currentID(Frame_t{});
    static std::queue<std::unique_ptr<PPFrame>> unused;
    static std::mutex mutex;
    static constexpr Frame_t cache_size{Frame_t::number_t(10)};
    
    for (auto i=0_f; i<cache_size; ++i)
        unused.emplace(std::make_unique<PPFrame>());
        
    //std::mutex stage1_mutex;
    //double time_stage1 = 0, time_stage2 = 0, stage1_samples = 0, stage2_samples = 0;
    static GenericThreadPool pool(cmn::hardware_concurrency(), "preprocess_main");
    
    //! Stages
    std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>> tasks =
    {
        [&video](std::unique_ptr<PPFrame>&& ptr, auto&) -> bool {
            auto idx = ptr->index();
            auto range = Tracker::analysis_range();
            if(!range.contains(idx) && idx != range.end() && (not Tracker::end_frame().valid() || idx > Tracker::end_frame())) {
                std::unique_lock<std::mutex> lock(mutex);
                unused.emplace(std::move(ptr));
                return false;
            }

            if(not range.contains(idx))
                throw U_EXCEPTION("Outside of analysis range: ", idx, " vs. ", range);

            Timer timer;
            pv::Frame frame;
            video.read_frame(frame, idx);
            Tracker::preprocess_frame(std::move(frame), *ptr, pool.num_threads() > 1 ? &pool : NULL, PPFrame::NeedGrid::NoNeed, video.header().resolution, false);

            ptr->set_loading_time(timer.elapsed());

            // clear stored blob data, so that the destructor is called
            // in a different thread (balancing) if they arent needed.
            IndividualManager::clear_pixels();
            return true;
        },

        [&tracker, &video, gui_instance, &gui](std::unique_ptr<PPFrame>&& ptr, auto&) -> bool {
            static Timer fps_timer;
            static Image empty(0, 0, 0);

            Timer timer;

            static Timing all_processing("Analysis::process()", 50);
            TakeTiming all(all_processing);

            LockGuard guard(w_t{}, "Analysis::process()");
            if(GUI_SETTINGS(terminate))
                return false;
            
            auto range = Tracker::analysis_range();

            auto idx = ptr->index();
            if (idx >= range.start()
                && max(range.start(), tracker.end_frame().valid() ? (tracker.end_frame() + 1_f) : 0_f) == idx
                && tracker.properties(idx) == nullptr
                && idx <= Tracker::analysis_range().end())
            {
                tracker.add(*ptr);

                static Timing after_track("Analysis::after_track", 10);
                TakeTiming after_trackt(after_track);
                
                if(idx + 1_f == video.length())
                    please_stop_analysis = true;

                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    data_kbytes += ptr->num_pixels() / 1024.0;
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

                        float percent = min(1, (ptr->index() - range.start()).get() / float(range.length().get() + 1)) * 100;
                        DurationUS us{ uint64_t(max(0, (double)(range.end() - ptr->index()).get() / double( frames_sec_average / frames_sec_samples ) * 1000 * 1000)) };
                        std::string str;
                        
                        if(FAST_SETTING(analysis_range).start != -1 || FAST_SETTING(analysis_range).end != -1)
                            str = format<FormatterType::NONE>("frame ", ptr->index(), "/", range.end(), "(", video.length(), ") (", dec<2>(data_sec / 1024.0), "MB/s @ ", dec<2>(frames_sec), "fps eta ", us, ") ", dec<2>(Tracker::average_seconds_per_individual() * 1000 * 1000),
#if defined(__APPLE__)
                                                              "µs/individual"
#else
                                                              "mus/individual"
#endif
                                                              );
                        else
                            str = format<FormatterType::NONE>("frame ", ptr->index(), "/", range.end(), " (", dec<2>(data_sec/1024.0), "MB/s @ ", dec<2>(frames_sec), "fps eta ", us, ") ", dec<2>(Tracker::average_seconds_per_individual() * 1000 * 1000),
#if defined(__APPLE__)
                                                              "µs/individual"
#else
                                                              "mus/individual"
#endif
                                                              );

                        {
                            // synchronize with debug messages
                            //std::lock_guard<std::mutex> debug_lock(DEBUG::debug_mutex());
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
                            print(dec<2>(percent),"% ", str.c_str());
                            print_timer.reset();
                        }
                    } 

                    gui::WorkProgress::add_queue("", [fs = frames_sec, &gui
#ifndef NDEBUG
                                                      ,&gui_instance
#endif
                                                     ](){
                        assert(gui_instance);
                        gui.frameinfo().current_fps = narrow_cast<int>(fs);
                    });
                }

                frames_count++;
            }

            static Timing procpush("Analysis::process::unused.push", 10);
            TakeTiming ppush(procpush);
            std::unique_lock<std::mutex> lock(mutex);
            unused.emplace(std::move(ptr));

            return true;
        }
    };
    
    ConnectedTasks analysis(std::move(tasks));
    analysis.start(// main thread
        [&tracker, &analysis, &video]() {
            auto endframe = tracker.end_frame();
            if(not currentID.load().valid()
               || not endframe.valid()
               || currentID.load() > endframe + cache_size
               || (analysis.stage_empty(0) && analysis.stage_empty(1))
               || currentID.load() < endframe)
            {
                currentID = endframe; // update current as well
            }
        
            auto range = Tracker::analysis_range();
            if(not currentID.load().valid()
               or currentID.load() < range.start())
                currentID = range.start() - 1_f;
            
            if(not endframe.valid())
                endframe = range.start();
            
            if(FAST_SETTING(analysis_range).end != -1
               && endframe >= Frame_t(sign_cast<Frame_t::number_t>(FAST_SETTING(analysis_range).end))
               && !SETTING(terminate)
               && !please_stop_analysis)
            {
                please_stop_analysis = true;
            }
            
            while(not currentID.load().valid()
                  || (currentID.load() < max(range.start(), endframe) + cache_size
                      && currentID.load() + 1_f < video.length()))
            {
                std::scoped_lock lock(mutex);
                if(unused.empty())
                    break;
                
                auto ptr = std::move(unused.front());
                unused.pop();
                
                if(not currentID.load().valid())
                    currentID = range.start();
                else
                    currentID = currentID.load() + 1_f;
                ptr->set_index(currentID.load());
                
                analysis.add(std::move(ptr));
            }
        }
    );
        
    gui.set_analysis(&analysis);
    gui_lock.unlock();
    
#if !COMMONS_NO_PYTHON
    CheckUpdates::init(&gui.gui());
#endif
    
    GlobalSettings::map().register_callbacks({"track_pause"}, [&](auto) {
        analysis.bump();
        
        bool pause = SETTING(track_pause).value<bool>();
        if(analysis.paused() != pause) {
            print("Adding to queue...");
            
            gui::WorkProgress::add_queue("pausing", [&analysis, pause](){
                if(analysis.paused() != pause) {
                    analysis.set_paused(pause);
                    print("Paused.");
                }
            });
            
            print("Added.");
        }
    });
    
    auto get_settings_from_results = [](const Path& filename) -> std::string {
        print("Trying to open results ",filename.str());
        ResultsFormat file(filename, NULL);
        file.start_reading();
        
        if(file.header().version >= ResultsFormat::V_14) {
            return file.header().settings;
        } else
            FormatExcept("Cannot load settings from results file < V_14");
        return "{}";
    };
    
    if(FAST_SETTING(track_pause) || load_results) {
        analysis.set_paused(true).get();
        
        if(load_results) {
            if(!executed_a_settings) {
                auto path = TrackingResults::expected_filename();
                auto str = get_settings_from_results(load_results_from.empty() ? path : load_results_from);
                print("Loading settings from ",path,"...");
                try {
                    default_config::warn_deprecated(path.str(), 
                        GlobalSettings::load_from_string(sprite::MapSource{path}, 
                            default_config::deprecations(), GlobalSettings::map(), str, AccessLevelType::STARTUP));
                    executed_a_settings = true;
                } catch(const UtilsException& e) {
                    FormatExcept("Cannot load settings from results file. Skipping.");
                }
            }
            
            gui.load_state(GUI::GUIType::TEXT, load_results_from);
            
            // explicitly set gui_frame if present in command-line
            if(cmd.settings_keys().find("gui_frame") != cmd.settings_keys().end()) {
                GUI::gui_frame_on_startup().frame = Meta::fromStr<Frame_t>(cmd.settings_keys().at("gui_frame"));
                gui::WorkProgress::add_queue("", [&](){
                    SETTING(gui_frame) = GUI::gui_frame_on_startup().frame;
                });
            }
            
            if(cmd.settings_keys().find("gui_focus_group") != cmd.settings_keys().end()) {
                GUI::gui_frame_on_startup().focus_group = Meta::fromStr<std::vector<Idx_t>>(cmd.settings_keys().at("gui_focus_group"));
                gui::WorkProgress::add_queue("", [&](){
                    SETTING(gui_focus_group) = GUI::gui_frame_on_startup().focus_group.value();
                });
            }
        }
    }
    
    if(!load_results && !settings_file.exists()) {
        FormatError("Settings file ",settings_file.str()," cannot be found.");
    }
    
    if(!load_settings_from_results.empty()) {
        auto path = TrackingResults::expected_filename();
        auto str = get_settings_from_results(path);
        sprite::Map defaults;
        
        GlobalSettings::docs_map_t docs;
        default_config::get(defaults, docs, NULL);
        auto added = GlobalSettings::load_from_string(sprite::MapSource{path}, default_config::deprecations(), defaults, str, AccessLevelType::STARTUP, true);
        
        DebugHeader("LOADING SETTINGS FROM ", path);
        
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
                FormatExcept("Cannot load ",name," from results file.");
            }
        }
        
        DebugHeader("/ LOADED SETTINGS FROM ", path);
    }
    
#if !COMMONS_NO_PYTHON
    if(SETTING(auto_train)) {
        FormatWarning("The application is going to attempt to automatically train the network upon finding a suitable consecutive segment.");
    }
    if(SETTING(auto_apply)) {
        if(SETTING(auto_train) || !py::VINetwork::weights_available()) {
            auto path = py::VINetwork::network_path();
            path = path.add_extension("npz");
            
            SETTING(error_terminate) = true;
            SETTING(terminate) = true;
            throw U_EXCEPTION("Cannot apply a network without network_weights available. (searching at ",path.str(),")");
        }
        
        FormatWarning("The application is going to apply a trained network after finishing the analysis and auto_correct it afterwards.");
    }
    if(SETTING(auto_categorize)) {
        if(!Categorize::weights_available()) {
            auto file = (std::string)SETTING(filename).value<file::Path>().filename();
            auto output = (std::string)file::DataLocation::parse("output").str();
            
            SETTING(error_terminate) = true;
            SETTING(terminate) = true;
            throw U_EXCEPTION("Make sure that a file called '",file,"_categories.npz' is located inside '",output,"'");
        }
        FormatWarning("The application is going to load a pretrained categories network and apply it after finishing the analysis (or loading).");
    }
#endif

    if(SETTING(auto_quit))
        FormatWarning("Application is going to quit after analysing and exporting data.");
    
    gui::IMGUIBase *imgui_base = nullptr;
    if(not (GlobalSettings::map().has("nowindow") 
            ? SETTING(nowindow).value<bool>()
            : false))
    {
        imgui_base = new gui::IMGUIBase(gui.window_title(), std::move(graph), [&](auto&){
            //std::lock_guard<std::recursive_mutex> lock(gui.gui().lock());
            if(SETTING(terminate))
                return false;
            
            return true;
        }, GUI::event);
        
        gui.set_base(imgui_base);
        imgui_base->platform()->set_icons({
            file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"Icon16.png"),
            file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"Icon32.png"),
            file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"Icon64.png")
        });
    }
    
    if(go_fullscreen)
        gui.toggle_fullscreen();
    
    //py::init();
    
    gui::SFLoop loop(gui.gui(), imgui_base, [&](gui::SFLoop&, gui::LoopStatus status){
        {
            auto guard = GUI_LOCK(gui.gui().lock());
            GUI::run_loop(status);
        }
        
        if(pause_stuff) {
            pause_stuff = false;
            
            std::string cmd;
            bool before;
            {
                before = analysis.is_paused();
                analysis.set_paused(true).get();
                
                LockGuard guard(w_t{}, "pause_stuff");
            
                print("Console opened.");
                print("Please enter command below (type help for available commands):");
                printf(">> ");
                std::getline(std::cin, cmd);
            }
            
            gui::WorkProgress::add_queue("", [&before, &analysis, cmd, &gui](){
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
                        print("You may type any of the following commands:");
                        print("\tinfo\t\t\t\tPrints information about the current file");
                        print("\tsave_results [force]\t\tSaves a .results file (if one already exists, force is required to overwrite).");
                        print("\texport_data\t\tExports the tracked data to CSV/NPZ files according to settings.");
                        print("\tsave_config [force]\t\tSaves the current settings (if settings exist, force to overwrite).");
                        print("\tauto_correct [force]\t\tGenerates auto_corrected manual_matches. If force is set, applies them.");
                        print("\ttrain_network [load]\t\tStarts network training with currently selected segment. If load is set, loads weights and applies them.");
                        print("\treanalyse\t\t\tReanalyses the whole video from frame 0.");
                    }
                    else if(command == "info") {
                        print(gui.info(false));
                    }
                    /*else if(command == "retrieve_matches") {
                        gui::WorkProgress::add_queue("retrieving matches", [](){
                            Settings::manual_matches_t manual_matches;
                            {
                                LockGuard guard(ro_t{}, "retrieving matches");
                                
                                for(auto && [id, fish] : Tracker::individuals()) {
                                    for(auto frame : fish->manually_matched()) {
                                        auto blob = fish->blob(frame);
                                        if(blob) {
                                            if(manual_matches[frame].find(id) != manual_matches[frame].end()
                                               && manual_matches[frame][id] != blob->blob_id())
                                            {
                                                print("Other blob (",manual_matches[frame][id]," != ",blob->blob_id(),") was assigned fish ",id," in frame ",frame);
                                            }
                                            for(auto && [fdx, bdx] : manual_matches[frame]) {
                                                if(fdx != id && bdx == blob->blob_id()) {
                                                    print("Other fish (",fdx," != ",id,") was assigned blob ",bdx," in frame ",frame);
                                                    break;
                                                }
                                            }
                                            
                                            manual_matches[frame][id] = blob->blob_id();
                                        }
                                    }
                                }
                            }
                            
                            auto str = prettify_array(Meta::toStr(manual_matches));
                            print(str);
                            
                            SETTING(manual_matches) = manual_matches;
                        });
                    }*/
                    else if(utils::beginsWith(command, "save_results")) {
                        gui.save_state(GUI::GUIType::TEXT, utils::endsWith(command, " force"));
                    }
                    else if(utils::beginsWith(command, "export_data")) {
                        gui.export_tracks();
                    }
#if !COMMONS_NO_PYTHON
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
                        
                        py::schedule([copy]() {
                            print("Executing ",copy);
                            try {
                                PythonIntegration::execute(copy);
                            } catch(const SoftExceptionImpl& e) {
                                print("Runtime error: ", e.what());
                            }
                        });
                    }
#endif
                    else if(utils::beginsWith(command, "continue")) {
                        before = true;
                        SETTING(track_pause) = false;
                    }
                    else if(utils::lowercase(command) == "print_memory") {
                        LockGuard guard(ro_t{}, "print_memory");
                        mem::IndividualMemoryStats overall;
                        IndividualManager::transform_all([&](auto, auto fish) {
                            mem::IndividualMemoryStats stats(fish);
                            stats.print();
                            overall += stats;
                        });
                    
                        overall.print();
                        
                        mem::TrackerMemoryStats stats;
                        stats.print();
                        
                        mem::OutputLibraryMemoryStats ol;
                        ol.print();
                    }
                    else if(utils::beginsWith(command, "save_config")) {
                        gui.write_config(utils::endsWith(command, " force"), GUI::GUIType::TEXT);
                    }
#if !COMMONS_NO_PYTHON
                    else if(utils::beginsWith(command, "auto_correct")) {
                        gui.auto_correct(GUI::GUIType::TEXT, utils::endsWith(command, " force"));
                    }
                    else if(utils::beginsWith(command, "train_network")) {
                        gui.training_data_dialog(GUI::GUIType::TEXT, utils::endsWith(command, " load"));    
                    } 
#endif
                    else if(utils::beginsWith(command, "reanalyse")) {
                        GUI::reanalyse_from(0_f, false);
                        SETTING(track_pause) = false;
                        /*{
                            LockGuard guard;
                            Tracker::instance()->remove_frames(0);
                        }
                        
                        if(SETTING(track_pause))
                            SETTING(track_pause) = false;*/
                        
                    } else if(GlobalSettings::map().has(command)) {
                        print("Object ",command);
                        auto str = GlobalSettings::get(command).toStr(),
                            val = GlobalSettings::get(command).get().valueString();
                        print(str.c_str(),"=",val.c_str());
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
                            print("Did you mean any of these settings keys? ", str);
                        }
                        
                        executed = false;
                    }
                }
                
                if(!executed)
                    default_config::warn_deprecated("input", GlobalSettings::load_from_string(sprite::MapSource::CMD, default_config::deprecations(), GlobalSettings::map(), cmd, AccessLevelType::PUBLIC));
                
                if(!before)
                    SETTING(track_pause) = false;
                analysis.bump();
            });
        }
        
        static std::atomic<bool> already_pausing = false;
        if(please_stop_analysis && !already_pausing) {
            already_pausing = true;
            please_stop_analysis = false;
            gui::WorkProgress::add_queue("pausing", [&](){
                analysis.set_paused(true).get();
                already_pausing = false;
                GUI::tracking_finished();
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
        auto guard = GUI_LOCK(gui.gui().lock());
        try {
            gui.do_recording();
        } catch(const std::exception& ex) {
            FormatExcept("Exception while recording ('", ex.what(),"').");
        }
    },
    [&](gui::SFLoop& loop){
#if !COMMONS_NO_PYTHON
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
#endif
    });
    
    print("Preparing for shutdown...");
    gui::WorkProgress::stop();
    tracker::gui::blob_view_shutdown();
    
#if !COMMONS_NO_PYTHON
    CheckUpdates::cleanup();
    Categorize::terminate();
#endif
    
    analysis.terminate();
    
    tracker.prepare_shutdown();
    
    {
        delete gui_instance;
        gui_instance = nullptr;
    }
    if(imgui_base)
        delete imgui_base;
    
    if(log_file)
        fclose(log_file);
    log_file = NULL;
    
    exit(SETTING(error_terminate) ? 1 : 0);
}
