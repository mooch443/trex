//===========================================================================//
//                                                                           //
// Project: TGrabs                                                           //
//                                                                           //
//===========================================================================//

//-Includes--------------------------------------------------------------------

#include <types.h>
#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
#include <execinfo.h>
#endif
#include <signal.h>

#include <misc/CommandLine.h>
#include <video/VideoSource.h>
#include <misc/Blob.h>
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>
#if WITH_MHD
#include <http/httpd.h>
#endif
#include <grabber.h>
#include <gui.h>
#include <gui/DrawCVBase.h>
#include <file/Path.h>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/ocl.hpp>
#include <opencv2/opencv_modules.hpp>
#if defined(VIDEOS_USE_CUDA)
#include <opencv2/cudacodec.hpp>
#include <cuda/dynlink_nvcuvid.h>
#endif
#endif

#include <video/Video.h>
#include "CropWindow.h"

#include "default_config.h"
#include <tracker/misc/default_config.h>
#include <misc/default_settings.h>

#if WITH_GITSHA1
#include "GitSHA1.h"
#endif

#include <gui/IMGUIBase.h>
#include <python/GPURecognition.h>
#include <opencv2/core/utils/logger.hpp>

#include <tracking/Recognition.h>

#if __linux__
#include <X11/Xlib.h>
#endif

//-Functions-------------------------------------------------------------------

ENUM_CLASS(Arguments,
           d, dir, i, input, o, output, settings, s, nowindow, mask, h, help, p)

#ifndef WIN32
struct sigaction sigact;
#endif
char *progname;

void panic(const char *fmt, ...) {
    CrashProgram::crash_pid = std::this_thread::get_id();
    
    char buf[50];
    va_list argptr;
    va_start(argptr, fmt);
    vsprintf(buf, fmt, argptr);
    va_end(argptr);
    fprintf(stderr, "%s", buf);
    exit(1);
}

#ifndef WIN32
static void dumpstack(void) {
    void *array[20];
    size_t size;
    size = backtrace(array, 20);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
}

static void signal_handler(int sig) {
    if (sig == SIGHUP) panic("FATAL: Program hanged up\n");
    if (sig == SIGSEGV || sig == SIGBUS){
        dumpstack();
        panic("FATAL: %s Fault. Logged StackTrace\n", (sig == SIGSEGV) ? "Segmentation" : ((sig == SIGBUS) ? "Bus" : "Unknown"));
    }
    if (sig == SIGQUIT) panic("QUIT signal ended program\n");
    if (sig == SIGKILL) panic("KILL signal ended program\n");
    if(sig == SIGINT) {
        if(!SETTING(terminate_error))
            SETTING(terminate_error) = true;
        if(!SETTING(terminate)) {
            SETTING(terminate) = true;
            print("Waiting for video to close.");
        }
            //std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}
#else
BOOL WINAPI consoleHandler(DWORD signal_code) {
    if (signal_code == CTRL_C_EVENT) {
        if (!SETTING(terminate)) {
            SETTING(terminate) = true;
            print("Waiting for video to close.");
            return TRUE;
        }
        else
            FormatExcept("Pressing CTRL+C twice immediately stops the program in an undefined state.");
    }

    return FALSE;
}
#endif

static void at_exit() {
    //GlobalSettings::get(GUI::setting_keys.at("terminate")).value<bool>() = true;
    
    if(FrameGrabber::instance) {
        printf("Didn't clean up FrameGrabber properly.\n");
        printf("Waiting for analysis to be paused...");
        
        if(FrameGrabber::instance && FrameGrabber::instance->processed().open()) {
            printf("Trying to close file...\n");
            FrameGrabber::instance->processed().stop_writing();
        }
        
        if(FrameGrabber::instance)
            FrameGrabber::instance->safely_close();
        
        printf("Closed.\n");
    }
    
#ifndef WIN32
    sigemptyset(&sigact.sa_mask);
#endif
}

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
#else
    if (!SetConsoleCtrlHandler(consoleHandler, TRUE)) {
        printf("\nERROR: Could not set control handler");
    }
#endif
}


std::string date_time() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];
    
    time (&rawtime);
    timeinfo = localtime(&rawtime);
    
    strftime(buffer,sizeof(buffer),"%d-%m-%Y %H:%M:%S",timeinfo);
    std::string str(buffer);
    return str;
}

#include <misc/TestCamera.h>
#include <file/CSVExport.h>
#include <processing/CPULabeling.h>
#include <processing/RawProcessing.h>
#include <misc/ocl.h>

using namespace file;

std::queue<std::function<void()>> _exec_main_queue;
std::mutex _mutex;

FILE *log_file = NULL;
std::mutex log_mutex;

template<class F, class... Args>
auto exec_main_queue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>>
{
    std::lock_guard<std::mutex> guard(_mutex);
    using return_type = typename std::invoke_result_t<F, Args...>;
    auto task = std::make_shared< std::packaged_task<return_type()> >(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    
    auto future = task->get_future();
    _exec_main_queue.push([task](){ (*task)(); });
    //_exec_main_queue.push(std::bind([](F& fn){ fn(); }, std::move(fn)));
    return future;
}

int main(int argc, char** argv)
{
#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
#if __linux__
    XInitThreads();
#endif
    
#ifdef __APPLE__
//char env[] = "OPENCV_OPENCL_DEVICE=:GPU:1";
//    putenv(env);
    
    std::string PATH = (std::string)getenv("PATH");
    if(!utils::contains(PATH, "/usr/local/bin")) {
        PATH += ":/usr/local/bin";
        setenv("PATH", PATH.c_str(), 1);
    }
#endif
    
    pv::DataLocation::register_path("input", [](file::Path filename) -> file::Path {
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
    
    pv::DataLocation::register_path("settings", [](file::Path path) -> file::Path {
        auto settings_file = path.str().empty() ? SETTING(settings_file).value<Path>() : path;
        if(settings_file.empty()) {
            print("The parameter settings_file (or -s) is empty. You can specify a settings file in the command-line by adding:\n\t-s 'path/to/file.settings'");
            return settings_file;
        }
		
        if(!settings_file.is_absolute()) {
            settings_file = SETTING(output_dir).value<file::Path>() / settings_file;
        }
        
        if(!settings_file.has_extension() || settings_file.extension() != "settings")
            settings_file = settings_file.add_extension("settings");
        
        return settings_file;
    });
    
    pv::DataLocation::register_path("output", [](file::Path filename) -> file::Path {
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
    
    pv::DataLocation::register_path("output_settings", [](file::Path) -> file::Path {
        file::Path settings_file(SETTING(filename).value<Path>().filename());
        if(settings_file.empty())
            throw U_EXCEPTION("settings_file (and like filename) is an empty string.");
        
        if(!settings_file.has_extension() || settings_file.extension() != "settings")
            settings_file = settings_file.add_extension("settings");
        
        return pv::DataLocation::parse("output", settings_file);
    });
    
    GlobalSettings::map().set_do_print(true);
    
    /*auto debug_callback = DEBUG::SetDebugCallback({
        DEBUG::DEBUG_TYPE::TYPE_ERROR,
        DEBUG::DEBUG_TYPE::TYPE_EXCEPTION,
        DEBUG::DEBUG_TYPE::TYPE_WARNING,
        DEBUG::DEBUG_TYPE::TYPE_INFO
    }, [](auto, const std::string& msg) {
        std::lock_guard<std::mutex> guard(log_mutex);
        if(log_file) {
            char nl = '\n';
            fwrite(msg.c_str(), 1, msg.length(), log_file);
            fwrite(&nl, 1, 1, log_file);
            fflush(log_file);
        }
    });
    */
    
    //!TODO: Error log_file not implemented
    gui::init_errorlog();
    ocl::init_ocl();
    
    using namespace grab;
    
#if CV_MAJOR_VERSION >= 3
#ifdef USE_GPU_MAT
#if defined(VIDEOS_USE_CUDA)
    //    void* hHandleDriver = 0;
    CUresult res; //= cuInit(0, __CUDA_API_VERSION, hHandleDriver);
    /*if (res != CUDA_SUCCESS) {
     throw std::exception();
     }*/
    res = cuvidInit(0);
    if (res != CUDA_SUCCESS) {
        throw std::exception();
    }
    
    print("Initialized CuVid.");
    
#endif
#endif
#endif
    
    progname = *(argv);
    std::atexit(at_exit);
    
#ifndef WIN32
    setenv("KMP_DUPLICATE_LIB_OK", "True", 1);
#endif
    
    init_signals();
    print("Starting Application...");
    
    srand (time(NULL));
        
    try {
        DebugHeader("LOADING DEFAULT SETTINGS");
        ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
        SETTING(recognition_enable) = false;
        
        grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
        grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
        
#if WITH_FFMPEG
#ifdef WIN32
        LPSTR lpFilePart;
        char filename[MAX_PATH];

        if(!SearchPath( NULL, "ffmpeg", ".exe", MAX_PATH, filename, &lpFilePart))
        {
            auto conda_prefix = ::default_config::conda_environment_path().str();
            print("Conda prefix: ", conda_prefix);
            if(!conda_prefix.empty()) {
                auto files = file::Path(conda_prefix+"/bin").find_files();
                for(auto file : files) {
                    if(file.filename() == "ffmpeg") {
                        print("Found ffmpeg in ",file.str());
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
                            print("Found ffmpeg in ",file.str());
                            SETTING(ffmpeg_path) = file;
                            break;
                        }
                    }
                }
                
                if(!SETTING(ffmpeg_path).value<file::Path>().empty())
                    break;
            }
        }
#endif
#endif
        
        // switch working directory
        DebugHeader("LOADING COMMANDLINE");
        CommandLine cmd(argc, argv, true, grab::default_config::deprecations());
        cmd.cd_home();
#if __APPLE__
        std::string _wd = "../Resources/";
    #if defined(WIN32)
        if (SetCurrentDirectoryA(_wd.c_str()))
    #else
        if (!chdir(_wd.c_str()))
    #endif
            print("Changed directory to ", _wd,".");
        else
            FormatError("Cannot change directory to ",_wd,".");
        
#elif defined(TREX_CONDA_PACKAGE_INSTALL)
        auto conda_prefix = ::default_config::conda_environment_path().str();
        if(!conda_prefix.empty()) {
            file::Path _wd(conda_prefix);
            _wd = _wd / "usr" / "share" / "trex";
            
#if defined(WIN32)
            if (!SetCurrentDirectoryA(_wd.c_str()))
#else
            if (chdir(_wd.c_str()))
#endif
                FormatExcept("Cannot change directory to ",_wd.str(),"");
        }
#endif
        
        for(auto &option : cmd.settings()) {
            if(utils::lowercase(option.name) == "output_prefix") {
                SETTING(output_prefix) = option.value;
            }
        }
        
        if(Path("default.settings").exists() && Path("default.settings").is_regular()) {
            DebugHeader("LOADING FROM 'default.settings'");
            GlobalSettings::load_from_file({}, "default.settings", AccessLevelType::STARTUP);
            DebugHeader("LOADED 'default.settings'");
        }
        
        for(auto &option : cmd) {
            if(Arguments::has(option.name)) {
                switch (Arguments::get(option.name)) {
                    case Arguments::nowindow:
                        SETTING(nowindow) = true;
                        break;
                        
                    case Arguments::i:
                    case Arguments::input: {
                        SETTING(video_source) = option.value;
                        break;
                    }
                        
                    case Arguments::o:
                    case Arguments::output: {
                        SETTING(filename) = Path(option.value);
                        break;
                    }
                        
                    case Arguments::p: {
                        SETTING(output_prefix) = std::string(option.value);
                        break;
                    }
                        
                    case Arguments::s:
                    case Arguments::settings:
                        SETTING(settings_file) = Path(option.value).add_extension("settings");
                        break;
                        
                    case Arguments::d:
                    case Arguments::dir:
                        SETTING(output_dir) = Path(option.value);
                        break;
                        
                    case Arguments::mask: {
                        Path dir(option.value);
                        SETTING(mask_path) = dir;
                        break;
                    }
                        
                    case Arguments::help:
                    case Arguments::h: {
                        if(option.value == "rst") {
                            cmd.load_settings();
                            
                            std::stringstream ss;
                            sprite::Map tracking_settings;
                            GlobalSettings::docs_map_t tracking_docs;
                            ::default_config::get(tracking_settings, tracking_docs, nullptr);
                            
                            for(auto &key : tracking_settings.keys()) {
                                if(key != "nowindow")
                                    ss << key << ";";
                            }
                            
                            auto rst = cmn::settings::help_restructured_text("TGrabs parameters", GlobalSettings::defaults(), GlobalSettings::docs(), GlobalSettings::access_levels(), "", ss.str(), ".. include:: names.rst\n\n.. NOTE::\n\t|grabs| has a live-tracking feature, allowing users to extract positions and postures of individuals while recording/converting. For this process, all parameters relevant for tracking are available in |grabs| as well -- for a reference of those, please refer to :doc:`parameters_trex`.\n");
                            
                            file::Path path = pv::DataLocation::parse("output", "parameters_tgrabs.rst");
                            auto f = path.fopen("wb");
                            if(!f)
                                throw U_EXCEPTION("Cannot open ",path.str());
                            fwrite(rst.data(), sizeof(char), rst.length(), f);
                            fclose(f);
                            
                            //printf("%s\n", rst.c_str());
                            print("Saved at ", path.str(),".");
                            
                            exit(0);
                        }
                        else if(option.value == "settings") {
                            cmd.load_settings();
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
#endif
                            }
                            
                        } else {
                            printf("\n");
                            DebugHeader("AVAILABLE OPTIONS");
                            print("-i <filename>       Input video source (basler/webcam/video path)");
                            print("-d <folder>         Set the default input/output folder");
                            print("-s <filename>       Set the .settings file to be used (default is <input>.settings)");
                            print("-o                  Output filename");
                            print("-h                  Prints this message");
                            print("-h settings         Displays all default settings with description in the default browser");
                            exit(0);
                        }
                        
                        exit(0);
                        break;
                    }
                        
                    default:
                        FormatWarning("Unknown option ", option.name," with value ",option.value);
                        break;
                }
            }
        }
        
        Path settings_file = pv::DataLocation::parse("settings");
        if(!settings_file.empty()) {
            if (settings_file.exists() && settings_file.is_regular()) {
                DebugHeader("LOADING FROM ", settings_file);
                std::map<std::string, std::string> deprecations {{"fish_minmax_size","blob_size_range"}};
                auto rejections = GlobalSettings::load_from_file(deprecations, settings_file.str(), AccessLevelType::STARTUP);
                for(auto && [key, val] : rejections) {
                    if(deprecations.find(key) != deprecations.end())
                        throw U_EXCEPTION("Parameter ",key," is deprecated. Please use ",deprecations.at(key),".");
                }
                DebugHeader("/LOADED ", settings_file);
            }
            else
                FormatError("Cannot find settings file ",settings_file,".");
        }
        
        /**
         * Try to load Settings from the command-line that have been
         * ignored previously.
         */
        cmd.load_settings();

        if (SETTING(tags_recognize) && !SETTING(enable_live_tracking)) {
            // need live tracking to track tags
            SETTING(enable_live_tracking) = true;
        }

        if (SETTING(enable_closed_loop) && !SETTING(enable_live_tracking)) {
            FormatWarning("Forcing enable_live_tracking = true because closed loop has been enabled.");
            SETTING(enable_live_tracking) = true;
        }

        SETTING(meta_source_path) = Path(SETTING(video_source).value<std::string>());
        std::vector<file::Path> filenames;
        
        // recognize keywords in video_source
        if(SETTING(video_source).value<std::string>() != "basler"
           && SETTING(video_source).value<std::string>() != "test_image"
           && SETTING(video_source).value<std::string>() != "webcam"
           && SETTING(video_source).value<std::string>() != "interactive")
        {
            auto video_source = SETTING(video_source).value<std::string>();
            try {
                filenames = Meta::fromStr<std::vector<file::Path>>(video_source);
                if(filenames.size() > 1) {
                    print("Found an array of filenames.");
                } else if(filenames.size() == 1) {
                    SETTING(video_source) = filenames.front();
                    filenames.clear();
                } else
                    throw U_EXCEPTION("Empty input filename ",video_source,". Please specify an input name.");
                
            } catch(const illegal_syntax& e) {
                // ... do nothing
            }
            
            /*if(filenames.empty()) {
                auto filepath = file::Path(SETTING(video_source).value<std::string>());
                if(filepath.remove_filename().empty()) {
                    std::string path = (SETTING(output_dir).value<Path>() / filepath).str();
                    SETTING(video_source) = path;
                } else
                    SETTING(video_source) = filepath.str();
            }*/
            
            if(SETTING(filename).value<file::Path>().empty()) {
                auto filename = file::Path(SETTING(video_source).value<std::string>());
                if(filename.has_extension())
                    filename = filename.remove_extension();
                
                if(utils::contains((std::string)filename.filename(), '%')) {
                    filename = filename.remove_filename();
                }
                
                SETTING(filename) = file::Path(file::Path(filename).filename());
                if(SETTING(filename).value<file::Path>().empty()) {
                    SETTING(filename) = file::Path("video");
                    FormatWarning("No output filename given. Defaulting to 'video'.");
                } else
                    print("Given empty filename, the program will default to using input basename ",SETTING(filename).value<file::Path>().str(),".");
            }
            
        } else if(SETTING(filename).value<file::Path>().empty()) {
            SETTING(filename) = file::Path("video");
            FormatWarning("No output filename given. Defaulting to 'video'.");
        }
        
        if(!SETTING(exec).value<file::Path>().empty()) {
            Path exec_settings = pv::DataLocation::parse("settings", SETTING(exec).value<file::Path>());
            if (exec_settings.exists() && exec_settings.is_regular()) {
                DebugHeader("LOADING FROM ", exec_settings);
                std::map<std::string, std::string> deprecations {{"fish_minmax_size","blob_size_range"}};
                auto rejections = GlobalSettings::load_from_file(deprecations, exec_settings.str(), AccessLevelType::STARTUP);
                for(auto && [key, val] : rejections) {
                    if(deprecations.find(key) != deprecations.end())
                        throw U_EXCEPTION("Parameter ",key," is deprecated. Please use ",deprecations.at(key),".");
                }
                DebugHeader("/LOADED ", exec_settings);
            }
            else
                FormatError("Cannot find settings file ",exec_settings.str(),".");
            
            SETTING(exec) = file::Path();
        }
        
        std::stringstream ss;
        for(int i=0; i<argc; ++i) {
            ss << " " << argv[i];
        }
        SETTING(meta_cmd) = ss.str();
#if WITH_GITSHA1
        SETTING(meta_build) = std::string(g_GIT_SHA1);
#else
        SETTING(meta_build) = std::string("<undefined>");
#endif
        SETTING(meta_conversion_time) = std::string(date_time());
        
        if(!SETTING(log_file).value<file::Path>().empty()) {
            auto path = pv::DataLocation::parse("output", SETTING(log_file).value<file::Path>());
            
            log_mutex.lock();
            log_file = fopen(path.str().c_str(), "wb");
            log_mutex.unlock();
            
            print("Logging to ", path.str(),".");
        }
        
        if(SETTING(manual_identities).value<std::set<track::Idx_t>>().empty() && SETTING(track_max_individuals).value<uint32_t>() != 0)
        {
            std::set<track::Idx_t> vector;
            for(uint32_t i=0; i<SETTING(track_max_individuals).value<uint32_t>(); ++i) {
                vector.insert(track::Idx_t(i));
            }
            SETTING(manual_identities) = vector;
        }

        std::shared_ptr<gui::IMGUIBase> imgui_base;
        
#if WITH_PYLON
        print("Starting with Basler Pylon support.");
        
        // Before using any pylon methods, the pylon runtime must be initialized.
        Pylon::PylonAutoInitTerm term;
        
        try {
#endif
        
        FrameGrabber grabber([&imgui_base](FrameGrabber& grabber){
            exec_main_queue([&](){}).get();
        });
            
        if (SETTING(crop_window) && grabber.video() && (!GlobalSettings::map().has("nowindow") || SETTING(nowindow).value<bool>() == false)) {
#if CMN_WITH_IMGUI_INSTALLED
            gui::CropWindow cropwindow(grabber);
#endif
        }
            
        auto gui = std::make_unique<GUI>(grabber);
#if WITH_MHD
        Httpd httpd([&](Httpd::Session*, const std::string& url){
            cv::Mat image;
            
            if(url == "/background.jpg") {
                image = grabber.average();
                
            } else if(utils::beginsWith(url, "/info")) {
                std::string str = gui.info_text();
                
                std::vector<uchar> buffer(str.begin(), str.end());
                return Httpd::Response(buffer, "text/html");
                
            } else if(utils::beginsWith(url, "/gui")) {
                return gui.render_html();
            
            } else if(url == "/do_terminate") {
                SETTING(terminate) = true;
            }
            
            if(image.empty()) {
                std::string str = "page not found";
                std::vector<uchar> buffer(str.begin(), str.end());
                return Httpd::Response(buffer, "text/html");
            }
            
            std::vector<uchar> buffer;
            cv::imencode(".jpg", image, buffer, { cv::IMWRITE_JPEG_QUALITY, SETTING(web_quality).value<int>() });
            
            return Httpd::Response(buffer);
        }, "grabber.html");
#endif
        while (!gui->terminated())
        {
            tf::show();
            
            {
                std::unique_lock guard(_mutex);
                while(!_exec_main_queue.empty()) {
                    auto f = std::move(_exec_main_queue.front());
                    _exec_main_queue.pop();
                    
                    guard.unlock();
                    f();
                    guard.lock();
                }
            }
            
            
            if(!SETTING(nowindow) && !imgui_base && grabber.task()._complete) {
                imgui_base = std::make_shared<gui::IMGUIBase>(SETTING(app_name).value<std::string>()+" ("+utils::split(SETTING(filename).value<file::Path>().str(),'/').back()+")", gui->gui(), [&](){
                    //std::lock_guard<std::recursive_mutex> lock(gui.gui().lock());
                    if(SETTING(terminate))
                        return false;
                    
                    return true;
                }, GUI::static_event);
                    
                gui->set_base(imgui_base.get());
                imgui_base->platform()->set_icons({
                    "gfx/"+SETTING(app_name).value<std::string>()+"Icon16.png",
                    "gfx/"+SETTING(app_name).value<std::string>()+"Icon32.png",
                    "gfx/"+SETTING(app_name).value<std::string>()+"Icon64.png"
                });
                
            } else if(imgui_base) {
                auto status = imgui_base->update_loop();
                if(status == gui::LoopStatus::END)
                    SETTING(terminate) = true;
                gui->update_loop();
            } else {
                std::chrono::milliseconds ms(75);
                std::this_thread::sleep_for(ms);
            }
        }

        gui = nullptr;
        imgui_base = nullptr;
        print("Ending the program.");
        
    } catch(const UtilsException& e) {
#ifdef WIN32
        if (!SETTING(nowindow)) {
            auto str = std::string("An error occurred, forcing TGrabs to quit:\n") + e.what();
            MessageBox(
                NULL,
                str.c_str(),
                "Error",
                MB_ICONERROR | MB_OK
            );
        }
#endif
        FormatExcept("Utils exception: ", e.what());
        exit(1);
    }
        
#if WITH_PYLON
    }
    catch (const Pylon::GenericException &e)
    {
        throw U_EXCEPTION("An exception occured: ",e.GetDescription());
        return 1;
    }
#endif
    
    //DEBUG::UnsetDebugCallback(debug_callback);
    gui::deinit_errorlog();
    
    if(log_file)
        fclose(log_file);
    log_file = NULL;

    int returncode = SETTING(terminate_error) ? 1 : 0;
    exit(returncode);
}
