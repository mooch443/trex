#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/IMGUIBase.h>
#include <gui/SFLoop.h>
#include <gui/types/Button.h>
#include <video/VideoSource.h>
#include <opencv2/dnn.hpp>
#include <pv.h>
#include <python/GPURecognition.h>
#include <misc/PythonWrapper.h>
#include <misc/CommandLine.h>
#include <file/DataLocation.h>
#include <misc/default_config.h>
#include <tracking/Tracker.h>
#include <tracking/IndividualManager.h>
#include <misc/PixelTree.h>
#include <gui/GUICache.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Textfield.h>
#include <gui/types/List.h>
#include <grabber/misc/default_config.h>
#include <gui/DynamicGUI.h>
#include <gui/SettingsDropdown.h>
#include <GitSHA1.h>
#include <grabber/misc/Webcam.h>
#include <opencv2/core/utils/logger.hpp>

#include <misc/TaskPipeline.h>
#include <gui/Scene.h>

#include <misc/AbstractVideoSource.h>
#include <misc/VideoVideoSource.h>
#include <misc/WebcamVideoSource.h>

#include <gui/LoadingScene.h>
#include <gui/ConvertScene.h>
#include <gui/StartingScene.h>
#include <gui/SettingsScene.h>
#include <gui/TrackingSettingsScene.h>
#include <gui/TrackingScene.h>
#include <gui/AnnotationScene.h>
#include <gui/TrackingState.h>
#include <gui/WorkProgress.h>
#include <tracking/Segmenter.h>
#include <tracking/OutputLibrary.h>

#include <python/Yolo8.h>
//#include <python/Yolo7InstanceSegmentation.h>
//#include <python/Yolo7ObjectDetection.h>

#include <file/PathArray.h>
#include <misc/SettingsInitializer.h>

#include <signal.h>
#include <misc/default_settings.h>

#if !COMMONS_NO_PYTHON
#include <gui/CheckUpdates.h>
#endif

using namespace cmn;
using namespace gui;
static_assert(ObjectDetection<Yolo8>);
static_assert(ObjectDetection<BackgroundSubtraction>);

namespace ind = indicators;
using namespace default_config;

void save_rst_files() {
    auto rst = cmn::settings::help_restructured_text("TRex parameters", GlobalSettings::defaults(), GlobalSettings::docs(), GlobalSettings::access_levels());
    file::Path path = file::DataLocation::parse("output", "parameters_trex.rst");
    auto f = path.fopen("wb");
    if(!f)
        throw U_EXCEPTION("Cannot open ",path.str());
    fwrite(rst.data(), sizeof(char), rst.length(), f.get());
    
    //printf("%s\n", rst.c_str());
    print("Saved at ",path,".");
}

TRexTask determineTaskType() {
    auto output_file = settings::find_output_name(GlobalSettings::map());
    print("output_name = ", output_file);
    
    if (auto array = SETTING(source).value<file::PathArray>();
        array.empty())
    {
        return TRexTask_t::none;
        
    } else if (not output_file.empty()
               && ((    output_file.has_extension()
                         && output_file.extension() == "pv"
                         && output_file.exists())
                   || output_file.add_extension("pv").exists()))
    {
        SETTING(filename) = file::Path(output_file);
        return TRexTask_t::track;
        
    } else if(output_file.empty()) {
        auto front = file::Path(file::find_basename(array));
        /*output_file = !front.has_extension() ?
                      file::DataLocation::parse("input", front.add_extension("pv")) :
                      file::DataLocation::parse("input", front.replace_extension("pv"));*/

        output_file = (not front.has_extension() || front.extension() != "pv") ?
                      file::DataLocation::parse("output", front.add_extension("pv")) :
                      file::DataLocation::parse("output", front.replace_extension("pv"));
        
        if (output_file.exists()) {
            if(array.size() == 1
               && array.get_paths().front() == "webcam")
            {
                SETTING(filename) = file::Path("webcam");
            } else {
                SETTING(filename) = file::Path(output_file);
            }
            
            return TRexTask_t::track;
        } else {
            return TRexTask_t::convert;
        }
    }
    
    return TRexTask_t::convert;
}

void launch_gui(std::future<void>& f) {
    IMGUIBase base(window_title(), {1024,850}, [&, ptr = &base](DrawStructure&)->bool {
        UNUSED(ptr);
        //graph.draw_log_messages(Bounds(Vec2(0, 80), graph.dialog_window_size()));
        return true;
    }, [ptr = &base](auto&, Event e) {
        if(not SceneManager::getInstance().on_global_event(e)) {
            if(e.type == EventType::KEY) {
                if(e.key.code == Keyboard::Escape) {
                    SETTING(terminate) = true;
                }
                
            } else if(e.type == EventType::WINDOW_RESIZED) {
                auto work_area = ptr->work_area();
                auto scale = 1920.f / work_area.width;
                if(scale != 1.f)
                    scale = 1.f + (scale - 1.f) * 0.35;
                //print("scale = ", 1920.f / work_area.width, " (",scale,") dpi = ", ptr->dpi_scale());
                SETTING(gui_interface_scale) = float(scale);
            }
        }
    });
    
#if !COMMONS_NO_PYTHON
    CheckUpdates::init(base.graph().get());
#endif
    
    /**
     * Get the SceneManager instance and register all scenes
     */
    auto& manager = SceneManager::getInstance();
    WorkProgress::instance().start();
    
    static std::unique_ptr<Segmenter> segmenter;
    ConvertScene converting(base, [&](ConvertScene& scene){
        segmenter = std::make_unique<Segmenter>(
        [&manager]() {
            if (SETTING(auto_quit)) {
                if (not SETTING(terminate))
                    SETTING(terminate) = true;
            }
            else {
                GlobalSettings::map().set_print_by_default(true);
                thread_print("Segmenter terminating and switching to tracking scene: ", segmenter->output_file_name());
                if(SETTING(gui_frame).value<Frame_t>().valid())
                    SETTING(gui_frame) = Frame_t(SETTING(gui_frame)).try_sub(10_f);
				manager.set_active("tracking-scene");
			}
        },
        [&manager](std::string error) {
            if(SETTING(nowindow))
                throw U_EXCEPTION("Error converting: ", error);
            
            manager.set_switching_error(error);
            if(manager.last_active())
                manager.set_active(manager.last_active());
            else manager.set_active("starting-scene");
        });

        if(f.valid())
            f.get();
        scene.set_segmenter(segmenter.get());
        
        // on activate
        if(not segmenter->output_size().empty())
            base.graph()->set_size(Size2(1024, segmenter->output_size().height / segmenter->output_size().width * 1024));
        
    }, [](auto&){
        // on deactivate
        segmenter = nullptr;
    });
    manager.register_scene(&converting);

    StartingScene start{ base };
    manager.register_scene(&start);
    manager.set_fallback(start.name());
    
    TrackingScene tracking_scene{ base };
    manager.register_scene(&tracking_scene);
    
    SettingsScene settings_scene{ base };
    manager.register_scene(&settings_scene);
    TrackingSettingsScene tsettings_scene{ base };
    manager.register_scene(&tsettings_scene);
    
    AnnotationScene annotations{base};
    manager.register_scene(&annotations);

    LoadingScene loading(base, file::DataLocation::parse("output"), ".pv", [](const file::Path&, std::string) {
        }, [](const file::Path&, std::string) {

        });
    manager.register_scene(&loading);
    
    using namespace default_config;
    std::unordered_map<TRexTask, Scene*> task_scenes {
        { TRexTask_t::none, &start },
		{ TRexTask_t::convert, &converting },
		{ TRexTask_t::track, &tracking_scene },
        { TRexTask_t::annotate, &annotations },
        { TRexTask_t::rst, &start }
	};

    if (const auto task = SETTING(task).value<TRexTask>();
        task == TRexTask_t::none)
    {
        TRexTask taskType = determineTaskType();
        settings::load(SETTING(source).value<file::PathArray>(),
                       SETTING(filename).value<file::Path>(),
                       taskType,
                       SETTING(detect_type),
                       {}, {});
        manager.set_active(task_scenes[taskType]);
        
    } else if(task == TRexTask_t::rst) {
        save_rst_files();
        manager.set_active(&start);
        
    } else {
        if (auto it = task_scenes.find(task);
            it != task_scenes.end())
        {
            if(it->second == &converting) {
                //SETTING(cm_per_pixel) = float(0.01);
                
                settings::load(SETTING(source).value<file::PathArray>(),
                               SETTING(filename).value<file::Path>(),
                               TRexTask_t::convert,
                               SETTING(detect_type),
                               {}, {});
            } else if(it->second == &tracking_scene) {
                settings::load(SETTING(source).value<file::PathArray>(),
                               SETTING(filename).value<file::Path>(),
                               TRexTask_t::track,
                               SETTING(detect_type),
                               {}, {});
                
            } else
                settings::load({}, {}, 
                               TRexTask_t::none,
                               SETTING(detect_type),
                               {}, {});
            
            manager.set_active(it->second);
        }
        else {
            settings::load({}, {}, 
                           TRexTask_t::none,
                           SETTING(detect_type),
                           {}, {});
            manager.set_active(&start);
        }
	}
    
    base.platform()->set_icons({
        //file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_16.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_32.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_48.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_64.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_128.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_256.png")
    });
    
    file::cd(file::DataLocation::parse("app"));
    
    gui::SFLoop loop(*base.graph(), &base, [&](gui::SFLoop&, LoopStatus) {
        if(f.valid()
           && f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            f.get();
        }
        manager.update(&base, *base.graph());
    });
    
    manager.clear();
    
#if !COMMONS_NO_PYTHON
    CheckUpdates::cleanup();
#endif
    
    base.graph()->root().set_stage(nullptr);
}

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

#if !defined(WIN32) && !defined(__EMSCRIPTEN__)
/*static void dumpstack(void) {
    void *array[20];
    auto size = backtrace(array, 20);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
}*/

struct sigaction sigact;

static void signal_handler(int sig) {
    if (sig == SIGHUP) panic("FATAL: Program hanged up\n");
    if (sig == SIGSEGV || sig == SIGBUS){
        //dumpstack();
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
#elif defined(WIN32)
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
#elif defined(WIN32)
    SetConsoleCtrlHandler(consoleHandler, TRUE);
#endif
}

std::string start_tracking(std::future<void>& f) {
    if(SETTING(filename).value<file::Path>().empty())
        SETTING(filename) = file::Path(settings::find_output_name(GlobalSettings::map()));
    
    std::atomic<bool> terminate{false};
    TrackingState state{nullptr};
    state._tracking_callbacks.push([&](){
        terminate = true;
    });
    state.init_video();
    
    RecentItems::open(SETTING(source).value<file::PathArray>().source(), GlobalSettings::current_defaults_with_config());
    
    //! get the python init future at this point
    if(f.valid())
        f.get();
    
    while(not terminate)
        std::this_thread::sleep_for(std::chrono::seconds(1));
    return {};
}

std::string start_converting(std::future<void>& f) {
    if(SETTING(filename).value<file::Path>().empty()) {
        SETTING(filename) = file::Path(settings::find_output_name(GlobalSettings::map()));
    }
    
    std::string last_error;
    Segmenter segmenter(
        [&f]() {
            //if(SETTING(auto_quit).value<bool>())
                SETTING(terminate) = true;
            //else
            //    start_tracking(f);
            //    throw InvalidArgumentException("What should I do now?");
        },
        [&last_error](std::string error) {
            SETTING(error_terminate) = true;
            SETTING(terminate) = true;
            last_error = error;
        });
    print("Loading source = ", SETTING(source).value<file::PathArray>());
    
    ind::ProgressBar bar{
        ind::option::BarWidth{50},
        ind::option::Start{"["},
/*#ifndef _WIN32
        ind::option::Fill{"█"},
        ind::option::Lead{"▂"},
        ind::option::Remainder{"▁"},
#else*/
        ind::option::Fill{"="},
        ind::option::Lead{">"},
        ind::option::Remainder{" "},
//#endif
        ind::option::End{"]"},
        ind::option::PostfixText{"Converting video..."},
        ind::option::ShowPercentage{true},
        ind::option::ForegroundColor{ind::Color::white},
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
    };
    
    bar.set_progress(0);
    
    ind::ProgressSpinner spinner{
        ind::option::PostfixText{"Recording..."},
        ind::option::ForegroundColor{ind::Color::white},
        ind::option::SpinnerStates{std::vector<std::string>{
            //"⣾","⣽","⣻","⢿","⡿","⣟","⣯","⣷"
            //"◢","◣","◤","◥",
            //"◜◞", "◟◝", "◜◞", "◟◝"
            " ◴"," ◷"," ◶"," ◵"
            //"⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂", "⠁"
        }},
        ind::option::FontStyles{std::vector<ind::FontStyle>{ind::FontStyle::bold}}
    };
    
    Timer last_tick;
    segmenter.set_progress_callback([&](float percent){
        if(std::isnan(percent)
           || std::isinf(percent))
        {
            spinner.tick();
            static std::once_flag flag;
            std::call_once(flag, [](){
                FormatWarning("Percent is infinity.");
            });
            return;
        }
        
        if(percent >= 0)
            bar.set_progress(saturate(percent, 0.f, 100.f));
        else if(last_tick.elapsed() > 1) {
            spinner.set_option(ind::option::PostfixText{"Recording ("+Meta::toStr(Tracker::end_frame())+")..."});
            spinner.set_option(ind::option::ShowPercentage{false});
            spinner.tick();
            last_tick.reset();
        }
    });
    
    if (SETTING(source).value<file::PathArray>() == file::PathArray("webcam"))
        segmenter.open_camera();
    else
        segmenter.open_video();
    
    auto finite = segmenter.is_finite();
    segmenter.start();
    
    //! get the python init future at this point
    if(f.valid())
        f.get();
    
    while(not SETTING(terminate))
        std::this_thread::sleep_for(std::chrono::seconds(1));
    
    if(not SETTING(error_terminate)) {
        spinner.set_option(ind::option::ForegroundColor{ind::Color::green});
        spinner.set_option(ind::option::PrefixText{"✔"});
        spinner.set_option(ind::option::ShowSpinner{false});
        spinner.set_option(ind::option::PostfixText{"Done."});
    } else {
        spinner.set_option(ind::option::ForegroundColor{ind::Color::red});
        spinner.set_option(ind::option::PrefixText{"X"});
        spinner.set_option(ind::option::ShowSpinner{false});
        spinner.set_option(ind::option::PostfixText{"Failed."});
    }
    
    if(finite) {
        bar.set_progress(100);
        bar.mark_as_completed();
    } else
        spinner.mark_as_completed();
    
    return last_error;
}

int main(int argc, char**argv) {
#ifdef NDEBUG
    cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_ERROR);
#endif
    
    const char* locale = "C";
    std::locale::global(std::locale(locale));
    
#ifndef WIN32
    setenv("LC_ALL", "C", 1);
#endif
    
    using namespace gui;
    init_signals();
#ifdef WIN32
    SetConsoleOutputCP( 65001 );
#endif
    default_config::register_default_locations();
    
    grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    ::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    gui::init_errorlog();
    set_thread_name("main");
    
    /**
     * Some settings related to tracking and
     * object detection
     */
    //SETTING(meta_video_scale) = float(1);
    
    using namespace cmn;
    namespace py = Python;
    auto cwd = file::cwd();
    if(cwd.empty())
        cwd = file::Path(default_config::homedir());
    
    print("CWD: ", cwd);
    DebugHeader("LOADING COMMANDLINE");
    GlobalSettings::map()["cwd"].get().set_do_print(true);
    CommandLine::init(argc, argv, true);
    CommandLine::instance().add_setting("cwd", cwd.str());
    SETTING(cwd) = cwd;
    file::cd(file::DataLocation::parse("app").absolute());
    print("CWD: ", file::cwd());
    
    /*GlobalSettings::map().register_callbacks({"source", "meta_source_path", "filename", "detect_type", "cm_per_pixel", "track_background_subtraction", "gui_interface_scale"}, [](auto key){
        if(key == "source")
            print("Changed source to ", SETTING(source).value<file::PathArray>());
        else if(key == "meta_source_path")
            print("Changed meta_source_path to ", SETTING(meta_source_path).value<std::string>());
        else if(key == "filename")
            print("Changed filename to ", SETTING(filename).value<file::Path>());
        else if(key == "detect_type")
            print("Changed detection type to ", SETTING(detect_type));
        else if(key == "cm_per_pixel")
            print("Changerd cm_per_pixel to ", SETTING(cm_per_pixel));
        else if(key == "track_background_subtraction")
            print("Changed track_background_subtraction to ", SETTING(track_background_subtraction));
    });*/
    
    for(auto a : CommandLine::instance()) {
        if(a.name == "s") {
            SETTING(settings_file) = file::Path(a.value).add_extension("settings");
            CommandLine::instance().add_setting("settings_file", a.value);
        }
        else if(a.name == "i") {
            SETTING(source) = file::PathArray(a.value);
            CommandLine::instance().add_setting("source", a.value);
        }
        else if(a.name == "m") {
            SETTING(detect_model) = file::Path(a.value);
            CommandLine::instance().add_setting("detect_model", a.value);
        }
        else if (a.name == "bm") {
            SETTING(region_model) = file::Path(a.value);
            CommandLine::instance().add_setting("region_model", a.value);
        }
        else if(a.name == "d") {
            SETTING(output_dir) = file::Path(a.value);
            CommandLine::instance().add_setting("output_dir", a.value);
        }
        else if(a.name == "dim") {
            SETTING(detect_resolution) = Meta::fromStr<uint16_t>(a.value);
            CommandLine::instance().add_setting("detect_resolution", a.value);
        }
        else if(a.name == "o") {
            auto path = file::Path(a.value);
            if(path.has_extension())
                path = path.remove_extension();
            SETTING(filename) = path;
            CommandLine::instance().add_setting("filename", path.str());
        }
        else if(a.name == "p") {
            SETTING(output_prefix) = std::string(a.value);
            CommandLine::instance().add_setting("output_prefix", a.value);
        }
    }
    
    std::future<void> f;
    try {
        py::init();
        f = py::schedule([](){
            print("Python = ", py::get_instance());
            track::PythonIntegration::set_settings(GlobalSettings::instance(), file::DataLocation::instance(), Python::get_instance());
            track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });
        });
    } catch(const std::exception& e) {
        FormatError("Cannot initialize python. Please refer to the above error messages prefixed with [py] to estimate the cause of this issue: ", e.what());
        exit(1);
    }
    
    using namespace track;
    
    GlobalSettings::map().set_print_by_default(true);
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    SETTING(app_name) = std::string("TRex");
    SETTING(meta_real_width) = 1000.f;
    
    std::stringstream ss;
    for(int i=0; i<argc; ++i) {
        if(i > 0)
            ss << " ";
        if(argv[i][0] == '-')
            ss << argv[i];
        else
            ss << "'" << argv[i] << "'";
    }
    SETTING(meta_cmd) = ss.str();
#if WITH_GITSHA1
    SETTING(meta_build) = std::string(g_GIT_SHA1);
#else
    SETTING(meta_build) = std::string("<undefined>");
#endif

    CommandLine::instance().load_settings();
    
    //if(not SETTING(source).value<file::PathArray>().empty())
    //    SETTING(scene_crash_is_fatal) = true;

    if (not SETTING(filename).value<file::Path>().empty()) {
        auto path = SETTING(filename).value<file::Path>();
        if (path.has_extension() && path.extension() == "pv")
            path = path.remove_extension();
        SETTING(filename) = file::DataLocation::parse("output", path);
    }
    
    std::string last_error;
    if(SETTING(nowindow)) {
        auto task = SETTING(task).value<TRexTask>();
        if(task == TRexTask_t::none)
            task = determineTaskType();
        if(task == TRexTask_t::none)
            throw U_EXCEPTION("Not sure what to do. Please specify a task (-task <name>) or an input file (-i <path>).");
        
        settings::load(SETTING(source).value<file::PathArray>(),
                       SETTING(filename).value<file::Path>(),
                       task,
                       SETTING(detect_type),
                       {}, {});

        Output::Library::InitVariables();
        Output::Library::Init();
        
        /// in terminal we dont want to async a GUI anyway.
        /// also, on windows we might get in trouble here
        /// if GlobalSettings isnt assigned the right instance
        /// yet in python_dll:
        print("Waiting for python...");
        if(f.valid())
            f.get();

        if(task == TRexTask_t::convert) {
            last_error = start_converting(f);
            
        } else if(task == TRexTask_t::track) {
            last_error = start_tracking(f);
        } else if(task == TRexTask_t::rst) {
            save_rst_files();
        } else {
            throw U_EXCEPTION("Unknown task type: ", task);
        }
        
    } else {
        // get the python init future
        launch_gui(f);
    }
    
    try {
        if (f.valid())
            f.get();

        Detection::manager().clean_up();
        Detection::deinit();
        py::deinit();
    } catch(const std::exception& e) {
        FormatExcept("Unknown deinit() error, quitting normally anyways. ", e.what());
    }
    
    if(SETTING(error_terminate)) {
        if(not last_error.empty())
            FormatError(last_error.c_str());
        return 1;
    }
    
    return 0;
}

