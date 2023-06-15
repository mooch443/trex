#include <commons.pc.h>
#include <gui/DrawStructure.h>
#include <gui/IMGUIBase.h>
#include <gui/SFLoop.h>
#include <gui/types/Button.h>
#include <video/VideoSource.h>
#include <opencv2/dnn.hpp>
#include <pv.h>
#include <python/GPURecognition.h>
#include <tracking/PythonWrapper.h>
#include <misc/CommandLine.h>
#include <file/DataLocation.h>
#include <misc/default_config.h>
#include <tracking/Tracker.h>
#include <tracking/IndividualManager.h>
#include <misc/PixelTree.h>
#include <gui/Timeline.h>
#include <gui/GUICache.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Textfield.h>
#include <gui/types/List.h>
#include <grabber/misc/default_config.h>
#include <gui/DynamicGUI.h>
#include <gui/SettingsDropdown.h>
#include "Alterface.h"
#include <GitSHA1.h>
#include <grabber/misc/Webcam.h>

#include <misc/TaskPipeline.h>
#include <Scene.h>

#include <misc/TileImage.h>
#include <misc/AbstractVideoSource.h>
#include <misc/VideoVideoSource.h>
#include <misc/WebcamVideoSource.h>

#include <gui/LoadingScene.h>
#include <gui/ConvertScene.h>

#include <tracking/Yolo8.h>
#include <tracking/Yolo7InstanceSegmentation.h>
#include <tracking/Yolo7ObjectDetection.h>

#include <signal.h>

using namespace cmn;

struct TileImage;

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

using namespace gui;

static_assert(ObjectDetection<Yolo7ObjectDetection>);
static_assert(ObjectDetection<Yolo7InstanceSegmentation>);
static_assert(ObjectDetection<Yolo8>);

namespace ind = indicators;

class StartingScene : public Scene {
    file::Path _image_path = file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_1024.png");

    // The image of the logo
    std::shared_ptr<ExternalImage> _logo_image;
    std::shared_ptr<Entangled> _title = std::make_shared<Entangled>();

    // The list of recent items
    std::shared_ptr<ScrollableList<>> _recent_items;
    std::shared_ptr<VerticalLayout> _buttons_and_items = std::make_shared<VerticalLayout>();
    
    std::shared_ptr<VerticalLayout> _logo_title_layout = std::make_shared<VerticalLayout>();
    std::shared_ptr<HorizontalLayout> _button_layout;
    
    // The two buttons for user interactions, now as Layout::Ptr
    std::shared_ptr<Button> _video_file_button;
    std::shared_ptr<Button> _camera_button;

    // The HorizontalLayout for the two buttons and the image
    HorizontalLayout _main_layout;
    
    dyn::Context context {
        .variables = {
            {
                "global",
                std::unique_ptr<VarBase_t>(new Variable([](std::string) -> sprite::Map& {
                    return GlobalSettings::map();
                }))
            }
        }
    };
    dyn::State state;
    std::vector<Layout::Ptr> objects;

public:
    StartingScene(Base& window)
    : Scene(window, "starting-scene", [this](auto&, DrawStructure& graph){ _draw(graph); }),
      _logo_image(std::make_shared<ExternalImage>(Image::Make(cv::imread(_image_path.str(), cv::IMREAD_UNCHANGED)))),
      _recent_items(std::make_shared<ScrollableList<>>(Bounds(0, 10, 310, 500))),
      _video_file_button(std::make_shared<Button>("Open file", attr::Size(150, 50))),
      _camera_button(std::make_shared<Button>("Camera", attr::Size(150, 50)))
    {
        auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
        print(window.window_dimensions().mul(dpi), " and logo ", _logo_image->size());
        
        // Callback for video file button
        _video_file_button->on_click([](auto){
            // Implement logic to handle the video file
            SceneManager::getInstance().set_active("converting");
        });

        // Callback for camera button
        _camera_button->on_click([](auto){
            // Implement logic to start recording from camera
            SETTING(source).value<std::string>() = "webcam";
            SceneManager::getInstance().set_active("converting");
        });
        
        // Create a new HorizontalLayout for the buttons
        _button_layout = std::make_shared<HorizontalLayout>(std::vector<Layout::Ptr>{
            Layout::Ptr(_video_file_button),
            Layout::Ptr(_camera_button)
        });
        //_button_layout->set_pos(Vec2(1024 - 10, 550));
        //_button_layout->set_origin(Vec2(1, 0));

        _buttons_and_items->set_children({
            Layout::Ptr(_recent_items),
            Layout::Ptr(_button_layout)
        });
        
        _logo_title_layout->set_children({
            Layout::Ptr(_title),
            Layout::Ptr(_logo_image)
        });
        
        // Set the list and button layout to the main layout
        _main_layout.set_children({
            Layout::Ptr(_logo_title_layout),
            Layout::Ptr(_buttons_and_items)
        });
        //_main_layout.set_origin(Vec2(1, 0));
    }

    void activate() override {
        // Fill the recent items list
        _recent_items->set_items({
            TextItem("olivia_momo-carrot_t4t_041822.mp4", 0),
            TextItem("DJI_0268.MOV", 1),
            TextItem("8guppies_20s", 2)
        });
    }

    void deactivate() override {
        // Logic to clear or save state if needed
    }

    void _draw(DrawStructure& graph) {
        dyn::update_layout("welcome_layout.json", context, state, objects);
        
        //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
        auto max_w = window()->window_dimensions().width * 0.65;
        auto max_h = window()->window_dimensions().height - _button_layout->height() - 25;
        auto scale = Vec2(max_w * 0.4 / _logo_image->width());
        _logo_image->set_scale(scale);
        _title->set_size(Size2(max_w, 25));
        _recent_items->set_size(Size2(_recent_items->width(), max_h));
        
        graph.wrap_object(_main_layout);
        
        std::vector<Layout::Ptr> _objs{objects.begin(), objects.end()};
        _objs.insert(_objs.begin(), Layout::Ptr(_title));
        _objs.push_back(Layout::Ptr(_logo_image));
        _logo_title_layout->set_children(_objs);
        _logo_title_layout->set_policy(VerticalLayout::Policy::CENTER);
        
        
        for(auto &obj : objects) {
            dyn::update_objects(graph, obj, context, state);
            //graph.wrap_object(*obj);
        }
        
        _buttons_and_items->auto_size(Margin{0,0});
        _logo_title_layout->auto_size(Margin{0,0});
        _main_layout.auto_size(Margin{0,0});
    }
};

void launch_gui() {
    DrawStructure graph(1024, 768);
    IMGUIBase base(window_title(), graph, [&, ptr = &base]()->bool {
        UNUSED(ptr);
        graph.draw_log_messages();
        
        return true;
    }, [](Event e) {
        if(e.type == EventType::KEY) {
            if(e.key.code == Keyboard::Escape) {
                SETTING(terminate) = true;
            }
        }
    });
    
    auto& manager = SceneManager::getInstance();
    
    StartingScene start(base);
    manager.register_scene(&start);

    if(SETTING(source).value<std::string>() == "")
        manager.set_active(&start);
    
    ConvertScene converting(base);
    manager.register_scene(&converting);
    //manager.set_active(&converting);
    if (SETTING(source).value<std::string>() != "")
        manager.set_active(&converting);

    LoadingScene loading(base, file::DataLocation::parse("output"), ".pv", [](const file::Path&, std::string) {
        }, [](const file::Path&, std::string) {

        });
    manager.register_scene(&loading);

    graph.set_size(Size2(1024, converting.output_size().height / converting.output_size().width * 1024));
    
    base.platform()->set_icons({
        //file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_16.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_32.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_48.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_64.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_128.png"),
        file::DataLocation::parse("app", "gfx/"+SETTING(app_name).value<std::string>()+"_256.png")
    });
    
    file::cd(file::DataLocation::parse("app"));
    
    gui::SFLoop loop(graph, &base, [&](gui::SFLoop&, LoopStatus) {
        manager.update(graph);
        
        if (graph.is_key_pressed(Keyboard::Right)) {
            SETTING(do_filter) = true;
        } else if (graph.is_key_pressed(Keyboard::Left)) {
            SETTING(do_filter) = false;
        }
    });
    
    manager.set_active(nullptr);
    manager.update_queue();
    graph.root().set_stage(nullptr);
    Detection::manager().~PipelineManager<TileImage>();
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

int main(int argc, char**argv) {
    using namespace gui;
    init_signals();
#ifdef WIN32
    SetConsoleOutputCP( 65001 );
#endif
    
    default_config::register_default_locations();
    
    ::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    grab::default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    grab::default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    
    gui::init_errorlog();
    set_thread_name("main");
    
    /**
     * Some settings related to tracking and
     * object detection
     */
    SETTING(meta_video_scale) = float(1);
    SETTING(source) = std::string("");
    SETTING(model) = file::Path("");
    SETTING(segmentation_resolution) = uint16_t(128);
    SETTING(segmentation_model) = file::Path("");
    SETTING(region_model) = file::Path("");
    SETTING(region_resolution) = uint16_t(320);
    SETTING(detection_resolution) = uint16_t(640);
    SETTING(filename) = file::Path("");
    SETTING(meta_classes) = std::vector<std::string>{ };
    SETTING(detection_type) = ObjectDetectionType::yolo8;
    SETTING(tile_image) = size_t(0);
    SETTING(batch_size) = uchar(10);
    
    SETTING(do_filter) = false;
    SETTING(filter_classes) = std::vector<uint8_t>{};
    SETTING(is_writing) = false;
    
    using namespace cmn;
    namespace py = Python;
    print("CWD: ", file::cwd());
    DebugHeader("LOADING COMMANDLINE");
    CommandLine cmd(argc, argv, true);
    file::cd(file::DataLocation::parse("app").absolute());
    print("CWD: ", file::cwd());
    
    for(auto a : cmd) {
        if(a.name == "i") {
            SETTING(source) = std::string(a.value);
        }
        if(a.name == "m") {
            SETTING(model) = file::Path(a.value);
        }
        if(a.name == "sm") {
            SETTING(segmentation_model) = file::Path(a.value);
        }
        if (a.name == "bm") {
            SETTING(region_model) = file::Path(a.value);
        }
        if(a.name == "d") {
            SETTING(output_dir) = file::Path(a.value);
        }
        if(a.name == "dim") {
            SETTING(detection_resolution) = Meta::fromStr<uint16_t>(a.value);
        }
        if(a.name == "o") {
            SETTING(filename) = file::Path(a.value);
        }
    }
    
    py::init();
    py::schedule([](){
        track::PythonIntegration::set_settings(GlobalSettings::instance(), file::DataLocation::instance());
        track::PythonIntegration::set_display_function([](auto& name, auto& mat) { tf::imshow(name, mat); });
    });
    
    
    using namespace track;
    
    GlobalSettings::map().set_do_print(true);
    GlobalSettings::map().dont_print("gui_frame");
    SETTING(app_name) = std::string("TRexA");
    SETTING(threshold) = int(100);
    SETTING(track_do_history_split) = false;
    SETTING(track_max_speed) = Settings::track_max_speed_t(300);
    SETTING(track_threshold) = Settings::track_threshold_t(0);
    SETTING(track_posture_threshold) = Settings::track_posture_threshold_t(0);
    SETTING(blob_size_ranges) = Settings::blob_size_ranges_t({
        Rangef(10,300)
    });
    SETTING(track_speed_decay) = Settings::track_speed_decay_t(1);
    SETTING(track_max_reassign_time) = Settings::track_max_reassign_time_t(1);
    SETTING(terminate) = false;
    SETTING(calculate_posture) = false;
    SETTING(gui_interface_scale) = float(1);
    SETTING(meta_source_path) = SETTING(source).value<std::string>();
    
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
    SETTING(meta_conversion_time) = std::string(date_time());
    SETTING(meta_encoding) = grab::default_config::meta_encoding_t::r3g3b2;

    cmd.load_settings();
    
    launch_gui();
    
    py::deinit();
    return 0;
}

