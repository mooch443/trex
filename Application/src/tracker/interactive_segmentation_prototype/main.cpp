#include <commons.pc.h>

#ifndef WIN32
__attribute__((constructor))
static void early_env_setup() {
    cmn::utf8loc::enable_utf8();
    setenv("KMP_DUPLICATE_LIB_OK", "TRUE", 1);
}
#else
#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU"))
static void (*windowsEarlyEnvSetup)(void) = []() {
    cmn::utf8loc::enable_utf8();
    SetEnvironmentVariable("KMP_DUPLICATE_LIB_OK", "TRUE");
    return;
};
#endif

#include <misc/CommandLine.h>
#include <core/PythonWrapper.h>
#include <tracking/Tracker.h>
#include <tracking/Segmenter.h>
#include <pv.h>
#include <GitSHA1.h>
#include <ui/Scene.h>
#include <gui/SFLoop.h>
#include <gui/DrawStructure.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <misc/GlobalSettings.h>
#include <gui/DynamicGUI.h>
#include <ui/Scene.h>
#include <ui/AnnotationScene.h>
#include <ui/Bowl.h>
#include "LiveSegmentation.h"
#include <python/GPURecognition.h>

using namespace cmn;

int main(int argc, char** argv) {
    GlobalSettings::write([](Configuration& config){
        default_config::get(config);
    });
    
    default_config::register_default_locations();
    
    cmn::CommandLine::init(argc, argv);
    auto& cmd = cmn::CommandLine::instance();
    cmd.cd_home();
    cmd.load_settings();

    SETTING(app_name) = std::string("TRex");
    SETTING(detect_sam3_prompt) = std::string("floor");
    
    Print("interactive_segmentation_prototype",
          "git:", std::string_view(g_GIT_DESCRIBE_TAG),
          "sha:", std::string_view(g_GIT_SHA1),
          "build:", std::string_view(g_TREX_BUILD_TYPE));
    
    namespace py = Python;
    std::future<void> f;
    try {
        py::init().get();
        f = py::schedule([](){
            //Print("Python = ", py::get_instance());
            track::PythonIntegration::set_settings(GlobalSettings::instance(), file::DataLocation::instance(), Python::get_instance());
            track::PythonIntegration::set_display_function([](auto& name, auto& mat) {
                tf::imshow(name, mat);
            },
            []() {
                tf::destroyAllWindows();
            });
        });
    } catch(const std::exception& e) {
        FormatError("Cannot initialize python. Please refer to the above error messages prefixed with [py] to estimate the cause of this issue: ", e.what());
        exit(1);
    }
    
    // Touch core types to ensure the prototype links against the main tracking stack.
    (void)sizeof(track::Tracker);
    (void)sizeof(track::Segmenter);
    (void)pv::V_7;
    
    using namespace gui;
    
    auto& manager = SceneManager::getInstance();
    
    static constexpr auto window_title = "InteractiveSegmentationPrototype";
    IMGUIBase base(window_title, {1024,850}, [&, ptr = &base](DrawStructure& graph)->bool {
        UNUSED(ptr);
        graph.draw_log_messages(Bounds(Vec2(0, 80), graph.dialog_window_size()));
        return true;
    }, [ptr = &base](DrawStructure&g, Event e) {
        if(not SceneManager::getInstance().on_global_event(e)) {
            if(e.type == EventType::KEY) {
                if(e.key.code == Keyboard::Escape) {
                    SETTING(terminate) = true;
                }
                
            } else if(e.type == EventType::WINDOW_RESIZED) {
                auto h = max(10u, g.height());
                auto w = max(10u, g.width());
#if defined(WIN32)
                Float2_t dpi = 1_F;
#else
                Float2_t dpi = ptr->dpi_scale();
#endif
                assert(not std::isinf(dpi));
                auto min_width = 1350_F * dpi;
                auto min_height = 1024_F * dpi;
                
                auto scale = max(0.9_F, sqrt(min_width / w));
                auto yscale = max(0.9_F, sqrt(min_height / h));
                
                //Print("scale=",scale, " yscale=",yscale, " w=",w," h=",h, " dpi=", dpi, " (", ptr->dpi_scale(), ") interface=", gui::interface_scale());
                SETTING(gui_interface_scale) = Float2_t(yscale > scale ? yscale : yscale);
                g.set_scale(gui::interface_scale());
            }
        }
    });
    
    file::cd(file::DataLocation::parse("app"));
    
    SETTING(source) = file::PathArray{"/Users/tristan/Downloads/test_videos/cam1/GX010004_recut.MP4"};
    
    LiveSegmentation live_scene(base);
    AnnotationScene annotation_scene(base);
    manager.register_scene(&annotation_scene);
    manager.register_scene(&live_scene);
    //manager.set_active(&annotation_scene);
    manager.set_active(&live_scene);
    
    gui::SFLoop loop(*base.graph(), &base, [&](gui::SFLoop&, LoopStatus) {
        if(f.valid()
           && f.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
        {
            try {
                f.get();
                f = {};
            } catch(const std::exception& ex) {
                SceneManager::set_switching_error(ex.what());
            }
        }
        
        manager.update(&base, *base.graph());
    });
    
    manager.clear();
    
    if(f.valid()) {
        try {
            f.get();
        } catch(const std::exception& ex) {
            FormatExcept(ex.what());
        }
    }
    
    Python::deinit();
    
    return 0;
}
