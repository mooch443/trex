#include "SettingsScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
//#include <misc/RecentItems.h>
//#include <gui/types/Dropdown.h>
//#include <gui/types/Checkbox.h>
#include <gui/dyn/Action.h>
//#include <video/VideoSource.h>
//#include <misc/AbstractVideoSource.h>
//#include <misc/VideoVideoSource.h>
//#include <misc/WebcamVideoSource.h>
#include <gui/types/SettingsTooltip.h>
#include <gui/DynamicGUI.h>
#include <misc/SettingsInitializer.h>
#include <misc/BlurryVideoLoop.h>
#include <gui/GUIVideoAdapterElement.h>
#include <misc/VideoInfo.h>
#include <gui/GUIVideoAdapter.h>

namespace gui {

struct SettingsScene::Data {
    //BlurryVideoLoop _video_loop;
    SettingsTooltip _settings_tooltip;
    Timer video_image_timer, animation_timer;
    double blur_target{1};
    double blur_percentage{0};
    
    Size2 window_size;
    dyn::DynamicGUI dynGUI;
    CallbackCollection callback;
    
    ExternalImage _preview_image;
    IMGUIBase *_window{nullptr};
    
    std::atomic<Size2> _next_video_size;
    Size2 _video_size;
    std::unordered_map<std::string, Layout::Ptr> _video_adapters;
    
    void draw(DrawStructure& graph) {
        using namespace dyn;
        
        if(not dynGUI) {
            dynGUI = DynamicGUI{
                .path = "settings_layout.json",
                .graph = &graph,
                .context = {
                    ActionFunc("set", [](Action action) {
                        if(action.parameters.size() != 2)
                            throw InvalidArgumentException("Invalid number of arguments for action: ",action);
                        
                        auto parm = Meta::fromStr<std::string>(action.first());
                        if(not GlobalSettings::has(parm))
                            throw InvalidArgumentException("No parameter ",parm," in global settings.");
                        
                        auto value = action.last();
                        GlobalSettings::get(parm).get().set_value_from_string(value);
                    }),
                    ActionFunc("go-back", [](auto){
                        auto prev = SceneManager::getInstance().last_active();
                        if(prev)
                            SceneManager::getInstance().set_active(prev);
                        print("Going back");
                    }),
                    ActionFunc("convert", [](auto){
                        DebugHeader("Converting ", SETTING(source).value<file::PathArray>());
                        sprite::Map copy = GlobalSettings::map();
                        settings::load(SETTING(source), {}, default_config::TRexTask_t::convert, SETTING(detect_type), {}, copy);
                        SceneManager::getInstance().set_active("convert-scene");
                    }),
                    ActionFunc("track", [](auto){
                        DebugHeader("Tracking ", SETTING(source).value<file::PathArray>());
                        //SETTING(filename) = file::Path();
                        SceneManager::getInstance().set_active("tracking-scene");
                    }),
                    ActionFunc("choose-source", [](auto){
                        print("choose-source");
                    }),
                    
                    ActionFunc("change_scene", [](Action action) {
                        if(action.parameters.empty())
                            throw U_EXCEPTION("Invalid arguments for ", action, ".");
                        
                        auto scene = Meta::fromStr<std::string>(action.first());
                        if(not SceneManager::getInstance().is_scene_registered(scene))
                            return false;
                        SceneManager::getInstance().set_active(scene);
                        return true;
                    }),
                    ActionFunc("choose-target", [](auto){
                        print("choose-target");
                    }),
                    ActionFunc("choose-model", [](auto){
                        print("choose-detection");
                    }),
                    ActionFunc("choose-region", [](auto){
                        print("choose-region");
                    }),
                    ActionFunc("choose-settings", [](auto){
                        print("choose-settings");
                    }),
                    ActionFunc("toggle-background-subtraction", [](auto){
                        SETTING(track_background_subtraction) = not SETTING(track_background_subtraction).value<bool>();
                    }),
                    VarFunc("settings_summary", [](const VarProps&) -> std::string {
                        return std::string(GlobalSettings::map().toStr());
                    }),
                    VarFunc("window_size", [this](const VarProps&) -> Vec2 {
                        return window_size;
                    }),
                    VarFunc("video_size", [this](const VarProps& props) -> Vec2 {
                        if(_video_size.empty())
                            return Size2(1);
                        return _video_size;
                    })
                }
            };
            
            dynGUI.context.custom_elements["video"] = std::unique_ptr<GUIVideoAdapterElement>{
                new GUIVideoAdapterElement(_window, [this]() {
                    return window_size;
                }, [this](VideoInfo info) {
                    _next_video_size = info.size;
                }, [this](const file::PathArray& path, IMGUIBase* window, std::function<void(VideoInfo)> callback) {
                    if(_video_adapters.contains(path.source())) {
                        return _video_adapters[path.source()];
                    } else {
                        Layout::Ptr ptr = Layout::Make<GUIVideoAdapter>(path, window, callback);
                        if(_video_adapters.size() >= 2)
                            _video_adapters.clear();
                        _video_adapters[path.source()] = ptr;
                        return ptr;
                    }
                })
            };
        }
        
        auto dt = saturate(animation_timer.elapsed(), 0.001, 1.0);
        blur_percentage += (blur_target - blur_percentage) * dt * 2.0;
        blur_percentage = saturate(blur_percentage, 0.0, 1.0);
        animation_timer.reset();
        
        //_video_loop.set_blur_value(blur_percentage);
        
        double limit = 0.1;
        if(abs(blur_target - blur_percentage) > 0.01)
            limit = 0.025;
        if(video_image_timer.elapsed() > limit) {
            /*auto ptr = _video_loop.get_if_ready();
            if(ptr) {
                _preview_image.exchange_with(std::move(ptr));
                
                /// did we get something back? yes if there was
                /// already a preview image. -> return it
                if(ptr)
                    _video_loop.move_back(std::move(ptr));
                
                video_image_timer.reset();
            }*/
        }
        
        auto scale = Vec2{
            max(window_size.width / max(_preview_image.width(), 1.f),
                window_size.height / max(_preview_image.height(), 1.f))
        };
        
        _preview_image.set_scale(scale);
        _preview_image.set_origin(Vec2(0.5));
        _preview_image.set_pos(window_size * 0.5);
        _preview_image.set_color(White.alpha(100));
        
        graph.wrap_object(_preview_image);
        //graph.wrap_object(_preview_image);
        dynGUI.update(nullptr/*, [this](auto &objs){
            objs.push_back(Layout::Ptr(_preview_image));
        }*/);
        _video_size = _next_video_size.load();
    }
};

SettingsScene::SettingsScene(Base& window)
    : Scene(window, "settings-scene", [this](auto&,DrawStructure& graph){
        _draw(graph);
      })
{
}

SettingsScene::~SettingsScene() {
    
}

void SettingsScene::activate() {
    //auto video_size = Size2(1200,920);
    auto work_area = ((const IMGUIBase*)window())->work_area();
    //auto window_size = video_size;
    
    auto window_size = Size2(work_area.width * 0.75, work_area.width * 0.75 * 0.7);

    Bounds bounds(
        Vec2(),
        window_size);

    print("Calculated bounds = ", bounds, " from window size = ", window_size, " and work area = ", work_area);
    bounds.restrict_to(work_area);
    bounds << Vec2((work_area.width - work_area.x) / 2 - bounds.width / 2,
        work_area.height / 2 - bounds.height / 2 + work_area.y);
    bounds.restrict_to(work_area);
    print("Restricting bounds to work area: ", work_area, " -> ", bounds);

    print("setting bounds = ", bounds);
    //window()->set_window_size(window_size);
    window()->set_window_bounds(bounds);
    
    dyn::Modules::add(dyn::Modules::Module{
        ._name = "follow",
        ._apply = [](size_t index, dyn::State& state, const Layout::Ptr& o) {
            state.display_fns[index] = [o = o.get()](DrawStructure& g){
                o->set_pos(g.mouse_position() + Vec2(5));
            };
        }
    });
    
    _data = std::make_unique<Data>();
    _data->_window = (IMGUIBase*)window();
    /*_data->callback = GlobalSettings::map().register_callbacks({"source"}, [this](auto name) {
        if(name == "source") {
            // changed source, need to update background images
            if(_data)
                _data->_video_loop.set_path(SETTING(source).value<file::PathArray>());
        }
    });
    
    _data->_video_loop.start();*/
}

void SettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    //_data->dynGUI.clear();
    
    if(_data && _data->callback)
        GlobalSettings::map().unregister_callbacks(std::move(_data->callback));
    _data->dynGUI.clear();
    _data = nullptr;
    dyn::Modules::remove("follow");
}

void SettingsScene::_draw(DrawStructure& graph) {
    using namespace dyn;

    
    auto w = Vec2(window()->window_dimensions().width, 
                  window()->window_dimensions().height);
    if(w != _data->window_size) {
        _data->window_size = w;
        //_data->_video_loop.set_target_resolution(w);
    }
    
    _data->draw(graph);
}

bool SettingsScene::on_global_event(Event e) {
    if(e.type == EventType::KEY
       && not e.key.pressed) 
    {
        if(e.key.code == Keyboard::T
           && _data)
        {
            if(_data->blur_target < 1)
                _data->blur_target = 1;
            else
                _data->blur_target = 0;
            return true;
        }
    }
    return false;
}

}

