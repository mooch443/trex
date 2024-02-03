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
#include <gui/WorkProgress.h>
#include <gui/Coordinates.h>
#include <python/Yolo8.h>

namespace gui {

struct SettingsScene::Data {
    //BlurryVideoLoop _video_loop;
    SettingsTooltip _settings_tooltip;
    Timer video_image_timer, animation_timer;
    double blur_target{1};
    double blur_percentage{0};
    
    dyn::DynamicGUI dynGUI;
    CallbackCollection callback;
    std::string layout_name { "choose_settings_layout.json" };
    
    std::unordered_map<std::string, std::future<bool>> _scheduled_exist_checks;
    std::unordered_map<std::string, bool> _done_exist_checks;
    
    sprite::Map _defaults;
    std::stack<std::string> _last_layouts;
    
    ExternalImage _preview_image;
    IMGUIBase *_window{nullptr};
    
    std::atomic<Size2> _next_video_size;
    Size2 _video_size;
    std::unordered_map<std::string, Layout::Ptr> _video_adapters;
    
    sprite::Map get_changed_props() const {
        sprite::Map copy = GlobalSettings::map();
        const auto &defaults = GlobalSettings::defaults();
        const auto &_defaults = GlobalSettings::current_defaults_with_config();
        print("current output_dir = ", _defaults.at("calculate_posture"));
        print("current output_dir = ", copy.at("calculate_posture"));
        print("keys = ", copy.keys());
        print("_defaults keys = ", _defaults.keys());
        
        for(auto &key : copy.keys()) {
            
            if(  ( (_defaults.has(key)
                     && copy.at(key).get() != _defaults.at(key).get())
                   || not defaults.has(key)
                   || defaults.at(key) != copy.at(key)
                  )
                //|| (this->_defaults.has(key) && copy.at(key).get() != this->_defaults.at(key).get()))
               && (GlobalSettings::access_level(key) < AccessLevelType::LOAD
                   || is_in(key, "output_dir", "output_prefix", "settings_file")))
            {
                if(_defaults.has(key))
                    print("Keeping ", key, ": default<", _defaults.at(key).get(), "> != assigned<", copy.at(key).get(),">");
                else
                    print("Keeping ", key, ": ", copy.at(key).get());
                
                continue;
            }
            
            copy.erase(key);
        }
        print("Maintaining: ", copy.keys());
        return copy;
    }
    
    void draw(DrawStructure& graph) {
        using namespace dyn;
        if(not dynGUI) {
            dynGUI = DynamicGUI{
                .gui = SceneManager::getInstance().gui_task_queue(),
                .path = layout_name,
                .graph = &graph,
                .context = {
                    ActionFunc("set", [](Action action) {
                        REQUIRE_EXACTLY(2, action);
                        
                        auto parm = Meta::fromStr<std::string>(action.first());
                        if(not GlobalSettings::has(parm))
                            throw InvalidArgumentException("No parameter ",parm," in global settings.");
                        
                        auto value = action.last();
                        GlobalSettings::get(parm).get().set_value_from_string(value);
                    }),
                    ActionFunc("go-back", [this](auto){
                        if(_last_layouts.empty()) {
                            SceneManager::getInstance().enqueue([](auto, auto&) {
                                SceneManager::getInstance().set_active("starting-scene");
                            });
                            return;
                        }
                        
                        layout_name = _last_layouts.top();
                        _last_layouts.pop();
                        
                        SceneManager::getInstance().enqueue([this](auto, auto&) {
                            dynGUI.clear();
                            dynGUI = {};
                        });
                    }),
                    ActionFunc("convert", [this](auto){
                        DebugHeader("Converting ", SETTING(source).value<file::PathArray>());
                        
                        WorkProgress::add_queue("loading...", [this, copy = get_changed_props()]() {
                            print("changed props = ", copy.keys());
                            sprite::Map before = GlobalSettings::map();
                            sprite::Map defaults = GlobalSettings::current_defaults();
                            sprite::Map defaults_with_config = GlobalSettings::current_defaults_with_config();
                            
                            settings::load(SETTING(source), {}, default_config::TRexTask_t::convert, SETTING(detect_type), {}, copy);
                            
                            SceneManager::getInstance().enqueue([this,
                                before = std::move(before),
                                defaults = std::move(defaults),
                                defaults_with_config = std::move(defaults_with_config)
                            ] (auto,DrawStructure& graph) {
                                using namespace track;
                                if (SETTING(detect_type).value<detect::ObjectDetectionType_t>() == detect::ObjectDetectionType::yolo8) 
                                {
                                    auto path = SETTING(detect_model).value<file::Path>();
                                    if (Yolo8::is_default_model(path)
                                        || (Yolo8::valid_model(path) && path.exists()))
                                    {
                                        /// we have a valid model
                                    }
                                    else {
                                        graph.dialog([this](Dialog::Result) {
                                            if (layout_name != "settings_layout.json") {
                                                _last_layouts.push(layout_name);
                                                layout_name = "settings_layout.json";

                                                SceneManager::getInstance().enqueue([this](auto, auto&) {
                                                    dynGUI.clear();
                                                    dynGUI = {};
                                                });
                                            }
                                        }, "The model file <c><cyan>"+path.str() + "</cyan></c> does not seem to exist and is not a default Yolo8 model name. Please choose a valid model file (a Yolo8 saved model <c><cyan>.pt</cyan></c>).", "Invalid model", "Okay");
                                        return;
                                    }
                                }

                                auto filename = SETTING(filename).value<file::Path>();
                                if(filename.empty())
                                    filename = settings::find_output_name(GlobalSettings::map());
                                if(not filename.has_extension() || filename.extension() != "pv")
                                    filename = filename.add_extension("pv");
                                
                                if(filename.exists()) {
                                    graph.dialog([
                                        before = std::move(before),
                                        defaults = std::move(defaults),
                                        defaults_with_config = std::move(defaults_with_config)
                                    ](Dialog::Result result) mutable {
                                        if(result == Dialog::Result::OKAY) {
                                            /// continue on to converting!
                                            SceneManager::getInstance().set_active("convert-scene");
                                            
                                        } else {
                                            /// we have to reset settings:
                                            GlobalSettings::map() = before;
                                            GlobalSettings::set_current_defaults(std::move(defaults));
                                            GlobalSettings::set_current_defaults_with_config(std::move(defaults_with_config));
                                        }
                                        
                                    }, "Starting the conversion would overwrite <cyan><c>"+filename.str()+"</c></cyan>, which already exists. Are you sure?", "Overwrite file", "Overwrite", "Cancel");
                                } else
                                    SceneManager::getInstance().set_active("convert-scene");
                            });
                        });
                    }),
                    ActionFunc("change_layout", [this](const Action& action) {
                        REQUIRE_EXACTLY(1, action);
                        auto &name = action.first();
                        _last_layouts.push(layout_name);
                        layout_name = name+".json";
                        
                        SceneManager::getInstance().enqueue([this](auto, auto&) {
                            dynGUI.clear();
                            dynGUI = {};
                        });
                    }),
                    ActionFunc("track", [this](auto){
                        DebugHeader("Tracking ", SETTING(source).value<file::PathArray>());
                        
                        WorkProgress::add_queue("loading...", [copy = get_changed_props()](){
                            print("changed props = ", copy.keys());
                            auto array = SETTING(source).value<file::PathArray>();
                            auto front = file::Path(file::find_basename(array));
                            /*output_file = !front.has_extension() ?
                                          file::DataLocation::parse("input", front.add_extension("pv")) :
                                          file::DataLocation::parse("input", front.replace_extension("pv"));*/

                            auto output_file = (not front.has_extension() || front.extension() != "pv") ?
                                          file::DataLocation::parse("output", front.add_extension("pv")) :
                                          file::DataLocation::parse("output", front.replace_extension("pv"));
                            if (output_file.exists()) {
                                SETTING(filename) = file::Path(output_file);
                            }
                            
                            settings::load(array, SETTING(filename), default_config::TRexTask_t::track, SETTING(detect_type), {}, copy);
                            
                            SceneManager::getInstance().enqueue([](){
                                SceneManager::getInstance().set_active("tracking-scene");
                            });
                        });
                    }),
                    ActionFunc("choose-source", [](auto){
                        print("choose-source");
                    }),
                    
                    ActionFunc("change_scene", [](Action action) {
                        REQUIRE_EXACTLY(1, action);
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
                    VarFunc("window_size", [](const VarProps&) -> Vec2 {
                        return FindCoord::get().screen_size();
                    }),
                    VarFunc("previous_stack_size", [this](const VarProps&) -> size_t {
                        return _last_layouts.size();
                    }),
                    VarFunc("video_size", [this](const VarProps&) -> Vec2 {
                        if(_video_size.empty())
                            return Size2(1);
                        return _video_size;
                    }),
                    VarFunc("file_exists", [this](const VarProps& props) -> bool {
                        REQUIRE_EXACTLY(1, props);
                        
                        auto path = props.first();
                        if(_done_exist_checks.contains(path))
                            return _done_exist_checks.at(path);
                        
                        if(not _scheduled_exist_checks.contains(path)) {
                            while(_scheduled_exist_checks.size() > 0)
                                _scheduled_exist_checks.erase(_scheduled_exist_checks.begin());
                            
                            _scheduled_exist_checks[path] = std::async(std::launch::async, [](const file::Path& path) -> bool {
                                //std::this_thread::sleep_for(std::chrono::seconds(2));
                                return path.exists();
                            }, path);
                        }
                        
                        if(_scheduled_exist_checks.at(path).wait_for(std::chrono::milliseconds(15)) == std::future_status::ready) 
                        {
                            /// limit size of the map
                            while(_done_exist_checks.size() > 25) {
                                _done_exist_checks.erase(_done_exist_checks.begin());
                            }
                            
                            _done_exist_checks[path] = _scheduled_exist_checks.at(path).get();
                            _scheduled_exist_checks.erase(path);
                            return _done_exist_checks.at(path);
                        }
                        
                        throw std::runtime_error("Still checking status...");
                    }),
                    VarFunc("resulting_path", [](const VarProps&) -> file::Path {
                        return settings::find_output_name(GlobalSettings::map());
                    })
                }
            };
            
            dynGUI.context.custom_elements["video"] = std::unique_ptr<GUIVideoAdapterElement>{
                new GUIVideoAdapterElement(_window, []() {
                    return FindCoord::get().screen_size();
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
        
        auto coords = FindCoord::get();
        Size2 window_size = coords.screen_size();
        
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
    WorkProgress::instance().start();
    //auto video_size = Size2(1200,920);
    auto work_area = ((const IMGUIBase*)window())->work_area(); 
#if defined(WIN32)
    work_area.y += 25;
#endif
    auto window_size = Size2(work_area.width * 0.75, work_area.width * 0.75 * 0.7);
    if(window_size.height > work_area.height) {
        auto ratio = window_size.width / window_size.height;
        window_size = Size2(work_area.height * ratio, work_area.height);
    }
    
    Bounds bounds(
        Vec2(),
        window_size);
    
    print("Calculated bounds = ", bounds, " from window size = ", window_size, " and work area = ", work_area);
    bounds.restrict_to(Bounds(work_area.size()));
    bounds << Vec2(work_area.width / 2 - bounds.width / 2,
                    work_area.height / 2 - bounds.height / 2);
    bounds.restrict_to(Bounds(work_area.size()));
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
    _data->callback = GlobalSettings::map().register_callbacks({"filename", "source"}, [](auto name) {
        if(name == "filename") {
            file::Path path = GlobalSettings::map().at("filename").value<file::Path>();
            if(not path.empty() && not path.remove_filename().empty()) {
                path = path.filename();
                SETTING(filename) = path;
            }
        } else if(name == "source") {
            file::PathArray source = GlobalSettings::map().at("source");
            settings::ExtendableVector exclude{
                "output_prefix",
                "filename",
                "source",
                "output_dir"
            };
            if(not source.empty()
               && file::DataLocation::parse("input", source.get_paths().front()).is_regular())
            {
                file::Path filename = GlobalSettings::map().at("filename");
                try {
                    settings::load(source, filename, default_config::TRexTask_t::convert, track::detect::ObjectDetectionType::none, exclude, GlobalSettings::current_defaults_with_config());
                    
                } catch(const std::exception& ex) {
                    FormatWarning("Ex = ", ex.what());
                }
            } else if(not source.empty()
                      && settings::find_output_name(GlobalSettings::map()).add_extension("pv").is_regular())
            {
                file::Path filename = settings::find_output_name(GlobalSettings::map());//GlobalSettings::map().at("filename");
                try {
                    settings::load(source, filename, default_config::TRexTask_t::track, track::detect::ObjectDetectionType::none, exclude, GlobalSettings::current_defaults_with_config());
                    
                } catch(const std::exception& ex) {
                    FormatWarning("Ex = ", ex.what());
                }
            }
        }
    });
    
    _data->_defaults = GlobalSettings::map();
}

void SettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    //_data->dynGUI.clear();
    WorkProgress::stop();
    if(_data && _data->callback)
        GlobalSettings::map().unregister_callbacks(std::move(_data->callback));
    _data->dynGUI.clear();
    _data = nullptr;
    dyn::Modules::remove("follow");
}

void SettingsScene::_draw(DrawStructure& graph) {
    using namespace dyn;

    /*auto w = Vec2(window()->window_dimensions().width,
                  window()->window_dimensions().height);
    if(w != _data->window_size) {
        _data->window_size = w;
        //_data->_video_loop.set_target_resolution(w);
    }*/
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

