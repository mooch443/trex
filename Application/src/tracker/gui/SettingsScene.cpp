#include "SettingsScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <misc/RecentItems.h>
//#include <gui/types/Dropdown.h>
//#include <gui/types/Checkbox.h>
#include <gui/dyn/Action.h>
#include <video/VideoSource.h>
//#include <misc/AbstractVideoSource.h>
//#include <misc/VideoVideoSource.h>
//#include <misc/WebcamVideoSource.h>
#include <gui/DynamicGUI.h>
#include <misc/SettingsInitializer.h>
#include <misc/BlurryVideoLoop.h>
#include <gui/GUIVideoAdapterElement.h>
#include <misc/VideoInfo.h>
#include <gui/GUIVideoAdapter.h>
#include <gui/WorkProgress.h>
#include <misc/Coordinates.h>
#include <python/Yolo8.h>
#include <tracking/Output.h>

#include <gui/TrackingScene.h>

#include <portable-file-dialogs.h>

namespace cmn::gui {

struct SettingsScene::Data {
    Timer video_image_timer, animation_timer;
    double blur_target{1};
    double blur_percentage{0};
    
    std::future<void> check_new_video_source;
    
    dyn::DynamicGUI dynGUI;
    CallbackCollection callback;
    std::string layout_name { "choose_settings_layout.json" };
    
    std::unordered_map<std::string, std::future<bool>> _scheduled_exist_checks;
    std::unordered_map<std::string, bool> _done_exist_checks;

    GuardedProperty<file::PathArray> current_path;
    
    std::atomic<bool> _selected_source_exists{false};
    
    sprite::Map _defaults;
    std::stack<std::string> _last_layouts;
    
    ExternalImage _preview_image;
    IMGUIBase *_window{nullptr};
    
    file::PathArray _initial_source;
    std::atomic<Size2> _next_video_size;
    Size2 _video_size;
    std::unordered_map<std::string, Layout::Ptr> _video_adapters;
    
    ~Data() {
        if(callback)
            GlobalSettings::map().unregister_callbacks(std::move(callback));
        dynGUI.clear();
        
        if(check_new_video_source.valid())
            check_new_video_source.get();
    }
    
    file::Path target_file(std::string_view ext = "pv") {
        if(ext == "results") {
            auto filename = Output::TrackingResults::expected_filename();
            return filename;
        }
        
        auto filename = SETTING(filename).value<file::Path>();
        if(filename.empty())
            filename = settings::find_output_name(GlobalSettings::map());
        if(not filename.has_extension() || filename.extension() != ext)
            filename = filename.add_extension(ext);
        return filename;
    }
    
    void check_video_source(file::PathArray source);
    void register_callbacks() {
        if(callback)
            GlobalSettings::map().unregister_callbacks(std::move(callback));
        
        callback = GlobalSettings::map().register_callbacks({
            "filename",
            "source",
            "detect_type"
            
        }, [this](auto name) {
            if(name == "filename") {
                file::Path path = GlobalSettings::map().at("filename").value<file::Path>();
                if(not path.empty() && not path.remove_filename().empty()) {
                    if(path.has_extension("pv"))
                        path = path.remove_extension();
                    path = path.filename();
                    SETTING(filename) = path;
                }
            } else if(name == "source") {
                //SETTING(filename) = file::Path();

                file::PathArray source = GlobalSettings::map().at("source");
                if(check_new_video_source.valid())
                    check_new_video_source.get();
                
                check_new_video_source = std::async(std::launch::async, [source, this](){
                    check_video_source(source);
                });
                
            } else if(name == "detect_type") {
                auto detect_type = SETTING(detect_type).value<track::detect::ObjectDetectionType_t>();
                
                ExtendableVector exclude;
                if(not SETTING(detect_model).value<file::Path>().empty()
                   && SETTING(detect_model).value<file::Path>() != file::Path(track::detect::yolo::default_model()).remove_extension())
                {
                    exclude = {
                        "detect_model"
                    };
                }
                
                settings::set_defaults_for(detect_type, GlobalSettings::map(), exclude);
            }
        });
    }
    
    sprite::Map get_changed_props() const {
        sprite::Map copy = GlobalSettings::map();
        const auto &defaults = GlobalSettings::defaults();
        const auto &_defaults = GlobalSettings::current_defaults_with_config();
        /*Print("current output_dir = ", _defaults.at("calculate_posture"));
        Print("current output_dir = ", copy.at("calculate_posture"));
        Print("keys = ", copy.keys());
        Print("_defaults keys = ", _defaults.keys());*/
        
        for(auto &key : copy.keys()) {
            
            if(  ( (_defaults.has(key)
                     && copy.at(key).get() != _defaults.at(key).get())
                   || not defaults.has(key)
                   || defaults.at(key) != copy.at(key)
                  )
                //|| (this->_defaults.has(key) && copy.at(key).get() != this->_defaults.at(key).get()))
               && (GlobalSettings::access_level(key) < AccessLevelType::INIT
                   || is_in(key, "output_dir", "output_prefix", "settings_file", "video_conversion_range")))
            {
                if(_defaults.has(key))
                    Print("Keeping ", key, "::",GlobalSettings::access_level(key),": default<", _defaults.at(key).get(), "> != assigned<", copy.at(key).get(),">");
                else
                    Print("Keeping ", key, "::",GlobalSettings::access_level(key),": ", copy.at(key).get());
                
                continue;
            }
            
            copy.erase(key);
        }
        //Print("Maintaining: ", copy.keys());
        return copy;
    }
    
    void load_video_settings(const file::PathArray& source);
    
    void draw(DrawStructure& graph) {
        using namespace dyn;
        if(not dynGUI) {
            dynGUI = DynamicGUI{
                .gui = SceneManager::getInstance().gui_task_queue(),
                .path = layout_name,
                .context = {
                    ActionFunc("set", [](Action action) {
                        REQUIRE_EXACTLY(2, action);
                        
                        auto parm = Meta::fromStr<std::string>(action.first());
                        if(not GlobalSettings::has(parm))
                            throw InvalidArgumentException("No parameter ",parm," in global settings.");
                        
                        auto value = action.last();
                        GlobalSettings::get(parm).get().set_value_from_string(value);
                    }),
                    ActionFunc("reset_settings", [](auto){
                        SceneManager::getInstance().enqueue([](auto, DrawStructure& graph) {
                            graph.dialog([](Dialog::Result result) mutable {
                                if(result == Dialog::Result::OKAY) {
                                    /// resets settings that come from the recentitems
                                    /// config array:
                                    sprite::Map cleared;
                                    
                                    SETTING(filename).get().copy_to(&cleared);
                                    SETTING(source).get().copy_to(&cleared);
                                    SETTING(output_prefix).get().copy_to(&cleared);
                                    SETTING(output_dir).get().copy_to(&cleared);
                                    SETTING(detect_type).get().copy_to(&cleared);
                                    
                                    settings::reset(cleared);
                                    
                                    //settings::load(SETTING(source).value<file::PathArray>(), SETTING(filename).value<file::Path>(), default_config::TRexTask_t::none, SETTING(detect_type), {}, cleared);
                                }
                            }, "This will reset all settings you have made here. Everything that is located inside <cyan><c>.settings</c></cyan> files or the video file itself (e.g. <cyan><c>frame_rate</c></cyan>) will remain. Are you sure?", "Reset settings", "Reset", "Cancel");
                            
                        });
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
                        
                        auto f = WorkProgress::add_queue("", [this, copy = get_changed_props()]() {
                            Print("changed props = ", copy.keys());
                            sprite::Map before = GlobalSettings::map();
                            sprite::Map defaults = GlobalSettings::current_defaults();
                            sprite::Map defaults_with_config = GlobalSettings::current_defaults_with_config();

                            /// if we determine that we are actually reconverting a pv file
                            /// we need to load the source settings from the pv file, since we
                            /// cant use the pv file and convert it to itself:
                            auto source = SETTING(source).value<file::PathArray>();
                            if (source.size() == 1
                                && source.get_paths().front().has_extension("pv"))
                            {
                                auto meta_source_path = SETTING(meta_source_path).value<std::string>();
                                if (not meta_source_path.empty()) {
                                    file::PathArray array(meta_source_path);
                                    auto parent = file::find_parent(source);
                                    auto basename = file::find_basename(source);

                                    if (parent && parent->exists()) {
                                        if(SETTING(output_dir).value<file::Path>().empty())
											SETTING(output_dir) = parent.value();
                                    }

                                    if(SETTING(filename).value<file::Path>().empty())
										SETTING(filename) = file::Path(basename);
                                    SETTING(source) = array;
                                }
                            }
                            
                            auto filename = SETTING(filename).value<file::Path>();
                            if (not filename.empty()) {
                                filename = settings::find_output_name(GlobalSettings::map());
                            }
                            settings::load(SETTING(source), filename, default_config::TRexTask_t::convert, SETTING(detect_type), {}, copy);
                            
                            SceneManager::getInstance().enqueue([this,
                                before = std::move(before),
                                defaults = std::move(defaults),
                                defaults_with_config = std::move(defaults_with_config)
                            ] (auto,DrawStructure& graph) {
                                using namespace track;
                                if (SETTING(detect_type).value<detect::ObjectDetectionType_t>() == detect::ObjectDetectionType::yolo8) 
                                {
                                    auto path = SETTING(detect_model).value<file::Path>();
                                    if (track::detect::yolo::valid_model(path))
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
                        if(f.wait_for(std::chrono::milliseconds(100)) == std::future_status::ready) {
                            f.get();
                        } else {
                            WorkProgress::set_item("loading...");
                        }
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
                        
                        WorkProgress::add_queue("loading...", [this, copy = get_changed_props()]()
                        {
                            sprite::Map before = GlobalSettings::map();
                            sprite::Map defaults = GlobalSettings::current_defaults();
                            sprite::Map defaults_with_config = GlobalSettings::current_defaults_with_config();
                            
                            Print("changed props = ", copy.keys());
                            auto array = SETTING(source).value<file::PathArray>();
                            auto front = file::Path(file::find_basename(array));
                            
                            auto output_file = (not front.has_extension() || front.extension() != "pv") ?
                            file::DataLocation::parse("output", front.add_extension("pv")) :
                            file::DataLocation::parse("output", front.replace_extension("pv"));
                            if (output_file.exists()) {
                                SETTING(filename) = file::Path(output_file);
                            }
                            
                            settings::load(array, SETTING(filename), default_config::TRexTask_t::track, track::detect::ObjectDetectionType::none, {}, copy);
                            
                            auto open_file = [](){
                                SceneManager::getInstance().enqueue([](){
                                    SceneManager::getInstance().set_active("tracking-scene");
                                });
                            };
                            
                            auto fname = target_file("results");
                            if(fname.exists()) {
                                SceneManager::getInstance().enqueue([open_file, fname, before = std::move(before), defaults = std::move(defaults), defaults_with_config = std::move(defaults_with_config)](auto,DrawStructure& graph)
                                                                    {
                                    graph.dialog([open_file, before = std::move(before), defaults = std::move(defaults), defaults_with_config = std::move(defaults_with_config)](Dialog::Result result) mutable {
                                        if(result == Dialog::Result::OKAY) {
                                            /// load results
                                            TrackingScene::request_load();
                                            open_file();
                                        } else if(result == Dialog::Result::SECOND) {
                                            open_file();
                                        } else {
                                            GlobalSettings::map() = before;
                                            GlobalSettings::set_current_defaults(std::move(defaults));
                                            GlobalSettings::set_current_defaults_with_config(std::move(defaults_with_config));
                                        }
                                        
                                    }, "Would you like to load <cyan><c>"+fname.str()+"</c></cyan>, which already exists?", "Results exist", "Load Results", "Cancel", "Track Again");
                                });
                                
                            } else {
                                open_file();
                            }
                        });
                    }),
                    ActionFunc("choose-source", [](auto){
                        Print("choose-source");
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
                        Print("choose-target");
                    }),
                    ActionFunc("choose-model", [](auto){
                        Print("choose-detection");
                    }),
                    ActionFunc("choose-region", [](auto){
                        Print("choose-region");
                    }),
                    ActionFunc("choose-settings", [](auto){
                        Print("choose-settings");
                    }),
                    ActionFunc("choose-folder", [](const Action& action) {
                        REQUIRE_AT_LEAST(1, action);
                        WorkProgress::add_queue("Selecting folder", [action](){
                            auto parm = action.parameters.front();
                            auto folder = action.parameters.size() == 1 ? action.parameters.back() : file::cwd().str();
                            if(not file::Path{folder}.is_folder())
                                folder = {};
                            
                            auto dir = pfd::select_folder("Select a folder", folder).result();
                            GlobalSettings::get(parm).get().set_value_from_string(dir);
                            std::cout << "Selected "<< parm <<": " << dir << "\n";
                        });
                    }),
                    ActionFunc("choose-file", [](const Action& action) {
                        REQUIRE_AT_LEAST(1, action);
                        WorkProgress::add_queue("Selecting file", [action](){
                            auto parm = action.parameters.front();
                            auto folder = action.parameters.size() > 1 ? action.parameters.at(1) : file::cwd().str();
                            if(not file::Path{folder}.is_folder())
                                folder = {};
                            
                            std::vector<std::string> filters;
                            if(action.parameters.size() > 2) {
                                filters.insert(filters.end(), action.parameters.begin() + 2, action.parameters.end());
                            }
                            
                            auto flags = GlobalSettings::get(parm).is_type<file::PathArray>() ? pfd::opt::multiselect : pfd::opt::none;
                            auto dir = pfd::open_file("Select a file", folder, filters, flags).result();
                            
                            if(GlobalSettings::get(parm).is_type<file::PathArray>())
                            {
                                if(not dir.empty())
                                    GlobalSettings::get(parm).get().set_value_from_string(Meta::toStr(dir));
                            } else {
                                if(not dir.empty()) {
                                    GlobalSettings::get(parm).get().set_value_from_string(dir.front());
                                }
                            }
                        });
                    }),
                    ActionFunc("choose-settings", [this](const Action& action) {
                        WorkProgress::add_queue("Selecting file", [this, action](){
                            auto folder = action.parameters.size() > 0 ? action.parameters.at(0) : file::cwd().str();
                            if(not file::Path{folder}.is_folder())
                                folder = {};
                            
                            std::vector<std::string> filters{
                                "Settings", "*.settings",
                                "PV Videos", "*.pv"
                            };
                            
                            auto flags = pfd::opt::none;
                            auto dir = pfd::open_file("Load Settings File", folder, filters, flags).result();
                            
                            if(not dir.empty()) {
                                file::Path path(dir.front());
                                if(path.has_extension() && utils::lowercase(path.extension()) == "pv") {
                                    load_video_settings(path);
                                    
                                } else {
                                    GlobalSettings::load_from_file({}, path.str(), AccessLevelType::LOAD);
                                }
                            }
                        });
                    }),
                    ActionFunc("reload_selected_source", [this](auto){
                        file::PathArray source = GlobalSettings::map().at("source");
                        load_video_settings(source);
                    }),
                    ActionFunc("reset_settings", [](auto){
                        sprite::Map cleared;
                        
                        SETTING(filename).get().copy_to(&cleared);
                        SETTING(source).get().copy_to(&cleared);
                        SETTING(output_prefix).get().copy_to(&cleared);
                        SETTING(output_dir).get().copy_to(&cleared);
                        SETTING(detect_type).get().copy_to(&cleared);
                        
                        settings::reset(cleared);
                    }),
                    ActionFunc("toggle-background-subtraction", [](auto){
                        SETTING(track_background_subtraction) = not SETTING(track_background_subtraction).value<bool>();
                    }),
                    VarFunc("video_file", [this](const VarProps&) -> file::PathArray {
                        return current_path.get();
                    }),
                    VarFunc("selected_source_exists", [this](const VarProps&) -> bool {
                        return _selected_source_exists.load();
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
        dynGUI.update(graph, nullptr/*, [this](auto &objs){
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

void SettingsScene::Data::check_video_source(file::PathArray source) {
    ExtendableVector exclude{
        "output_prefix",
        "filename",
        "source",
        "output_dir"
    };

    _selected_source_exists = false;
    
    try {
        if (source != _initial_source
            && source.get_paths().size() == 1
            && source.get_paths().front().has_extension("pv"))
        {
            //auto output = settings::find_output_name(GlobalSettings::map());
            auto output = source.get_paths().front();
            pv::File video(output);
            video.header();
            
            _selected_source_exists = true;
            
            /// escape!
            //return;

            /*if (metadata.has("meta_source_path")) {
                auto source_path = file::PathArray(metadata.at("meta_source_path").value<std::string>());
                if (not source_path.empty()
                    && source_path.get_paths().front().exists())
                {
                    try {
                        //WorkProgress::add_queue("loading...", [source_path, output]() {
                            //SETTING(output_dir) = output.remove_filename();
                            file::Path filename = output.remove_extension();
                            //source = source_path;

                            try {
                                //settings::load(source, filename, default_config::TRexTask_t::track, track::detect::ObjectDetectionType::none, exclude, {});

                            }
                            catch (const std::exception& ex) {
                                FormatWarning("Ex = ", ex.what());
                            }
                        //});
                        
                        return;
                    }
                    catch (...) {
                        /// do nothing
                    }
                }
            }*/
            
        } else if(source == file::PathArray("webcam")) {
            /// should be okay
            
        } else if(source != _initial_source) {
            VideoSource v(source);
            Print("VideoSource for ",source," of size ", v.size(),".");
        }
        
        SceneManager::getInstance().enqueue([this, source](){
            try {
                if(_initial_source.empty()) {
                    load_video_settings(source);
                    
                    // set initial source for the first time
                    _initial_source = source;
                }
                
                current_path.set(std::move(source));
                
            } catch(...) {
                /// do nothing
            }
        });
    }
    catch (...) {
        /// do nothing
    }
    
    /// some stuff regarding reloading settings when the source changes:
    /// *currently disabled since it feels bad*
    //if (source == initial_source) {
    //    return;
    //}
}

void SettingsScene::Data::load_video_settings(const file::PathArray& source) {
    ExtendableVector exclude{
        //"output_prefix",
        "filename",
        "source",
        "load",
        "task"
        //"output_dir"
    };
    
    if(callback)
        GlobalSettings::map().unregister_callbacks(std::move(callback));
    
    if (auto source_path = source.empty() ? file::Path{} : file::DataLocation::parse("input", source.get_paths().front());
        not source.empty()
        && source_path.is_regular()
        && (not source_path.has_extension()
            || utils::lowercase(source_path.extension()) != "pv"))
    {
        file::Path filename = GlobalSettings::map().at("filename");
        try {
            settings::load(source, filename, default_config::TRexTask_t::convert, track::detect::ObjectDetectionType::none, exclude, GlobalSettings::current_defaults_with_config());
        }
        catch (const std::exception& ex) {
            FormatWarning("Ex = ", ex.what());
        }
    }
    else if (not source_path.empty()
             && source_path.is_regular()
             //auto output_path = settings::find_output_name(GlobalSettings::map()).add_extension("pv");
             //not source.empty()
             //&& output_path.is_regular())
             )
    {
        try {
            pv::File file(source_path.remove_extension());
            auto str = file.header().metadata;
            
            sprite::Map map;
            try {
                sprite::parse_values(sprite::MapSource{ source_path }, map, str, &GlobalSettings::defaults(), exclude, default_config::deprecations());
            }
            catch (...) {
                /// do nothing
            }
            
            auto filename = SETTING(filename).value<file::Path>();
            auto csource = SETTING(source).value<file::PathArray>();
            settings::load(source, {}, default_config::TRexTask_t::none, track::detect::ObjectDetectionType::none, exclude, map);
            SETTING(source) = csource;
            SETTING(filename) = filename;
            
            //settings::load(source, filename, default_config::TRexTask_t::track, track::detect::ObjectDetectionType::none, exclude, GlobalSettings::current_defaults_with_config());
        }
        catch (const std::exception& ex) {
            FormatWarning("Ex = ", ex.what());
        }
    }
    
    register_callbacks();
}

void SettingsScene::activate() {
    WorkProgress::instance().start();
    
    _data = std::make_unique<Data>();
    _data->_window = (IMGUIBase*)window();

    _data->_initial_source = SETTING(source).value<file::PathArray>();
    _data->register_callbacks();
    _data->_defaults = GlobalSettings::map();
}

void SettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    //_data->dynGUI.clear();
    WorkProgress::stop();

    if(_data) {
        Print("_data is set, need to unregister callbacks...");
        if(_data->callback)
            GlobalSettings::map().unregister_callbacks(std::move(_data->callback));
        
        Print("Clearing _data->dynGUI");
        _data->dynGUI.clear();

        Print("_checking new video source: ", _data->check_new_video_source.valid());
        if(_data->check_new_video_source.valid())
            _data->check_new_video_source.get();

        /// need to clear queue in case we got something pushed in the check_new_video_source
        SceneManager::getInstance().update_queue();
        
        Print("Deleting _data...");
        _data = nullptr;

        Print("Done.");
    }
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

