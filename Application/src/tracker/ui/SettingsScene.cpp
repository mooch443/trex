#include "SettingsScene.h"
#include <file/PathArray.h>
#include <misc/CommandLine.h>
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <ui/RecentItems.h>
//#include <gui/types/Dropdown.h>
//#include <gui/types/Checkbox.h>
#include <gui/dyn/Action.h>
#include <video/VideoSource.h>
#include <core/SettingsPaths.h>
#include <gui/DynamicGUI.h>
#include <core/SettingsInitializer.h>
#include <core/BlurryVideoLoop.h>
#include <ui/GUIVideoAdapterElement.h>
#include <core/VideoInfo.h>
#include <ui/GUIVideoAdapter.h>
#include <ui/WorkProgress.h>
#include <ui/Coordinates.h>
#include <core/DetectionTypes.h>
#include <python/Detection.h>
#include <tracking/Output.h>
#include <python/PythonWrapper.h>
#include <python/BackendRegistry.h>

#include <ui/TrackingScene.h>
#include <ui/ConvertScene.h>

#include <portable-file-dialogs.h>
#include <misc/ProtectedProperty.h>

namespace cmn::gui {

ProtectedProperty<std::string> last_opened_tab{"choose_settings_layout.json"};

void SettingsScene::reset_last_opened_tab() {
    last_opened_tab.set("choose_settings_layout.json");
}

namespace {
thread_local bool restoring_settings = false;

struct SettingsRestoreGuard {
    bool previous = std::exchange(restoring_settings, true);
    ~SettingsRestoreGuard() { restoring_settings = previous; }
};

void set_global_setting_from_path_string(const std::string& parm, const std::string& path) {
    auto ref = GlobalSettings::get(parm);
    auto& setting = ref.get();
    if(setting.is_type<file::Path>()) {
        setting.toProperty<file::Path>() = file::Path(path);
    } else {
        setting.set_value_from_string(path);
    }
}

void copy_file_locations(sprite::Map& values, const sprite::Map& source) {
    for(auto key : {"output_dir", "output_prefix", "settings_file"})
        source.at(key).get().copy_to(values);
}
}

struct SettingsScene::Data {
    Timer video_image_timer, animation_timer;
    double blur_target{1};
    double blur_percentage{0};
    
    std::mutex _video_source_mutex;
    using CheckedSource = std::expected<file::PathArray, std::string>;
    std::future<CheckedSource> check_new_video_source;
    std::atomic<bool> _video_source_changed{true};
    
    dyn::DynamicGUI dynGUI;
    cmn::CallbackFuture callback;
    std::string layout_name { "choose_settings_layout.json" };
    
    std::unordered_map<std::string, std::future<bool>> _scheduled_exist_checks;
    std::unordered_map<std::string, bool> _done_exist_checks;
    
    Timer source_paths_timer;
    std::optional<file::Path> last_output_name;
    std::optional<file::Path> existing_pv_path;
    std::optional<file::PathArray> conversion_input;

    // Only a checked source with its settings loaded is available to the preview and conversion.
    std::optional<file::PathArray> current_path;
    
    std::mutex _task_lock;
    std::vector<std::future<void>> _running_tasks;
    std::atomic<size_t> _are_python_tasks_running{0};
    std::mutex _detection_model_task_mutex;

    struct DetectionModelRequest {
        track::detect::ObjectDetectionType_t type;
        file::Path detect_model;
        file::Path region_model;

        bool is_current() const {
            return type == READ_SETTING_WITH_DEFAULT(detect_type, track::detect::ObjectDetectionType_t{})
                && detect_model == detect::yolo::current_detect_model_path()
                && region_model == detect::yolo::current_region_model_path();
        }
    };

    struct DetectionModelResult {
        uint64_t generation;
        DetectionModelRequest request;
        std::vector<track::detect::ModelConfig> models;
    };

    std::atomic<uint64_t> _model_generation{0};
    std::atomic<bool> _detection_models_changed{false};
    ProtectedProperty<std::optional<DetectionModelResult>> _model_configs;
    std::map<std::pair<std::string, std::string>, std::vector<track::detect::ModelConfig>> _cached_model_configs;
    track::detect::TemporaryClassNames _model_class_names{[this] {
        if(auto config = model_config(track::detect::ModelTaskType::detect))
            return config->classes;
        return track::detect::yolo::names::owner_map_t{};
    }};
    std::stack<std::string> _last_layouts;
    
    IMGUIBase *_window{nullptr};
    
    file::PathArray _loaded_source;
    std::atomic<Size2> _video_size;
    std::unordered_map<std::string, Layout::Ptr> _video_adapters;
    
    ~Data() {
        update_running_tasks(true);
        
        if(callback)
            GlobalSettings::unregister_callbacks(std::move(callback));
        dynGUI.clear();
        
        if(check_new_video_source.valid())
            check_new_video_source.wait();
    }
    
    CheckedSource check_video_source(file::PathArray source);

    void invalidate_source_paths() {
        last_output_name.reset();
        existing_pv_path.reset();
        conversion_input.reset();
        source_paths_timer.reset();
    }

    bool checks_running() const {
        return _are_python_tasks_running.load() > 0 || _video_source_changed.load()
            || check_new_video_source.valid()
            || (dynGUI.current_object_handler
                && dynGUI.current_object_handler->has_pending_global_path_lookup("source"));
    }

    bool source_ready() const {
        return current_path.has_value() && not checks_running();
    }

    file::Path existing_pv_file() {
        if(checks_running())
            return {};
        if(not existing_pv_path) {
            try {
                const auto path = settings::resolve_recording_path();
                existing_pv_path = path.is_regular() ? path : file::Path{};
            } catch(const std::exception&) {
                existing_pv_path = file::Path{};
            }
        }
        return *existing_pv_path;
    }
    
    void update_running_tasks(bool wait = false) {
#ifndef NDEBUG
        FormatWarning("Waiting for Python tasks...");
#endif
        std::unique_lock guard(_task_lock);
        for(auto it = _running_tasks.begin(); it != _running_tasks.end();) {
            if(wait || cmn::is_future_ready(*it)) {
                try {
                    it->get();
                } catch(...) {
#ifndef NDEBUG
                    FormatWarning("Failed running a task.");
#endif
                }
                it = _running_tasks.erase(it);
                
            } else {
                ++it;
            }
        }
        
        if(_running_tasks.empty()) {
            _are_python_tasks_running = 0;
            //Print("// Setting python tasks = 0");
        }
    }

    std::optional<track::detect::ModelConfig> model_config(track::detect::ModelTaskType task) const {
        auto result = _model_configs.get();
        if(result
           && result->generation == _model_generation.load()
           && result->request.is_current())
        {
            for(const auto& config : result->models) {
                if(config.task == task)
                    return config;
            }
        }
        return std::nullopt;
    }

    bool detection_model_ready() const {
        auto detect_type = READ_SETTING_WITH_DEFAULT(detect_type, track::detect::ObjectDetectionType_t{});
        if(detect_type == track::detect::ObjectDetectionType::precomputed)
            return not READ_SETTING(detect_precomputed_file, file::PathArray).empty();
        if(detect_type != track::detect::ObjectDetectionType::yolo)
            return true;

        const auto config = model_config(track::detect::ModelTaskType::detect);
        return config
            && config->output_format != track::detect::ObjectDetectionFormat::none
            && detect::yolo::valid_model(detect::yolo::current_detect_model_path());
    }

    file::PathArray current_conversion_source(const sprite::Map* values = nullptr) const {
        try {
            if(not values) {
                return GlobalSettings::read([&](const Configuration& config) {
                    return current_conversion_source(&config.values);
                });
            }
            const auto usable = [&](const file::PathArray& requested) -> file::PathArray {
                if(settings::is_camera_source(requested.source()))
                    return requested;
                auto input = file::DataLocation::parse("input", requested, values);
                const auto& paths = input.get_paths();
                if(not paths.empty() && std::ranges::all_of(paths, [](const auto& path) {
                    return not path.has_extension("pv") && path.is_regular();
                }))
                    return input;
                return {};
            };
            const auto selected = values->at("source").value<file::PathArray>().resolved_at(CommandLine::instance().launch_dir());
            if(selected.empty() || settings::video_source_error(selected))
                return {};
            auto input = usable(selected);
            if(not input.empty())
                return input;

            const auto original = values->value_with_default("meta_source_path", std::string{});
            if(original.empty())
                return {};
            input = usable(file::PathArray(original));
            if(not input.empty() || file::Path(original).is_absolute())
                return input;
            return usable(file::PathArray(file::DataLocation::parse("output", original, values)));
        } catch(const std::exception&) {
            return {};
        }
    }

    void detection_models_updated() {
        update_running_tasks();

        const auto generation = _model_generation.load();
        const auto type = READ_SETTING_WITH_DEFAULT(detect_type, track::detect::ObjectDetectionType_t{});
        if(type != track::detect::ObjectDetectionType::yolo) {
            return;
        }

        const DetectionModelRequest request{
            .type = type,
            .detect_model = detect::yolo::current_detect_model_path(),
            .region_model = detect::yolo::current_region_model_path()
        };

        // Text fields may briefly publish incomplete paths while they are
        // edited. Do not start Python, or accept an arbitrary regular file,
        // until the detector path is a supported model.
        if(not track::detect::yolo::valid_model(request.detect_model))
            return;
        
        std::unique_lock guard(_task_lock);
        ++_are_python_tasks_running;
        //Print("// Python tasks running = ", _are_python_tasks_running.load());
        
        _running_tasks.emplace_back(Python::schedule(Python::PackagedTask{
            ._network = nullptr,
            ._task = [this, request, generation]()
            {
                std::unique_lock model_guard(_detection_model_task_mutex);

                try {
                    if(generation == _model_generation.load() && request.is_current())
                    {
                        const auto key = std::pair{request.detect_model.str(), request.region_model.str()};
                        const bool cacheable = request.region_model.empty() || request.region_model.is_regular();
                        if(cacheable && _cached_model_configs.contains(key))
                        {
                            _model_configs.set(DetectionModelResult{
                                generation, request, _cached_model_configs.at(key)
                            });
                            
                        } else {
                            /// for no cache, reinit:
                            try {
                                std::vector<track::detect::ModelConfig> models;
                                if(const auto* hooks = track::detect::ensure_backend(track::detect::ObjectDetectionType::yolo); hooks && hooks->init) {
                                    models = hooks->init();
                                } else {
                                    throw U_EXCEPTION("YOLO backend is unavailable.");
                                }
                                
                                /// dont need to keep it
                                if(const auto* hooks = track::detect::ensure_backend(track::detect::ObjectDetectionType::yolo); hooks && hooks->deinit) {
                                    hooks->deinit();
                                }

                                if(generation == _model_generation.load() && request.is_current()) {
                                    if(cacheable)
                                        _cached_model_configs[key] = models;
                                    _model_configs.set(DetectionModelResult{generation, request, std::move(models)});
                                }
                                
                            } catch(...) {
                                try {
                                    if(const auto* hooks = track::detect::ensure_backend(track::detect::ObjectDetectionType::yolo);
                                       hooks && hooks->deinit)
                                    {
                                        hooks->deinit();
                                    }
                                } catch(const std::exception& ex) {
                                    FormatWarning("Failed to clean up the detection model after initialization failed: ", ex.what());
                                } catch(...) {
                                    FormatWarning("Failed to clean up the detection model after initialization failed.");
                                }

                                FormatWarning("Failed to initialize ", request.detect_model);
                            }
                        }
                    }

                    --_are_python_tasks_running;
                    Print("// Python tasks running(normal end) = ", _are_python_tasks_running.load());
                    
                } catch(...) {
                    --_are_python_tasks_running;
                    Print("// Python tasks running (exception) = ", _are_python_tasks_running.load());
                    throw;
                }
            },
            ._can_run_before_init = false
        }));
    }
    
    void register_callbacks() {
        if(callback)
            GlobalSettings::unregister_callbacks(std::move(callback));
        
        auto fn = [this](std::string_view name) {
            if(is_in(name, "detect_type", "detect_model", "region_model", "output_dir", "output_prefix"))
                ++_model_generation;

            if(restoring_settings && is_in(name, "detect_type", "filename", "source")) {
                if(name == "detect_type")
                    _detection_models_changed = true;
                return;
            }

            if(name == "averaging_method") {
                SETTING(reset_average) = true;
                
            } else if(name == "source") {
                //SETTING(filename) = file::Path();
                _video_source_changed = true;
                
            } else if(name == "detect_type") {
                auto detect_type = READ_SETTING(detect_type, track::detect::ObjectDetectionType_t);
                
                ExtendableVector exclude;
                if(auto detect_model = detect::yolo::current_detect_model_path();
                   not detect_model.empty()
                   && detect_model != file::Path(track::detect::yolo::default_model()).remove_extension())
                {
                    exclude = {
                        "detect_model"
                    };
                }
                if(not READ_SETTING(region_model, file::Path).empty()){
                    exclude += std::vector<std::string_view>{
                        "region_model"
                    };
                }
                
                GlobalSettings::write([&](Configuration& config){
                    settings::set_defaults_for(detect_type, config.values, exclude, config.values.at("cm_per_pixel").value<track::Settings::cm_per_pixel_t>());
                });

                if(detect_type == track::detect::ObjectDetectionType::yolo)
                    _detection_models_changed = true;
                else {
                    if(is_in(detect_type, track::detect::ObjectDetectionType::background_subtraction, track::detect::ObjectDetectionType::precomputed, track::detect::ObjectDetectionType::none)) {
                        SETTING(detect_model) = file::Path{};
                        SETTING(region_model) = file::Path{};
                    }
                }
                
            } else if(is_in(name, "detect_model", "region_model", "output_dir", "output_prefix")) {
                _detection_models_changed = true;
            }
        };
        
        callback = GlobalSettings::register_callbacks<sprite::RegisterInit::DONT_TRIGGER>({
            "filename",
            "source",
            "detect_type",
            "detect_model",
            "region_model",
            "averaging_method",
            "output_dir",
            "output_prefix"

        }, fn);
        
        if(READ_SETTING(source, file::PathArray).has_source())
            fn("detect_model");
    }
    
    sprite::Map get_changed_props(const GlobalSettings::MutableSnapshot& snapshot) const {
        auto copy = snapshot.config.values;
        const auto& defaults = snapshot.config.defaults;
        const auto& _defaults = snapshot.current_defaults_with_config;
        //const auto &defaults = GlobalSettings::defaults();
        //const auto &_defaults = GlobalSettings::current_defaults_with_config();
        
#ifndef NDEBUG
        /*Print("current video_conversion_range = ", _defaults.at("video_conversion_range"));
        Print("current video_conversion_range = ", copy.at("video_conversion_range"));
        Print("keys = ", copy.keys());
        Print("_defaults keys = ", _defaults.keys());*/
#endif
        
        for(auto &key : copy.keys()) {
            if(is_in(key, "output_dir", "output_prefix", "settings_file"))
                continue;
            if(  ( (_defaults.has(key)
                     && copy.at(key).get() != _defaults.at(key).get())
                   || not defaults.has(key)
                   || defaults.at(key) != copy.at(key)
                  )
               && (snapshot.config._access_level(key) < AccessLevelType::INIT
                   || is_in(key, "video_conversion_range", "detect_type", "detect_format")))
            {
#ifndef NDEBUG
                /*if(_defaults.has(key))
                    Print("Keeping ", key, "::",GlobalSettings::access_level(key),": default<", _defaults.at(key).get(), "> != assigned<", copy.at(key).get(),">");
                else
                    Print("Keeping ", key, "::",GlobalSettings::access_level(key),": ", copy.at(key).get());*/
#endif
                
                continue;
            }
            
            copy.erase(key);
        }
#ifndef NDEBUG
        Print("Maintaining: ", copy.keys());
#endif
        return copy;
    }
    
    bool load_video_settings(const file::PathArray& source, bool keep_selected_source = true);
    
    void update_video_source() {
        std::unique_lock guard{_video_source_mutex, std::try_to_lock};
        if(not guard)
            return;
        if(not check_new_video_source.valid()) {
            if(not _video_source_changed.exchange(false))
                return;

            current_path.reset();
            auto source = READ_SETTING(source, file::PathArray);
            check_new_video_source = std::async(std::launch::async, &Data::check_video_source,
                                                this, std::move(source));
            return;
        }
        if(not cmn::is_future_ready(check_new_video_source))
            return;

        invalidate_source_paths();
        try {
            auto checked = check_new_video_source.get();
            // The selection can change while the worker is opening the video.
            // in this case just wait for next iteration to restart the update.
            if(_video_source_changed)
                return;
            if(not checked) {
                SceneManager::set_switching_error(checked.error());
                return;
            }
            auto source = std::move(*checked);
            if(source != READ_SETTING(source, file::PathArray))
                return;

            // otherwise we can go on committing the results...
            // unless we received empty?
            if(not source.has_source())
                return;
            
            if(source != _loaded_source) {
                if(not load_video_settings(source))
                    return;
            }
            
            current_path = std::move(source);
            
        } catch(const std::exception& e) {
            if(not callback)
                register_callbacks();
            
            FormatWarning("Cannot load the selected video source: ", e.what());
        }
    }

    void draw(DrawStructure& graph) {
        using namespace dyn;
        update_video_source();
        if(source_paths_timer.elapsed() > 1)
            invalidate_source_paths();
        // Model inspection may initialize Python and must run outside settings callbacks.
        if(_detection_models_changed.exchange(false))
            detection_models_updated();

        if(not dynGUI) {
            dynGUI = DynamicGUI{
                .gui = SceneManager::getInstance().gui_task_queue(),
                .path = layout_name,
                .context = {
                    ActionFunc("set", [](Action action) {
                        REQUIRE_EXACTLY(2, action);
                        
                        auto parm = Meta::fromStr<std::string>(action.first());
                        if(not GlobalSettings::has_value(parm))
                            throw InvalidArgumentException("No parameter ",parm," in global settings.");
                        
                        auto value = action.last();
                        GlobalSettings::write([&](Configuration& config) {
                            config.values[parm].get().set_value_from_string(value);
                        });
                    }),
                    ActionFunc("reset_settings", [](auto){
                        SceneManager::enqueue([](auto, DrawStructure& graph) {
                            graph.dialog([](Dialog::Result result) mutable {
                                if(result == Dialog::Result::OKAY) {
                                    track::detect::TemporaryClassNames loading;
                                    /// resets settings that come from the recentitems
                                    /// config array:
                                    sprite::Map cleared;
                                    
                                    SETTING(filename).copy_to(cleared);
                                    SETTING(source).copy_to(cleared);
                                    SETTING(output_prefix).copy_to(cleared);
                                    SETTING(output_dir).copy_to(cleared);
                                    SETTING(settings_file).copy_to(cleared);
                                    SETTING(meta_source_path).copy_to(cleared);
                                    SETTING(detect_type).copy_to(cleared);
                                    
                                    settings::reset(cleared);
                                    
                                    //settings::load(READ_SETTING(source).value<file::PathArray>(), SETTING(filename, file::Path), default_config::TRexTask_t::none, SETTING(detect_type), {}, cleared);
                                }
                            }, "This will reset all settings you have made here to defaults, except for the file, output prefix and folder.\n\nAre you sure you want to do this?", "Reset settings", "Reset", "Cancel");
                            
                        });
                    }),
                    ActionFunc("go-back", [this](auto){
                        if(_last_layouts.empty()) {
                            SceneManager::enqueue(SceneManager::AlwaysAsync{}, []() {
                                SceneManager::getInstance().set_active("starting-scene");
                            });
                            return;
                        }
                        
                        layout_name = _last_layouts.top();
                        last_opened_tab.set(layout_name);
                        _last_layouts.pop();
                        
                        SceneManager::enqueue(SceneManager::AlwaysAsync{}, [this]() {
                            dynGUI.clear();
                            dynGUI = {};
                        });
                    }),
                    ActionFunc("convert", [this](auto){
                        std::unique_lock guard{_video_source_mutex, std::try_to_lock};
                        if(not guard || not source_ready() || not detection_model_ready())
                        {
                            FormatWarning("Ignoring conversion request while the selected detection model is not ready.");
                            return;
                        }

                        auto output_map = GlobalSettings::mutable_snapshot();
                        auto input = current_conversion_source(&output_map->config.values);
                        if(input.empty()) {
                            FormatWarning("Cannot convert because the original input video could not be found.");
                            return;
                        }

                        DebugHeader("Converting ", utils::ShortenText(current_path->toStr(), 100));
                        
                        auto f = WorkProgress::add_queue("", [this, source = *current_path, input = std::move(input), output_map = std::move(output_map)]() mutable {
                            if(_video_source_changed || source != READ_SETTING(source, file::PathArray))
                                return;
                            auto copy = get_changed_props(*output_map);
                            
                            auto filename = output_map->config.at("filename").value<file::Filename>();

                            /// if we determine that we are actually reconverting a pv file
                            /// we need to load the source settings from the pv file, since we
                            /// cant use the pv file and convert it to itself:
                            if(input != source && input != file::DataLocation::parse("input", source, &output_map->config.values)) {
                                auto parent = file::find_parent(source);

                                if(parent && copy.at("output_dir").value<file::Path>().empty()) {
                                    auto path = file::DataLocation::parse("input", *parent, &output_map->config.values);
                                    if(path.is_folder())
                                        copy["output_dir"] = path;
                                }

                                if(filename.path().empty()) {
                                    const auto name = settings::choose_output_name(output_map->config.values);
                                    filename = file::Filename(name.path().remove_extension());
                                }
                            }
                            source = std::move(input);
                            
                            settings::load(settings::LoadContext{
                                .source = source,
                                .filename = filename.path(),
                                .task = default_config::TRexTask_t::convert,
                                .type = output_map->config.at("detect_type").value<track::detect::ObjectDetectionType_t>(),
                                .source_map = std::move(copy),
                                .output_map = output_map
                            });
                            
                            SceneManager::enqueue([this, output_map] (auto,DrawStructure& graph) {
                                using namespace track;
                                const auto& config = output_map->config;
                                if(auto detect_type = config.at("detect_type").value<detect::ObjectDetectionType_t>();
                                   detect_type && *detect_type == detect::ObjectDetectionType::yolo)
                                {
                                    auto path = detect::yolo::current_detect_model_path(&config.values);
                                    if(not detect::yolo::valid_model(path)) {
                                        graph.dialog([this](Dialog::Result) {
                                            if (layout_name != "settings_layout.json") {
                                                _last_layouts.push(layout_name);
                                                layout_name = "settings_layout.json";
                                                last_opened_tab.set(layout_name);

                                                SceneManager::enqueue(SceneManager::AlwaysAsync{}, [this]() {
                                                    dynGUI.clear();
                                                    dynGUI = {};
                                                });
                                            }
                                        }, "The model file <c><cyan>"+path.str() + "</cyan></c> does not seem to exist and is not a default YOLO model name. Please choose a valid saved model (<c><cyan>.pt</cyan></c> or <c><cyan>.pth</cyan></c>).", "Invalid model", "Okay");
                                        return;
                                    }
                                }

                                auto open_file = [](const std::shared_ptr<GlobalSettings::MutableSnapshot>& output_map, bool start_over) {
                                    auto rollback = GlobalSettings::read([](const Configuration& config) {
                                        return std::shared_ptr<GlobalSettings::Snapshot>(
                                            new GlobalSettings::Snapshot(config),
                                            [](GlobalSettings::Snapshot* snapshot) {
                                                const SettingsRestoreGuard guard;
                                                delete snapshot;
                                            });
                                    });
                                    SceneManager::getInstance().set_active("convert-scene", [output_map, start_over]() {
                                        output_map->apply();
                                        if(start_over)
                                            ConvertScene::force_start_over.set(true);
                                    }, [rollback]() {
                                        rollback->commit();
                                    });
                                };
                                const auto name = settings::choose_output_name(config.values);
                                const auto filename = file::DataLocation::parse("output", name.path(), &config.values);
                                
                                if(filename.exists()) {
                                    graph.dialog([open_file, pending_map = output_map](Dialog::Result result) mutable {
                                        auto pending = std::move(pending_map);
                                        if(pending && (result == Dialog::Result::OKAY || result == Dialog::Result::SECOND))
                                            open_file(pending, result == Dialog::Result::OKAY);
                                    }, "Starting the conversion would overwrite <cyan><c>"+filename.str()+"</c></cyan>, which already exists. Are you sure?", "Overwrite file", "Start over (overwrite)", "Cancel", "Continue (if possible)");
                                } else {
                                    open_file(output_map, false);
                                }
                            });
                        });
                        guard.unlock();
                        if(cmn::is_future_ready<100ms>(f)) {
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
                        last_opened_tab.set(layout_name);
                        
                        SceneManager::enqueue(SceneManager::AlwaysAsync{}, [this]() {
                            dynGUI.clear();
                            dynGUI = {};
                        });
                    }),
                    ActionFunc("track", [this](auto){
                        std::unique_lock guard{_video_source_mutex, std::try_to_lock};
                        if(not guard || checks_running())
                            return;
                        auto output_map = GlobalSettings::mutable_snapshot();
                        const auto& config = output_map->config;
                        auto array = config.at("source").value<file::PathArray>();
                        existing_pv_path.reset();
                        if(existing_pv_file().empty())
                            return;
                        WorkProgress::add_queue("loading...", [this, array = std::move(array), output_map = std::move(output_map)]() mutable
                        {
                            if(_video_source_changed || array != READ_SETTING(source, file::PathArray))
                                return;
                            auto copy = get_changed_props(*output_map);
                            
                            /*copy.register_callbacks({
                                "detect_format",
                                "detect_skeleton"
                            }, [](auto key){
                                if(key == "detect_format")
                                    Print("Changed detect_format");
                                else if(key == "detect_skeleton")
                                    Print("Changed detect_skeleton");
                            });*/
                            
                            if(copy.has("detect_type")) {
                                /// we do not allow changing the detect type when tracking
                                /// a file, so the detection type will be loaded from the
                                /// original file that we are opening.
                                copy.erase("detect_type");
                            }
                            
                            //Print("changed props = ", copy.keys());
                            
                            settings::load(settings::LoadContext{
                                .source = array,
                                .filename = output_map->config.at("filename").value<file::Filename>().path(),
                                .task = default_config::TRexTask_t::track,
                                .type = {},
                                .source_map = std::move(copy),
                                .quiet = true,
                                .output_map = output_map
                            });
                            
                            output_map->config.values["source"] = array;
                            auto open_file = [output_map](){
                                SceneManager::enqueue(SceneManager::AlwaysAsync{}, [output_map](){
                                    SceneManager::getInstance().set_active("tracking-scene", [output_map]() {
                                        output_map->apply();
                                    });
                                });
                            };
                            
                            auto fname = Output::TrackingResults::expected_filename(output_map->config.values);
                            if(fname.exists()) {
                                SceneManager::enqueue([open_file, fname, output_map](auto,DrawStructure& graph) {
                                    graph.dialog([open_file, output_map](Dialog::Result result) mutable {
                                        if(is_in(result, Dialog::Result::OKAY, Dialog::Result::SECOND)) {
                                            const auto source = output_map->config.at("source").value<file::PathArray>();
                                            DebugHeader("Tracking ", no_quotes(utils::ShortenText(source.toStr(), 1000)));
                                        }
                                        if(result == Dialog::Result::OKAY) {
                                            /// load results
                                            TrackingScene::request_load();
                                            open_file();
                                        } else if(result == Dialog::Result::SECOND) {
                                            open_file();
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
                            auto folder = action.parameters.size() > 1 ? action.parameters.at(1) : std::string{};
                            if(not file::Path{folder}.is_folder())
                                folder = {};
                            
                            auto dir = pfd::select_folder("Select a folder", folder).result();
                            set_global_setting_from_path_string(parm, dir);
                            std::cout << "Selected "<< parm <<": " << dir << "\n";
                        });
                    }),
                    ActionFunc("choose-file", [this](const Action& action) {
                        REQUIRE_AT_LEAST(1, action);
                        WorkProgress::add_queue("Selecting file", [this, action](){
                            auto parm = action.parameters.front();
                            auto folder = action.parameters.size() > 1 ? action.parameters.at(1) : std::string{};
                            if(not file::Path{folder}.is_folder())
                                folder = {};
                            
                            std::vector<std::string> filters;
                            if(action.parameters.size() > 2) {
                                filters.insert(filters.end(), action.parameters.begin() + 2, action.parameters.end());
                            }
                            
                            auto flags = GlobalSettings::get(parm).is_type<file::PathArray>()
                                && parm != "detect_precomputed_file" ? pfd::opt::multiselect : pfd::opt::none;
                            auto dir = pfd::open_file("Select a file", folder, filters, flags).result();
                            
                            if(GlobalSettings::get(parm).is_type<file::PathArray>())
                            {
                                if(not dir.empty()) {
                                    std::unique_lock guard{_video_source_mutex};
                                    GlobalSettings::write([&](Configuration& config) {
                                        auto& property = config.values[parm].toProperty<file::PathArray>();
                                        // Selecting the current file also supersedes a draft in its editor.
                                        property.assign_if(file::PathArray(Meta::toStr(dir)));
                                        property.triggerCallbacks();
                                    });
                                }
                            } else {
                                if(not dir.empty()) {
                                    set_global_setting_from_path_string(parm, dir.front());
                                }
                            }
                        });
                    }),
                    ActionFunc("choose-settings", [this](const Action& action) {
                        WorkProgress::add_queue("Selecting file", [this, action](){
                            auto folder = action.parameters.size() > 0 ? action.parameters.at(0) : std::string{};
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
                                std::unique_lock guard{_video_source_mutex};
                                if(path.has_extension() && utils::lowercase(path.extension()) == "pv") {
                                    _video_source_changed = true;
                                    load_video_settings(file::PathArray{path}, false);
                                    
                                } else {
                                    path = file::DataLocation::parse("input", path);
                                    GlobalSettings::load_from_file(path.str(), {
                                        .deprecations = default_config::deprecations(),
                                        .access = AccessLevelType::LOAD
                                    });
                                    //GlobalSettings::load_from_file({}, path.str(), AccessLevelType::LOAD);
                                }
                            }
                        });
                    }),
                    ActionFunc("reload_selected_source", [this](auto){
                        auto source = GlobalSettings::read_value<file::PathArray>("source");
                        if(source) {
                            WorkProgress::add_queue("loading...", [this, source = *source] {
                                std::unique_lock guard{_video_source_mutex};
                                if(source == READ_SETTING(source, file::PathArray)) {
                                    _video_source_changed = true;
                                    load_video_settings(source);
                                }
                            });
                        } else
                            FormatWarning("No source selected for reload_selected_source.");
                    }),
                    ActionFunc("toggle-background-subtraction", [](auto){
                        SETTING(track_background_subtraction) = not BOOL_SETTING(track_background_subtraction);
                    }),
                    VarFunc("video_file", [this](const VarProps&) -> file::PathArray {
                        if(not current_path)
                            return {};
                        if(settings::is_camera_source(current_path->source()))
                            return *current_path;
                        return file::DataLocation::parse("input", *current_path);
                    }),
                    VarFunc("current_conversion_source", [this](const VarProps&) -> file::PathArray {
                        if(not conversion_input)
                            conversion_input = current_conversion_source();
                        return *conversion_input;
                    }),
                    VarFunc("selected_source_exists", [this](const VarProps&) -> bool {
                        return current_path && current_path->has_single_match("pv");
                    }),
                    VarFunc("basler_support_compiled", [](const VarProps&) -> bool {
#if WITH_PYLON
                        return true;
#else
                        return false;
#endif
                    }),
                    VarFunc("settings_summary", [](const VarProps&) -> std::string {
                        static Timer last_update;
                        static std::string last_update_text;
                        
                        if(last_update.elapsed() > 1
                           || last_update_text.empty())
                        {
                            auto delta = default_config::generate_delta_config(AccessLevelType::PUBLIC);
                            std::stringstream ss;
                            for(auto& [key, prop] : delta.map) {
                                if(utils::beginsWith(key, "gui_")
                                   || utils::beginsWith(key, "cam_")
                                   || utils::beginsWith(key, "export_")
                                   || utils::beginsWith(key, "gpu_")
                                   //|| utils::beginsWith(key, "manual_")
                                   || utils::beginsWith(key, "heatmap_"))
                                {
                                    continue;
                                }
                                ss << "<key>" << key << "</key>: <c>" << utils::ShortenText(prop->valueString(), 100, 0.5, "<gray>…</gray>") << "</c>\n";
                            }
                            
                            last_update.reset();
                            last_update_text = ss.str();
                        }
                        
                        return last_update_text;
                    }),
                    VarFunc("model_search_path", [](const VarProps&) -> std::vector<file::Path> {
                        return detect::yolo::detect_model_candidates(READ_SETTING_WITH_DEFAULT(detect_model, file::Path{}));
                    }),
                    VarFunc("window_size", [](const VarProps&) -> Vec2 {
                        return FindCoord::get().screen_size();
                    }),
                    VarFunc("previous_stack_size", [this](const VarProps&) -> size_t {
                        return _last_layouts.size();
                    }),
                    VarFunc("video_size", [this](const VarProps&) -> Vec2 {
                        auto size = _video_size.load();
                        return size.empty() ? Size2(1) : size;
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
                        
                        if(cmn::is_future_ready<15ms>(_scheduled_exist_checks.at(path)))
                        {
                            /// limit size of the map
                            while(_done_exist_checks.size() > 25) {
                                _done_exist_checks.erase(_done_exist_checks.begin());
                            }
                            
                            _done_exist_checks[path] = _scheduled_exist_checks.at(path).get();
                            _scheduled_exist_checks.erase(path);
                            return _done_exist_checks.at(path);
                        }
                        
                        return false;
                    }),
                    VarFunc("resulting_path", [this](const VarProps&) -> file::Path {
                        if(not last_output_name) {
                            last_output_name = GlobalSettings::read([](const Configuration& config) {
                                const auto name = settings::choose_output_name(config.values);
                                if(name.path().empty())
                                    return file::Path{};
                                return file::DataLocation::parse("output", name.path(), &config.values).remove_extension();
                            });
                        }
                        return last_output_name.value();
                    }),
                    VarFunc("checks_running", [this](const VarProps&) -> bool {
                        return checks_running();
                    }),
                    VarFunc("valid_detection_model", [this](const VarProps&) -> bool {
                        return detection_model_ready();
                    }),
                    VarFunc("valid_conversion_source", [this](const VarProps&) -> bool {
                        if(not source_ready())
                            return false;
                        if(not conversion_input)
                            conversion_input = current_conversion_source();
                        return not conversion_input->empty();
                    }),
                    VarFunc("valid_pv_file", [this](const VarProps&) -> bool {
                        return not existing_pv_file().empty();
                    }),
                    VarFunc("existing_pv_file", [this](const VarProps&) -> file::Path {
                        return existing_pv_file();
                    }),
                    VarFunc("model_detect_format", [this](const VarProps&) {
                        const auto config = model_config(track::detect::ModelTaskType::detect);
                        return track::detect::ObjectDetectionFormat_t{
                            config ? config->output_format : track::detect::ObjectDetectionFormat_t::none
                        };
                    }),
                    VarFunc("model_detect_resolution", [this](const VarProps&) {
                        const auto config = model_config(track::detect::ModelTaskType::detect);
                        return config ? config->trained_resolution : track::detect::DetectResolution{};
                    }),
                    VarFunc("model_region_resolution", [this](const VarProps&) {
                        const auto config = model_config(track::detect::ModelTaskType::region);
                        return config ? config->trained_resolution : track::detect::DetectResolution{};
                    }),
                    VarFunc("season", [](const VarProps&) {
                        return GlobalSettings::currentSeason().toStr();
                    })
                }
            };
            
            dynGUI.context.custom_elements["video"] = std::unique_ptr<GUIVideoAdapterElement>{
                new GUIVideoAdapterElement(_window, []() {
                    return FindCoord::get().screen_size();
                }, [this](VideoInfo info) {
                    _video_size = info.resolution;
                }, [this](const file::PathArray& path, IMGUIBase* window, std::function<void(VideoInfo)> callback) {
                    if(_video_adapters.contains(path.source())) {
                        return _video_adapters[path.source()];
                    } else {
                        Layout::Ptr ptr = Layout::Make<GUIVideoAdapter>{path, window, callback};
                        //Print("Making new video adapter for ", path);
                        if(_video_adapters.size() >= 2) {
                            Print("Clearing video adapter history...");
                            _video_adapters.clear();
                        }
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
        
        //graph.wrap_object(_preview_image);
        dynGUI.update(graph, nullptr/*, [this](auto &objs){
            objs.push_back(Layout::Ptr(_preview_image));
        }*/);
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

SettingsScene::Data::CheckedSource SettingsScene::Data::check_video_source(file::PathArray source) {
    update_running_tasks(true);
    if(not source.has_source() || settings::is_camera_source(source.source()))
        return source;

    auto lookup = GlobalSettings::read([](const Configuration& config) { return config.values; });
    const auto input = file::DataLocation::parse("input", source, &lookup);
    lookup["source"] = input;
    if(auto error = settings::video_source_error(input))
        return std::unexpected(*error);

    if(input.has_single_match("pv")) {
        pv::File video(input[0]);
        video.header();
    } else {
        // An existing PV can still be tracked when its original video has been moved.
        if(input.has_single_match() && not input[0].is_regular())
        {
            const auto name = settings::choose_output_name(lookup);
            if(file::DataLocation::parse("output", name.path(), &lookup).is_regular())
                return source;
        }
        
        if(input.empty())
            return {};
        
        VideoSource v(input);
        Print("VideoSource for ", no_quotes(utils::ShortenText(source.toStr(), 1000)), " of size ", v.size(), ".");
    }
    
    return source;
}

bool SettingsScene::Data::load_video_settings(const file::PathArray& source, bool keep_selected_source) {
    bool loaded = true;
    track::detect::TemporaryClassNames loading;

    if(callback)
        GlobalSettings::unregister_callbacks(std::move(callback));
    
    try {
        const auto snapshot = GlobalSettings::mutable_snapshot();
        auto output_map = keep_selected_source ? snapshot : nullptr;
        const auto& config = snapshot->config;
        // A new source selection derives its filename from the selected input.
        const auto filename = not keep_selected_source || source == _loaded_source
            ? config.at("filename").value<file::Filename>().path()
            : file::Path{};
        sprite::Map promote;
        copy_file_locations(promote, config.values);
        
        const auto selected = settings::is_camera_source(source.source())
            ? source : file::DataLocation::parse("input", source, &config.values);
        file::Path source_path;
        if(selected.has_single_match("pv")) {
            source_path = selected[0];
        }

        if(selected.has_source() && not selected.all_files_are("pv")) {
            settings::load(settings::LoadContext{
                .source = source,
                .filename = filename,
                .task = default_config::TRexTask_t::convert,
                .type = {},
                .source_map = std::move(promote),
                .output_map = output_map
            });
            
        } else if(source_path.is_regular()) {
            pv::File file(source_path);
            auto& str = file.header().metadata;
            
            sprite::Map map;
            try {
                if(str.has_value()) {
                    sprite::parse_values(sprite::MapSource{ source_path }, map, str.value(), &config.defaults, {"filename"}, default_config::deprecations());
                }
            }
            catch (...) {
                /// do nothing
            }
            
            if(map.has("source"))
                map.at("source").get().copy_to(promote);
            
            track::detect::ObjectDetectionType_t detect_type;
            if(map.has("detect_type")) {
                map.at("detect_type").get().copy_to(promote);
                detect_type = map.at("detect_type").value<track::detect::ObjectDetectionType_t>();
            }
            promote["meta_encoding"] = file.header().encoding;
            
            settings::load(settings::LoadContext{
                .source = source,
                .filename = filename,
                .task = default_config::TRexTask_t::none,
                .type = detect_type,
                .source_map = promote,
                .output_map = output_map
            });
            
        } else if(source.has_source())
            loaded = false;

        if(output_map && loaded && source.has_source()) {
            GlobalSettings::write([&](Configuration& config) {
                if(config.at("source").value<file::PathArray>() == source) {
                    output_map->config.values["source"] = source;
                    output_map->apply();
                }
            });
        }
        
    } catch(const std::exception& ex) {
        FormatWarning("Cannot load video settings: ", ex.what());
        loaded = false;
    }

    register_callbacks();
    if(source != READ_SETTING(source, file::PathArray)) {
        _video_source_changed = true;
        return false;
    }
    
    if(loaded) {
        _loaded_source = source;
    }
    return loaded;
}

void SettingsScene::activate() {
    WorkProgress::instance().start();
    
    _data = std::make_unique<Data>();
    _data->_window = (IMGUIBase*)window();

    _data->_loaded_source = READ_SETTING(source, file::PathArray);
    _data->register_callbacks();
    _data->layout_name = last_opened_tab.get();
}

void SettingsScene::deactivate() {
    // Logic to clear or save state if needed
    //RecentItems::set_select_callback(nullptr);
    //_data->dynGUI.clear();
    WorkProgress::stop();

    if(_data) {
        //Print("_data is set, need to unregister callbacks...");
        if(_data->callback)
            GlobalSettings::unregister_callbacks(std::move(_data->callback));
        
        //Print("Clearing _data->dynGUI");
        _data->dynGUI.clear();

        SceneManager::getInstance().update_queue();
        
        //Print("Deleting _data...");
        _data = nullptr;

        //Print("Done.");
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
