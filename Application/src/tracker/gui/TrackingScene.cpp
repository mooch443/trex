#include "TrackingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
#include <nlohmann/json.hpp>
#include <misc/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>
#include <tracking/Tracker.h>
#include <misc/Timer.h>
#include <tracking/PPFrame.h>
#include <tracking/IndividualManager.h>
#include <file/DataLocation.h>
#include <grabber/misc/default_config.h>
#include <misc/OutputLibrary.h>
#include <tracking/Categorize.h>
#include <gui/CheckUpdates.h>

using namespace track;

namespace gui {

static constexpr Frame_t cache_size{Frame_t::number_t(10)};

TrackingScene::Data::Data(Image::Ptr&& average, pv::File&& video, std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>>&& stages)
:
video(std::move(video)),
tracker(std::move(average), video),
analysis(std::move(stages)),
pool(4u, "preprocess_main")

{ }

TrackingScene::TrackingScene(Base& window)
: Scene(window, "tracking-scene", [this](auto&, DrawStructure& graph){ _draw(graph); })
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    print("window dimensions", window.window_dimensions().mul(dpi));
}

bool TrackingScene::stage_0(ConnectedTasks::Type && ptr) {
    auto idx = ptr->index();
    auto range = _data->tracker.analysis_range();
    if(!range.contains(idx) && idx != range.end() && (not Tracker::end_frame().valid() || idx > Tracker::end_frame())) {
        std::unique_lock lock(_task_mutex);
        _data->unused.emplace(std::move(ptr));
        return false;
    }

    if(not range.contains(idx))
        throw U_EXCEPTION("Outside of analysis range: ", idx, " vs. ", range);

    Timer timer;
    pv::Frame frame;
    _data->video.read_frame(frame, idx);
    Tracker::preprocess_frame(std::move(frame), *ptr, _data->pool.num_threads() > 1 ? &_data->pool : NULL, PPFrame::NeedGrid::NoNeed, _data->video.header().resolution, false);

    ptr->set_loading_time(timer.elapsed());

    // clear stored blob data, so that the destructor is called
    // in a different thread (balancing) if they arent needed.
    IndividualManager::clear_pixels();
    return true;
}

bool TrackingScene::stage_1(ConnectedTasks::Type && ptr) {
    static Timer fps_timer;
    static Image empty(0, 0, 0);

    Timer timer;

    static Timing all_processing("Analysis::process()", 50);
    TakeTiming all(all_processing);

    LockGuard guard(w_t{}, "Analysis::process()");
    if(SETTING(terminate))
        return false;
    
    auto range = Tracker::analysis_range();

    auto idx = ptr->index();
    if (idx >= range.start()
        && max(range.start(), _data->tracker.end_frame().valid() ? (_data->tracker.end_frame() + 1_f) : 0_f) == idx
        && _data->tracker.properties(idx) == nullptr
        && idx <= Tracker::analysis_range().end())
    {
        _data->tracker.add(*ptr);

        static Timing after_track("Analysis::after_track", 10);
        TakeTiming after_trackt(after_track);
        
        if(idx + 1_f == _data->video.length())
            _data->please_stop_analysis = true;

        /*{
            std::lock_guard lock(data_mutex);
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
                
                if(FAST_SETTING(analysis_range).first != -1 || FAST_SETTING(analysis_range).second != -1)
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

        frames_count++;*/
    }

    static Timing procpush("Analysis::process::unused.push", 10);
    TakeTiming ppush(procpush);
    std::unique_lock lock(_task_mutex);
    _data->unused.emplace(std::move(ptr));

    return true;
}

void TrackingScene::init_video() {
    SettingsMaps combined;
    
    const auto set_combined_access_level = [&combined](auto& name, AccessLevel level) {
        combined.access_levels[name] = level;
    };
    
    combined.map.set_do_print(false);
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    //default_config::get(combined.map, combined.docs, set_combined_access_level);
    
    std::vector<std::string> save = combined.map.has("meta_write_these") ? combined.map.get<std::vector<std::string>>("meta_write_these").value() : std::vector<std::string>{};
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
    
    GlobalSettings::map().set_do_print(true);
    default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    GlobalSettings::map().dont_print("gui_frame");
    GlobalSettings::map().dont_print("gui_focus_group");
    
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
    
    bool executed_a_settings{false};
    pv::File video(file::DataLocation::parse("input", SETTING(source).value<file::PathArray>().source()), pv::FileMode::READ);
    
    if(video.header().version <= pv::Version::V_2) {
        SETTING(crop_offsets) = CropOffsets();
        
        file::Path settings_file = file::DataLocation::parse("settings");
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
            sprite::parse_values(GlobalSettings::map(), video.header().metadata, &combined);
    } catch(const UtilsException& e) {
        // dont do anything, has been printed already
    }
    
    Image::Ptr average = Image::Make(video.average());
    SETTING(video_size) = Size2(average->cols, average->rows);
    SETTING(video_mask) = video.has_mask();
    SETTING(video_length) = uint64_t(video.length().get());
    SETTING(video_info) = std::string(video.get_info());
    
    if(SETTING(frame_rate).value<uint32_t>() <= 0) {
        FormatWarning("frame_rate == 0, calculating from frame tdeltas.");
        video.generate_average_tdelta();
        SETTING(frame_rate) = (uint32_t)max(1, int(video.framerate()));
    }
    
    Output::Library::InitVariables();
    
    auto settings_file = file::DataLocation::parse("settings");
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
    cmd.load_settings(&combined);
    
    SETTING(gui_interface_scale) = float(1);
    
    //! Stages
    _data = std::make_unique<Data>(
       std::move(average),
       std::move(video),
       std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>>
       {
            [this](ConnectedTasks::Type&& ptr, auto&) -> bool {
                return stage_0(std::move(ptr));
            },
            [this](ConnectedTasks::Type&& ptr, auto&) -> bool {
                return stage_1(std::move(ptr));
            }
       }
    );
}

void TrackingScene::activate() {
    using namespace dyn;
    for(size_t i=0; i<1000; ++i) {
        sprite::Map map;
        map.set_do_print(false);
        map["i"] = i;
        map["pos"] = Vec2(100, 100 + i * 10);
        map["name"] = std::string("Text");
        map["detail"] = std::string("detail");
        map["radius"] = float(rand()) / float(RAND_MAX) * 50 + 50;
        map["angle"] = float(rand()) / float(RAND_MAX) * RADIANS(180);
        map["acceleration"] = Vec2();
        map["velocity"] = Vec2();
        map["angular_velocity"] = float(0);
        map["mass"] = float(rand()) / float(RAND_MAX) * 1 + 0.1f;
        map["inertia"] = float(rand()) / float(RAND_MAX) * 1 + 0.1f;
        map.set_do_print(false);
        _fish_data.emplace_back(std::move(map));
        
        _individuals.emplace_back(new Variable([i, this](VarProps) -> sprite::Map& {
            return _fish_data.at(i);
        }));
    }
    
    init_video();
    
    for (auto i=0_f; i<cache_size; ++i)
        _data->unused.emplace(std::make_unique<PPFrame>(_data->tracker.average().bounds().size()));
    
    _data->analysis.start(// main thread
        [this, &analysis = _data->analysis, &please_stop_analysis = _data->please_stop_analysis, &currentID = _data->currentID, &tracker = _data->tracker, &video = _data->video]() {
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
            
            if(FAST_SETTING(analysis_range).second != -1
               && endframe >= Frame_t(sign_cast<Frame_t::number_t>(FAST_SETTING(analysis_range).second))
               && !SETTING(terminate)
               && !please_stop_analysis)
            {
                please_stop_analysis = true;
            }
            
            while(not currentID.load().valid()
                  || (currentID.load() < max(range.start(), endframe) + cache_size
                      && currentID.load() + 1_f < video.length()))
            {
                std::scoped_lock lock(_task_mutex);
                if(_data->unused.empty())
                    break;
                
                auto ptr = std::move(_data->unused.front());
                _data->unused.pop();
                
                if(not currentID.load().valid())
                    currentID = range.start();
                else
                    currentID = currentID.load() + 1_f;
                ptr->set_index(currentID.load());
                
                analysis.add(std::move(ptr));
            }
        }
    );
}

void TrackingScene::deactivate() {
    dynGUI.clear();
    
    print("Preparing for shutdown...");
#if !COMMONS_NO_PYTHON
    CheckUpdates::cleanup();
    Categorize::terminate();
#endif
    _data->analysis.terminate();
    _data = nullptr;
}

void TrackingScene::_draw(DrawStructure& graph) {
    static Timer timer;
    auto dt = saturate(timer.elapsed(), 0.001, 0.1);
    
    using namespace dyn;
    if(not dynGUI)
        dynGUI = {
            .path = "tracking_layout.json",
            .graph = &graph,
            .context = {
                ActionFunc("set", [](Action action) {
                    if(action.parameters.size() != 2)
                        throw InvalidArgumentException("Invalid number of arguments for action: ",action);
                    
                    auto parm = Meta::fromStr<std::string>(action.parameters.front());
                    if(not GlobalSettings::has(parm))
                        throw InvalidArgumentException("No parameter ",parm," in global settings.");
                    
                    auto value = action.parameters.back();
                    GlobalSettings::get(parm).get().set_value_from_string(value);
                }),
                ActionFunc("change_scene", [](Action action) {
                    if(action.parameters.empty())
                        throw U_EXCEPTION("Invalid arguments for ", action, ".");

                    auto scene = Meta::fromStr<std::string>(action.parameters.front());
                    if(not SceneManager::getInstance().is_scene_registered(scene))
                        return false;
                    SceneManager::getInstance().set_active(scene);
                    return true;
                }),
                
                VarFunc("window_size", [this](VarProps) -> Vec2 {
                    return window_size;
                }),
                
                VarFunc("fishes", [this](VarProps)
                    -> std::vector<std::shared_ptr<VarBase_t>>&
                {
                    return _individuals;
                }),
                
                VarFunc("tracker", [this](VarProps) -> auto& {
                    static sprite::Map map = [](){
                        sprite::Map map;
                        map.set_do_print(false);
                        return map;
                    }();
                    static Range<Frame_t> last;
                    Range<Frame_t> current{ _data->tracker.start_frame(), _data->tracker.end_frame() };
                    if(current != last) {
                        map["range"] = current;
                        last = current;
                    }
                    return map;
                })
            }
        };
    
    //auto dpi = ((const IMGUIBase*)window())->dpi_scale();
    auto max_w = window()->window_dimensions().width * 0.65;
    window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height);
    element_size = Size2((window()->window_dimensions().width - max_w - 50), window()->window_dimensions().height - 25 - 50);
    
    dynGUI.update(nullptr);
    
    const float target_angular_velocity = 0.01f * 2.0f * static_cast<float>(M_PI);
    auto mouse_position = graph.mouse_position();
    float time_factor = 0.5f; // Smaller values make the system more 'inertial', larger values make it more 'responsive'
    const float linear_damping = 0.95f;  // Close to 1: almost no damping, close to 0: strong damping
    const float angular_damping = 0.95f;
    
    /*for(auto &i : _data) {
        auto v = i["pos"].value<Vec2>();
        auto velocity = i["velocity"].value<Vec2>();
        auto radius = i["radius"].value<float>();
        auto angle = i["angle"].value<float>();
        auto angular_velocity = i["angular_velocity"].value<float>();
        auto mass = i["mass"].value<float>();
        auto inertia = i["inertia"].value<float>();
        
        // Apply damping
        velocity *= linear_damping;
        angular_velocity *= angular_damping;

        // Calculate new target angle based on mouse position
        float dx = mouse_position.x - v.x;
        float dy = mouse_position.y - v.y;
        float target_angle = atan2(dy, dx);
        
        // Calculate angular direction
        float angular_direction = target_angle - angle;

        // Update angular acceleration, angular_velocity and angle
        float angular_acceleration = angular_direction / inertia;
        angular_velocity += angular_acceleration * dt;
        angle += angular_velocity * dt;

        // Ensure angle is within bounds
        while (angle > 2.0f * static_cast<float>(M_PI)) {
            angle -= 2.0f * static_cast<float>(M_PI);
        }

        // Calculate new target position
        float target_x = mouse_position.x + cos(angle) * radius;
        float target_y = mouse_position.y + sin(angle) * radius;
        Vec2 target(target_x, target_y);

        // Calculate direction and update linear acceleration, velocity and position
        Vec2 direction = target - v;
        Vec2 acceleration = direction / mass;
        velocity += acceleration * dt;
        v += velocity * dt;

        // Update the attributes
        i["pos"] = v;
        i["velocity"] = velocity;
        i["angle"] = angle;
        i["angular_velocity"] = angular_velocity;
    }*/
    timer.reset();
    graph.root().set_dirty();
}

}
