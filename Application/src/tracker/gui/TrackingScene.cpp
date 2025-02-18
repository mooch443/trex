#include "TrackingScene.h"
#include <misc/GlobalSettings.h>
#include <gui/IMGUIBase.h>
#include <gui/DynamicGUI.h>
#include <misc/RecentItems.h>
#include <misc/CommandLine.h>
#include <file/PathArray.h>
#include <tracking/Tracker.h>
//#include <grabber/misc/default_config.h>
#include <tracking/OutputLibrary.h>
#include <ml/Categorize.h>
#include <gui/WorkProgress.h>
#include <gui/DrawBlobView.h>
#include <tracking/Output.h>
#include <gui/dyn/Action.h>
#include <gui/GUICache.h>
#include <gui/AnimatedBackground.h>
#include <misc/Coordinates.h>
#include <gui/ScreenRecorder.h>
#include <gui/Bowl.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/TrackingState.h>
#include <misc/SettingsInitializer.h>
#include <ml/CategorizeInterface.h>
#include <gui/DrawPreviewImage.h>
#include <gui/DrawPosture.h>
#include <misc/SettingsInitializer.h>
#include <tracking/FilterCache.h>
#include <misc/FOI.h>
#include <gui/dyn/ParseText.h>
#include <gui/ParseLayoutTypes.h>
#include <gui/InfoCard.h>
#include <tracking/AutomaticMatches.h>
#include <gui/DrawDataset.h>
#include <gui/DrawExportOptions.h>
#include <misc/PythonWrapper.h>
#include <python/GPURecognition.h>
#include <tracking/MemoryStats.h>
#include <grabber/misc/default_config.h>
#include <gui/GuiSettings.h>
#include <gui/PreviewAdapterElement.h>
#include <gui/DrawUniqueness.h>
#include <gui/TimingStatsElement.h>

using namespace track;

namespace cmn::gui {

std::atomic<bool> _load_requested{false};

struct TrackingScene::Data {
    std::unique_ptr<GUICache> _cache;
    std::unique_ptr<DrawDataset> _dataset;
    std::unique_ptr<DrawExportOptions> _export_options;
    std::unique_ptr<DrawUniqueness> _uniqueness;
    
    /// these will help updating some visual stuff whenever
    /// the tracker has added a new frame:
    std::optional<std::size_t> _frame_callback;
    std::atomic<bool> _tracker_has_added_frames{false};
    
    std::unique_ptr<Bowl> _bowl;
    //std::unordered_map<Idx_t, Bounds> _last_bounds;
    
    std::unique_ptr<AnimatedBackground> _background;
    std::unique_ptr<ExternalImage> _gui_mask;
    
    std::function<void(Vec2, bool, std::string)> _clicked_background;
    double _time_since_last_frame{0};

    sprite::Map _primary_selection;
    sprite::Map _keymap;
    
    struct {
        uint64_t last_change;
        FOI::foi_type::mapped_type changed_frames;
        std::string name;
        Color color;
    } _foi_state;
    
    std::atomic<FrameRange> _analysis_range;
    CallbackCollection _callback;
    Vec2 _last_mouse;
    Vec2 _bowl_mouse;
    bool _zoom_dirty{false};
    //pv::Frame _frame;
    size_t _last_active_individuals{0};
    size_t _last_live_individuals{0};
    
    // The dynamic part of the gui that is live-loaded from file
    dyn::DynamicGUI dynGUI;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _individuals;
    std::vector<sprite::Map> _fish_data;
    
    /// Variables for "consec" VarFunc:
    std::vector<sprite::Map> tracklets;
    std::vector<std::shared_ptr<dyn::VarBase_t>> variables;
    
    ScreenRecorder _recorder;
    
    /**
     * @brief Constructor for the Data struct.
     *
     * Initializes the Data object with provided average image, video, and analysis functions.
     *
     * @param average Pointer to the average image.
     * @param video The video file to be analyzed.
     * @param functions A list of functions representing the analysis stages.
     */
    Data(Image::Ptr&& average,
         pv::File& video);
    
    void redraw_all();
    void handle_zooming(Event);
};

TrackingScene::~TrackingScene() {
    
}

void TrackingScene::request_load() {
    _load_requested = true;
}

const static std::unordered_map<std::string_view, gui::Keyboard::Codes> _key_map {
    {"esc", Keyboard::Escape},
    {"left", Keyboard::Left},
    {"right", Keyboard::Right},
    {"lshift", Keyboard::LShift},
    {"lctrl", Keyboard::LControl},
    {"rctrl", Keyboard::RControl},
    {"system", Keyboard::LSystem},
    {"lalt", Keyboard::LAlt},
    {"ralt", Keyboard::RAlt}
};

TrackingScene::Data::Data(Image::Ptr&& average, pv::File& video)
{
    if(average->dims == 3) {
        auto rgba = Image::Make(average->rows, average->cols, 4);
        cv::cvtColor(average->get(), rgba->get(), cv::COLOR_BGR2BGRA);
        average = std::move(rgba);
    }
    
    _background = std::make_unique<AnimatedBackground>(std::move(average), &video);
    
    _background->add_event_handler(EventType::MBUTTON, [this](Event e){
        if(e.mbutton.pressed) {
            if(_clicked_background)
                _clicked_background(Vec2(e.mbutton.x, e.mbutton.y).map<round>(), e.mbutton.button == 1, "");
            else
                Print("Clicked, but nobody is around.");
        }
    });
    
    /// Deal with zooming
    _background->add_event_handler(EventType::SCROLL, [this](Event e) {
        handle_zooming(e);
    });
    _background->add_event_handler(EventType::DRAG, [this](Event e) {
        Print("Drag: ", e.drag.rx, ",",e.drag.ry);
    });
    /*SceneManager::getInstance().enqueue([](DrawStructure& graph) {
        graph.root().add_event_handler(EventType::SCROLL, [this](Event e) {
            handle_zooming(e);
        });
    });*/
    
    if(video.has_mask()) {
        cv::Mat mask = video.mask().mul(cv::Scalar(255));
        mask.convertTo(mask, CV_8UC1);
        _gui_mask = std::make_unique<ExternalImage>(Image::Make(mask), Vec2(0, 0), Vec2(1), Color(255, 255, 255, 125));
    }

    for (auto& [key, code] : _key_map)
        _keymap[key] = false;
}

void TrackingScene::Data::handle_zooming(Event e) {
    auto video_size = SETTING(meta_video_size).value<Size2>();
    auto mp = _cache->previous_mouse_position;
    mp = FindCoord::get().convert(HUDCoord{mp});
    
    if(SETTING(gui_focus_group).value<std::vector<track::Idx_t>>().empty()) {
        auto gui_zoom_polygon = SETTING(gui_zoom_polygon).value<std::vector<Vec2>>();
        if(gui_zoom_polygon.empty() || gui_zoom_polygon.size() != 4)
        {
            gui_zoom_polygon = {
                Vec2(0_F, 0_F),
                Vec2(video_size.width, 0_F),
                Vec2(video_size.width, video_size.height),
                Vec2(0_F, video_size.height),
            };
            
            // Calculate the scaling factor
            auto scale_factor = e.scroll.dy > 0 ? 0.95_F : 1.05_F;

            // Scale the polygon around the mouse position
            auto new_gui_zoom_polygon = gui_zoom_polygon;
            for (auto& v : new_gui_zoom_polygon) {
                v = mp + (v - mp) * scale_factor;
            }
            
            // Calculate new dimensions
            Size2 dims = new_gui_zoom_polygon.at(2) - new_gui_zoom_polygon.front();

            // Enforce zoom limits
            if (dims.width < video_size.width * 2
                && dims.height < video_size.height * 2
                && dims.width > 10
                && dims.height > 10)
            {
                gui_zoom_polygon = new_gui_zoom_polygon;
            }

            // Apply existing transformations if necessary
            // gui_zoom_polygon = apply_existing_transformations(gui_zoom_polygon);

            //SETTING(gui_zoom_polygon) = gui_zoom_polygon;
            
        }
        
        //else
        {
            assert(gui_zoom_polygon.size() == 4);

            // Calculate the scaling factor
            auto scale_factor = e.scroll.dy > 0 ? 0.95_F : 1.05_F;

            // Scale the polygon around the mouse position
            auto new_gui_zoom_polygon = gui_zoom_polygon;
            for (auto& v : new_gui_zoom_polygon) {
                v = mp + (v - mp) * scale_factor;
            }

            // Calculate new dimensions
            Size2 dims = new_gui_zoom_polygon.at(2) - new_gui_zoom_polygon.front();

            // Enforce zoom limits
            if (dims.width < video_size.width * 2
                && dims.height < video_size.height * 2
                && dims.width > 10
                && dims.height > 10)
            {
                gui_zoom_polygon = new_gui_zoom_polygon;
            }
        }

        GlobalSettings::map().do_print("gui_zoom_polygon", false);
        GlobalSettings::map().do_print("gui_zoom_limit", false);
        SETTING(gui_zoom_polygon) = gui_zoom_polygon;
        
        Size2 dims = gui_zoom_polygon.at(2) - gui_zoom_polygon.front();
        auto zoom_limit = SETTING(gui_zoom_limit).value<Size2>();
        if(zoom_limit.width > dims.width) {
            zoom_limit = dims;
            SETTING(gui_zoom_limit) = zoom_limit;
        }
        GlobalSettings::map().do_print("gui_zoom_polygon", true);
        GlobalSettings::map().do_print("gui_zoom_limit", true);
        
    } else {
        auto zoom_limit = SETTING(gui_zoom_limit).value<Size2>();
        if(e.scroll.dy > 0) {
            zoom_limit *= 0.95_F;
            if(zoom_limit.width > 10) {
                GlobalSettings::map().do_print("gui_zoom_limit", false);
                SETTING(gui_zoom_limit) = zoom_limit;
                GlobalSettings::map().do_print("gui_zoom_limit", true);
            }
            
        } else if(e.scroll.dy < 0) {
            zoom_limit *= 1.05_F;
            if(zoom_limit.width < video_size.width * 2
               && zoom_limit.height < video_size.height * 2)
            {
                GlobalSettings::map().do_print("gui_zoom_limit", false);
                SETTING(gui_zoom_limit) = zoom_limit;
                GlobalSettings::map().do_print("gui_zoom_limit", true);
            }
        }
    }
}

TrackingScene::TrackingScene(Base& window)
: Scene(window, "tracking-scene", [this](auto&, DrawStructure& graph){ _draw(graph); })
{
    auto dpi = ((const IMGUIBase*)&window)->dpi_scale();
    Print("window dimensions", window.window_dimensions().mul(dpi));
}

// Utility function to find the appropriate Idx_t based on a given comparator
template<typename Comparator, typename Set>
Idx_t find_wrapped_id(const Set& ids, track::Idx_t current_id, Comparator comp) {
    if (ids.empty()) {
        return Idx_t(); // Return invalid Idx_t if the set is empty.
    }

    Idx_t result_id;
    for (auto id : ids) {
        // id > current_id && id > result_id
        if (comp(id, current_id) && (not result_id.valid() || comp(result_id, id))) {
            result_id = id;
        }
    }
    
    if(result_id.valid())
        return result_id;

    // Wrap-around logic
    if constexpr(is_set<Set>::value) {
        if constexpr(Comparator{}(track::Idx_t(0), track::Idx_t(1))) {
            // smaller than
            return *ids.rbegin();
        } else {
            // greater than
            return *ids.begin();
        }
    } else {
        if constexpr(Comparator{}(track::Idx_t(0), track::Idx_t(1))) {
            // smaller than
            return *std::max_element(ids.begin(), ids.end());
        } else {
            // greater than
            return *std::min_element(ids.begin(), ids.end());
        }
    }
}

bool TrackingScene::on_global_event(Event event) {
    if(event.type == EventType::MBUTTON) {
        _data->_zoom_dirty = true;
    }
    if(event.type == EventType::WINDOW_RESIZED) {
        _data->_zoom_dirty = true;
    }
    if(event.type == EventType::KEY) {
        if(event.key.code == Keyboard::LShift)
            _data->_zoom_dirty = true;
        
        if(not event.key.pressed)
            return true;
        
        switch (event.key.code) {
            case Keyboard::Escape:
                SceneManager::getInstance().enqueue([](auto, DrawStructure& graph) {
                    graph.dialog([](Dialog::Result result) {
                        if(result == Dialog::Result::OKAY) {
                            SETTING(terminate) = true;
                        }
                    }, "Are you sure you want to exit the application? Any unsaved changes will be discarded.", "Exit", "Quit", "Cancel");
                });
                break;
            case Keyboard::Space:
                SETTING(gui_run) = not SETTING(gui_run).value<bool>();
                break;
            case Keyboard::B:
                SETTING(gui_show_posture) = not SETTING(gui_show_posture).value<bool>();
                break;
            case Keyboard::M:
                //if(_data) {
                //    next_poi(_data->_cache->primary_selected_id());
                //} else
                    next_poi(Idx_t());
                break;
            case Keyboard::N:
                //if(_data) {
                //    prev_poi(_data->_cache->primary_selected_id());
                //} else
                    prev_poi(Idx_t());
                break;
            case Keyboard::C:
                if(_data && _data->_cache && _data->_cache->has_selection())
                    prev_poi(_data->_cache->primary_selected_id());
                else
                    prev_poi(Idx_t());
                break;
            case Keyboard::V:
                if(_data && _data->_cache && _data->_cache->has_selection())
                    next_poi(_data->_cache->primary_selected_id());
                else
                    next_poi(Idx_t());
                break;
            case Keyboard::L:
                _state->load_state(SceneManager::getInstance().gui_task_queue(), Output::TrackingResults::expected_filename());
                break;
            case Keyboard::Z:
                _state->save_state(SceneManager::getInstance().gui_task_queue(), false);
                break;
            case Keyboard::T:
                SETTING(gui_show_timeline) = not SETTING(gui_show_timeline).value<bool>();
                break;
            case Keyboard::Comma:
                WorkProgress::add_queue("Pausing...", [this](){
                    _state->analysis.set_paused(not _state->analysis.paused()).get();
                });
                break;
            case Keyboard::S:
                if(GUI_SETTINGS(gui_show_export_options)) {
                    WorkProgress::add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { _state->_controller->export_tracks(); });
                    SETTING(gui_show_export_options) = false;
                } else {
                    SETTING(gui_show_export_options) = true;
                }
                break;
            case Keyboard::Left:
                set_frame(GUI_SETTINGS(gui_frame).try_sub(1_f));
                break;
            case Keyboard::Right:
                set_frame(GUI_SETTINGS(gui_frame) + 1_f);
                break;
            case Keyboard::P: {
                Idx_t id = _data->_cache->primary_selected_id();
                if (!_data->_cache->active_ids.empty()) {
                    Idx_t next_id = find_wrapped_id(_data->_cache->active_ids, id, std::greater<Idx_t>());
                    if(next_id.valid())
                        SETTING(gui_focus_group) = std::vector<Idx_t>{next_id};
                }
                break;
            }
            case Keyboard::O: {
                Idx_t id = _data->_cache->primary_selected_id();
               if (!_data->_cache->active_ids.empty()) {
                   Idx_t prev_id = find_wrapped_id(_data->_cache->active_ids, id, std::less<Idx_t>());
                   if(prev_id.valid())
                       SETTING(gui_focus_group) = std::vector<Idx_t>{prev_id};
               }
                break;
            }
            case Keyboard::D:
                SETTING(gui_mode) = GUI_SETTINGS(gui_mode) == mode_t::tracking ? mode_t::blobs : mode_t::tracking;
                _data->_cache->set_tracking_dirty();
                _data->_cache->set_blobs_dirty();
                _data->_cache->set_redraw();
                break;
                
            case Keyboard::F: {
                SceneManager::getInstance().enqueue([](IMGUIBase* base, DrawStructure& graph){
                    if(graph.is_key_pressed(Codes::LSystem))
                    {
                        base->toggle_fullscreen(graph);
                    }
                });
                break;
            }
            case Keyboard::F11:
                SceneManager::getInstance().enqueue([](IMGUIBase* base, DrawStructure& graph){
                    base->toggle_fullscreen(graph);
                });
                break;
            case Keyboard::R: {
                if(_data) {
                    SceneManager::getInstance().enqueue([this](IMGUIBase* base, DrawStructure& graph){
                        if(_data->_recorder.recording()) {
                            _data->_recorder.stop_recording(base, &graph);
                            _data->_background->set_strict(false);
                        } else {
                            _data->_background->set_strict(true);
                            _data->_recorder.start_recording(base, GUI_SETTINGS(gui_frame));
                        }
                    });
                }
                break;
            }
            default:
                break;
        }
        return true;
    }
    return false;
}

void TrackingScene::activate() {
    WorkProgress::instance().start();
    
    settings::initialize_filename_for_tracking();
    
    _state = std::make_unique<TrackingState>(SceneManager::getInstance().gui_task_queue());
    
    //! Stages
    _data = std::unique_ptr<Data>{
        new Data{
            Image::Make(_state->video->average()),
            *_state->video
        }
    };
    
    _data->_callback = GlobalSettings::map().register_callbacks({
        "gui_focus_group",
        "gui_run",
        "track_pause",
        "analysis_range",
        "frame_rate",
        
        "cam_matrix",
        "cam_undistort",
        "cam_undistort_vector",
        
        "manual_matches",
        "manual_splits",
        "track_ignore_bdx",
        
        "gui_show_texts",
        "gui_show_probabilities",
        "gui_show_visualfield",
        "gui_show_outline",
        "gui_show_midline",
        "gui_show_posture",
        "gui_show_heatmap",
        "gui_show_blobs",
        "gui_show_selections",
        "gui_zoom_polygon",
        "gui_zoom_limit",
        
        "gui_fish_label",
        
        "individual_image_normalization",
        "individual_image_size",
        "individual_image_scale",
        
        "track_background_subtraction",
        "meta_encoding",
        "track_threshold",
        "track_posture_threshold",
        "track_size_filter",
        "track_include", "track_ignore",
        
        "detect_skeleton",
        
        "heatmap_resolution", "heatmap_dynamic",
        "heatmap_ids", "heatmap_value_range",
        "heatmap_normalization", "heatmap_frames",
        "heatmap_source",
        
        "output_prefix",
        
        "gui_wait_for_background"
        
    }, [this](std::string_view key) {
        if(key == "gui_wait_for_background") {
            //if(_data && _data->_background)
               // _data->_background->set_strict(SETTING(gui_wait_for_background).value<bool>());
        }
        else if(is_in(key,
                 "cam_matrix",
                 "cam_undistort",
                 "cam_undistort_vector"))
        {
            if(_data
               && _data->_background)
            {
                init_undistortion();
            }
            
        } else if((key == "gui_zoom_polygon"
                   || key == "gui_zoom_limit"
                   || key == "gui_focus_group")
                  && _data->_bowl)
        {
            _data->_bowl->_screen_size = Vec2();
            _data->_zoom_dirty = true;
            _data->_cache->set_fish_dirty(true);
        } else if(key == "gui_run") {
            
        } else if(key == "track_pause") {
            /*gui::WorkProgress::add_queue("pausing...", [this](){
                _state->analysis.bump();
                bool pause = SETTING(track_pause).value<bool>();
                if(_state->analysis.paused() != pause) {
                    _state->analysis.set_paused(pause).get();
                }
            });*/
        } else if(key == "analysis_range") {
            _data->_analysis_range = Tracker::analysis_range();
            
        } else if(is_in(key, "track_ignore_bdx", "manual_splits", "manual_matches")
                  && _data
                  && _data->_cache)
        {
            if(Tracker::end_frame().valid()
               && _data->_cache->frame_idx.valid()
               && Tracker::end_frame() >= _data->_cache->frame_idx)
            {
                WorkProgress::add_queue("", [frame = _data->_cache->frame_idx, this](){
                    Tracker::instance()->_remove_frames(frame);
                    if(_state)
                        _state->analysis.set_paused(false);
                });
            }
        }
        
        if(utils::beginsWith(key, "gui_show_")
           || is_in(key,
                 "cam_matrix",
                 "cam_undistort",
                 "cam_undistort_vector",
                 "analysis_range", 
                 "track_threshold", 
                 "track_posture_threshold",
                 "track_size_filter",
                 "frame_rate",
                 "track_background_subtraction",
                 "meta_encoding",
                 "individual_image_normalization",
                 "individual_image_size",
                 "individual_image_scale",
                 "gui_zoom_polygon",//"gui_zoom_limit",
                 "detect_skeleton",
                 "track_include", "track_ignore"))
        {
            redraw_all();
        }
        
        if(key == "gui_focus_group"
           || key == "gui_fish_label"
           || key == "detect_skeleton"
           || utils::beginsWith(key, "heatmap_"))
        {
            SceneManager::getInstance().enqueue([this, frame = _data && _data->_cache ? _data->_cache->frame_idx : Frame_t{}](){
                if(_data && _data->_cache) {
                    _data->_primary_selection = {};
                    _data->_cache->set_tracking_dirty();
                    //_data->_cache->set_raw_blobs_dirty();
                    _data->_cache->set_fish_dirty(true);
                    //if(frame.valid())
                    //    _data->_cache->set_reload_frame(frame);
                    _data->_cache->set_redraw();
                    //_data->_cache->set_blobs_dirty();
                    //_data->_cache->frame_idx = {};
                    //SETTING(gui_frame) = Frame_t{};
                    //set_frame(frame);
                }
            });
        }
        
        if(key == "output_prefix") {
            SceneManager::getInstance().enqueue([this](){
                window()->set_title(window_title());
            });
        }
    });
    
    _data->_analysis_range = Tracker::analysis_range();
    _state->init_video();
    
    _state->tracker->register_add_callback([this](Frame_t frame){
        if(GUI_SETTINGS(gui_frame) == frame) {
            redraw_all();
        }
    });
    
    assert(not _data->_frame_callback);
    _data->_frame_callback = _state->tracker->register_add_callback([this](Frame_t) {
        if(_data)
            _data->_tracker_has_added_frames = true;
    });
    
    RecentItems::open(SETTING(source).value<file::PathArray>().source(), GlobalSettings::current_defaults_with_config());
    
    if(_load_requested) {
        bool exchange = true;
        if(_load_requested.compare_exchange_strong(exchange, false)) 
        {
            _state->load_state(nullptr, Output::TrackingResults::expected_filename());
        }
    }
}

void TrackingScene::redraw_all() {
    if(not _data || not _data->_cache) {
#ifndef NDEBUG
        FormatWarning("No data set, cannot redraw all.");
#endif
        return;
    }
    
    SceneManager::getInstance().enqueue([this, frame = _data->_cache->frame_idx](){
        if(_data && _data->_cache) {
            _data->_primary_selection = {};
            _data->_cache->set_tracking_dirty();
            _data->_cache->set_raw_blobs_dirty();
            _data->_cache->set_fish_dirty(true);
            if(frame.valid())
                _data->_cache->set_reload_frame(frame);
            _data->_cache->set_redraw();
            _data->_cache->set_blobs_dirty();
            //_data->_cache->frame_idx = {};
            //SETTING(gui_frame) = Frame_t{};
            //set_frame(frame);
        }
    });
}

void TrackingScene::init_undistortion() {
    if(not SETTING(cam_undistort)) {
        _data->_background->set_undistortion(std::nullopt, std::nullopt);
    } else {
        
        auto cam_data = SETTING(cam_matrix).value<std::vector<double>>();
        auto undistort_data = SETTING(cam_undistort_vector).value<std::vector<double>>();
        _data->_background->set_undistortion(std::move(cam_data), std::move(undistort_data));
    }
}

void TrackingScene::deactivate() {
#ifndef NDEBUG
    ThreadManager::getInstance().printThreadTree();
#endif
    if(_data && _data->_recorder.recording()) {
        _data->_recorder.stop_recording(nullptr, nullptr);
        if(_data->_background)
            _data->_background->set_strict(false);
    }
    
    /// unregister callback that tracks the tracker adding frames
    if(_data && _data->_frame_callback) {
        if(_state->tracker)
            _state->tracker->unregister_add_callback(_data->_frame_callback.value());
        _data->_frame_callback = std::nullopt;
    }
    
    Categorize::Work::terminate_prediction() = true;
    WorkProgress::stop();
    Categorize::Work::terminate_prediction() = false;

    cmn::gui::tracker::blob_view_shutdown();

    if(_data)
        _data->dynGUI.clear();
    
    if(_data && _data->_callback)
        GlobalSettings::map().unregister_callbacks(std::move(_data->_callback));
    
    auto config = default_config::generate_delta_config(AccessLevelType::LOAD, _state ? _state->video.get() : nullptr);
    for(auto &[key, value] : config.map) {
        Print(" * ", no_quotes(utils::ShortenText(Meta::toStr(*value), 1000)));
    }
    Print();

    _data = nullptr;
    _state = nullptr;
    
    for(auto &key : GlobalSettings::current_defaults_with_config().keys()) {
        auto value = GlobalSettings::map().at(key);
        if(/*contains(config.excluded.toVector(), key)
           && GlobalSettings::access_level(key) < AccessLevelType::LOAD*/
           is_in(key, "filename", "source", "output_dir", "output_prefix"))
        {
            Print(" . ", no_quotes(utils::ShortenText(Meta::toStr(value.get()), 1000)));
            value.get().copy_to(GlobalSettings::current_defaults_with_config());
            continue;
        }
        //Print(" - ", value.get());
        GlobalSettings::current_defaults_with_config().erase(key);
    }
    
    //GlobalSettings::current_defaults_with_config() = {}
    
    config.write_to(GlobalSettings::current_defaults_with_config());
    RecentItems::open(SETTING(source).value<file::PathArray>().source(), GlobalSettings::current_defaults_with_config());
    
    SETTING(filename) = file::Path();
    SETTING(source) = file::PathArray();
    
    SettingsMaps combined;
    combined.map.set_print_by_default(false);
    
    grab::default_config::get(combined.map, combined.docs, [&](auto& name, auto level){
        combined.access_levels[name] = level;
    });
    default_config::get(combined.map, combined.docs, [&](auto& name, auto level){
        combined.access_levels[name] = level;
    });
    
    for(auto &key : combined.map.keys()) {
        if(GlobalSettings::access_level(key) > AccessLevelType::LOAD) {
            //combined.map.at(key)->valueString();
            //Print(" - Ignoring ", key, " ", no_quotes(combined.map.at(key)->valueString()), " vs. ", no_quotes(GlobalSettings::at(key)->valueString()));
            continue;
        }
        
        //Print(" * Resetting ", key);
        if(GlobalSettings::map().has(key)) {
            auto p = GlobalSettings::map().at(key).get().do_print();
            GlobalSettings::map().do_print(key, false);
            combined.map.at(key).get().copy_to(GlobalSettings::map());
            GlobalSettings::map().do_print(key, p);
        } else {
            combined.map.at(key).get().copy_to(GlobalSettings::map());
        }
    }
    
    GlobalSettings::set_current_defaults(combined.map);
    GlobalSettings::set_current_defaults_with_config(combined.map);
}

void TrackingScene::set_frame(Frame_t frameIndex) {
    if(frameIndex < _state->video->length()
       && GUI_SETTINGS(gui_frame) != frameIndex)
    {
        SETTING(gui_frame) = frameIndex;
        _data->_cache->request_frame_change_to(frameIndex);
    }
}

/**
 * Update the run loop for the tracking scene.
 *
 * This function updates the playback frame based on elapsed time and GUI settings,
 * handling both recording (fixed dt) and non-recording (elapsed dt) modes.
 */
void TrackingScene::update_run_loop() {
    const auto redraw_dt = last_redraw.elapsed();
    last_redraw.reset();  /// Restart the timer for the next frame.
    
    /// Ensure required data is available.
    if (not _data || not _data->_cache)
        return;
    
    /// Cache frequently used GUI settings.
    const uint32_t gui_playback_speed = GUI_SETTINGS(gui_playback_speed);
    const double frame_rate = GUI_SETTINGS(frame_rate); // Frames per second.
    
    // Set the time delta (dt) for frame updates.
    if (_data->_recorder.recording()) {
        /// For recording, use a fixed dt based on frame rate.
        _data->_cache->set_dt(1.0 / double(frame_rate));
    } else {
        /// For playback, use the elapsed time since the last redraw.
        _data->_cache->set_dt(redraw_dt);
    }
    
    /// Exit early if the run loop is disabled.
    if (!GUI_SETTINGS(gui_run))
        return;
    
    /// Retrieve current dt and the frame index.
    /// This is based on either constant intervals or real-time,
    /// depending on the current recording mode.
    /// NOTE: cache->dt is used elsewhere, too.
    const auto dt = _data->_cache->dt();
    Frame_t index = GUI_SETTINGS(gui_frame);
    
    if (_data->_recorder.recording()) {
        /// Recording mode: load frames synchronously.
        _data->_cache->set_load_frames_blocking(true);
        
        /// Advance the frame index by either the playback speed or one frame.
        /// This has to be a conditional to protect against wrong user-set parameters.
        index += (gui_playback_speed > 1)
                    ? Frame_t(gui_playback_speed)
                    : 1_f;
        
        /// If the new frame exceeds video length, clamp it and stop running.
        if (auto L = _state->video->length();
            index >= L)
        {
            index = L.try_sub(1_f);
            SETTING(gui_run) = false;
        }
        set_frame(index);
        
        /// Update the background increment to keep visual sync.
        if (_data->_background)
            _data->_background->set_increment((gui_playback_speed > 1) ? Frame_t(gui_playback_speed) : 1_f);
        
    } else {
        /// Non-recording mode: accumulate elapsed time.
        /// Cache settings to avoid redundant calls.
        const bool gui_wait_for_background = SETTING(gui_wait_for_background).value<bool>();
        const bool gui_wait_for_pv = SETTING(gui_wait_for_pv).value<bool>();
        const Frame_t gui_displayed_frame = SETTING(gui_displayed_frame).value<Frame_t>();
        
        _data->_cache->set_load_frames_blocking(false);
        _data->_time_since_last_frame += dt;
        
        /// Determine how many frames to advance based on the accumulated time.
        double advances = _data->_time_since_last_frame * frame_rate * gui_playback_speed;
        if (advances >= 1) {
            Frame_t rounded_advances{uint32_t(std::round(advances))};
            
            //Print("* displayed frame = ", gui_displayed_frame, " vs. ",
                  _data->_background ? _data->_background->displayed_frame() : Frame_t(), " vs. index=", index, " advance=", rounded_advances);
            
            if (gui_wait_for_background) {
                if(_data->_background)
                    _data->_background->set_enable_fade(false);
                
                /// Only advance if both the GUI and background are synchronized.
                if (not _data->_background
                    || _data->_background->displayed_frame() != index)
                {
                    rounded_advances = {};
                } else {
                    rounded_advances = Frame_t(gui_playback_speed);
                }
            } else {
                if(_data->_background)
                    _data->_background->set_enable_fade(true);
            }
            
            if(gui_wait_for_pv
               && rounded_advances.valid())
            {
                if(gui_displayed_frame != index) {
                    rounded_advances = {};
                }
            }
            
            if(rounded_advances.valid()) {
                index += rounded_advances;
                
                /// Clamp frame index if needed.
                if (auto L = _state->video->length();
                    index >= L)
                {
                    index = L.try_sub(1_f);
                    SETTING(gui_run) = false;
                }
                
                set_frame(index);
                
                /// Update the background increment using the calculated advances.
                if (_data->_background)
                    _data->_background->set_increment(rounded_advances);
            }
            
            /// Reset the accumulated time.
            _data->_time_since_last_frame = 0;
        }
    }
}

void TrackingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not _data->dynGUI) {
        init_gui(_data->dynGUI, graph);
        //auto size = window()->get_window_bounds().size();
        //window()->set_window_size(Size2(1024,759));
        //window()->set_window_size(size);
    }
    
    update_run_loop();
    if(not _data)
        return;
    
    if(_data->_tracker_has_added_frames
       //&& _state && _state->analysis.is_paused()
       && _data->_cache)
    {
        auto result = _data->_cache->update_slow_tracker_stuff();
        if(result) {
            _data->_tracker_has_added_frames = false;
            redraw_all();
        }
    }
    
    //window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height).div(((IMGUIBase*)window())->dpi_scale()) * gui::interface_scale();
    
    auto coords = FindCoord::get();
    
    if(not _data->_cache) {
        _data->_cache = std::make_unique<GUICache>(&graph, _state->video);
        _data->_bowl = std::make_unique<Bowl>(_data->_cache.get());
        _data->_bowl->set_video_aspect_ratio(_state->video->size().width, _state->video->size().height);
        _data->_bowl->fit_to_screen(coords.screen_size());
        
        _data->_clicked_background = [&](const Vec2& pos, bool v, std::string key) {
            gui::tracker::clicked_background(graph, *_data->_cache, pos, v, key);
        };
    }
    
    auto mouse = graph.mouse_position();
    if(mouse != _data->_last_mouse 
       //|| _data->_cache->is_animating()
       || graph.root().is_animating())
    {
        if(((IMGUIBase*)window())->focussed()) {
            //_data->_cache->set_blobs_dirty();
            //_data->_cache->set_tracking_dirty();
            _data->_zoom_dirty = true;
        }
        _data->_last_mouse = mouse;
        
        //if(graph.root().is_animating())
        //    Print("Is animating", graph.root().is_animating());
    }
    
    //if(false)
    {
        uint64_t last_change = FOI::last_change();
        auto name = SETTING(gui_foi_name).value<std::string>();

        if (last_change != _data->_foi_state.last_change
            || name != _data->_foi_state.name)
        {
            _data->_foi_state.name = name;

            if (!_data->_foi_state.name.empty()) {
                long_t id = FOI::to_id(_data->_foi_state.name);
                if (id != -1) {
                    auto changed = FOI::foi(id);
                    if(changed.has_value()) {
                        _data->_foi_state.changed_frames = std::move(changed.value());
                        _data->_foi_state.color = FOI::color(_data->_foi_state.name);
                    }
                }
            }

            _data->_foi_state.last_change = last_change;
        }
    }

    for (auto& [key, code] : _key_map) {
        _data->_keymap[key] = graph.is_key_pressed(code);
    }
    
    Frame_t loaded;
    if(_data->_cache) {
        auto frameIndex = GUI_SETTINGS(gui_frame);
        //do {
        //Print("Loading ", frameIndex);
            loaded = _data->_cache->update_data(frameIndex);
            
        //} while(_data->_recorder.recording() && loaded.valid() && loaded != frameIndex);
        
        if(loaded.valid() || _data->_cache->fish_dirty()) {
            //Print("Update all... ", loaded, "(",frameIndex,")");
            
            if(loaded.valid())
                SETTING(gui_displayed_frame) = loaded;
            using namespace dyn;
            
            _data->_individuals.resize(_data->_cache->raw_blobs.size());
            _data->_fish_data.resize(_data->_individuals.size());
            
            /*size_t i = 0;
            std::set<pv::bid> bdxes;
            for(auto &[fdx, fish] : _data->_cache->fish_selected_blobs) {
                if(_data->_individuals.size() <= i) {
                    _data->_individuals.resize(i + 1);
                    _data->_fish_data.resize(i + 1);
                }
                
                auto speed = fish.basic_stuff->centroid.speed<Units::CM_AND_SECONDS>();
                auto &var = _data->_individuals[i];
                if(not var)
                    var = std::unique_ptr<VarBase_t>(new Variable([i, this](const VarProps&) -> sprite::Map& {
                        return _data->_fish_data.at(i);
                    }));
                
                auto &map = _data->_fish_data.at(i);
                map["pos"] = Vec2(fish.basic_stuff->blob.calculate_bounds().pos());
                map["speed"] = speed;
                bdxes.insert(fish.bdx);
                ++i;
            }
            
            for(size_t j=0; j<_data->_cache->raw_blobs.size(); ++j) {
                auto &blob = _data->_cache->raw_blobs[j];
                if(bdxes.contains(blob->blob->blob_id()))
                {
                    continue;
                }
                
                if(_data->_individuals.size() <= i) {
                    _data->_individuals.resize(i + 1);
                    _data->_fish_data.resize(i + 1);
                }
                
                auto &var = _data->_individuals[i];
                if(not var)
                    var = std::unique_ptr<VarBase_t>(new Variable([i, this](const VarProps&) -> sprite::Map& {
                        return _data->_fish_data.at(i);
                    }));
                
                auto &map = _data->_fish_data.at(i);
                map["pos"] = Vec2(blob->blob->bounds().pos());
                if(map.has("speed"))
                    map.erase("speed");
                bdxes.insert(blob->blob->blob_id());
                ++i;
            }*/
            
            _data->_zoom_dirty = true;
        }
    }
    
    if(_data->_zoom_dirty
       || _data->_cache->is_tracking_dirty()
       || _data->_cache->fish_dirty()
        )
    {
        std::vector<Vec2> targets;
        if(_data->_cache->has_selection()
           && not graph.is_key_pressed(Keyboard::LShift))
        {
            for(auto fdx : _data->_cache->selected) {
                /*bool found = false;
                if (auto it = _data->_cache->fish_selected_blobs.find(fdx);
                    it != _data->_cache->fish_selected_blobs.end()) 
                {
                    auto const &blob = it->second;
                    if(blob.basic_stuff) {
                        auto bds = blob.basic_stuff->blob.calculate_bounds();
                        targets.push_back(bds.pos());
                        targets.push_back(bds.pos() + bds.size());
                        targets.push_back(bds.pos() + bds.size().mul(0, 1));
                        targets.push_back(bds.pos() + bds.size().mul(1, 0));
                        //_data->_last_bounds[fdx] = bds;
                        found = true;
                    }
                }

                if (not found) {*/
                    if(_data->_cache->fish_last_bounds.contains(fdx)) {
                        auto& bds = _data->_cache->fish_last_bounds.at(fdx);
                        targets.push_back(bds.pos());
                        targets.push_back(bds.pos() + bds.size());
                        targets.push_back(bds.pos() + bds.size().mul(0, 1));
                        targets.push_back(bds.pos() + bds.size().mul(1, 0));
                    }
                //}
            }
            
            /*if(_data->_last_bounds.size() > 100) {
                std::vector<Idx_t> remove;
                for(auto &[fdx, bds] : _data->_last_bounds) {
                    if(not contains(_data->_cache->selected, fdx)
                       || (not contains(_data->_cache->active_ids, fdx)
                           && not contains(_data->_cache->inactive_ids, fdx)))
                    {
                        remove.push_back(fdx);
#ifndef NDEBUG
                        Print("* removing individual ", fdx);
#endif
                    }
                }
                
                //Print("active = ", _data->_cache->active_ids);
                //Print("inactive = ", _data->_cache->inactive_ids);
                
                for(auto fdx: remove)
                    _data->_last_bounds.erase(fdx);
            }*/
        }
        
        _data->_bowl->fit_to_screen(coords.screen_size());
        if(GUI_SETTINGS(gui_auto_scale_focus_one))
            _data->_bowl->set_target_focus(targets);
        else
            _data->_bowl->set_target_focus({});
        
        if(loaded.valid())
            _data->_zoom_dirty = false;
    }
    
    _data->_bowl->update_scaling(_data->_cache->dt());
    
    auto alpha = SETTING(gui_background_color).value<Color>().a;
    _data->_background->set_color(Color(255, 255, 255, alpha ? alpha : 1));
    
    graph.wrap_object(*_data->_background);
    
    _data->_background->set_scale(_data->_bowl->scale());
    _data->_background->set_pos(_data->_bowl->pos());
    _data->_background->set_video_scale(SETTING(meta_video_scale).value<float>());
    
    if (alpha > 0) {
        /*if(PD(gui_mask)) {
            PD(gui_mask)->set_color(PD(background)->color().alpha(PD(background)->color().a * 0.5));
            PD(gui).wrap_object(*PD(gui_mask));
        }*/
    }
    
    //if(GUI_SETTINGS(gui_mode) == mode_t::tracking)
    {
        graph.wrap_object(*_data->_bowl);
    }
    
    if(_data->_cache->frame_idx.valid())
        _data->_bowl->update(_data->_cache->frame_idx, graph, coords);
    _data->_bowl_mouse = coords.convert(HUDCoord(graph.mouse_position())); //_data->_bowl->global_transform().getInverse().transformPoint(graph.mouse_position());
    
    /*const auto mode = GUI_SETTINGS(gui_mode);
    const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
    update_display_blobs(draw_blobs);*/
    
    //_data->_bowl.auto_size({});
    //_data->_bowl->set(LineClr{Cyan});
    //_data->_bowl.set(FillClr{Yellow});
    
    if(GUI_SETTINGS(gui_mode) == mode_t::blobs) {
        cmn::gui::tracker::draw_blob_view({
            .graph = graph,
            .cache = *_data->_cache,
            .coord = coords
        });
    }
    
    cmn::gui::tracker::draw_boundary_selection(graph, window(), *_data->_cache, _data->_bowl.get());
    
    _data->dynGUI.update(graph, nullptr);
    
    Categorize::draw(_state->video, (IMGUIBase*)window(), graph);
    
    //DrawPreviewImage::draw(_state->tracker->background(), _data->_cache->processed_frame(), GUI_SETTINGS(gui_frame), graph);
    
    if(GUI_SETTINGS(gui_show_posture) && not _data->_cache->selected.empty()) {
        _data->_cache->draw_posture(graph, GUI_SETTINGS(gui_frame));
    }
    
    if(GUI_SETTINGS(gui_show_dataset)) {
        if(not _data->_dataset) {
            _data->_dataset = std::make_unique<DrawDataset>();
        }
        
        _data->_dataset->set_data(_data->_cache->frame_idx, *_data->_cache);
        graph.wrap_object(*_data->_dataset);
    }
    
    if(GUI_SETTINGS(gui_show_export_options)) {
        if(not _data->_export_options) {
            _data->_export_options = std::make_unique<DrawExportOptions>();
        }
        
        _data->_export_options->draw(graph, _state.get());
    }
    
    if(GUI_SETTINGS(gui_show_uniqueness)) {
        if(not _data->_uniqueness) {
            _data->_uniqueness = std::make_unique<DrawUniqueness>(_data->_cache.get(), _state->video);
        }
        
        graph.wrap_object(*_data->_uniqueness);
    }
    
    //if(not graph.root().is_dirty() && not graph.root().is_animating())
    //    std::this_thread::sleep_for(std::chrono::milliseconds(((IMGUIBase*)window())->focussed() ? 10 : 200));
    //Print("dirty = ", graph.root().is_dirty());
    if(graph.root().is_dirty())
        last_dirty.reset();
    else if(last_dirty.elapsed() > 0.25)
    {
        graph.root().set_dirty();
        last_dirty.reset();
    }
}

void TrackingScene::next_poi(Idx_t _s_fdx) {
    auto frame = _data->_cache->frame_idx;
    auto next_frame = frame;
    std::set<FOI::fdx_t> fdx;
    
    {
        for(const FOI& foi : _data->_foi_state.changed_frames) {
            if(_s_fdx.valid()) {
                if(not foi.fdx().contains(FOI::fdx_t(_s_fdx)))
                    continue;
            }
            
            if(not frame.valid() || foi.frames().start > frame) {
                next_frame = foi.frames().start;
                fdx = foi.fdx();
                break;
            }
        }
    }
    
    if(frame != next_frame && next_frame.valid()) {
        set_frame(next_frame);
        
        if(!_s_fdx.valid())
        {
            if(!fdx.empty()) {
                _data->_cache->deselect_all();
                for(auto id : fdx) {
                    if(!_data->_cache->is_selected(Idx_t(id.id)))
                        _data->_cache->do_select(Idx_t(id.id));
                }
            }
        }
    }
}

void TrackingScene::prev_poi(Idx_t _s_fdx) {
    auto frame = _data->_cache->frame_idx;
    Frame_t next_frame;
    std::set<FOI::fdx_t> fdx;
    
    {
        for(const FOI& foi : _data->_foi_state.changed_frames) {
            if(_s_fdx.valid()) {
                if(not foi.fdx().contains(FOI::fdx_t(_s_fdx)))
                    continue;
            }
            
            if(frame.valid() && foi.frames().end < frame && (not next_frame.valid() || foi.frames().end > next_frame)) {
                next_frame = foi.frames().end;
                fdx = foi.fdx();
            }
        }
    }
    
    if(next_frame.valid() && frame != next_frame) {
        set_frame(next_frame);
        
        if(!_s_fdx.valid())
        {
            if(!fdx.empty()) {
                _data->_cache->deselect_all();
                for(auto id : fdx) {
                    if(!_data->_cache->is_selected(Idx_t(id.id)))
                        _data->_cache->do_select(Idx_t(id.id));
                }
            }
        }
    }
}

void TrackingScene::init_gui(dyn::DynamicGUI& dynGUI, DrawStructure& ) {
    using namespace dyn;
    dyn::DynamicGUI g{
        .gui = SceneManager::getInstance().gui_task_queue(),
        .path = "tracking_layout.json",
        .context = {
            ActionFunc("print_memory_stats", [this](Action){
                mem::IndividualMemoryStats overall;
                {
                    auto lock = _data->_cache->lock_individuals();
                    for(auto && [fdx, fish] : lock.individuals) {
                        mem::IndividualMemoryStats stats(fish);
                        stats.print();
                        overall += stats;
                    }
                }
            
                overall.print();
                
                mem::TrackerMemoryStats stats;
                stats.print();
                
                mem::OutputLibraryMemoryStats ol;
                ol.print();
            }),
            ActionFunc("quit", [](Action) {
                SceneManager::getInstance().enqueue([](auto, DrawStructure& graph) {
                    graph.dialog([](Dialog::Result result) {
                        if(result == Dialog::Result::OKAY) {
                            SETTING(terminate) = true;
                        }
                    }, "Are you sure you want to exit the application? Any unsaved changes will be discarded.", "Exit", "Quit", "Cancel");
                });
            }),
            ActionFunc("set", [this](Action action) {
                if(action.parameters.size() != 2)
                    throw InvalidArgumentException("Invalid number of arguments for action: ",action);
                
                auto parm = Meta::fromStr<std::string>(action.first());
                if(not GlobalSettings::has(parm))
                    throw InvalidArgumentException("No parameter ",parm," in global settings.");
                
                auto value = action.last();
                
                if(parm == "gui_frame") {
                    set_frame(Meta::fromStr<Frame_t>(value));
                } else
                    GlobalSettings::get(parm).get().set_value_from_string(value);
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
            ActionFunc("reanalyse", [this](Action action) {
                if(action.parameters.empty())
                    _state->tracker->_remove_frames(Frame_t{});
                else {
                    auto frame = Meta::fromStr<Frame_t>(action.first());
                    _state->tracker->_remove_frames(frame);
                }
                _state->analysis.set_paused(false);
            }),
            ActionFunc("load_results", [this](Action){
                _state->load_state(SceneManager::getInstance().gui_task_queue(), Output::TrackingResults::expected_filename());
            }),
            ActionFunc("save_results", [this](Action) {
                _state->save_state(SceneManager::getInstance().gui_task_queue(), false);
            }),
            ActionFunc("clear_ml", [this](Action) {
                SceneManager::getInstance().enqueue([this](auto, DrawStructure& graph) {
                    graph.dialog([this](Dialog::Result result) {
                        if(result == Dialog::Result::OKAY
                           && _state && _state->tracker)
                        {
                            _state->tracker->clear_tracklets_identities();
                            _state->tracker->clear_vi_predictions();
                        }
                    }, "Are you sure you want to clear all VI predictions?\n<i>This does not touch the current tracking state, but deletes all information generated by previous <c>Visual Identification</c>. If you reanalyse afterwards, identities can not be automatically corrected and you are left with regular tracking until you re-apply VI.</i>", "Clear VI predictions", "Yes", "Cancel");
                });
            }),
            ActionFunc("export_data", [](Action){
                SETTING(gui_show_export_options) = true;
            }),
            ActionFunc("python", [](Action action){
                REQUIRE_EXACTLY(1, action);
                
                Python::schedule(Python::PackagedTask{
                    ._network = nullptr,
                    ._task = Python::PromisedTask(
                        [action](){
                            using py = PythonIntegration;
                            Print("Executing: ", action.first());
                            py::execute(action.first());
                        }
                    ),
                    ._can_run_before_init = false
                });
            }),
            ActionFunc("set_paused", [this](Action action) {
                REQUIRE_EXACTLY(1, action);
                
                bool value = Meta::fromStr<bool>(action.first());
                WorkProgress::add_queue("Pausing...", [this, value](){
                    if(_state)
                        _state->analysis.set_paused(value).get();
                });
            }),
            ActionFunc("write_config", [video = _state->video](Action){
                WorkProgress::add_queue("", [video]() {
                    settings::write_config(video.get(), false, SceneManager::getInstance().gui_task_queue());
                });
            }),
            ActionFunc("categorize", [this](Action){
                _state->_controller->_busy = true;
                Categorize::show(_state->video,
                    [this](){
                        _state->_controller->_busy = false;
                        if(SETTING(auto_quit))
                            _state->_controller->auto_quit(SceneManager::getInstance().gui_task_queue());
                        //GUI::instance()->auto_quit();
                    },
                    [this](const std::string& text, double percent){
                        if(percent < 0) {
                            _state->_controller->_busy = false;
                            return;
                        }
                        Print(text.c_str());
                        _state->_controller->_current_percent = percent;
                        //GUI::instance()->set_status(text);
                    }
                );
            }),
            ActionFunc("auto_correct", [this](Action){
                _state->_controller->auto_correct(SceneManager::getInstance().gui_task_queue(), false);
            }),
            ActionFunc("visual_identification", [this](Action) {
                vident::training_data_dialog(SceneManager::getInstance().gui_task_queue(), false, [](){
                    Print("callback ");
                }, _state->_controller.get());
            }),
            ActionFunc("remove_automatic_matches", [this](const Action& action) {
                REQUIRE_EXACTLY(2, action);
                auto fdx = Meta::fromStr<Idx_t>(action.parameters.front());
                if(not action.parameters.back().empty() 
                   && action.parameters.back().front() == '[')
                {
                    auto range = FrameRange(Meta::fromStr<Range<Frame_t>>(action.parameters.back()));
                    Print("Erasing automatic matches for fish ", fdx," in range ", range.start(),"-",range.end());
                    if(_state && _state->tracker) {
                        LockGuard guard(w_t{}, "automatic assignments");
                        AutoAssign::delete_automatic_assignments(fdx, range);
                        _state->tracker->_remove_frames(range.start());
                        _state->analysis.set_paused(false);
                    }
                } else {
                    auto frame = Meta::fromStr<Frame_t>(action.parameters.back());
                    Print("Erasing automatic matches for fish ", fdx," in frame ", frame);
                    if(_state && _state->tracker) {
                        LockGuard guard(w_t{}, "automatic assignments");
                        AutoAssign::delete_automatic_assignments(fdx, FrameRange(Range<Frame_t>{frame, frame}));
                        _state->tracker->_remove_frames(frame);
                        _state->analysis.set_paused(false);
                    }
                }
                
                Print("Got ", action.name, ": ", action.parameters);
            }),
            
            VarFunc("is_segment", [this](const VarProps& props) -> bool {
                if(props.parameters.size() != 1)
                    throw InvalidArgumentException("Need exactly one argument for ", props);
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                for(auto &seg : _data->_cache->global_tracklet_order()) {
                    if(FrameRange(seg).contains(frame)) {
                        return true;
                    }
                }
                return false;
            }),
            VarFunc("is_paused", [this](const VarProps&) -> bool {
                if(not _state)
                    return false;
                return _state->analysis.is_paused();
            }),
            VarFunc("get_tracklet", [this](const VarProps& props) -> FrameRange {
                if(props.parameters.size() != 1)
                    throw InvalidArgumentException("Need exactly one argument for ", props);
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                for(auto &seg : _data->_cache->global_tracklet_order()) {
                    if(FrameRange(seg).contains(frame)) {
                        return FrameRange(seg);
                    }
                }
                throw InvalidArgumentException("No frame range contains ", frame," for ", props);
            }),
            VarFunc("is_tracked", [this](const VarProps& props) -> bool {
                if (props.parameters.size() != 1)
                    throw InvalidArgumentException("Need exactly one argument for ", props);

                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                return _data->_cache->tracked_frames.contains(frame);
            }),
            VarFunc("active_individuals", [this](const VarProps& props) -> size_t {
                if(props.parameters.size() != 1)
                    throw std::invalid_argument("Need exactly one argument for "+props.toStr());
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                {
                    LockGuard guard(ro_t{}, "active", 10);
                    if(not guard.locked()) {
                        return _data->_last_active_individuals;
                    }
                    
                    if(_state->tracker->properties(frame)) {
                        _data->_last_active_individuals = Tracker::active_individuals(frame).size();
                        return _data->_last_active_individuals;
                    }
                }
                
                throw std::invalid_argument("Frame "+Meta::toStr(frame)+" not tracked.");
            }),
            VarFunc("segments_for", [this](const VarProps& props) -> std::vector<ShadowTracklet>{
                REQUIRE_EXACTLY(1, props);
                auto idx = Meta::fromStr<Idx_t>(props.first());
                if(auto it = _data->_cache->_individual_ranges.find(idx);
                   it != _data->_cache->_individual_ranges.end())
                {
                    return it->second;
                }
                
                throw InvalidArgumentException("Cannot find individual ", props);
            }),
            VarFunc("live_individuals", [this](const VarProps& props) -> size_t {
                if(props.parameters.size() != 1)
                    throw std::invalid_argument("Need exactly one argument for "+props.toStr());
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                {
                    LockGuard guard(ro_t{}, "active", 10);
                    if(not guard.locked()) {
                        return _data->_last_live_individuals;
                    }
                    
                    if(_state->tracker->properties(frame)) {
                        size_t count{0u};
                        for(auto fish : Tracker::active_individuals(frame)) {
                            if(fish->has(frame))
                                ++count;
                        }
                        
                        _data->_last_live_individuals = count;
                        return count;
                    }
                }
                
                throw std::invalid_argument("Frame "+Meta::toStr(frame)+" not tracked.");
            }),
            VarFunc("window_size", [](const VarProps&) -> Vec2 {
                return FindCoord::get().screen_size();
            }),
            
            VarFunc("video_size", [this](const VarProps&) -> Vec2 {
                return _data->_bowl->_video_size;
            }),
            
            VarFunc("fps", [this](const VarProps&) -> double {
                return _state->_stats.frames_per_second.load();
            }),
            
            VarFunc("fishes", [this](const VarProps&)
                -> const std::vector<std::shared_ptr<VarBase_t>>&
            {
                if(not _data)
                    throw std::runtime_error("_data is nullptr");
                return _data->_individuals;
            }),
            
            VarFunc("consec", [this](const VarProps&) -> auto& {
                //auto consec = _data->tracker->global_tracklet_order();
                auto &consec = _data->_cache->global_tracklet_order();
                
                ColorWheel wheel;
                for(size_t i=0; i<5 && i < consec.size(); ++i) {
                    if(_data->tracklets.size() <= i) {
                        _data->tracklets.emplace_back();
                        _data->variables.emplace_back(new Variable([i, this](const VarProps&) -> sprite::Map& {
                            return _data->tracklets.at(i);
                        }));
                        assert(_data->variables.size() == _data->tracklets.size());
                    }
                    
                    auto& map = _data->tracklets.at(i);
                    //if(map.print_by_default())
                    //    map.set_print_by_default(false);
                    map["color"] = wheel.next();
                    map["start"] = consec.at(i).start;
                    map["end"] = consec.at(i).end + 1_f;
                }
                
                if(_data->tracklets.size() >= consec.size()) {
                    _data->tracklets.resize(consec.size());
                    _data->variables.resize(consec.size());
                }
                
                return _data->variables;
            }),
            
            VarFunc("primary_selection", [this](const VarProps&) -> sprite::Map& {
                if(not _data)
                    throw InvalidArgumentException("_data not set.");
                
                auto & map = _data->_primary_selection;
                Idx_t fdx = _data->_cache->primary_selected_id();
                auto frame = _data->_cache->frame_idx;
                if(not map.has("fdx")
                   || not map.has("frame")
                   || map["fdx"].value<Idx_t>() != fdx
                   || map["frame"].value<Frame_t>() != frame)
                {
                    map["fdx"] = fdx;
                    map["frame"] = frame;
                    map["has_neighbor"] = false;
                    map["bdx"] = pv::bid();
                    map["px"] = Float2_t{};
                    map["size"] = Float2_t{};
                    map["ps"] = std::vector<std::tuple<pv::bid, Probability, Probability>>{};
                    map["p"] = 0.0;
                    map["p_time"] = 0.0;
                    auto _id = Identity::Temporary(fdx);
                    map["color"] = _id.color();
                    map["is_automatic"] = false;
                    map["segment"] = Range<Frame_t>{};
                    map["name"] = _id.name();
                    
                    auto probs = fdx.valid()
                        ? _data->_cache->probs(fdx)
                        : nullptr;
                    if(probs) {
                        std::vector<std::tuple<pv::bid, Probability, Probability>> ps;
                        for(auto &[bdx, value] : *probs) {
                            ps.emplace_back(bdx, value.p, value.p_time);
                        }
                        std::sort(ps.begin(), ps.end(), [](const auto &A, const auto& B){
                            return std::make_tuple(std::get<1>(A), std::get<0>(A)) > std::make_tuple(std::get<1>(B), std::get<0>(B));
                        });
                        map["ps"] = ps;
                    }
                    
                    if(fdx.valid()) {
                        if(auto it = _data->_cache->fish_selected_blobs.find(fdx);
                           it != _data->_cache->fish_selected_blobs.end())
                        {
                            auto& stuff = it->second.basic_stuff;
                            if(stuff) {
                                /// add curve speed
                                map["speed"] = stuff->centroid.speed<Units::CM_AND_SECONDS>();
                                
                                auto query = _data->_cache->blob_grid().query(stuff->centroid.pos<Units::PX_AND_SECONDS>(), FAST_SETTING(track_max_speed) / FAST_SETTING(cm_per_pixel));
                                auto min_d = 0.f;
                                pv::bid min_bdx;
                                Idx_t min_fdx;
                                for(auto &[d, bdx] : query) {
                                    if(bdx != it->second.bdx
                                       && (d < min_d || not min_bdx.valid()))
                                    {
                                        auto fit = _data->_cache->blob_selected_fish.find(bdx);
                                        if(fit != _data->_cache->blob_selected_fish.end()) {
                                            min_fdx = fit->second;
                                            min_d = d;
                                            min_bdx = bdx;
                                        }
                                    }
                                }
                                
                                if(min_fdx.valid()) {
                                    map["nearest_neighbor"] = min_fdx;
                                    map["has_neighbor"] = true;
                                } else if(map.has("nearest_neighbor"))
                                    map.erase("nearest_neighbor");
                                
                                map["nearest_neighbor_distance"] = min_d * FAST_SETTING(cm_per_pixel);
                                map["bdx"] = it->second.bdx;
                                map["predictions"] = it->second.pred ? it->second.pred.value() : std::vector<float>{};
                                map["is_automatic"] = it->second.automatic_match;
                                map["segment"] = it->second.tracklet;
                                map["size"] = Float2_t(it->second.basic_stuff.has_value() ? it->second.basic_stuff->thresholded_size : uint64_t(0)) * SQR(FAST_SETTING(cm_per_pixel));
                                map["px"].toProperty<Float2_t>() = it->second.basic_stuff.has_value() ? it->second.basic_stuff->thresholded_size : uint64_t(0);
                                
                                if(probs) {
                                    if(auto pit = probs->find(it->second.bdx);
                                       pit != probs->end())
                                    {
                                        
                                        map["p"] = pit->second.p;
                                        map["p_time"] = pit->second.p_time;
                                        map["p_pos"] = pit->second.p_pos;
                                        map["p_angle"] = pit->second.p_angle;
                                    }
                                }
                            }
                        }
                    }
                }
                //map["speed"] = _data->_cache->lock_individuals().individuals.at(fdx)->centroid(_data->_cache->frame_idx)->speed<Units::CM_AND_SECONDS>();
                return map;
            }),
            
            VarFunc("tracker", [this](const VarProps&) -> Range<Frame_t> {
                if(not _state->tracker->start_frame().valid())
                    return Range<Frame_t>(_data->_analysis_range.load().start(), _data->_analysis_range.load().start());
                return Range<Frame_t>{ _state->tracker->start_frame(), _state->tracker->end_frame() + 1_f };
            }),
            
            VarFunc("analysis_range", [this](const VarProps&) -> Range<Frame_t> {
                auto range = _state->tracker->analysis_range().range;
                range.end += 1_f;
                return range;
            }),

            VarFunc("key", [this](const VarProps&) -> auto& {
                return _data->_keymap;
            }),
            
            VarFunc("mouse_in_bowl", [this](const VarProps&) -> Vec2 {
                return _data->_bowl_mouse;
            }),
            
            VarFunc("mouse", [this](const VarProps&) -> Vec2 {
                return this->_data->_last_mouse;
            }),
            VarFunc("vi_apply_percent", [this](const VarProps&) {
                return _state->_controller->_current_percent.load();
            }),
            VarFunc("vi_busy", [this](const VarProps&) -> bool {
                return _state->_controller->_busy.load();
            })
        }
    };
    
    g.context.custom_elements["preview"] = std::unique_ptr<CustomElement>(new PreviewAdapterElement([this]() -> const track::PPFrame* {
        if(not _data || not _data->_cache)
            return nullptr;
        return &_data->_cache->processed_frame();
        
    }, [this](Idx_t fdx) -> std::tuple<const constraints::FilterCache*, std::optional<BdxAndPred>>
    {
        if(not _data || not _data->_cache)
            return {nullptr, std::nullopt};
        
        const constraints::FilterCache* filters{nullptr};
        if(auto it = _data->_cache->filter_cache.find(fdx);
           it != _data->_cache->filter_cache.end())
        {
            filters = it->second.get();
        }
        
        if(auto it = _data->_cache->fish_selected_blobs.find(fdx);
           it != _data->_cache->fish_selected_blobs.end())
        {
            return {filters, it->second.clone()};
        }
        
        return {filters, std::nullopt};
    }));
    
    g.context.custom_elements["drawsegments"] = std::unique_ptr<CustomElement>(new CustomElement{
        "drawsegments",
        [](LayoutContext& context) -> Layout::Ptr {
            [[maybe_unused]] auto fdx = context.get(Idx_t(), "fdx");
            auto pad = context.get(Bounds(), "pad");
            auto limit = context.get(Size2(), "max_size");
            auto font = parse_font(context.obj);
            auto ptr = Layout::Make<DrawSegments>();
            ptr.to<DrawSegments>()->set(font);
            ptr.to<DrawSegments>()->set(attr::Margins{pad});
            ptr.to<DrawSegments>()->set(attr::SizeLimit{limit});
            return ptr;
        },
        [this](Layout::Ptr&o, const Context& context, State& state, const auto& patterns) {
            auto ptr = o.to<DrawSegments>();
            
            Idx_t fdx;
            Frame_t frame = _data->_cache->frame_idx;
            
            if(patterns.contains("fdx")) {
                try {
                    fdx = Meta::fromStr<Idx_t>(parse_text(patterns.at("fdx").original, context, state));
                } catch(...) {
                    
                }
            }
            
            SizeLimit limit;
            if(patterns.contains("max_size")) {
                try {
                    limit = Meta::fromStr<SizeLimit>(parse_text(patterns.at("max_size").original, context, state));
                    ptr->set(limit);
                } catch(...) {
                    
                }
            }
            
            if(fdx != ptr->fdx()
               || frame != ptr->frame())
            {
                std::vector<ShadowTracklet> segments;
                if(fdx.valid() && frame.valid()) {
                    if(auto it = _data->_cache->_individual_ranges.find(fdx);
                       it != _data->_cache->_individual_ranges.end())
                    {
                        segments = it->second;
                    }
                }
                ptr->set(fdx, frame, segments);
            }
            
            return false;
        }
    });
    
    g.context.custom_elements["timingstats"] = std::unique_ptr<TimingStatsElement>{
        new TimingStatsElement(TimingStatsCollector::getInstance())
    };
    
    dynGUI = std::move(g);
}

}
