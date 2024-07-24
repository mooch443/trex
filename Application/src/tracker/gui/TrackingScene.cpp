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

using namespace track;

namespace cmn::gui {

std::atomic<bool> _load_requested{false};

class IndividualImage : public Entangled {
    GETTER(Idx_t, fdx);
    Image::Ptr ptr;
    GETTER(Frame_t, frame);
    ExternalImage _display;

    static constexpr inline std::array<std::string_view, 10> _setting_names {
        "individual_image_normalization",
        "individual_image_size",
        "individual_image_scale",
        
        "track_background_subtraction",
        "meta_encoding",
        "track_threshold",
        "track_posture_threshold",
        "track_size_filter",
        "track_include", "track_ignore"
    };
    std::unordered_map<std::string_view, std::string> _settings;
    
public:
    using Entangled::set;
    void set_data(Idx_t fdx, Frame_t frame, pv::BlobWeakPtr blob, const Background* background, const constraints::FilterCache* filters, const Midline* midline) {
        // already set
        if(fdx == _fdx && _frame == frame && not settings_changed())
            return;
        
        this->_fdx = fdx;
        this->_frame = frame;
        
        auto &&[image, pos] = DrawPreviewImage::make_image(blob, midline, filters, background);
        _display.set_source(std::move(image));
        update_settings();
        update();
    }

    bool settings_changed() const {
        if(_settings.empty())
            return true;
        for(auto&[key, value] : _settings) {
            if(GlobalSettings::map().at(key).get().valueString() != value) {
                return true;
            }
        }
        return false;
    }
    void update_settings() {
        for(auto key : _setting_names) {
            _settings[key] = GlobalSettings::map().at(key).get().valueString();
        }
    }
    
    void update() override {
        begin();
        advance_wrap(_display);
        end();
        auto_size({});
    }
};

struct TrackingScene::Data {
    std::unique_ptr<GUICache> _cache;
    std::unique_ptr<DrawDataset> _dataset;
    std::unique_ptr<DrawExportOptions> _export_options;
    
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
    std::vector<sprite::Map> segments;
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
    
    if(video.has_mask()) {
        cv::Mat mask = video.mask().mul(cv::Scalar(255));
        mask.convertTo(mask, CV_8UC1);
        _gui_mask = std::make_unique<ExternalImage>(Image::Make(mask), Vec2(0, 0), Vec2(1), Color(255, 255, 255, 125));
    }

    for (auto& [key, code] : _key_map)
        _keymap[key] = false;
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
        "manual_ignore_bdx",
        
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
        
        "individual_image_normalization",
        "individual_image_size",
        "individual_image_scale",
        
        "track_background_subtraction",
        "meta_encoding",
        "track_threshold",
        "track_posture_threshold",
        "track_size_filter",
        "track_include", "track_ignore"
        
    }, [this](std::string_view key) {
        if(is_in(key, 
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
            
        } else if(is_in(key, "manual_ignore_bdx", "manual_splits", "manual_matches")
                  && _data
                  && _data->_cache)
        {
            WorkProgress::add_queue("", [frame = _data->_cache->frame_idx](){
                Tracker::instance()->_remove_frames(frame);
                SETTING(track_pause) = false;
            });
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
                 "gui_focus_group",
                 "track_background_subtraction",
                 "meta_encoding",
                 "individual_image_normalization",
                 "individual_image_size",
                 "individual_image_scale",
                "track_include", "track_ignore"))
        {
            redraw_all();
        }
    });
    
    _data->_analysis_range = Tracker::analysis_range();
    _state->init_video();
    
    _state->tracker->register_add_callback([this](Frame_t frame){
        if(GUI_SETTINGS(gui_frame) == frame) {
            redraw_all();
        }
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
    ThreadManager::getInstance().printThreadTree();
    if(_data && _data->_recorder.recording()) {
        _data->_recorder.stop_recording(nullptr, nullptr);
        if(_data->_background)
            _data->_background->set_strict(false);
    }
    
    Categorize::Work::terminate_prediction() = true;
    WorkProgress::stop();
    Categorize::Work::terminate_prediction() = false;
    
    if(_data)
        _data->dynGUI.clear();
    cmn::gui::tracker::blob_view_shutdown();
    
    if(_data && _data->_callback)
        GlobalSettings::map().unregister_callbacks(std::move(_data->_callback));
    
    auto config = default_config::generate_delta_config(_state ? _state->video.get() : nullptr);
    for(auto &[key, value] : config.map) {
        Print(" * ", *value);
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
            Print(" . ", value.get());
            value.get().copy_to(&GlobalSettings::current_defaults_with_config());
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
}

void TrackingScene::set_frame(Frame_t frameIndex) {
    if(frameIndex <= _state->video->length()
       && GUI_SETTINGS(gui_frame) != frameIndex)
    {
        SETTING(gui_frame) = frameIndex;
        _data->_cache->request_frame_change_to(frameIndex);
    }
}

void TrackingScene::update_run_loop() {
    //if(!recording())
    if(not _data || not _data->_cache)
        return;
    
    if(_data->_recorder.recording()) {
        _data->_cache->set_dt(0.75 / double(FAST_SETTING(frame_rate)));
    } else {
        _data->_cache->set_dt(last_redraw.elapsed());
    }
    last_redraw.reset();
    
    //else
    //    _data->_cache->set_dt(0.75f / (float(GUI_SETTINGS(frame_rate))));
    
    if(not GUI_SETTINGS(gui_run))
        return;
    
    const auto dt = _data->_cache->dt();
    const double frame_rate = GUI_SETTINGS(frame_rate) * GUI_SETTINGS(gui_playback_speed);
    
    Frame_t index = GUI_SETTINGS(gui_frame);
    
    if(_data->_recorder.recording()) {
        index += 1_f;
        
        if(auto L = _state->video->length();
           index >= L) 
        {
            index = L.try_sub(1_f);
            SETTING(gui_run) = false;
        }
        set_frame(index);
        if(_data && _data->_background)
            _data->_background->set_increment(1_f);
        
    } else {
        _data->_time_since_last_frame += dt;
        
        double advances = _data->_time_since_last_frame * frame_rate;
        if(advances >= 1) {
            index += Frame_t(uint(advances));
            if(auto L = _state->video->length();
               index >= L)
            {
                index = L.try_sub(1_f);
                SETTING(gui_run) = false;
            }
            set_frame(index);
            if(_data && _data->_background)
                _data->_background->set_increment(Frame_t(uint(advances)));
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
    }
    
    //if(false)
    {
        uint64_t last_change = FOI::last_change();
        auto name = SETTING(gui_foi_name).value<std::string>();

        if (last_change != _data->_foi_state.last_change || name != _data->_foi_state.name) {
            _data->_foi_state.name = name;

            if (!_data->_foi_state.name.empty()) {
                long_t id = FOI::to_id(_data->_foi_state.name);
                if (id != -1) {
                    _data->_foi_state.changed_frames = FOI::foi(id);//_tracker->changed_frames();
                    _data->_foi_state.color = FOI::color(_data->_foi_state.name);
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
    
    _data->dynGUI.update(nullptr);
    
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

void TrackingScene::init_gui(dyn::DynamicGUI& dynGUI, DrawStructure& graph) {
    using namespace dyn;
    dyn::DynamicGUI g{
        .gui = SceneManager::getInstance().gui_task_queue(),
        .path = "tracking_layout.json",
        .graph = &graph,
        .context = {
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
            ActionFunc("export_data", [this](Action){
                SETTING(gui_show_export_options) = true;
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
                for(auto &seg : _data->_cache->global_segment_order()) {
                    if(FrameRange(seg).contains(frame)) {
                        return true;
                    }
                }
                return false;
            }),
            VarFunc("get_segment", [this](const VarProps& props) -> FrameRange {
                if(props.parameters.size() != 1)
                    throw InvalidArgumentException("Need exactly one argument for ", props);
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                for(auto &seg : _data->_cache->global_segment_order()) {
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
            VarFunc("segments_for", [this](const VarProps& props) -> std::vector<ShadowSegment>{
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
                //auto consec = _data->tracker->global_segment_order();
                auto &consec = _data->_cache->global_segment_order();
                
                ColorWheel wheel;
                for(size_t i=0; i<5 && i < consec.size(); ++i) {
                    if(_data->segments.size() <= i) {
                        _data->segments.emplace_back();
                        _data->variables.emplace_back(new Variable([i, this](const VarProps&) -> sprite::Map& {
                            return _data->segments.at(i);
                        }));
                        assert(_data->variables.size() == _data->segments.size());
                    }
                    
                    auto& map = _data->segments.at(i);
                    //if(map.print_by_default())
                    //    map.set_print_by_default(false);
                    map["color"] = wheel.next();
                    map["start"] = consec.at(i).start;
                    map["end"] = consec.at(i).end + 1_f;
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
                    map["ps"] = std::vector<std::tuple<pv::bid, Probability, Probability>>{};
                    map["p"] = 0.0;
                    map["p_time"] = 0.0;
                    map["color"] = Identity::Temporary(fdx).color();
                    map["is_automatic"] = false;
                    map["segment"] = Range<Frame_t>{};
                    
                    auto probs = fdx.valid()
                        ? _data->_cache->probs(fdx)
                        : nullptr;
                    if(probs) {
                        std::vector<std::tuple<pv::bid, Probability, Probability>> ps;
                        for(auto &[bdx, value] : *probs) {
                            ps.emplace_back(bdx, value.p, value.p_time);
                        }
                        std::sort(ps.begin(), ps.end(), [](const auto &A, const auto& B){
                            return std::make_tuple(std::get<1>(A), std::get<0>(A)) < std::make_tuple(std::get<1>(B), std::get<0>(B));
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
                                map["is_automatic"] = it->second.automatic_match;
                                map["segment"] = it->second.segment;
                                
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
    
    g.context.custom_elements["preview"] = std::unique_ptr<CustomElement>(new CustomElement{
        .name = "preview",
        .create = [](LayoutContext& context) -> Layout::Ptr {
            [[maybe_unused]] auto fdx = context.get(Idx_t(), "fdx");
            auto ptr = Layout::Make<IndividualImage>();
            return ptr;
        },
        .update = [this](Layout::Ptr&o, const Context& context, State& state, const auto& patterns) {
            auto ptr = o.to<IndividualImage>();
            
            Idx_t fdx;
            Frame_t frame = _data->_cache->frame_idx;
            
            if(patterns.contains("fdx")) {
                fdx = Meta::fromStr<Idx_t>(parse_text(patterns.at("fdx").original, context, state));
            }
            /*if(patterns.contains("frame")) {
                frame = Meta::fromStr<Frame_t>(parse_text(patterns.at("frame").original, context, state));
            }*/
            
            if(fdx != ptr->fdx()
               || frame != ptr->frame()
               || ptr->settings_changed())
            {
                const constraints::FilterCache* cache{nullptr};
                if(auto it = _data->_cache->filter_cache.find(fdx);
                   it != _data->_cache->filter_cache.end())
                {
                    cache = it->second.get();
                }
                
                pv::BlobWeakPtr blob_ptr{nullptr};
                if(auto it = _data->_cache->fish_selected_blobs.find(fdx);
                   it != _data->_cache->fish_selected_blobs.end())
                {
                    _data->_cache->processed_frame().transform_blobs_by_bid(std::array{it->second.bdx}, [&blob_ptr](pv::Blob& blob) {
                        blob_ptr = &blob;
                    });
                    
                    if(blob_ptr) {
                        if(blob_ptr->encoding() == Background::meta_encoding())
                            ptr->set_data(fdx, frame, blob_ptr, _data->_cache->background(), cache, it->second.midline.get());
                        else
                            FormatWarning("Not displaying image yet because of the wrong encoding: ", blob_ptr->encoding(), " vs. ", Background::meta_encoding());
                    }
                    //else
                    //    throw InvalidArgumentException("Cannot find pixels for ", fdx, " and ", it->second.bdx);
                } //else
                  //  throw InvalidArgumentException("Cannot find individual ", fdx, " in cache.");
            }
            
            return false;
        }
    });
    
    g.context.custom_elements["drawsegments"] = std::unique_ptr<CustomElement>(new CustomElement{
        .name = "drawsegments",
        .create = [](LayoutContext& context) -> Layout::Ptr {
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
        .update = [this](Layout::Ptr&o, const Context& context, State& state, const auto& patterns) {
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
            /*if(patterns.contains("frame")) {
                frame = Meta::fromStr<Frame_t>(parse_text(patterns.at("frame").original, context, state));
            }*/
            
            if(fdx != ptr->fdx()
               || frame != ptr->frame())
            {
                std::vector<ShadowSegment> segments;
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
    
    dynGUI = std::move(g);
}

}
