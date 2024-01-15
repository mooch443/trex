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
#include <tracking/Categorize.h>
#include <gui/CheckUpdates.h>
#include <gui/WorkProgress.h>
#include <gui/DrawBlobView.h>
#include <tracking/Output.h>
#include <gui/dyn/Action.h>
#include <gui/GUICache.h>
#include <gui/AnimatedBackground.h>
#include <gui/Coordinates.h>
#include <gui/ScreenRecorder.h>
#include <gui/Bowl.h>
#include <gui/types/Button.h>
#include <gui/types/Layout.h>
#include <gui/TrackingState.h>
#include <misc/SettingsInitializer.h>
#include <tracking/CategorizeInterface.h>

using namespace track;

namespace gui {

struct TrackingScene::Data {
    
    std::unique_ptr<GUICache> _cache;
    
    std::unique_ptr<Bowl> _bowl;
    std::unordered_map<Idx_t, Bounds> _last_bounds;
    
    std::unique_ptr<AnimatedBackground> _background;
    std::unique_ptr<ExternalImage> _gui_mask;
    
    std::function<void(Vec2, bool, std::string)> _clicked_background;
    double _time_since_last_frame{0};

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
    pv::Frame _frame;
    
    // The dynamic part of the gui that is live-loaded from file
    dyn::DynamicGUI dynGUI;
    std::vector<std::shared_ptr<dyn::VarBase_t>> _individuals;
    std::vector<sprite::Map> _fish_data;
    
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
         const pv::File& video);
};

TrackingScene::~TrackingScene() {
    
}

const static std::unordered_map<std::string_view, gui::Keyboard::Codes> _key_map {
    {"esc", Keyboard::Escape},
    {"left", Keyboard::Left},
    {"right", Keyboard::Right},
    {"lshift", Keyboard::LShift},
    {"lctrl", Keyboard::LControl},
    {"rctrl", Keyboard::RControl},
    {"lalt", Keyboard::LAlt},
    {"ralt", Keyboard::RAlt}
};

TrackingScene::Data::Data(Image::Ptr&& average, const pv::File& video)
{
    _background = std::make_unique<AnimatedBackground>(std::move(average), &video);
    
    _background->add_event_handler(EventType::MBUTTON, [this](Event e){
        if(e.mbutton.pressed) {
            if(_clicked_background)
                _clicked_background(Vec2(e.mbutton.x, e.mbutton.y).map<round>(), e.mbutton.button == 1, "");
            else
                print("Clicked, but nobody is around.");
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
    print("window dimensions", window.window_dimensions().mul(dpi));
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
                SETTING(terminate) = true;
                break;
            case Keyboard::Space:
                SETTING(gui_run) = not SETTING(gui_run).value<bool>();
                break;
            case Keyboard::M:
                next_poi(Idx_t());
                break;
            case Keyboard::N:
                prev_poi(Idx_t());
                break;
            case Keyboard::L:
                load_state(Output::TrackingResults::expected_filename());
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
                WorkProgress::add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { _state->export_tracks("", {}, {}); });
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
                _exec_main_queue.enqueue([](IMGUIBase* base, DrawStructure& graph){
                    if(graph.is_key_pressed(Codes::LSystem))
                    {
                        base->toggle_fullscreen(graph);
                    }
                });
                break;
            }
            case Keyboard::F11:
                _exec_main_queue.enqueue([](IMGUIBase* base, DrawStructure& graph){
                    base->toggle_fullscreen(graph);
                });
                break;
            case Keyboard::R: {
                if(_data) {
                    _exec_main_queue.enqueue([this](IMGUIBase* base, DrawStructure& graph){
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
    _state = std::make_unique<TrackingState>(&_exec_main_queue);
    
    //! Stages
    _data = std::unique_ptr<Data>{
        new Data{
            Image::Make(_state->video.average()),
            _state->video
        }
    };
    
    _data->_callback = GlobalSettings::map().register_callbacks({
        "gui_focus_group",
        "gui_run",
        "analysis_paused",
        "analysis_range"
        
    }, [this](std::string_view key) {
        if(key == "gui_focus_group" && _data->_bowl)
            _data->_bowl->_screen_size = Vec2();
        else if(key == "gui_run") {
            
        } else if(key == "analysis_paused") {
            gui::WorkProgress::add_queue("pausing...", [this](){
                _state->analysis.bump();
                bool pause = SETTING(analysis_paused).value<bool>();
                if(_state->analysis.paused() != pause) {
                    _state->analysis.set_paused(pause).get();
                }
            });
        } else if(key == "analysis_range") {
            _data->_analysis_range = Tracker::analysis_range();
        }
        
    });
    
    _data->_analysis_range = Tracker::analysis_range();
    _state->init_video();
    
    RecentItems::open(SETTING(source).value<file::PathArray>().source(), GlobalSettings::map());
}

void TrackingScene::deactivate() {
    ThreadManager::getInstance().printThreadTree();
    if(_data && _data->_recorder.recording()) {
        _data->_recorder.stop_recording(nullptr, nullptr);
        if(_data->_background)
            _data->_background->set_strict(false);
    }
    
    WorkProgress::stop();
    
    if(_data)
        _data->dynGUI.clear();
    tracker::gui::blob_view_shutdown();
    
    print("Preparing for shutdown...");
#if !COMMONS_NO_PYTHON
    CheckUpdates::cleanup();
    Categorize::terminate();
#endif
    
    if(_data && _data->_callback)
        GlobalSettings::map().unregister_callbacks(std::move(_data->_callback));
    
    if(_state)
        _state->analysis.terminate();
    _state = nullptr;
    _data = nullptr;
    
    SETTING(filename) = file::Path();
    SETTING(source) = file::PathArray();
}

void TrackingScene::set_frame(Frame_t frameIndex) {
    if(frameIndex <= _state->video.length()
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
        _data->_cache->set_dt(1.0 / double(FAST_SETTING(frame_rate)));
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
        
        if(auto L = _state->video.length();
           index >= L) 
        {
            index = L.try_sub(1_f);
            SETTING(gui_run) = false;
        }
        set_frame(index);
        
    } else {
        _data->_time_since_last_frame += dt;
        
        double advances = _data->_time_since_last_frame * frame_rate;
        if(advances >= 1) {
            index += Frame_t(uint(advances));
            if(auto L = _state->video.length();
               index >= L)
            {
                index = L.try_sub(1_f);
                SETTING(gui_run) = false;
            }
            set_frame(index);
            _data->_time_since_last_frame = 0;
        }
    }
}

void TrackingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not _data->dynGUI)
         init_gui(_data->dynGUI, graph);
    
    update_run_loop();
    if(_data)
        _exec_main_queue.processTasks(static_cast<IMGUIBase*>(window()), graph);
    else
        return;
    
    if(window()) {
        auto update = FindCoord::set_screen_size(graph, *window());
        if(update != window_size)
            window_size = update;
    }
    //window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height).div(((IMGUIBase*)window())->dpi_scale()) * gui::interface_scale();
    
    if(not _data->_cache) {
        _data->_cache = std::make_unique<GUICache>(&graph, &_state->video);
        _data->_bowl = std::make_unique<Bowl>(_data->_cache.get());
        _data->_bowl->set_video_aspect_ratio(_state->video.size().width, _state->video.size().height);
        _data->_bowl->fit_to_screen(window_size);
        
        _data->_clicked_background = [&](const Vec2& pos, bool v, std::string key) {
            tracker::gui::clicked_background(graph, *_data->_cache, pos, v, key);
        };
    }
    
    auto mouse = graph.mouse_position();
    if(mouse != _data->_last_mouse || _data->_cache->is_animating() || graph.root().is_animating()) {
        if(((IMGUIBase*)window())->focussed()) {
            _data->_cache->set_blobs_dirty();
            _data->_cache->set_tracking_dirty();
        }
        _data->_last_mouse = mouse;
    }
    
    if(false) {
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
    
    if(_data->_cache) {
        auto frameIndex = GUI_SETTINGS(gui_frame);
        Frame_t loaded;
        //do {
            loaded = _data->_cache->update_data(frameIndex);
            
        //} while(_data->_recorder.recording() && loaded.valid() && loaded != frameIndex);
        
        if(loaded.valid() || _data->_cache->fish_dirty()) {
            //print("Update all... ", loaded, "(",frameIndex,")");
            
            if(loaded.valid())
                SETTING(gui_displayed_frame) = loaded;
            using namespace dyn;
            
            _data->_individuals.resize(_data->_cache->raw_blobs.size());
            _data->_fish_data.resize(_data->_individuals.size());
            for(size_t i=0; i<_data->_cache->raw_blobs.size(); ++i) {
                auto &var = _data->_individuals[i];
                if(not var)
                    var = std::unique_ptr<VarBase_t>(new Variable([i, this](const VarProps&) -> sprite::Map& {
                        return _data->_fish_data.at(i);
                    }));
                
                auto &map = _data->_fish_data.at(i);
                auto &fish = _data->_cache->raw_blobs[i];
                map["pos"] = Vec2(fish->blob->bounds().pos());
            }
            
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
                bool found = false;
                if (auto it = _data->_cache->fish_selected_blobs.find(fdx);
                    it != _data->_cache->fish_selected_blobs.end()) 
                {
                    auto bdx = _data->_cache->fish_selected_blobs.at(fdx);
                    for (auto& blob : _data->_cache->raw_blobs) {
                        if (blob->blob &&
                            (blob->blob->blob_id() == bdx || blob->blob->parent_id() == bdx)) {
                            auto& bds = blob->blob->bounds();
                            targets.push_back(bds.pos());
                            targets.push_back(bds.pos() + bds.size());
                            targets.push_back(bds.pos() + bds.size().mul(0, 1));
                            targets.push_back(bds.pos() + bds.size().mul(1, 0));
                            _data->_last_bounds[fdx] = bds;
                            found = true;
                            break;
                        }
                    }
                }

                if (not found) {
                    if (_data->_last_bounds.contains(fdx)) {
                        auto& bds = _data->_last_bounds.at(fdx);
                        targets.push_back(bds.pos());
                        targets.push_back(bds.pos() + bds.size());
                        targets.push_back(bds.pos() + bds.size().mul(0, 1));
                        targets.push_back(bds.pos() + bds.size().mul(1, 0));
                    }
                }
            }
            
            std::vector<Idx_t> remove;
            for(auto &[fdx, bds] : _data->_last_bounds) {
                if(not contains(_data->_cache->selected, fdx)
                   || (not contains(_data->_cache->active_ids, fdx)
                   && not contains(_data->_cache->inactive_ids, fdx)))
                {
                    remove.push_back(fdx);
                    print("* removing individual ", fdx);
                }
            }
            
            //print("active = ", _data->_cache->active_ids);
            //print("inactive = ", _data->_cache->inactive_ids);
            
            for(auto fdx: remove)
                _data->_last_bounds.erase(fdx);
        }
        
        _data->_bowl->fit_to_screen(window_size);
        _data->_bowl->set_target_focus(targets);
        _data->_zoom_dirty = false;
        _data->_cache->updated_tracking();
    }
    
    _data->_bowl->update_scaling();
    
    auto coords = FindCoord::get(); 
    
    auto alpha = SETTING(gui_background_color).value<Color>().a;
    _data->_background->set_color(Color(255, 255, 255, alpha ? alpha : 1));

    if (alpha > 0) {
        graph.wrap_object(*_data->_background);
        /*if(PD(gui_mask)) {
            PD(gui_mask)->set_color(PD(background)->color().alpha(PD(background)->color().a * 0.5));
            PD(gui).wrap_object(*PD(gui_mask));
        }*/
        _data->_background->set_scale(_data->_bowl->scale());
        _data->_background->set_pos(_data->_bowl->pos());
    }
    
    //if(GUI_SETTINGS(gui_mode) == mode_t::tracking)
    {
        graph.wrap_object(*_data->_bowl);
    }

    _data->_bowl->update(_data->_cache->frame_idx, graph, coords);
    _data->_bowl_mouse = coords.convert(HUDCoord(graph.mouse_position())); //_data->_bowl->global_transform().getInverse().transformPoint(graph.mouse_position());
    
    /*const auto mode = GUI_SETTINGS(gui_mode);
    const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
    update_display_blobs(draw_blobs);*/
    
    //_data->_bowl.auto_size({});
    //_data->_bowl->set(LineClr{Cyan});
    //_data->_bowl.set(FillClr{Yellow});
    
    if(GUI_SETTINGS(gui_mode) == mode_t::blobs) {
        tracker::gui::draw_blob_view({
            .graph = graph,
            .cache = *_data->_cache,
            .coord = coords
        });
    }
    
    tracker::gui::draw_boundary_selection(graph, window(), *_data->_cache, _data->_bowl.get());
    
    _data->dynGUI.update(nullptr);
    
    Categorize::draw(_state->video, (IMGUIBase*)window(), graph);
    
    graph.section("loading", [this](DrawStructure& base, auto section) {
        WorkProgress::update((IMGUIBase*)window(), base, section, window_size);
    });
    //
    
    if(not graph.root().is_dirty())
        std::this_thread::sleep_for(std::chrono::milliseconds(((IMGUIBase*)window())->focussed() ? 10 : 200));
    //print("dirty = ", graph.root().is_dirty());
    if(graph.root().is_dirty())
        last_dirty.reset();
    else if((_data->_cache->is_animating()
       && last_dirty.elapsed() > 0.1)
            || last_dirty.elapsed() > 0.25)
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
    auto next_frame = frame;
    std::set<FOI::fdx_t> fdx;
    
    {
        for(const FOI& foi : _data->_foi_state.changed_frames) {
            if(_s_fdx.valid()) {
                if(not foi.fdx().contains(FOI::fdx_t(_s_fdx)))
                    continue;
            }
            
            if(not frame.valid() || foi.frames().end < frame) {
                next_frame = foi.frames().end;
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

void TrackingScene::init_gui(dyn::DynamicGUI& dynGUI, DrawStructure& graph) {
    using namespace dyn;
    dyn::DynamicGUI g{
        .path = "tracking_layout.json",
        .graph = &graph,
        .context = {
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
                load_state(Output::TrackingResults::expected_filename());
            }),
            ActionFunc("save_results", [this](Action) {
                save_state(false);
            }),
            ActionFunc("export_data", [this](Action){
                WorkProgress::add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { _state->export_tracks("", {}, {}); });
            }),
            ActionFunc("write_config", [this](Action){
                WorkProgress::add_queue("", [this]() { settings::write_config(false, &_exec_main_queue); });
            }),
            ActionFunc("categorize", [this](Action){
                Categorize::show(&_state->video,
                    [this](){
                        if(SETTING(auto_quit))
                            _state->auto_quit();
                        //GUI::instance()->auto_quit();
                    },
                    [](const std::string& text){
                        print(text);
                        //GUI::instance()->set_status(text);
                    }
                );
            }),
            ActionFunc("auto_correct", [this](Action){
                _state->auto_correct(&_exec_main_queue);
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
                    LockGuard guard(ro_t{}, "active");
                    if(_state->tracker->properties(frame))
                        return Tracker::active_individuals(frame).size();
                }
                
                throw std::invalid_argument("Frame "+Meta::toStr(frame)+" not tracked.");
            }),
            VarFunc("live_individuals", [this](const VarProps& props) -> size_t {
                if(props.parameters.size() != 1)
                    throw std::invalid_argument("Need exactly one argument for "+props.toStr());
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                {
                    LockGuard guard(ro_t{}, "active");
                    if(_state->tracker->properties(frame)) {
                        size_t count{0u};
                        for(auto fish : Tracker::active_individuals(frame)) {
                            if(fish->has(frame))
                                ++count;
                        }
                        return count;
                    }
                }
                
                throw std::invalid_argument("Frame "+Meta::toStr(frame)+" not tracked.");
            }),
            VarFunc("window_size", [this](const VarProps&) -> Vec2 {
                return window_size;
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
                static std::vector<sprite::Map> segments;
                static std::vector<std::shared_ptr<VarBase_t>> variables;
                
                ColorWheel wheel;
                for(size_t i=0; i<3 && i < consec.size(); ++i) {
                    if(segments.size() <= i) {
                        segments.emplace_back();
                        variables.emplace_back(new Variable([i](const VarProps&) -> sprite::Map& {
                            return segments.at(i);
                        }));
                        assert(variables.size() == segments.size());
                    }
                    
                    auto& map = segments.at(i);
                    //if(map.print_by_default())
                    //    map.set_print_by_default(false);
                    map["color"] = wheel.next();
                    map["start"] = consec.at(i).start;
                    map["end"] = consec.at(i).end + 1_f;
                }
                
                return variables;
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
            })
        }
    };
    
    g.context.custom_elements["image"] = CustomElement{
        .name = "image",
        .create = [](LayoutContext&) -> Layout::Ptr {
            return Layout::Make<ExternalImage>();
        },
        .update = [](Layout::Ptr&, const Context&, State&, const robin_hood::unordered_map<std::string, Pattern>&) {
            
        }
    };
    
    dynGUI = std::move(g);
}

void TrackingScene::save_state(bool force_overwrite) {
    static bool save_state_visible = false;
    if(save_state_visible)
        return;
    
    save_state_visible = true;
    static file::Path file;
    file = Output::TrackingResults::expected_filename();
    
    auto fn = [this]() {
        bool before = _state->analysis.is_paused();
        _state->analysis.set_paused(true).get();
        
        LockGuard guard(w_t{}, "GUI::save_state");
        try {
            Output::TrackingResults results(*_state->tracker);
            results.save([](const std::string& title, float x, const std::string& description){ WorkProgress::set_progress(title, x, description); }, file);
        } catch(const UtilsException&e) {
            auto what = std::string(e.what());
            _exec_main_queue.enqueue([what](auto, DrawStructure& graph){
                graph.dialog([](Dialog::Result){}, "Something went wrong saving the program state. Maybe no write permissions? Check out this message, too:\n<i>"+what+"</i>", "Error");
            });
            
            FormatExcept("Something went wrong saving program state. Maybe no write permissions?"); }
        
        if(!before)
            _state->analysis.set_paused(false).get();
        
        save_state_visible = false;
    };
    
    if(file.exists() && !force_overwrite) {
        _exec_main_queue.enqueue([fn](auto, DrawStructure& graph){
            graph.dialog([fn](Dialog::Result result) {
                if(result == Dialog::Result::OKAY) {
                    WorkProgress::add_queue("Saving results...", fn);
                } else if(result == Dialog::Result::SECOND) {
                    do {
                        if(file.remove_filename().empty()) {
                            file = file::Path("backup_" + file.str());
                        } else
                            file = file.remove_filename() / ("backup_" + (std::string)file.filename());
                    } while(file.exists());
                    
                    auto expected = Output::TrackingResults::expected_filename();
                    if(expected.move_to(file)) {
                        file = expected;
                        WorkProgress::add_queue("Saving backup...", fn);
                    //if(std::rename(expected.str().c_str(), file->str().c_str()) == 0) {
//                          *file = expected;
//                            work().add_queue("Saving backup...", fn);
                    } else {
                        FormatExcept("Cannot rename ",expected," to ",file,".");
                        save_state_visible = false;
                    }
                } else
                    save_state_visible = false;
                
            }, "Overwrite tracking previous results at <i>"+file.str()+"</i>?", "Overwrite", "Yes", "Cancel", "Backup old one");
        });
        
    } else
        WorkProgress::add_queue("Saving results...", fn);
}

void TrackingScene::load_state(file::Path from) {
    static bool state_visible = false;
    if(state_visible)
        return;
    
    state_visible = true;

    auto fn = [this, from]() {
        bool before = _state->analysis.is_paused();
        _state->analysis.set_paused(true).get();
        
        Categorize::DataStore::clear();
        
        LockGuard guard(w_t{}, "GUI::load_state");
        Output::TrackingResults results{*_state->tracker};
        
        try {
            auto header = results.load([](const std::string& title, float value, const std::string& desc) {
                WorkProgress::set_progress(title, value, desc);
            }, from);
            
            if(header.version <= Output::ResultsFormat::Versions::V_33
               && !Tracker::instance()->vi_predictions().empty())
            {
                // probably need to convert blob ids
                pv::Frame f;
                size_t found = 0;
                size_t N = 0;
                
                for (auto &[k, v] : Tracker::instance()->vi_predictions()) {
                    _state->video.read_frame(f, k);
                    auto blobs = f.get_blobs();
                    N += v.size();
                    
                    for(auto &[bid, ps] : v) {
                        auto it = std::find_if(blobs.begin(), blobs.end(), [&bid=bid](auto &a)
                        {
                            return a->blob_id() == bid || a->parent_id() == bid;
                        });
                        
                        auto id = uint32_t(bid);
                        auto x = id >> 16;
                        auto y = id & 0x0000FFFF;
                        auto center = Vec2(x, y);
                        
                        if(it != blobs.end() || x > Tracker::average().cols || y > Tracker::average().rows) {
                            // blobs are probably fine
                            ++found;
                        } else {
                            
                        }
                    }
                    
                    if(found * 2 > N) {
                        // blobs are probably fine!
                        print("blobs are probably fine ",found,"/",N,".");
                        break;
                    } else if(N > 0) {
                        print("blobs are probably not fine.");
                        break;
                    }
                }
                
                if(found * 2 <= N && N > 0) {
                    print("fixing...");
                    WorkProgress::set_item("Fixing old blob_ids...");
                    WorkProgress::set_description("This is necessary because you are loading an <b>old</b> .results file with <b>visual identification data</b> and, since the format of blob_ids has changed, we would otherwise be unable to associate the objects with said visual identification info.\n<i>If you want to avoid this step, please use the older TRex version to load the file or let this run and overwrite the old .results file (so you don't have to wait again). Be careful, however, as information might not transfer over perfectly.</i>\n");
                    auto old_id_from_position = [](Vec2 center) {
                        return (uint32_t)( uint32_t((center.x))<<16 | uint32_t((center.y)) );
                    };
                    auto old_id_from_blob = [&old_id_from_position](const pv::Blob &blob) -> uint32_t {
                        if(!blob.lines() || blob.lines()->empty())
                            return -1;
                        
                        const auto start = Vec2(blob.lines()->front().x0,
                                                blob.lines()->front().y);
                        const auto end = Vec2(blob.lines()->back().x1,
                                              blob.lines()->size());
                        
                        return old_id_from_position(start + (end - start) * 0.5);
                    };
                    
                    grid::ProximityGrid proximity{ Tracker::average().bounds().size() };
                    size_t i=0, all_found = 0, not_found = 0;
                    const size_t N = Tracker::instance()->vi_predictions().size();
                    ska::bytell_hash_map<Frame_t, ska::bytell_hash_map<pv::bid, std::vector<float>>> next_recognition;
                    
                    for (auto &[k, v] : Tracker::instance()->vi_predictions()) {
                        auto & active = Tracker::active_individuals(k);
                        ska::bytell_hash_map<pv::bid, const pv::CompressedBlob*> blobs;
                        
                        for(auto fish : active) {
                            auto b = fish->compressed_blob(k);
                            if(b) {
                                auto bounds = b->calculate_bounds();
                                auto center = bounds.pos() + bounds.size() * 0.5;
                                blobs[b->blob_id()] = b;
                                proximity.insert(center.x, center.y, b->blob_id());
                            }
                        }
                        /*GUI::instance()->video_source()->read_frame(f, k.get());
                        auto & blobs = f.blobs();
                        proximity.clear();
                        for(auto &b : blobs) {
                            auto c = b->bounds().pos() + b->bounds().size() * 0.5;
                            proximity.insert(c.x, c.y, (uint32_t)b->blob_id());
                        }*/
                        
                        ska::bytell_hash_map<pv::bid, std::vector<float>> tmp;
                        
                        for(auto &[bid, ps] : v) {
                            auto id = uint32_t(bid);
                            auto x = id >> 16;
                            auto y = id & 0x0000FFFF;
                            auto center = Vec2(x, y);
                            
                            auto r = proximity.query(center, 1);
                            if(r.size() == 1) {
                                auto obj = std::get<1>(*r.begin());
                                assert(obj.valid());
                                /*auto ptr = std::find_if(blobs.begin(), blobs.end(), [obj](auto &b){
                                    return obj == (uint32_t)b->blob_id();
                                });*/
                                /*auto ptr = blobs.find(pv::bid(obj));
                                
                                if(ptr == blobs.end()) {
                                    FormatError("Cannot find actual blob for ", obj);
                                } else {
                                    //auto unpack = ptr->second->unpack();
                                    //print("Found ", center, " as ", obj, " vs. ", id, "(", old_id_from_blob(*unpack) ," / ", *unpack ,")");
                                }*/
                                    tmp[obj] = ps;
                                    ++all_found;
                                
                            } else {
                                const pv::CompressedBlob* found = nullptr;
                                _state->video.read_frame(f, k);
                                for(auto &b : f.get_blobs()) {
                                    auto c = b->bounds().pos() + b->bounds().size() * 0.5;
                                    if(sqdistance(c, center) < 2) {
                                        //print("Found blob close to ", center, " at ", c, ": ", *b);
                                        for(auto &fish : active) {
                                            auto b = fish->compressed_blob(k);
                                            if(b && (b->blob_id() == bid || b->parent_id == bid))
                                            {
                                                //print("Equal IDS1 ", b->blob_id(), " and ", id);
                                                tmp[b->blob_id()] = ps;
                                                found = b;
                                                break;
                                            }
                                            
                                            if(b) {
                                                auto bounds = b->calculate_bounds();
                                                auto center = bounds.pos() + bounds.size() * 0.5;
                                                
                                                auto distance = sqdistance(c, center);
                                                //print("\t", fish->identity(), ": ", b->blob_id(), "(",b->parent_id,") at ", center, " (", distance, ")", (distance < 5 ? "*" : ""));
                                                
                                                if(distance < 2) {
                                                    tmp[b->blob_id()] = ps;
                                                    found = b;
                                                    break;
                                                }
                                            }
                                        }
                                        
                                        tmp[b->blob_id()] = ps;
                                        break;
                                    }
                                }
                                
                                if(found == nullptr) {
                                    //print("Not found for ", center, " size=", r.size(), " with id ", bid);
                                    ++not_found;
                                } else {
                                    ++all_found;
                                }
                            }
                        }
                        
                        //v = tmp;
                        next_recognition[k] = tmp;
                        
                        ++i;
                        if(i % uint64_t(N * 0.1) == 0) {
                            print("Correcting old-format pv::bid: ", dec<2>(double(i) / double(N) * 100), "%");
                            WorkProgress::set_percent(double(i) / double(N));
                        }
                    }
                    
                    print("Found:", all_found, " not found:", not_found);
                    if(all_found > 0)
                        Tracker::instance()->set_vi_data(next_recognition);
                }
            }
            
            {
                sprite::Map config;
                GlobalSettings::docs_map_t docs;
                default_config::get(config, docs, NULL);
                try {
                    default_config::load_string_with_deprecations(from.str(), header.settings, config, AccessLevelType::STARTUP, {}, true);
                    
                } catch(const cmn::illegal_syntax& e) {
                    print("Illegal syntax in .results settings (",e.what(),").");
                }
                
                std::vector<Idx_t> focus_group;
                if(config.has("gui_focus_group"))
                    focus_group = config["gui_focus_group"].value<std::vector<Idx_t>>();
                
                //if(GUI::instance() && !gui_frame_on_startup().frame.valid()) {
                //    WorkProgress::add_queue("", [f = Frame_t(header.gui_frame)](){
                        SETTING(gui_frame) = Frame_t(header.gui_frame);
                //    });
                //}
                
                //if(GUI::instance() && !gui_frame_on_startup().focus_group.has_value()) {
                //    WorkProgress::add_queue("", [focus_group](){
                        SETTING(gui_focus_group) = focus_group;
                //    });
                //}
                
            }
            
            if((header.analysis_range.start != -1 || header.analysis_range.end != -1) && SETTING(analysis_range).value<std::pair<long_t, long_t>>() == std::pair<long_t,long_t>{-1,-1})
            {
                SETTING(analysis_range) = std::pair<long_t, long_t>(header.analysis_range.start, header.analysis_range.end);
            }
            
            WorkProgress::add_queue("", [](){
                Tracker::instance()->check_segments_identities(false, IdentitySource::VisualIdent, [](float ) { },
                [](const std::string&t, const std::function<void()>& fn, const std::string&b)
                {
                    WorkProgress::add_queue(t, fn, b);
                });
            });
            
        } catch(const UtilsException& e) {
            FormatExcept("Cannot load results. Crashed with exception: ", e.what());
            
            auto what = std::string(e.what());
            _exec_main_queue.enqueue([from, what](IMGUIBase* base, DrawStructure& graph) {
                graph.dialog([](Dialog::Result){}, "Cannot load results from '"+from.str()+"'. Loading crashed with this message:\n<i>"+what+"</i>", "Error");
            });
            
            auto start = Tracker::start_frame();
            Tracker::instance()->_remove_frames(start);
                //removed_frames(start);
        }
        
        //PD(analysis).reset_PD(cache);
        Output::Library::clear_cache();
        
        /*auto range = PD(tracker).analysis_range();
        bool finished = (PD(tracker).end_frame().valid() && PD(tracker).end_frame() == range.end()) || PD(tracker).end_frame() >= range.end();
#if !COMMONS_NO_PYTHON
        if(finished && SETTING(auto_categorize)) {
            auto_categorize();
        } else if(finished && SETTING(auto_train)) {
            auto_train();
        }
        else if(finished && SETTING(auto_apply)) {
            auto_apply();
        }
        else if(finished && SETTING(auto_tags)) {
            auto_tags();
        }
        else if(finished && SETTING(auto_quit)) {
#else
        if(finished && SETTING(auto_quit)) {
#endif
#if WITH_SFML
            if(has_window())
                window().setVisible(false);
#endif
            
            try {
                this->export_tracks();
            } catch(const UtilsException&) {
                SETTING(error_terminate) = true;
            }
            
            SETTING(terminate) = true;
        }
        
        if(GUI::instance() && (!before || (!finished && SETTING(auto_quit))))
            PD(analysis).set_paused(false).get();*/
        
        state_visible = false;
    };
    
    /*if (type == GRAPHICAL) {
        PD(gui).dialog([fn](Dialog::Result result) {
            if(result == Dialog::Result::OKAY) {
                WorkProgress::add_queue("Loading results...", fn, PD(video_source).filename().str());
            } else {
                state_visible = false;
            }
            
        }, "Are you sure you want to load results?\nThis will discard any unsaved changes.", "Load results", "Yes", "Cancel");
    } else {
        WorkProgress::add_queue("Loading results...", fn, PD(video_source).filename().str());
    }*/
    WorkProgress::add_queue("Loading results...", fn);
}

}
