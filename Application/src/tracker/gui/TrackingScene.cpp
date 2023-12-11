#include "TrackingScene.h"
#include <misc/GlobalSettings.h>
#include <file/DataLocation.h>
#include <gui/IMGUIBase.h>
#include <gui/types/ListItemTypes.h>
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
#include <gui/WorkProgress.h>
#include <gui/DrawBlobView.h>
#include <misc/Output.h>
#include <tracking/Export.h>
#include <misc/IdentifiedTag.h>

using namespace track;

namespace gui {

static constexpr Frame_t cache_size{Frame_t::number_t(10)};

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

TrackingScene::Data::Data(Image::Ptr&& average, pv::File&& video, std::vector<std::function<bool(ConnectedTasks::Type&&, const ConnectedTasks::Stage&)>>&& stages)
    : video(std::move(video)),
      tracker(std::move(average), this->video),
      analysis(std::move(stages)),
      pool(4u, "preprocess_main")
{
    gpuMat bg;
    video.average().copyTo(bg);
    video.processImage(bg, bg, false);
    cv::Mat original;
    bg.copyTo(original);
    
    _background = std::make_unique<AnimatedBackground>(Image::Make(original), &this->video);
    
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

void TrackingScene::Data::Statistics::calculateRates(double elapsed) {
    const auto frames_sec = frames_count / elapsed;
    frames_per_second = frames_sec;
    individuals_per_second = acc_individuals / sample_individuals;
    
    acc_frames += frames_sec;
    ++sample_frames;
    
    frames_count = 0;
    sample_individuals = 0;
    acc_individuals = 0;
}

void TrackingScene::Data::Statistics::printProgress(float percent, const std::string& status) {
    // Assuming we have a terminal width of 50 characters for the progress bar.
    constexpr int bar_width = 50;
    int pos = int(bar_width * (percent / 100.0f));

    printf("\r["); // Carriage return to overwrite the previous line
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) printf("=");
        else if (i == pos) printf(">"); // Indicator for current position
        else printf(" ");
    }
    printf("] %.2f%% %s", percent, status.c_str()); // Print the percentage and status message
    fflush(stdout); // Flush the output to ensure it appears immediately
}

void TrackingScene::Data::Statistics::logProgress(float percent, const std::string& status) {
    if (print_timer.elapsed() > 30) {
        // Here we use print(...) as if it's a member function similar to printf, but with
        // the behavior as specified (e.g., taking arbitrary arguments and producing reasonable
        // textual representations of objects, ending on a newline automatically).
        print("[Statistics] Progress: ", dec<2>(percent), "% ", status.c_str());
        print_timer.reset(); // Reset the timer to log again after the specified interval
    }
}

constexpr const char* time_unit() {
#if defined(__APPLE__)
    return "µs";
#else
    return "mus";
#endif
}

void TrackingScene::Data::Statistics::updateProgress(Frame_t frame, const FrameRange& analysis_range, Frame_t video_length, bool) {
    float percent = min(1.f, (frame - analysis_range.start()).get() / float(analysis_range.length().try_sub(1_f).get())) * 100.f;
    DurationUS us{ uint64_t(max(0, (double)(analysis_range.end() - frame).get() / double( acc_frames / sample_frames ) * 1000 * 1000)) };

    // Construct status string
    std::string status;
    auto prefix = FAST_SETTING(individual_prefix);
    if(frame == analysis_range.end()) {
        status = format<FormatterType::NONE>("Done (",
            dec<2>(frames_per_second.load()), "fps ",
            dec<2>(individuals_per_second.load()),"ind/s ",dec<2>(Tracker::average_seconds_per_individual() * 1000 * 1000), time_unit(), "/", prefix.c_str(),").") + "\n";
        printf("\r\n");

    } else if(FAST_SETTING(analysis_range).first != -1
       || FAST_SETTING(analysis_range).second != -1)
    {
        status = format<FormatterType::NONE>("frame ", frame, "/", analysis_range.end(), "(", video_length, ") @ ",
             dec<2>(frames_per_second.load()), "fps ", dec<2>(individuals_per_second.load()),"ind/s, eta ", us, ") ",
             dec<2>(Tracker::average_seconds_per_individual() * 1000 * 1000), time_unit(), "/", prefix.c_str()
        );
    } else {
        status = format<FormatterType::NONE>("frame ", frame, "/", analysis_range.end(), " (", dec<2>(frames_per_second.load()), "fps ",
             dec<2>(individuals_per_second.load()),"ind/s, eta ", us, ") ", dec<2>(Tracker::average_seconds_per_individual() * 1000 * 1000), time_unit(), "/", prefix.c_str()
        );
    }

    // Print progress to console
    printProgress(percent, status);

    // Log progress to file or wherever necessary
    logProgress(percent, status);
}


void TrackingScene::Data::Statistics::update(Frame_t frame, const FrameRange& analysis_range, Frame_t video_length, uint32_t num_individuals, bool force)
{
    frames_count++;
    acc_individuals += num_individuals;
    sample_individuals++;
    
    double elapsed = timer.elapsed();
    if ((elapsed >= 1 || force) && not analysis_range.empty()) {
        timer.reset();
        
        calculateRates(elapsed);
        updateProgress(frame, analysis_range, video_length, force);
    }
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

void TrackingScene::export_tracks(const file::Path& , Idx_t fdx, Range<Frame_t> range) {
    bool before = _data->analysis.is_paused();
    _data->analysis.set_paused(true).get();
    
    track::export_data(_data->video, _data->tracker, fdx, range);
    
    if(not before)
        _data->analysis.set_paused(false).get();
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
                    _data->analysis.set_paused(not _data->analysis.paused()).get();
                });
                break;
            case Keyboard::S:
                WorkProgress::add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { export_tracks("", {}, {}); });
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
                _data->_exec_main_queue.enqueue([](IMGUIBase* base, DrawStructure& graph){
                    if(graph.is_key_pressed(Codes::LSystem))
                    {
                        base->toggle_fullscreen(graph);
                    }
                });
                break;
            }
            case Keyboard::F11:
                _data->_exec_main_queue.enqueue([](IMGUIBase* base, DrawStructure& graph){
                    base->toggle_fullscreen(graph);
                });
                break;
            case Keyboard::R: {
                if(_data) {
                    _data->_exec_main_queue.enqueue([this](IMGUIBase* base, DrawStructure& graph){
                        if(_data->_recorder.recording()) {
                            _data->_recorder.stop_recording(base, &graph);
                        } else {
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
    try {
        pv::Frame frame;
        _data->video.read_frame(frame, idx);
        Tracker::preprocess_frame(std::move(frame), *ptr, _data->pool.num_threads() > 1 ? &_data->pool : NULL, PPFrame::NeedGrid::NoNeed, _data->video.header().resolution, false);

        ptr->set_loading_time(timer.elapsed());
    }
    catch (const std::exception& e) {
		print("Error while preprocessing frame ", idx, ": ", e.what());
        return false;
	}

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
        
        if(idx + 1_f == _data->video.length()
           || idx + 1_f > Tracker::analysis_range().end())
        {
            on_tracking_done();
        }
        
        static Timer last_added;
        if(last_added.elapsed() > 10) {
            _data->tracker.global_segment_order();
            last_added.reset();
        }
        
        //print(_data->tracker.active_individuals(idx));
        _data->_stats.update(idx, range, _data->video.length(), _data->tracker.statistics().at(idx).number_fish, idx == range.end());
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
    
    grab::default_config::get(combined.map, combined.docs, set_combined_access_level);
    default_config::get(combined.map, combined.docs, nullptr);
    
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

    thread_print("source = ", SETTING(source).value<file::PathArray>(), " ", (uint64_t)&GlobalSettings::map());
    GlobalSettings::map().set_print_by_default(true);
    //default_config::get(GlobalSettings::map(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    //default_config::get(GlobalSettings::set_defaults(), GlobalSettings::docs(), &GlobalSettings::set_access_level);
    GlobalSettings::map()["gui_frame"].get().set_do_print(false);
    GlobalSettings::map()["gui_focus_group"].get().set_do_print(false);
    
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
    
    //cmd.load_settings(&combined);
    
    //! TODO: have to delegate this to another thread
    //! otherwise we will get stuck here
    bool executed_a_settings{false};
    thread_print("source = ", SETTING(source).value<file::PathArray>(), " ", (uint64_t)&GlobalSettings::map());
    auto path = SETTING(source).value<file::PathArray>().empty()
        ? file::Path()
        : SETTING(source).value<file::PathArray>().get_paths().front();

    file::Path filename = file::DataLocation::parse("input", path);
    SETTING(filename) = filename.remove_extension();
    pv::File video(filename, pv::FileMode::READ);
    
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
    }
    
    std::vector<std::string> exclude_parameters{
        "source"
    };
    for (auto& [key, value] : cmd.settings_keys()) {
        exclude_parameters.push_back(key);
    }
    print("meta_source_path = ", SETTING(meta_source_path).value<std::string>());
    print("track_max_individuals = ", SETTING(track_max_individuals).value<uint32_t>());
    print("exclude_parameters = ", exclude_parameters);

    try {
        if (!video.header().metadata.empty()) {
            sprite::parse_values(GlobalSettings::map(), video.header().metadata, &combined, exclude_parameters);
        }
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
    Output::Library::Init();
    
    auto settings_file = file::DataLocation::parse("settings");
    if(settings_file.exists()) {
        if(default_config::execute_settings_file(settings_file, AccessLevelType::STARTUP, exclude_parameters))
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
    //cmd.load_settings(&combined);
    
    SETTING(gui_interface_scale) = float(1);
    print("cm_per_pixel = ", SETTING(cm_per_pixel).value<float>());
    
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
    
    RecentItems::open(SETTING(source).value<file::PathArray>().source(), GlobalSettings::map());
}

void TrackingScene::activate() {
    using namespace dyn;
    
    init_video();
    
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
                _data->analysis.bump();
                bool pause = SETTING(analysis_paused).value<bool>();
                if(_data->analysis.paused() != pause) {
                    _data->analysis.set_paused(pause).get();
                }
            });
        } else if(key == "analysis_range") {
            _data->_analysis_range = Tracker::analysis_range();
        }
        
    });
    
    _data->_analysis_range = Tracker::analysis_range();
    
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
                SETTING(analysis_paused) = true;
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
    ThreadManager::getInstance().printThreadTree();
    
    WorkProgress::stop();
    dynGUI.clear();
    tracker::gui::blob_view_shutdown();
    
    print("Preparing for shutdown...");
#if !COMMONS_NO_PYTHON
    CheckUpdates::cleanup();
    Categorize::terminate();
#endif
    
    if(_data)
        _data->analysis.terminate();
    _data = nullptr;
}

void TrackingScene::set_frame(Frame_t frameIndex) {
    if(frameIndex <= _data->video.length()
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
        if(index >= _data->video.length()) {
            index = _data->video.length().try_sub(1_f);
            SETTING(gui_run) = false;
        }
        set_frame(index);
        
    } else {
        _data->_time_since_last_frame += dt;
        
        double advances = _data->_time_since_last_frame * frame_rate;
        if(advances >= 1) {
            index += Frame_t(uint(advances));
            if(index >= _data->video.length()) {
                index = _data->video.length().try_sub(1_f);
                SETTING(gui_run) = false;
            }
            set_frame(index);
            _data->_time_since_last_frame = 0;
        }
    }
}

void TrackingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not dynGUI)
        dynGUI = init_gui(graph);
    
    update_run_loop();
    if(_data)
        _data->_exec_main_queue.processTasks(static_cast<IMGUIBase*>(window()), graph);
    else
        return;
    
    if(window()) {
        auto update = FindCoord::set_screen_size(graph, *window());
        if(update != window_size)
            window_size = update;
    }
    //window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height).div(((IMGUIBase*)window())->dpi_scale()) * gui::interface_scale();
    
    if(not _data->_cache) {
        _data->_cache = std::make_unique<GUICache>(&graph, &_data->video);
        _data->_bowl = std::make_unique<Bowl>(_data->_cache.get());
        _data->_bowl->set_video_aspect_ratio(_data->video.size().width, _data->video.size().height);
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
            
            _individuals.resize(_data->_cache->raw_blobs.size());
            _fish_data.resize(_individuals.size());
            for(size_t i=0; i<_data->_cache->raw_blobs.size(); ++i) {
                auto &var = _individuals[i];
                if(not var)
                    var = std::unique_ptr<VarBase_t>(new Variable([i, this](const VarProps&) -> sprite::Map& {
                        return _fish_data.at(i);
                    }));
                
                auto &map = _fish_data.at(i);
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
    
    dynGUI.update(nullptr);
    
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

dyn::DynamicGUI TrackingScene::init_gui(DrawStructure& graph) {
    using namespace dyn;
    return dyn::DynamicGUI {
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
                    _data->tracker._remove_frames(Frame_t{});
                else {
                    auto frame = Meta::fromStr<Frame_t>(action.first());
                    _data->tracker._remove_frames(frame);
                }
                _data->analysis.set_paused(false);
            }),
            ActionFunc("load_results", [this](Action){
                load_state(Output::TrackingResults::expected_filename());
            }),
            ActionFunc("save_results", [this](Action) {
                save_state(false);
            }),
            ActionFunc("export_data", [this](Action){
                WorkProgress::add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { export_tracks("", {}, {}); });
            }),
            ActionFunc("auto_correct", [this](Action){
                auto_correct();
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
            VarFunc("active_individuals", [this](const VarProps& props) -> size_t {
                if(props.parameters.size() != 1)
                    throw std::invalid_argument("Need exactly one argument for "+props.toStr());
                
                auto frame = Meta::fromStr<Frame_t>(props.parameters.front());
                {
                    LockGuard guard(ro_t{}, "active");
                    if(_data->tracker.properties(frame))
                        return Tracker::active_individuals(frame).size();
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
                return _data->_stats.frames_per_second.load();
            }),
            
            VarFunc("fishes", [this](const VarProps&)
                -> std::vector<std::shared_ptr<VarBase_t>>&
            {
                return _individuals;
            }),
            
            VarFunc("consec", [this](const VarProps&) -> auto& {
                //auto consec = _data->tracker.global_segment_order();
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
                if(not _data->tracker.start_frame().valid())
                    return Range<Frame_t>(_data->_analysis_range.load().start(), _data->_analysis_range.load().start());
                return Range<Frame_t>{ _data->tracker.start_frame(), _data->tracker.end_frame() + 1_f };
            }),
            
            VarFunc("analysis_range", [this](const VarProps&) -> Range<Frame_t> {
                auto range = _data->tracker.analysis_range().range;
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
}

void TrackingScene::save_state(bool force_overwrite) {
    static bool save_state_visible = false;
    if(save_state_visible)
        return;
    
    save_state_visible = true;
    static file::Path file;
    file = Output::TrackingResults::expected_filename();
    
    auto fn = [this]() {
        bool before = _data->analysis.is_paused();
        _data->analysis.set_paused(true).get();
        
        LockGuard guard(w_t{}, "GUI::save_state");
        try {
            Output::TrackingResults results(_data->tracker);
            results.save([](const std::string& title, float x, const std::string& description){ WorkProgress::set_progress(title, x, description); }, file);
        } catch(const UtilsException&e) {
            auto what = std::string(e.what());
            _data->_exec_main_queue.enqueue([what](auto, DrawStructure& graph){
                graph.dialog([](Dialog::Result){}, "Something went wrong saving the program state. Maybe no write permissions? Check out this message, too:\n<i>"+what+"</i>", "Error");
            });
            
            FormatExcept("Something went wrong saving program state. Maybe no write permissions?"); }
        
        if(!before)
            _data->analysis.set_paused(false).get();
        
        save_state_visible = false;
    };
    
    if(file.exists() && !force_overwrite) {
        _data->_exec_main_queue.enqueue([fn](auto, DrawStructure& graph){
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
        bool before = _data->analysis.is_paused();
        _data->analysis.set_paused(true).get();
        
        Categorize::DataStore::clear();
        
        LockGuard guard(w_t{}, "GUI::load_state");
        Output::TrackingResults results{_data->tracker};
        
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
                    _data->video.read_frame(f, k);
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
                                _data->video.read_frame(f, k);
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
            _data->_exec_main_queue.enqueue([from, what](IMGUIBase* base, DrawStructure& graph) {
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

void TrackingScene::auto_correct() {
    static constexpr const char* message_only_ml = "Automatic correction uses machine learning based predictions to correct potential tracking mistakes. Make sure that you have trained the visual identification network prior to using auto-correct.\n<i>Apply and retrack</i> will overwrite your <key>manual_matches</key> and replace any previous automatic matches based on new predictions made by the visual identification network. If you just want to see averages for the predictions without changing your tracks, click the <i>review</i> button.";
    static constexpr const char* message_both = "Automatic correction uses machine learning based predictions to correct potential tracking mistakes (visual identification, or physical tag data). Make sure that you have trained the visual identification network prior to using auto-correct, or that tag information is available.\n<i>Visual identification</i> and <i>Tags</i> will overwrite your <key>manual_matches</key> and replace any previous automatic matches based on new predictions made by the visual identification network/the tag data. If you just want to see averages for the visual identification predictions without changing your tracks, click the <i>Review VI</i> button.";
        
    _data->_exec_main_queue.enqueue([this](auto, DrawStructure& graph) {
        const bool tags_available = tags::available();
        graph.dialog([this, tags_available](gui::Dialog::Result r) {
            if(r == Dialog::ABORT)
                return;
            
            correct_identities(r != Dialog::SECOND, tags_available && r == Dialog::THIRD ? IdentitySource::QRCodes : IdentitySource::VisualIdent);
            
        }, tags_available ? message_both : message_only_ml, "Auto-correct", tags_available ? "Apply visual identification" : "Apply and retrack", "Cancel", "Review VI", tags_available ? "Apply tags" : "");
    });

}

void TrackingScene::correct_identities(bool force_correct, IdentitySource source) {
    WorkProgress::add_queue("checking identities...", [this, force_correct, source](){
        Tracker::instance()->check_segments_identities(force_correct, source, [](float x) { WorkProgress::set_percent(x); }, [this, source](const std::string&t, const std::function<void()>& fn, const std::string&b) {
            _data->_exec_main_queue.enqueue([this, fn, source](auto, DrawStructure&) {
                _data->_tracking_callbacks.push([this, source](){
                    correct_identities(false, source);
                });
                
                fn();
            });
        });
    });
}

void TrackingScene::on_tracking_done() {
    _data->please_stop_analysis = true;
    _data->tracker.global_segment_order();
    SETTING(analysis_paused) = true;
    
    // tracking has ended
    while(not _data->_tracking_callbacks.empty()) {
        _data->_tracking_callbacks.front()();
        _data->_tracking_callbacks.pop();
    }
}

}
