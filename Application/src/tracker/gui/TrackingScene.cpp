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
#include <gui/DrawFish.h>
#include <tracking/VisualField.h>
#include <gui/WorkProgress.h>

using namespace track;

namespace gui {

static constexpr Frame_t cache_size{Frame_t::number_t(10)};

void VisualFieldWidget::set_parent(SectionInterface * parent) {
    if(this->parent() == parent)
        return;
    
    _polygons.clear();
    Entangled::set_parent(parent);
}

void VisualFieldWidget::update() {
    begin();
    
    if(not _cache->has_selection()
       || not GUI_SETTINGS(gui_show_visualfield)) 
    {
        end();
        return;
    }
    
    const auto& frame = _cache->frame_idx;
    size_t poly_idx{0u};
    
    for(auto id : _cache->selected) {
        auto fish = _cache->individuals.at(id);
        
        VisualField* ptr = (VisualField*)fish->custom_data(frame, VisualField::custom_id);
        if(!ptr && fish->head(frame)) {
            ptr = new VisualField(id, frame, *fish->basic_stuff(frame), fish->posture_stuff(frame), true);
            fish->add_custom_data(frame, VisualField::custom_id, ptr, [](void* ptr) {
                //std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
                delete (VisualField*)ptr;
            });
        }
        
        if(ptr) {
            using namespace gui;
            double max_d = SQR(_cache->_video_resolution.width) + SQR(_cache->_video_resolution.height);
            
            std::vector<Vertex> crosses;
            
            for(auto &eye : ptr->eyes()) {
                crosses.emplace_back(eye.pos, eye.clr);
                
                for (size_t i=6; i<VisualField::field_resolution-6; i++) {
                    if(eye._depth[i] < FLT_MAX) {
                        //auto w = (1 - sqrt(eye._depth[i]) / (sqrt(max_d) * 0.5));
                        crosses.emplace_back(eye._visible_points[i], eye.clr);
                        
                        //if(eye._visible_ids[i] != fish->identity().ID())
                        //    base.line(eye.pos, eye._visible_points.at(i), eye.clr.alpha(100 * w * w + 10));
                    } else {
                        static const Rangef fov_range(-VisualField::symmetric_fov, VisualField::symmetric_fov);
                        static const double len = fov_range.end - fov_range.start;
                        double percent = double(i) / double(VisualField::field_resolution) * len + fov_range.start + eye.angle;
                        crosses.emplace_back(eye.pos + Vec2(Float2_t(cos(percent)), Float2_t( sin(percent))) * sqrtf(max_d) * 0.5f, eye.clr);
                        
                        //if(&eye == &_eyes[0])
                        //    base.line(eye.pos, eye.pos + Vec2(cos(percent), sin(percent)) * max_d, Red.alpha(100));
                    }
                    
                    if(eye._depth[i + VisualField::field_resolution] < FLT_MAX && eye._visible_ids[i + VisualField::field_resolution] != (long_t)id.get())
                    {
                        auto w = (1 - sqrt(eye._depth[i + VisualField::field_resolution]) / (sqrt(max_d) * 0.5));
                        //crosses.push_back(eye._visible_points[i + VisualField::field_resolution]);
                        add<Line>(eye.pos, eye._visible_points[i + VisualField::field_resolution], Black.alpha((uint8_t)saturate(50 * w * w + 10)));
                    }
                }
                
                crosses.emplace_back(eye.pos, eye.clr);
                add<Circle>(Loc(eye.pos), Radius{3}, LineClr{White.alpha(125)});
                //if(&eye == &_eyes[0])
                //auto poly = new gui::Polygon(crosses);
                //poly->set_fill_clr(Transparent);
                if(_polygons.size() <= poly_idx) {
                    auto ptr = std::make_unique<Polygon>(std::move(crosses));
                    _polygons.emplace_back(std::move(ptr));
                } else {
                    _polygons[poly_idx]->set_vertices(std::move(crosses));
                }
                
                //    base.add_object(poly);
                advance_wrap(*_polygons[poly_idx++]);
                crosses.clear();
            }
            
            for(auto &eye : ptr->eyes()) {
                Vec2 straight(cos(eye.angle), sin(eye.angle));
                
                add<Line>(eye.pos, eye.pos + straight * 11, Black, 1);
                
                auto left = Vec2((Float2_t)cos(eye.angle - VisualField::symmetric_fov),
                                 (Float2_t)sin(eye.angle - VisualField::symmetric_fov));
                auto right = Vec2((Float2_t)cos(eye.angle + VisualField::symmetric_fov),
                                  (Float2_t)sin(eye.angle + VisualField::symmetric_fov));
                
                add<Line>(eye.pos, eye.pos + left * 100, eye.clr.exposure(0.65f), 1);
                add<Line>(eye.pos, eye.pos + right * 100, eye.clr.exposure(0.65f), 1);
            }
        }
    }
    
    end();
    
    set_bounds(Bounds(Vec2(), _cache->_video_resolution));
    
    if(_polygons.size() > poly_idx) {
        _polygons.resize(poly_idx);
    }
}

Bowl::Bowl(GUICache* cache) : _cache(cache), _vf_widget(cache) {
    _current_scale = Vec2(1.0f, 1.0f);
    _target_scale = Vec2(1.0f, 1.0f);
    _current_pos = Vec2(0.0f, 0.0f);
    _target_pos = Vec2(0.0f, 0.0f);
    _aspect_ratio = Vec2(1.0f, 1.0f);
    _max_zoom = Vec2(300.0f, 300.0f);
    _screen_size = Vec2(0.0f, 0.0f);
    _center_of_screen = Vec2(0.0f, 0.0f);
    _timer.reset();
}

bool Bowl::has_target_points_changed(const std::vector<Vec2>& new_target_points) const {
    return _target_points != new_target_points;
}

bool Bowl::has_screen_size_changed(const Vec2& new_screen_size) const {
    return _screen_size != new_screen_size;
}

void Bowl::set_video_aspect_ratio(float video_width, float video_height) {
    if (video_width == 0 or video_height == 0) {
        return;
    }
    _aspect_ratio = Vec2(video_width, video_height);
    _video_size = {video_width, video_height};
}

void Bowl::fit_to_screen(const Vec2& screen_size) {
    if (not has_screen_size_changed(screen_size)) {
        return;
    }
    _screen_size = screen_size;
    _center_of_screen = _screen_size / 2;
    
    set_size(_video_size);
    update_goals();
}

void Bowl::set_target_focus(const std::vector<Vec2>& target_points) {
    if (not has_target_points_changed(target_points)) {
        return;
    }
    _target_points = target_points;
    update_goals();
}

void Bowl::update_goals() {
    if (_target_points.empty()) {
        float width_scale = _screen_size.x / _aspect_ratio.x;
        float height_scale = _screen_size.y / _aspect_ratio.y;
        float scale_factor = std::min(width_scale, height_scale);
        _target_scale = Vec2(scale_factor, scale_factor);
        _target_pos = _center_of_screen - _video_size.mul(_target_scale) / 2;
        
        return;
    }

    // Compute the bounding box and its center of mass
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::min();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::min();

    Vec2 sum_of_points(0.0f, 0.0f);
    for (const auto& point : _target_points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
        sum_of_points = sum_of_points + point;
    }
    
    Bounds bounding_box(Vec2(min_x, min_y), Size2(max_x - min_x + 1, max_y - min_y + 1));
    if(bounding_box.width < _max_zoom.x) {
        auto diff = _max_zoom.x - bounding_box.width;
        bounding_box.x -= diff / 2;
        bounding_box.width = _max_zoom.x;
    }
    if(bounding_box.height < _max_zoom.y) {
        auto diff = _max_zoom.y - bounding_box.height;
        bounding_box.y -= diff / 2;
        bounding_box.height = _max_zoom.y;
    }
    
    Vec2 theory_scale = _screen_size.div(bounding_box.size());
    auto o = _screen_size.div(theory_scale).max() * 0.1;
    bounding_box << bounding_box.pos() - o;
    bounding_box << bounding_box.size() + o * 2;
    
    // Find the required scaling factor, but don't exceed the max zoom
    Vec2 scale_required = _screen_size.div(bounding_box.size());
    Vec2 new_target_scale = Vec2( scale_required.min(), scale_required.min() );

    // Update the target scale
    _target_scale = new_target_scale;

    // Calculate the target position
    _target_pos = _center_of_screen - bounding_box.center().mul(_target_scale);
}

void Bowl::update_blobs(const Frame_t& frame) {
    const auto mode = GUI_SETTINGS(gui_mode);
    const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
    
    bool draw_blobs_separately = true;
    if(draw_blobs_separately)
    {
        if(GUI_SETTINGS(gui_mode) == gui::mode_t::tracking
           && _cache->tracked_frames.contains(frame))
        {
            for(auto &&[k,fish] : _cache->_fish_map) {
                auto obj = fish->shadow();
                if(obj)
                    advance_wrap(*obj);
            }
        }
        
        if(GUI_SETTINGS(gui_mode) != gui::mode_t::blobs) {
            for(auto & [b, ptr] : _cache->display_blobs) {
                //if(blob_fish.find(b->blob_id()) == blob_fish.end())
                {
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                    if(GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
                    {
                        ptr->ptr->tag(Effects::blur);
                    }
#endif
#endif
                    advance_wrap(*(ptr->ptr));
                }
            }
            
        } else {
            for(auto &[b, ptr] : _cache->display_blobs) {
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                if(GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
                {
                    ptr->ptr->untag(Effects::blur);
                }
#endif
#endif
                advance_wrap(*(ptr->ptr));
            }
        }
        
    } else if(draw_blobs
              && GUI_SETTINGS(gui_mode) == gui::mode_t::tracking
              && _cache->tracked_frames.contains(frame))
    {
        for(auto &&[k,fish] : _cache->_fish_map) {
            auto obj = fish->shadow();
            if(obj)
                advance_wrap(*obj);
        }
    }
}

void Bowl::update(Frame_t frame, DrawStructure &graph, const Size2& window_size) {
    update([this, &frame, &graph, window_size](auto&) {
        add<Circle>(LineClr{Red}, Loc{}, Radius{10});
        
        auto props = Tracker::properties(frame);
        if(not props)
            return;
        
        update_blobs(frame);
        advance_wrap(_vf_widget);
        
        set_of_individuals_t source;
        if(Tracker::has_identities() && GUI_SETTINGS(gui_show_inactive_individuals))
        {
            for(auto [id, fish] : _cache->individuals)
                source.insert(fish);
            //! TODO: Tracker::identities().count(id) ?
            
        } else {
            for(auto fish : _cache->active)
                source.insert(fish);
        }
        
        for (auto& fish : (source.empty() ? _cache->active : source)) {
            if (fish->empty()
                || fish->start_frame() > frame)
                continue;

            auto segment = fish->segment_for(frame);
            if (!GUI_SETTINGS(gui_show_inactive_individuals)
                && (!segment || (segment->end() != Tracker::end_frame()
                    && segment->length().get() < (long_t)GUI_SETTINGS(output_min_frames))))
            {
                continue;
            }

            /*auto it = container->map().find(fish);
            if (it != container->map().end())
                empty_map = &it->second;
            else
                empty_map = NULL;*/

            if (_cache->_fish_map.find(fish) == _cache->_fish_map.end()) {
                _cache->_fish_map[fish] = std::make_unique<gui::Fish>(*fish);
                fish->register_delete_callback(_cache->_fish_map[fish].get(), [&graph, this](Individual* f) {
                    std::lock_guard guard(graph.lock());

                    auto it = _cache->_fish_map.find(f);
                    if (it != _cache->_fish_map.end()) {
                        _cache->_fish_map.erase(f);
                    }
                    _cache->set_tracking_dirty();
                });
            }

            _cache->_fish_map[fish]->set_data(frame, props->time, nullptr);

            {
                std::unique_lock guard(Categorize::DataStore::cache_mutex());
                _cache->_fish_map[fish]->update(window_size, this, *this, graph);
            }
            //base.wrap_object(*PD(cache)._fish_map[fish]);
            //PD(cache)._fish_map[fish]->label(ptr, e);
        }
        
        
        for(auto &target : _target_points) {
            add<Circle>(FillClr{Cyan}, Radius{10}, Origin{0.5}, Loc{target});
        }
    });
}

void Bowl::update() {
    if(not content_changed())
        return;
    
    const auto dt = saturate(_timer.elapsed(), 0.001, 0.1);
    
    const float lerp_speed = 3.0f;
    float lerp_factor = 1.0f - std::exp(-lerp_speed * dt);

    _current_pos = _current_pos + (_target_pos - _current_pos) * lerp_factor;
    _current_scale = _current_scale + (_target_scale - _current_scale) * lerp_factor;

    _current_size = _current_size + (_video_size.mul(_current_scale) - _current_size) * lerp_factor;
    
    set_scale(_current_scale);
    set_pos(_current_pos);
    _timer.reset();
    
    set_content_changed(not _current_pos.Equals(_target_pos)
                        || not _current_scale.Equals(_target_scale)
                        || not _current_size.Equals(_video_size.mul(_current_scale)));
}

void Bowl::set_max_zoom_size(const Vec2& max_zoom) {
    _max_zoom = max_zoom;
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
    
    _background = std::make_unique<AnimatedBackground>(Image::Make(original));
    
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

bool TrackingScene::on_global_event(Event event) {
    if(event.type == EventType::KEY) {
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
        
        if(idx + 1_f == _data->video.length()) {
            _data->please_stop_analysis = true;
            SETTING(analysis_paused) = true;
        }
        
        //print(_data->tracker.active_individuals(idx));

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
    file::Path filename = file::DataLocation::parse("input", SETTING(source).value<file::PathArray>().source());
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
    
    init_video();
    
    _data->_callback = GlobalSettings::map().register_callbacks({
        "gui_focus_group",
        "gui_run",
        "analysis_paused"
        
    }, [this](std::string_view key) {
        if(key == "gui_focus_group" && _data->_bowl)
            _data->_bowl->_screen_size = Vec2();
        else if(key == "gui_run") {
            
        } else if(key == "analysis_paused") {
            gui::WorkProgress::add_queue("pausing...", [this](){
                _data->analysis.bump();
                bool pause = SETTING(analysis_paused).value<bool>();
                if(_data->analysis.paused() != pause) {
                    print("Adding to queue...");
                    _data->analysis.set_paused(pause).get();
                    print("Added.");
                }
            });
        }
    });
    
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
    WorkProgress::stop();
    dynGUI.clear();
    
    print("Preparing for shutdown...");
#if !COMMONS_NO_PYTHON
    CheckUpdates::cleanup();
    Categorize::terminate();
#endif
    _data->analysis.terminate();
    _data = nullptr;
}

void TrackingScene::update_display_blobs(bool draw_blobs) {
    if((_data->_cache->raw_blobs_dirty() || _data->_cache->display_blobs.size() != _data->_cache->raw_blobs.size()) && draw_blobs)
    {
        static std::mutex vector_mutex;
        auto screen_bounds = Bounds(Vec2(), window_size);
        //auto copy = PD(cache).display_blobs;
        size_t gpixels = 0;
        double gaverage_pixels = 0, gsamples = 0;
        
        distribute_indexes([&](auto, auto start, auto end, auto){
            std::unordered_map<pv::bid, SimpleBlob*> map;
            //std::vector<std::unique_ptr<gui::ExternalImage>> vector;
            
            const bool gui_show_only_unassigned = SETTING(gui_show_only_unassigned).value<bool>();
            const bool tags_dont_track = SETTING(tags_dont_track).value<bool>();
            size_t pixels = 0;
            double average_pixels = 0, samples = 0;
            
            for(auto it = start; it != end; ++it) {
                if(!*it || (tags_dont_track && (*it)->blob->is_tag())) {
                    continue;
                }
                
                //bool found = copy.count((*it)->blob.get());
                //if(!found) {
                    //auto bds = bowl.transformRect((*it)->blob->bounds());
                    //if(bds.overlaps(screen_bounds))
                    //{
                if(!gui_show_only_unassigned ||
                   (!_data->_cache->display_blobs.contains((*it)->blob->blob_id()) && !contains(_data->_cache->active_blobs, (*it)->blob->blob_id())))
                {
                    (*it)->convert();
                    //vector.push_back((*it)->convert());
                    map[(*it)->blob->blob_id()] = it->get();
                }
                    //}
                //}
                
                pixels += (*it)->blob->num_pixels();
                average_pixels += (*it)->blob->num_pixels();
                ++samples;
            }
            
            std::lock_guard guard(vector_mutex);
            gpixels += pixels;
            gaverage_pixels += average_pixels;
            gsamples += samples;
            _data->_cache->display_blobs.insert(map.begin(), map.end());
            //std::move(vector.begin(), vector.end(), std::back_inserter(PD(cache).display_blobs_list));
            //PD(cache).display_blobs_list.insert(PD(cache).display_blobs_list.end(), vector.begin(), vector.end());
            
        }, _data->pool, _data->_cache->raw_blobs.begin(), _data->_cache->raw_blobs.end());
        
        _data->_cache->_current_pixels = gpixels;
        _data->_cache->_average_pixels = gsamples > 0 ? gaverage_pixels / gsamples : 0;
        _data->_cache->updated_raw_blobs();
    }
}

void TrackingScene::set_frame(Frame_t frame) {
    if(frame <= _data->video.length()) {
        SETTING(gui_frame) = frame;
        
        if(_data->_cache) {
            _data->_cache->update_data(frame);
            
            using namespace dyn;
            
            const auto mode = GUI_SETTINGS(gui_mode);
            const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
            update_display_blobs(draw_blobs);
            
            _individuals.resize(_data->_cache->raw_blobs.size());
            _fish_data.resize(_individuals.size());
            for(size_t i=0; i<_data->_cache->raw_blobs.size(); ++i) {
                auto &var = _individuals[i];
                if(not var)
                    var = std::unique_ptr<VarBase_t>(new Variable([i, this](VarProps) -> sprite::Map& {
                        return _fish_data.at(i);
                    }));
                
                auto &map = _fish_data.at(i);
                auto &fish = _data->_cache->raw_blobs[i];
                map["pos"] = Vec2(fish->blob->bounds().pos());
            }
        }
    }
}

void TrackingScene::update_run_loop() {
    //if(!recording())
    if(not _data || not _data->_cache)
        return;
    
    _data->_cache->set_dt(last_redraw.elapsed());
    last_redraw.reset();
    //else
    //    _data->_cache->set_dt(0.75f / (float(GUI_SETTINGS(frame_rate))));
    
    if(not GUI_SETTINGS(gui_run))
        return;
    
    const auto dt = _data->_cache->dt();
    const double frame_rate = GUI_SETTINGS(frame_rate);
    
    Frame_t index = GUI_SETTINGS(gui_frame);
    _data->_time_since_last_frame += dt;
    
    double advances = _data->_time_since_last_frame * frame_rate;
    if(advances >= 1) {
        index += Frame_t(uint(advances));
        if(index >= _data->video.length())
            index = _data->video.length().try_sub(1_f);
        set_frame(index);
        _data->_time_since_last_frame = 0;
    }
}

void TrackingScene::_draw(DrawStructure& graph) {
    using namespace dyn;
    if(not dynGUI)
        dynGUI = init_gui(graph);
    
    update_run_loop();
    
    auto gui_scale = graph.scale();
    if(gui_scale.x == 0)
        gui_scale = Vec2(1);
    window_size = window()->window_dimensions().div(gui_scale) * gui::interface_scale();
    //window_size = Vec2(window()->window_dimensions().width, window()->window_dimensions().height).div(((IMGUIBase*)window())->dpi_scale()) * gui::interface_scale();
    
    if(not _data->_cache) {
        _data->_cache = std::make_unique<GUICache>(&graph, &_data->video);
        _data->_bowl = std::make_unique<Bowl>(_data->_cache.get());
        _data->_bowl->set_video_aspect_ratio(_data->video.size().width, _data->video.size().height);
        _data->_bowl->fit_to_screen(window_size);
        _data->_vf_widget = std::make_unique<VisualFieldWidget>(_data->_cache.get());
    }

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
    
    std::vector<Vec2> targets;
    if(_data->_cache->has_selection() 
       && not graph.is_key_pressed(Keyboard::LShift))
    {
        for(auto fdx : _data->_cache->selected) {
            if(not _data->_cache->fish_selected_blobs.contains(fdx))
                continue;
            
            auto bdx = _data->_cache->fish_selected_blobs.at(fdx);
            for(auto &blob: _data->_cache->raw_blobs) {
                if(blob->blob->blob_id() == bdx || blob->blob->parent_id() == bdx) {
                    auto& bds = blob->blob->bounds();
                    targets.push_back(bds.pos());
                    targets.push_back(bds.pos() + bds.size());
                    targets.push_back(bds.pos() + bds.size().mul(0, 1));
                    targets.push_back(bds.pos() + bds.size().mul(1, 0));
                    break;
                }
            }
        }
    }
    //_data->_bowl.set_video_aspect_ratio(_data->video.size().width, _data->video.size().height);
    _data->_bowl->fit_to_screen(window_size);
    _data->_bowl->set_target_focus(targets);
    _data->_bowl->set_content_changed(true);
    
    if(LockGuard guard(ro_t{}, "Update Gui", 100); guard.locked())
        _data->_bowl->update(_data->_cache->frame_idx, graph, window_size);
    
    //_data->_bowl.auto_size({});
    _data->_bowl->set(LineClr{Cyan});
    //_data->_bowl.set(FillClr{Yellow});
    
    auto alpha = SETTING(gui_background_color).value<Color>().a;
    _data->_background->set_color(Color(255, 255, 255, alpha ? alpha : 1));
    
    if(alpha > 0) {
        graph.wrap_object(*_data->_background);
        /*if(PD(gui_mask)) {
            PD(gui_mask)->set_color(PD(background)->color().alpha(PD(background)->color().a * 0.5));
            PD(gui).wrap_object(*PD(gui_mask));
        }*/
        _data->_background->set_scale(_data->_bowl->scale());
        _data->_background->set_pos(_data->_bowl->pos());
    }
    
    graph.wrap_object(*_data->_bowl);
    
    dynGUI.update(nullptr);
    
    graph.section("loading", [this](DrawStructure& base, auto section) {
        WorkProgress::update(base, section, window_size);
    });
    graph.root().set_dirty();
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
    return {
        .path = "tracking_layout.json",
        .graph = &graph,
        .context = {
            ActionFunc("set", [this](Action action) {
                if(action.parameters.size() != 2)
                    throw InvalidArgumentException("Invalid number of arguments for action: ",action);
                
                auto parm = Meta::fromStr<std::string>(action.parameters.front());
                if(not GlobalSettings::has(parm))
                    throw InvalidArgumentException("No parameter ",parm," in global settings.");
                
                auto value = action.parameters.back();
                
                if(parm == "gui_frame") {
                    set_frame(Meta::fromStr<Frame_t>(value));
                } else
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
            
            VarFunc("consec", [this](VarProps props) -> auto& {
                auto consec = _data->tracker.global_segment_order();
                static std::vector<sprite::Map> segments;
                static std::vector<std::shared_ptr<VarBase_t>> variables;
                segments.clear();
                variables.clear();
                
                ColorWheel wheel;
                for(size_t i=0; i<3 && i < consec.size(); ++i) {
                    sprite::Map map;
                    map.set_do_print(false);
                    map["color"] = wheel.next();
                    map["from"] = consec.at(i).start;
                    map["to"] = consec.at(i).end + 1_f;
                    segments.emplace_back(std::move(map));
                    variables.emplace_back(new Variable([i, this](VarProps) -> sprite::Map& {
                        return segments.at(i);
                    }));
                }
                
                return variables;
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
                    map["from"] = current.start;
                    map["to"] = current.end;
                    last = current;
                }
                return map;
            }),

            VarFunc("key", [this](VarProps) -> auto& {
                return _data->_keymap;
            })
        }
    };
}

}
