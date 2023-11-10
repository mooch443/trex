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
#include <gui/DrawBlobView.h>

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
    FindCoord::set_video(_video_size);
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
    
    _max_zoom = GUI_SETTINGS(gui_zoom_limit);
    
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
    auto o = _screen_size.div(theory_scale).max() * 0.25;
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
    //graph.section("blobs", [&, &_cache = _data->_cache, frame = GUI_SETTINGS(gui_frame)](DrawStructure &graph, Section* s)
    {
        //s->set_scale(_data->_bowl->scale());
        //s->set_pos(_data->_bowl->pos());
        
        const auto mode = GUI_SETTINGS(gui_mode);
        const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
        
        bool draw_blobs_separately = true;
        if(draw_blobs_separately)
        {
            if(GUI_SETTINGS(gui_mode) == gui::mode_t::tracking
               && _cache->tracked_frames.contains(frame))
            {
                std::unique_lock guard(_cache->_fish_map_mutex);
                for(auto &&[k,fish] : _cache->_fish_map) {
                    auto obj = fish->shadow();
                    if(obj)
                        advance_wrap(*obj);
                }
            }
            
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
            const bool gui_macos_blur = GUI_SETTINGS(gui_macos_blur);
#endif
            if(GUI_SETTINGS(gui_mode) != gui::mode_t::blobs) {
                for(auto & [b, ptr] : _cache->display_blobs) {
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                    if constexpr(std::is_same<MetalImpl, default_impl_t>::value) {
                        if(gui_macos_blur)
                            ptr->ptr->tag(Effects::blur);
                    }
#endif
                    advance_wrap(*(ptr->ptr));
                }
                
            } else {
                for(auto &[b, ptr] : _cache->display_blobs) {
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                    if constexpr(std::is_same<MetalImpl, default_impl_t>::value) {
                        if(gui_macos_blur)
                            ptr->ptr->untag(Effects::blur);
                    }
#endif
                    advance_wrap(*(ptr->ptr));
                }
            }
            
        } else if(draw_blobs
                  && GUI_SETTINGS(gui_mode) == gui::mode_t::tracking
                  && _cache->tracked_frames.contains(frame))
        {
            std::unique_lock guard(_cache->_fish_map_mutex);
            for(auto &&[k,fish] : _cache->_fish_map) {
                auto obj = fish->shadow();
                if(obj)
                    advance_wrap(*obj);
            }
        }
    }
}

void Bowl::set_data(Frame_t frame) {
}

void Bowl::update_scaling() {
    const auto dt = saturate(_timer.elapsed(), 0.001, 0.1);
    
    const float lerp_speed = 3.0f;
    float lerp_factor = 1.0f - std::exp(-lerp_speed * dt);

    _current_pos = _current_pos + (_target_pos - _current_pos) * lerp_factor;
    _current_scale = _current_scale + (_target_scale - _current_scale) * lerp_factor;

    _current_size = _current_size + (_video_size.mul(_current_scale) - _current_size) * lerp_factor;
    
    if(not scale().Equals(_current_scale)
       || not pos().Equals(_current_pos))
    {
        set_scale(_current_scale);
        set_pos(_current_pos);
        
        FindCoord::set_bowl_transform(global_transform());
    }
    
    _timer.reset();
}

void Bowl::update(Frame_t frame, DrawStructure &graph, const FindCoord& coord) {
    update([this, &frame, &graph, &coord](auto&) {
        update_blobs(frame);
        
        if(GUI_SETTINGS(gui_mode) != gui::mode_t::tracking)
            return;
        
        advance_wrap(_vf_widget);
        
        std::scoped_lock guard(Categorize::DataStore::cache_mutex(), _cache->_fish_map_mutex);
        for(auto &[id, fish] : _cache->_fish_map) {
            if(fish->frame() == frame)
                fish->update(coord, *this, graph);
        }
    });
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

bool TrackingScene::on_global_event(Event event) {
    if(event.type == EventType::MBUTTON) {
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
            case Keyboard::T:
                SETTING(gui_show_timeline) = not SETTING(gui_show_timeline).value<bool>();
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
            _data->tracker.global_segment_order();
            SETTING(analysis_paused) = true;
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
    
    SETTING(cm_per_pixel) = float(0);
    
    cmd.load_settings(&combined);
    
    //! TODO: have to delegate this to another thread
    //! otherwise we will get stuck here
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
    if(settings_file.exists()) {
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
        _data->_vf_widget = std::make_unique<VisualFieldWidget>(_data->_cache.get());
        
        _data->_clicked_background = [&](const Vec2& pos, bool v, std::string key) {
            tracker::gui::clicked_background(graph, *_data->_cache, pos, v, key);
        };
    }
    
    auto mouse = graph.mouse_position();
    if(mouse != _data->_last_mouse || _data->_cache->is_animating()) {
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
       || _data->_cache->fish_dirty())
    {
        std::vector<Vec2> targets;
        if(_data->_cache->has_selection()
           && not graph.is_key_pressed(Keyboard::LShift))
        {
            for(auto fdx : _data->_cache->selected) {
                if(not _data->_cache->fish_selected_blobs.contains(fdx)) {
                    if(_data->_last_bounds.contains(fdx)) {
                        auto &bds = _data->_last_bounds.at(fdx);
                        targets.push_back(bds.pos());
                        targets.push_back(bds.pos() + bds.size());
                        targets.push_back(bds.pos() + bds.size().mul(0, 1));
                        targets.push_back(bds.pos() + bds.size().mul(1, 0));
                    }
                    continue;
                }
                
                auto bdx = _data->_cache->fish_selected_blobs.at(fdx);
                for(auto &blob: _data->_cache->raw_blobs) {
                    if(blob->blob &&
                       (blob->blob->blob_id() == bdx || blob->blob->parent_id() == bdx)) {
                        auto& bds = blob->blob->bounds();
                        targets.push_back(bds.pos());
                        targets.push_back(bds.pos() + bds.size());
                        targets.push_back(bds.pos() + bds.size().mul(0, 1));
                        targets.push_back(bds.pos() + bds.size().mul(1, 0));
                        _data->_last_bounds[fdx] = bds;
                        break;
                    }
                }
            }
            
            std::vector<Idx_t> remove;
            for(auto &[fdx, bds] : _data->_last_bounds) {
                if(not contains(_data->_cache->selected, fdx))
                    remove.push_back(fdx);
            }
            
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
    _data->_bowl->update(_data->_cache->frame_idx, graph, coords);
    _data->_bowl_mouse = coords.convert(HUDCoord(graph.mouse_position())); //_data->_bowl->global_transform().getInverse().transformPoint(graph.mouse_position());
    
    /*const auto mode = GUI_SETTINGS(gui_mode);
    const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
    update_display_blobs(draw_blobs);*/
    
    //_data->_bowl.auto_size({});
    //_data->_bowl->set(LineClr{Cyan});
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
    
    //if(GUI_SETTINGS(gui_mode) == mode_t::tracking)
    {
        graph.wrap_object(*_data->_bowl);
    }
    
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
        WorkProgress::update(base, section, window_size);
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
    return {
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
            
            VarFunc("consec", [this](const VarProps& props) -> auto& {
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
                    if(map.do_print())
                        map.set_do_print(false);
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
            
            VarFunc("filename", [](const VarProps& props) -> file::Path {
                if(props.parameters.size() != 1) {
                    throw InvalidArgumentException("Need one argument for ", props,".");
                }
                return file::Path(Meta::fromStr<file::Path>(props.first()).filename());
            }),
            VarFunc("basename", [](const VarProps& props) -> file::Path {
                if(props.parameters.size() != 1) {
                    throw InvalidArgumentException("Need one argument for ", props,".");
                }
                return file::Path(Meta::fromStr<file::Path>(props.first()).filename()).remove_extension();
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

}
