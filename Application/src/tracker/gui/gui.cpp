#include "gui.h"
#include <misc/Timer.h>
#include <misc/detail.h>
#include <misc/cnpy_wrapper.h>

#include <gui/types/Drawable.h>
#include <gui/types/List.h>
#include <gui/types/Button.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Textfield.h>
#include <gui/IMGUIBase.h>
#include <gui/types/Tooltip.h>
#include <gui/GuiTypes.h>
#include <gui/IdentityHeatmap.h>
#include <gui/GUICache.h>
#include <gui/WorkProgress.h>
#include <gui/SFLoop.h>
#include <gui/DrawBase.h>
#include <gui/DrawCVBase.h>

#include <tracking/Tracker.h>
#include <tracking/Recognition.h>
#include <tracking/DatasetQuality.h>
#include <tracking/VisualField.h>
#include <tracking/Export.h>
#include <tracking/Accumulation.h>
#include <tracking/Categorize.h>

#include <misc/ConnectedTasks.h>
#include <misc/default_settings.h>
#include <misc/Output.h>
#include <misc/MemoryStats.h>
#include <processing/PadImage.h>
#include <misc/Results.h>

#include <pv.h>

#if WIN32
#include <Shellapi.h>

#define access(X, Y) _access(X, Y)
#define W_OK 2
#endif

IMPLEMENT(GUI::_instance) = NULL;

std::shared_ptr<gui::List> _settings_choice;
std::vector<gui::Drawable*> _static_pointers;

std::vector<gui::Drawable*>& GUI::static_pointers() {
    return _static_pointers;
}

GUI* GUI::instance() {
    return _instance;
}

#include <gui/types/Histogram.h>
#include <gui/types/StaticText.h>

#include <gui/DrawGraph.h>
#include <gui/DrawPosture.h>
#include <gui/Timeline.h>
#include <gui/DrawMenu.h>
#include <gui/DrawDataset.h>
#include <gui/RecognitionSummary.h>
#include <gui/DrawFish.h>
#include <gui/Label.h>
#include <gui/InfoCard.h>
#ifndef NDEBUG
#include <gui/FlowMenu.h>
#endif
using namespace gui;

enum class SelectedSettingType {
    ARRAY_OF_BOUNDS,
    ARRAY_OF_VECTORS,
    POINTS,
    NONE
};

struct PrivateData {
    GUICache _cache;

    cv::VideoWriter* _recording_capture = nullptr;
    cv::Size _recording_size;
    file::Path _recording_path;
    default_config::gui_recording_format_t::Class _recording_format;

    Frame_t _recording_start;
    Frame_t _recording_frame;
    Frame_t _last_recording_frame;
    std::atomic_bool _recording = false;

    Frame_t _flow_frame;
    Frame_t _next_frame;
    cv::Mat _cflow, _cflow_next;

    float _tdelta_gui;
    Frame_t _gui_last_frame;
    Timer _gui_last_frame_timer;
    Timer _gui_last_backup;

    pv::File *_video_source = nullptr;

    //! Reference to the Tracker
    Tracker* _tracker = nullptr;

    //! Timer for determining speed of increase by keypress
    Timer _last_increase_timer, _last_direction_change;
    bool _real_update = false;
    std::shared_ptr<gui::DrawDataset> _dataset;
    std::shared_ptr<gui::Timeline> _timeline;

    std::vector<gui::PropertiesGraph*> _fish_graphs;
    gui::Posture _posture_window;

    gui::WorkProgress* _work_progress = nullptr;
    gui::DrawStructure _gui;

    ConnectedTasks* _analysis = nullptr;

    gui::ExternalImage _recognition_image;
    std::map<std::vector<Vec2>, std::shared_ptr<gui::Drawable>> _ignore_shapes, _include_shapes;
    Timer _last_frame_change;

    gui::Histogram<float, gui::Hist::Options::NORMED> _histogram{ "Event energy", Bounds(200, 450, 800, 300), Filter::FixedBins(40), Display::LimitY(0.45) };
    gui::Histogram<float, gui::Hist::Options::NORMED> _midline_histogram{ "Midline length", Bounds(0, 0, 800, 300), Filter::FixedBins(0.6, 1.4, 50) };

    gui::StaticText _info;
    gui::Histogram<size_t, gui::Hist::Options::NORMED, std::vector<std::map<long_t, size_t>>> _length_histogram{ "Event length", Bounds(1, 800, 800, 300), Filter::FixedBins(0, 100, 50), Display::LimitY(0.45) };

    std::function<void(const Vec2&, bool, std::string)> _clicked_background;
    std::vector<std::vector<Vec2>> _current_boundary;
    std::string _selected_setting_name;
    SelectedSettingType _selected_setting_type;

};

template<typename T>
requires _is_dumb_pointer<T>
std::remove_pointer_t<T>& access_private(T& t) {
    return *t;
}

template<typename T>
    requires (!_is_dumb_pointer<T>)
T& access_private(T& t) {
    return t;
}

#define PD(X) access_private( GUI::instance()->_private_data->_ ## X )
#define PDP(X) ( GUI::instance()->_private_data->_ ## X )

bool GUI::recording() {
    return PD(recording);
}

const gui::Timeline& GUI::timeline() {
    assert(PDP(timeline));
    return *PD(timeline);
}

using namespace Hist;

template<globals::Cache::Variables M>
class DirectSettingsItem : public List::Item {
protected:
    GETTER_SETTER(std::string, description)
    
public:
    DirectSettingsItem(const std::string& description = "", long idx = -1) {
        if(description.empty())
            _description = Meta::toStr(M);
        else
            _description = description;
        
        set_selected(globals::Cache::get<M>());
    }
    
    operator const std::string&() const override {
        return _description;
    }
    
private:
    void operator=(const gui::List::Item&) override {
        assert(false);
    }
    
public:
    void set_selected(bool s) override {
        if(s != selected()) {
            List::Item::set_selected(s);
            GlobalSettings::get(globals::Cache::name<M>()) = s;
        }
    }
    void update() override {
        set_selected(globals::Cache::get<M>());
    }
};

class OuterBlobs {
    Image::UPtr image;
    Vec2 pos;
    std::unique_ptr<gui::ExternalImage> ptr;
    
public:
    OuterBlobs(Image::UPtr&& image = nullptr, std::unique_ptr<gui::ExternalImage>&& available = nullptr, const Vec2& pos = Vec2(), long_t id = -1) : image(std::move(image)), pos(pos), ptr(std::move(available)) {
        
    }
    
    std::unique_ptr<gui::ExternalImage> convert() {
        if(!ptr)
            ptr = std::make_unique<ExternalImage>(std::move(image), pos);
        else
            ptr->set_source(std::move(image));
        
        ptr->set_color(Red.alpha(255));
        return std::move(ptr);
    }
};

static std::unique_ptr<gui::heatmap::HeatmapController> heatmapController;

void drawOptFlowMap (const cv::Mat& flow, cv::Mat& map) {
    assert(flow.isContinuous());
    assert(map.isContinuous());
    Color* out = (Color*)map.data;
    
    for(const Vec2* ptr = (const Vec2*)flow.data; ptr != (const Vec2*)flow.data + flow.cols * flow.rows; ++ptr, ++out)
    {
        float c = DEGREE(normalize_angle(atan2(ptr->y, ptr->x)));
        float mag = saturate(length(*ptr)*850.f);
        float hue = c / 360.f * 255;
        
        *out = Color(hue, 255, mag).HSV2RGB();
    }
}

bool GUI::execute_settings(file::Path settings_file, AccessLevelType::Class accessLevel) {
    if(settings_file.exists()) {
        DebugHeader("LOADING '%S'", &settings_file.str());
        try {
            auto content = utils::read_file(settings_file.str());
            default_config::load_string_with_deprecations(settings_file, content, GlobalSettings::map(), accessLevel);
            
        } catch(const cmn::illegal_syntax& e) {
            Error("Illegal syntax in settings file.");
            return false;
        }
        DebugHeader("LOADED '%S'", &settings_file.str());
        return true;
    }
    
    return false;
}

GUI::GUI(pv::File& video_source, const Image& average, Tracker& tracker)
  :
    _average_image(average),
    _direction_change(false), _play_direction(1),
    _base(NULL),
    _blob_thread_pool(cmn::hardware_concurrency(), [](std::exception_ptr e) {
        GUI::work().add_queue("", [e](){
            try {
                std::rethrow_exception(e);
            } catch(const std::exception& ex) {
                GUI::instance()->gui().dialog("An error occurred in the blob thread pool:\n<i>"+std::string(ex.what())+"</i>", "Error");
            }
        });
    }),
    _properties_visible(false),
    _private_data(new PrivateData),
#if WITH_MHD
    _http_gui(NULL),
#endif
    _info_visible(false)
{
    GUI::_instance = this;

    PDP(tracker) = &tracker;
    PD(posture_window).set_bounds(Bounds(average.cols - 550 - 10, 100, 550, 400));
    PD(gui).set_size(Size2(average.cols, average.rows));
    PDP(video_source) = &video_source;

    gui::globals::Cache::init();
    
    PDP(timeline) = std::make_shared<Timeline>(*this, _frameinfo);
    PD(gui).root().insert_cache(_base, std::make_shared<CacheObject>());
    
    PD(info).set_pos(Vec2(_average_image.cols * 0.5, _average_image.rows * 0.5));
    PD(info).set_max_size(Size2(min(_average_image.cols * 0.75, 700), min(_average_image.rows * 0.75, 700)));
    PD(info).set_origin(Vec2(0.5, 0.5));
    PD(info).set_background(Color(50, 50, 50, 150), Black.alpha(150));

    PD(histogram).set_bounds(Bounds(_average_image.cols * 0.5, 450, 800, 300));
    PD(midline_histogram).set_bounds(Bounds(_average_image.cols * 0.5, _average_image.rows * 0.5, 800, 300));
    PD(length_histogram).set_bounds(Bounds(_average_image.cols * 0.5, 800, 800, 300));

    PD(histogram).set_origin(Vec2(0.5, 0.5));
    PD(midline_histogram).set_origin(Vec2(0.5, 0.5));
    PD(length_histogram).set_origin(Vec2(0.5, 0.5));
    
    for(size_t i=0; i<2; ++i)
        PD(fish_graphs).push_back(new PropertiesGraph(PD(tracker), PD(gui).mouse_position()));
    
    auto callback = "TRex::GUI";
    auto changed = [callback, this](sprite::Map::Signal signal, sprite::Map& map, const std::string& name, const sprite::PropertyType& value)
    {
        if(signal == sprite::Map::Signal::EXIT) {
            map.unregister_callback(callback);
            return;
        }
        
        // ignore gui frame
        if(name == "gui_frame") {
            return;
        }
        if(name == "auto_train") {
            Debug("Changing");
        }
        if(!GUI::instance())
            return;
        
        this->work().add_queue("", [this, name, &value](){
            if(!GUI::instance())
                return;
            
            std::lock_guard<std::recursive_mutex> lock_guard(this->gui().lock());
            
            /*if(name == "track_max_speed") {
                _setting_animation.name = name;
                _setting_animation.display = nullptr;
            }*/
            
            if(name == "app_name" || name == "output_prefix") {
                if(_base)
                    _base->set_title(window_title());
            } //else if(name == "gui_run")
                //globals::_settings.gui_run = value.value<bool>();
            //else if(name == "nowindow")
                //globals::_settings.nowindow = value.value<bool>();
                
            if(name == "output_graphs" || name == "limit" || name == "event_min_peak_offset" || name == "output_normalize_midline_data") {//name != "gui_frame" && name != "analysis_paused") {
                Output::Library::clear_cache();
                for(auto &graph : PD(fish_graphs))
                    graph->reset();
                GUI::set_redraw();
            }
            
            if(name == "exec") {
                if(!SETTING(exec).value<file::Path>().empty()) {
                    file::Path settings_file = pv::DataLocation::parse("settings", SETTING(exec).value<file::Path>());
                    execute_settings(settings_file, AccessLevelType::PUBLIC);
                    SETTING(exec) = file::Path();
                }
            }
            
            if(name == "gui_connectivity_matrix_file") {
                try {
                    this->load_connectivity_matrix();
                } catch(const UtilsException&) { }
                GUI::set_redraw();
            }
        
            if((name == "track_threshold" || name == "grid_points" || name == "recognition_shapes" || name == "grid_points_scaling" || name == "recognition_border_shrink_percent" || name == "recognition_border" || name == "recognition_coeff" || name == "recognition_border_size_rescale") && Tracker::instance())
            {
                this->work().add_queue("updating border", [this, name](){
                    if(name == "recognition_coeff" || name == "recognition_border_shrink_percent" || name == "recognition_border_size_rescale" || name == "recognition_border") {
                        PD(tracker).border().clear();
                    }
                    PD(tracker).border().update(PD(video_source));
                    
                    {
                        Tracker::LockGuard guard("setting_changed_"+name);
                        if(Tracker::recognition())
                            Tracker::recognition()->clear_filter_cache();
                        if(Tracker::recognition() && Tracker::recognition()->dataset_quality()) {
                            auto start = Tracker::start_frame();
                            Tracker::recognition()->dataset_quality()->remove_frames(start);
                        }
                    }
                    
                    std::lock_guard<std::recursive_mutex> lock_guard(this->gui().lock());
                    PD(recognition_image).set_source(Image::Make());
                    PD(cache).set_tracking_dirty();
                    PD(cache).set_blobs_dirty();
                    PD(cache).recognition_updated = true;
                    GUI::set_redraw();
                    if(PD(dataset))
                        PD(dataset)->clear_cache();
                });
            }
        
            std::vector<std::string> display_fields {
                "gui_show_paths",
                "gui_auto_scale",
                "gui_show_selections",
                "gui_foi_name",
                "gui_focus_group",
                "gui_auto_scale_focus_one",
                "gui_show_visualfield_ts",
                "gui_show_visualfield",
                "gui_show_posture",
                "gui_show_outline",
                "gui_show_midline",
                "gui_show_texts",
                "gui_show_recognition_summary",
                "gui_show_recognition_bounds",
                "gui_zoom_limit",
                "whitelist",
                "blacklist",
                "gui_background_color",
                "gui_show_detailed_probabilities"
            };
            
            if(name == "gui_equalize_blob_histograms") {
                PD(cache).set_tracking_dirty();
                PD(cache).set_blobs_dirty();
                PD(cache).set_raw_blobs_dirty();
                PD(cache).last_frame = {};
                GUI::set_redraw();
            }
        
            if(contains(display_fields, name)) {
                PD(cache).set_tracking_dirty();
                GUI::set_redraw();
            }
            
            if(name == "output_normalize_midline_data") {
                PD(posture_window).set_fish(NULL);
                GUI::set_redraw();
            }
            
            if(name == "gui_mode") {
                //globals::_settings.mode = (Mode)value.value<int>();
                GUI::set_redraw();
            }
            
            if(name == "gui_background_color") {
                _base->set_background_color(value.value<Color>());
            }
            
            if(name == "gui_interface_scale") {
                if(_base) {
                    PD(cache).recognition_updated = false;
                    
                    //auto size = _base ? _base->window_dimensions() : Size2(_average_image);
                    auto size = (screen_dimensions() / gui::interface_scale()).mul(PD(gui).scale());
                    
                    Event e(WINDOW_RESIZED);
                    e.size.width = size.width;
                    e.size.height = size.height;
                    this->local_event(e);
                }
            }
            
            if(name == "manual_matches") {
                auto matches = value.value<track::Settings::manual_matches_t>();
                static bool first_run = true;
                static track::Settings::manual_matches_t compare;
                
                //auto str = Meta::toStr(matches);
                //Debug("Starting matches thread: %S", &str);
                //str = Meta::toStr(FAST_SETTINGS(manual_matches));
                //Debug("Old: %S", &str);
                
                if(matches != compare || first_run) {
                    if(first_run)
                        first_run = false;
                    compare = matches;
                    
                    this->work().add_queue("updating with new manual matches...", [matches](){
                        //Tracker::LockGuard tracker_lock;
                        auto first_change = Tracker::instance()->update_with_manual_matches(matches);
                        
                        std::lock_guard<std::recursive_mutex> guard(PD(gui).lock());
                        if(first_change.valid())
                            PD(timeline)->reset_events(first_change);
                        
                        if(GUI::analysis())
                            GUI::analysis()->bump();
                    });
                }
                
            }
            
            if(name == "manual_splits") {
                static bool first_run = true;
                static track::Settings::manual_splits_t old;
                
                auto next = value.template value<track::Settings::manual_splits_t>();
                if(old != next || first_run) {
                    first_run = false;
                    auto itn = next.begin(), ito = old.begin();
                    for(; itn != next.end() && ito != old.end(); ++itn, ++ito) {
                        if(itn->first != ito->first || itn->second != ito->second) {
                            auto frame = min(itn->first, ito->first);
                            if(frame == this->frame().get()) {
                                PD(cache).last_threshold = -1;
                                PD(cache).set_tracking_dirty();
                            }
                            //reanalyse_from(frame);
                            break;
                        }
                    }
                    
                    old = next;
                    
                } else
                    Debug("Nothing changed.");
            }
        });
    };
    
    PDP(work_progress) = new WorkProgress;
    
    GlobalSettings::map().register_callback(callback, changed);
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "manual_matches", SETTING(manual_matches).get());
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "manual_splits", SETTING(manual_splits).get());
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "grid_points", SETTING(grid_points).get());
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "recognition_shapes", SETTING(recognition_shapes).get());
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "gui_run", SETTING(gui_run).get());
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "gui_mode", SETTING(gui_mode).get());
    changed(sprite::Map::Signal::NONE, GlobalSettings::map(), "nowindow", SETTING(nowindow).get());
    
#if WITH_MHD
    _http_gui = new HttpGui(_gui);
#endif
    
    { // do this in order to trigger calculating pixel percentages
        Tracker::LockGuard guard("GUI::update_data(-1)");
        PD(cache).update_data(Frame_t(FAST_SETTINGS(analysis_range).first));
    }
    
    while(!PD(timeline)->update_thread_updated_once()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    //static bool did_init_map = false;
    if (Recognition::python_available()) {
        track::PythonIntegration::set_settings(GlobalSettings::instance());
        track::PythonIntegration::set_display_function([](const std::string& name, const cv::Mat& image)
        {
            GUI::work().set_image(name, Image::Make(image));
        });
    }
}

GUI::~GUI() {
    DrawMenu::close();
    
#if WITH_MHD
    if(_http_gui)
        delete _http_gui;
#endif
    
    _private_data->_timeline = nullptr;
    set_base(nullptr);
    
    {
        std::lock_guard<std::recursive_mutex> lock(_private_data->_gui.lock());
        GUI::_instance = NULL;
    }
    
    delete _private_data->_work_progress;
        
    {
        std::lock_guard<std::recursive_mutex> lock(_private_data->_gui.lock());
        for(auto d : _static_pointers) {
            d->clear_parent_dont_check();
        }
    }
    
    if(_private_data->_recording_capture) {
        std::lock_guard<std::recursive_mutex> guard(_private_data->_gui.lock());
        _private_data->_recording = false;
        delete _private_data->_recording_capture;
    }

    delete _private_data;
}

void GUI::set_base(gui::Base* base) {
    std::lock_guard<std::recursive_mutex> guard(PD(gui).lock());
    _base = base;
        
    if(_base) {
        auto size = (screen_dimensions() / gui::interface_scale()).mul(PD(gui).scale());
            
        Event e(EventType::WINDOW_RESIZED);
        e.size.width = size.width;
        e.size.height = size.height;
        local_event(e);
            
        _base->set_title(window_title());
    }
}
    
bool GUI::run() {
    return GUI_SETTINGS(gui_run);
}

gui::GUICache& GUI::cache() {
    return PD(cache);
}

gui::DrawStructure& GUI::gui() {
    return PD(gui);
}

gui::FrameInfo& GUI::frameinfo() {
    return instance()->_frameinfo;
}

WorkProgress& GUI::work() {
    if (!instance())
        U_EXCEPTION("No instance.");
    return PD(work_progress);
}

void GUI::run(bool r) {
    if(r != GUI_SETTINGS(gui_run))
        SETTING(gui_run) = r;
}

void GUI::load_connectivity_matrix() {
    Debug("Updating connectivity matrix...");
    auto path = SETTING(gui_connectivity_matrix_file).value<file::Path>();
    path = pv::DataLocation::parse("input", path);
    
    if(!path.exists())
        U_EXCEPTION("Cannot find connectivity matrix file '%S'.", &path.str());
    
    auto contents = utils::read_file(path.str());
    auto rows = utils::split(contents, '\n');
    size_t expected_number = 1 + SQR(FAST_SETTINGS(track_max_individuals));
    std::map<long_t, std::vector<float>> matrix;
    std::vector<float> array;
    
    float maximum = 0;
    long_t min_frame = std::numeric_limits<long_t>::max(), max_frame = -1;
    for(size_t index = 0; index < rows.size(); ++index) {
        auto values = utils::split(rows[index], ',');
        if(values.size() == expected_number) {
            auto frame = Meta::fromStr<long_t>(values[0]);
            array.resize(values.size()-1);
            
            for(size_t i=1; i<values.size(); ++i) {
                array[i-1] = cmn::abs(Meta::fromStr<float>(values[i]));
                if(array[i-1] > maximum)
                    maximum = array[i-1];
            }
            
            matrix[frame] = array;
            
            if(frame < min_frame)
                min_frame = frame;
            if(frame > max_frame)
                max_frame = frame;
            
        } else {
            Warning("Row %d doesnt have enough columns (%d / %d), skipping.", index, values.size(), expected_number);
        }
    }
    
    if(maximum > 0) {
        for(auto && [frame, array] : matrix)
            for(auto &v : array)
                v /= maximum;
    }
    
    Debug("%d frames read (%d-%d)", matrix.size(), min_frame, max_frame);
    SETTING(gui_connectivity_matrix) = matrix;
    
    SETTING(gui_frame) = Frame_t(min_frame);
    PD(cache).connectivity_reload = true;
}

void GUI::run_loop(gui::LoopStatus status) {
    static Frame_t image_index;
    static float t = 0.0;
    static Timer timer, redraw_timer;
    
    image_index = GUI::frame();
    
    t += timer.elapsed();
    timer.reset();
    bool is_automatic = false;
#if WITH_MHD
    Base* base = GUI::instance()->base(); //? _base : (_http_gui ? &_http_gui->base() : nullptr);
#else
    Base* base = GUI::instance()->base();
#endif
    
    if(!GUI::run()) {
        t = 0;
        
        if(!GUI_SETTINGS(nowindow) && PD(cache).is_animating() &&  redraw_timer.elapsed() >= 0.2) {
            redraw_timer.reset();
            //set_redraw();
            GUI::gui().set_dirty(base);
            is_automatic = true;
            
        } else if((!GUI_SETTINGS(nowindow) && redraw_timer.elapsed() >= 0.30) || PD(recording)) {
            redraw_timer.reset();
            //set_redraw();
            GUI::gui().set_dirty(base);
            is_automatic = true;
        }
        
    } else if (image_index.valid() && !PD(recording)) {
        const float frame_rate = 1.f / (float(GUI_SETTINGS(frame_rate)) * GUI_SETTINGS(gui_playback_speed));
        Frame_t inc = Frame_t(t / frame_rate);
        bool is_required = false;
        
        if(inc >= 1_f) {
            auto before = image_index;
            if (PD(tracker).start_frame().valid()) {
                image_index = max(0_f, PD(tracker).start_frame(), min(PD(tracker).end_frame(), image_index + inc));
            }
            
            t = 0;
            if(before != image_index) {
                GUI::set_redraw();
                GUI::gui().set_dirty(base);
                is_required = true;
            }
        }
        
        if(redraw_timer.elapsed() >= 0.1) {
            redraw_timer.reset();
            //set_redraw();
            PD(gui).set_dirty(base);
            if(!is_required)
                is_automatic = true;
        }
        
        /*if (image_index > PD(tracker).end_frame()) {
            image_index = PD(tracker).end_frame();
        }*/
        
    } else if(!image_index.valid())
        image_index = PD(tracker).start_frame();
    
    if(PD(recording)) {
        //! playback_speed can only make it faster
        const float frames_per_second = max(1, GUI_SETTINGS(gui_playback_speed));
        image_index += Frame_t(frames_per_second);
        
        if (image_index > PD(tracker).end_frame()) {
            image_index = PD(tracker).end_frame();
            GUI::stop_recording();
        }
    }
    else if(GUI::run() && image_index >= PD(tracker).end_frame()) {
        GUI::run(false);
    }
    
    const bool changed = (base && (!GUI::gui().root().cached(base) || GUI::gui().root().cached(base)->changed())) || PD(cache).must_redraw() || status == LoopStatus::UPDATED;
    PD(real_update) = changed && (!is_automatic || GUI::run() || PD(recording));
    
    if(changed || PD(last_frame_change).elapsed() < 0.5) {
        if(changed) {
            CacheObject::Ptr ptr = PD(gui).root().cached(base);
            if(base && !ptr) {
                ptr = std::make_shared<CacheObject>();
                PD(gui).root().insert_cache(base, ptr);
            }
            if(ptr)
                ptr->set_changed(false);
        }
        
        if(GUI::frame() != image_index) {
            SETTING(gui_frame) = image_index;
        }
        
        //std::vector<std::string> changed_objects_str;
        size_t changed_objects = 0;
        if(!is_automatic && changed) {
            auto o = PD(gui).collect();
            for(auto obj : o) {
                if(obj->type() == Type::SINGLETON) {
                    obj = static_cast<SingletonObject*>(obj)->ptr();
                }
                if(base && obj->cached(base) && obj->cached(base)->changed() && obj->visible()) {
                    ++changed_objects;
                    //changed_objects_str.push_back(Meta::toStr(obj->type()) + " / " + obj->name() + " " + Meta::toStr((size_t)obj));
                }
            }
        }
        
        GUI::frameinfo().frameIndex = GUI::frame();
        
        static Timer last_redraw;
        if(!PD(recording))
            PD(cache).set_dt(last_redraw.elapsed());
        else
            PD(cache).set_dt(0.75f / (float(GUI_SETTINGS(frame_rate))));
        
        if(GUI::instance()->base())
            PD(gui).set_dialog_window_size(GUI::instance()->base()->window_dimensions().div(PD(gui).scale()) * gui::interface_scale());
        GUI::redraw();
        
        PD(cache).on_redraw();
        last_redraw.reset();
        
        {
            auto o = PD(gui).collect();
            for(auto obj : o) {
                if(obj->type() == Type::SINGLETON) {
                    obj = static_cast<SingletonObject*>(obj)->ptr();
                }
                if(base && obj->cached(base) && obj->cached(base)->changed() && obj->visible()) {
                    ++changed_objects;
                    //changed_objects_str.push_back(Meta::toStr(obj->type()) + " / " + obj->name() + " " + Meta::toStr((size_t)obj));
                }
            }
        }
        
        if(changed_objects) {
            /*auto str = Meta::toStr(changed_objects_str);
            Debug("changed: %S", &str);
            Debug("%d changed objects", changed_objects);*/
            PD(last_frame_change).reset();
        }
        
        //Debug("Timer/frame %f", timer.elapsed());
    }
    
    if(PD(recording))
        PD(recording_frame) = image_index;
    
    GUI::update_backups();
}

void GUI::do_recording() {
    if(!PD(recording) || PD(recording_frame) == PD(last_recording_frame) || !_base)
        return;
    
    assert(_base->frame_recording());
    static Timing timing("recording_timing");
    TakeTiming take(timing);
    
    if(!PD(last_recording_frame).valid()) {
        PD(last_recording_frame) = PD(recording_frame);
        return; // skip first frame
    }
    PD(last_recording_frame) = PD(recording_frame);
    
    auto& image = _base->current_frame_buffer();
    if(!image || image->empty() || !image->cols || !image->rows) {
        Warning("Expected image, but there is none.");
        return;
    }
    
    auto mat = image->get();
    
    if(PDP(recording_capture)) {
        static cv::Mat output;
        auto bounds = Bounds(0, 0, PD(recording_size).width, PD(recording_size).height);
        if(output.size() != PD(recording_size)) {
            output = cv::Mat::zeros(PD(recording_size).height, PD(recording_size).width, CV_8UC3);
        }
        
        auto input_bounds = bounds;
        input_bounds.restrict_to(Bounds(mat));
        auto output_bounds = input_bounds;
        output_bounds.restrict_to(Bounds(output));
        input_bounds << output_bounds.size();
        
        if(output_bounds.size() != Size2(output))
            output.mul(cv::Scalar(0));
        
        cv::cvtColor(mat(input_bounds), output(output_bounds), cv::COLOR_RGBA2RGB);
        PD(recording_capture).write(output);
        
    } else {
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << PD(recording_frame).toStr() << "." << PD(recording_format).name();
        //image.saveToFile((PD(recording_path) / ss.str()).str());
        auto filename = PD(recording_path) / ss.str();
        
        if(PD(recording_format) == "jpg") {
            cv::Mat output;
            cv::cvtColor(mat, output, cv::COLOR_RGBA2RGB);
            if(!cv::imwrite(filename.str(), output, { cv::IMWRITE_JPEG_QUALITY, 100 })) {
                Except("Cannot save to '%S'. Stopping recording.", &filename.str());
                PD(recording) = false;
            }
            
        } else if(PD(recording_format) == "png") {
            static std::vector<uchar> binary;
            static Image image;
            if(image.cols != (uint)mat.cols || image.rows != (uint)mat.rows)
                image.create(mat.rows, mat.cols, 4);
            
            cv::Mat output = image.get();
            cv::cvtColor(mat, output, cv::COLOR_BGRA2RGBA);
            
            to_png(image, binary);
            
            FILE *f = fopen(filename.str().c_str(), "wb");
            if(f) {
                fwrite(binary.data(), sizeof(char), binary.size(), f);
                fclose(f);
            } else {
                Except("Cannot write to '%S'. Stopping recording.", &filename.str());
                PD(recording) = false;
            }
        }
    }
    
    static Timer last_print;
    if(last_print.elapsed() > 2) {
        DurationUS duration{static_cast<uint64_t>((PD(recording_frame) - PD(recording_start)).get() / float(FAST_SETTINGS(frame_rate)) * 1000) * 1000};
        auto str = ("frame "+Meta::toStr(PD(recording_frame))+"/"+Meta::toStr(PD(cache).tracked_frames.end)+" length: "+Meta::toStr(duration));
        auto playback_speed = GUI_SETTINGS(gui_playback_speed);
        if(playback_speed > 1) {
            duration.timestamp = uint64_t(double(duration.timestamp) / double(playback_speed));
            str += " (real: "+Meta::toStr(duration)+")";
        }
        Debug("[rec] %S", &str);
        last_print.reset();
    }
}

bool GUI::is_recording() const {
    return PD(recording);
}

void GUI::start_recording() {
    if(GUI::instance()->base()) {
        PD(recording_start) = frame() + 1_f;
        PD(last_recording_frame) = {};
        PD(recording) = true;
        GUI::instance()->base()->set_frame_recording(true);
        
        file::Path frames = GUI::frame_output_dir();
        if(!frames.exists()) {
            if(!frames.create_folder()) {
                Error("Cannot create folder '%S'. Cannot record.", &frames.str());
                PD(recording) = false;
                return;
            }
        }
        
        size_t max_number = 0;
        try {
            for(auto &file : frames.find_files()) {
                auto name = std::string(file.filename());
                if(utils::beginsWith(name, "clip")) {
                    try {
                        if(utils::endsWith(name, ".avi"))
                            name = name.substr(0, name.length() - 4);
                        auto number = Meta::fromStr<size_t>(name.substr(std::string("clip").length()));
                        if(number > max_number)
                            max_number = number;
                        
                    } catch(const std::exception& e) {
                        Except("%S not a number ('%s').", &name, e.what());
                    }
                }
            }
            
            ++max_number;
            
        } catch(const UtilsException& ex) {
            Warning("Cannot iterate on folder '%S'. Defaulting to index 0.", &frames.str());
        }
        
        Debug("Clip index is %d. Starting at frame %d.", max_number, frame());
        
        frames = frames / ("clip" + Meta::toStr(max_number));
        cv::Size size(GUI::instance()->base() 
            && dynamic_cast<IMGUIBase*>(GUI::instance()->base()) 
            ? static_cast<IMGUIBase*>(GUI::instance()->base())->real_dimensions()
            : GUI::instance()->base()->window_dimensions());
        
        using namespace default_config;
        auto format = SETTING(guiPD(recording_format)).value<gui_recording_format_t::Class>();
        
        if(format == gui_recording_format_t::avi) {
            auto original_dims = size;
            if(size.width % 2 > 0)
                size.width -= size.width % 2;
            if(size.height % 2 > 0)
                size.height -= size.height % 2;
            Debug("Trying to record with size %dx%d instead of %fx%f @ %d", size.width, size.height, original_dims.width, original_dims.height, FAST_SETTINGS(frame_rate));
            
            frames = frames.add_extension("avi").str();
            PDP(recording_capture) = new cv::VideoWriter(frames.str(),
                cv::VideoWriter::fourcc('F','F','V','1'),
                                                     //cv::VideoWriter::fourcc('H','2','6','4'), //cv::VideoWriter::fourcc('I','4','2','0'),
                                                     FAST_SETTINGS(frame_rate), size, true);
            
            if(!PD(recording_capture).isOpened()) {
                Except("Cannot open video writer for path '%S'.", &frames.str());
                PD(recording) = false;
                delete PDP(recording_capture);
                PDP(recording_capture) = NULL;
                
                return;
            }
            
        } else if(format == gui_recording_format_t::jpg || format == gui_recording_format_t::png) {
            if(!frames.exists()) {
                if(!frames.create_folder()) {
                    Error("Cannot create folder '%S'. Cannot record.", &frames.str());
                    PD(recording) = false;
                    return;
                } else
                    Debug("Created folder '%S'.", &frames.str());
            }
        }
        
        Debug("Recording to '%S'... (%s)", &frames.str(), format.name());
        
        PD(recording_size) = size;
        PD(recording_path) = frames;
        PD(recording_format) = format;
    }
}

void GUI::stop_recording() {
    if(!GUI::instance()->base())
        return;
    GUI::instance()->base()->set_frame_recording(false);
    
    if(PDP(recording_capture)) {
        //PD(recording_capture)->release();
        delete PDP(recording_capture);
        PDP(recording_capture) = NULL;
        
        file::Path ffmpeg = SETTING(ffmpeg_path);
        if(!ffmpeg.empty()) {
            file::Path save_path = PD(recording_path).replace_extension("mov");
            std::string cmd = ffmpeg.str()+" -i "+PD(recording_path).str()+" -vcodec h264 -pix_fmt yuv420p -crf 15 -y "+save_path.str();
            
            PD(gui).dialog([save_path, cmd](Dialog::Result result){
                if(result == Dialog::OKAY) {
                    GUI::work().add_queue("converting video...", [cmd=cmd, save_path=save_path](){
                        Debug("Running '%S'..", &cmd);
                        if(system(cmd.c_str()) == 0)
                            Debug("Saved video at '%S'.", &save_path.str());
                        else
                            Error("Cannot save video at '%S'.", &save_path.str());
                    });
                }
                
            }, "Do you want to convert it, using <str>"+cmd+"</str>?", "Recording finished", "Yes", "No");
        }
        
    } else {
        auto clip_name = std::string(PD(recording_path).filename());
        printf("ffmpeg -start_number %d -i %s/%%06d.%s -vcodec h264 -crf 13 -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -profile:v main -pix_fmt yuv420p %s.mp4\n", PD(recording_start).get(), PD(recording_path).str().c_str(), PD(recording_format).name(), clip_name.c_str());
    }
    
    PD(recording) = false;
    PD(last_recording_frame).invalidate();
    
    DebugCallback("Stopped recording to '%S'.", &PD(recording_path).str());
}

void GUI::trigger_redraw() {
    instance()->redraw();
}

std::string GUI::window_title() const {
    auto output_prefix = SETTING(output_prefix).value<std::string>();
    return SETTING(app_name).value<std::string>()
        + (SETTING(version).value<std::string>().empty() ? "" : (" " + SETTING(version).value<std::string>()))
        + " (" + (std::string)SETTING(filename).value<file::Path>().filename() + ")"
        + (output_prefix.empty() ? "" : (" ["+output_prefix+"]"));
}

pv::File* GUI::video_source() {
    return PDP(video_source);
}

void GUI::redraw() {
    static bool added = false;
    static ExternalImage* gui_background = NULL, //*corrected_bg = NULL,
    *gui_mask = NULL;
    
    std::unique_lock<std::recursive_mutex> lock(PD(gui).lock());
    
    if(!added) {
        added = true;
        
        gpuMat bg;
        PD(video_source).average().copyTo(bg);
        PD(video_source).processImage(bg, bg, false);
        cv::Mat original;
        bg.copyTo(original);
        /*cv::Mat corrected;
        bg.copyTo(corrected);
        if(Tracker::instance()->grid())
            Tracker::instance()->grid()->correct_image(corrected);*/
        
        gui_background = new ExternalImage(Image::Make(original), Vec2(0, 0), Vec2(1), Color(255, 255, 255, 125));
        //corrected_bg = new ExternalImage(corrected, Vec2(0, 0));
        
        gui_background->add_event_handler(EventType::MBUTTON, [](Event e){
            if(e.mbutton.pressed) {
                if(e.mbutton.button == 1)
                    PD(current_boundary).clear();
                PD(clicked_background)(Vec2(e.mbutton.x, e.mbutton.y).map<round>(), e.mbutton.button == 1, "");
            }
        });
        gui_background->set_clickable(true);
        
        /*corrected_bg->add_event_handler(EventType::MBUTTON, [this](Event e){
            if(e.mbutton.pressed)
                PD(clicked_background)(Vec2(e.mbutton.x, e.mbutton.y).map(roundf), e.mbutton.button == 1);
        });
        corrected_bg->set_clickable(true);*/
        
        gui_background->set_name("gui_background");
        //corrected_bg->set_name("corrected_bg");
        
        if(PD(video_source).has_mask()) {
            cv::Mat mask = PD(video_source).mask().mul(cv::Scalar(255));
            mask.convertTo(mask, CV_8UC1);
            
            gui_mask = new ExternalImage(Image::Make(mask), Vec2(0, 0), Vec2(1), Color(255, 255, 255, 125));
        }
    }
    
    auto image = gui_background;//SETTING(correct_luminance) ? corrected_bg : gui_background;
    auto alpha = SETTING(gui_background_color).value<Color>().a;
    image->set_color(Color(255, 255, 255, alpha ? alpha : 1));
    
    if(alpha > 0) {
        PD(gui).wrap_object(*image);
        if(gui_mask) {
            gui_mask->set_color(image->color().alpha(image->color().a * 0.5));
            PD(gui).wrap_object(*gui_mask);
        }
    }
    
    //const Mode mode = (Mode)VALUE(mode).value<int>();
    auto ptr = PD(gui).find("fishbowl");
    if(ptr && (PD(cache).is_animating(ptr) || PD(cache).blobs_dirty() || PD(cache).is_tracking_dirty())) {
        assert(dynamic_cast<Section*>(ptr));
        
        auto pos = static_cast<Section*>(ptr)->pos();
        image->set_scale(static_cast<Section*>(ptr)->scale());
        image->set_pos(pos);
        
        if(gui_mask) {
            gui_mask->set_scale(image->scale());
            gui_mask->set_pos(image->pos());
        }
    }
    
    instance()->draw(PD(gui));
    //PD(gui).print(_base ? _base : &_http_gui->base());
}

void GUI::draw(DrawStructure &base) {
    const auto mode = GUI_SETTINGS(gui_mode);
    
    if(PD(gui_last_frame) != frame()) {
        PD(tdelta_gui) = PD(gui_last_frame_timer).elapsed() / (frame() - PD(gui_last_frame)).get();
        PD(gui_last_frame) = frame();
        PD(gui_last_frame_timer).reset();
    }
    
    PD(gui).section("show", [this, mode](DrawStructure &base, auto* section) {
        Tracker::LockGuard guard("show()", 100);
        if(!guard.locked() || !PD(real_update)) {
            section->reuse_objects();
        } else {
            PD(cache).update_data(this->frame());
            this->draw_raw(base, this->frame());
            PD(cache).set_mode(mode);
            
            if(mode == gui::mode_t::tracking)
                this->draw_tracking(base, this->frame());
            else if(mode == gui::mode_t::blobs)
                this->draw_raw_mode(base, this->frame());
            
            PD(cache).updated_blobs();
        }
    });
    
    
    if(mode == gui::mode_t::optical_flow) {
        PD(gui).section("optical", [this](auto& base, auto) {
            this->debug_optical_flow(base, this->frame());
        });
    }
    
    if(PD(timeline)->visible()) {
        DrawStructure::SectionGuard section(base, "head");
        auto scale = base.scale().reciprocal();
        auto dim = _base ? _base->window_dimensions().mul(scale * gui::interface_scale()) : Tracker::average().bounds().size();
        base.draw_log_messages(Bounds(Vec2(0, 85).mul(scale * gui::interface_scale()), dim - Size2(10, 85).mul(scale * gui::interface_scale())));
        
        if(PD(cache).has_selection()) {
            /*****************************
             * display the fishX info card
             *****************************/
            static InfoCard* e = nullptr;
            if(!e) {
                e = new InfoCard;
                _static_pointers.push_back(e);
            }
            e->update(base, this->frame());
            base.wrap_object(*e);
        }
        
        /**
         * -----------------------------
         * DISPLAY TIMELINE
         * -----------------------------
         */
        PD(timeline)->draw(base);
        
        /**
         * -----------------------------
         * DISPLAY RIGHT SIDE MENU
         * -----------------------------
         */
        if(SETTING(gui_show_export_options))
            draw_export_options(base);
        draw_menu(base);
        
        if(FAST_SETTINGS(calculate_posture) && GUI_SETTINGS(gui_show_midline_histogram)) {
            PD(midline_histogram).set_scale(base.scale().reciprocal());
            base.wrap_object(PD(midline_histogram));
        }
        
        if(FAST_SETTINGS(calculate_posture) && GUI_SETTINGS(gui_show_histograms)) {
            PD(histogram).set_scale(base.scale().reciprocal());
            PD(length_histogram).set_scale(base.scale().reciprocal());
            
            Size2 window_size(_average_image.cols, _average_image.rows);
            Vec2 pos = window_size * 0.5 - Vec2(0, (PD(histogram).global_bounds().height + PD(length_histogram).global_bounds().height + 10) * 0.5);
            PD(histogram).set_pos(pos);
            PD(length_histogram).set_pos(pos + Vec2(0, PD(histogram).global_bounds().height + 5));
            
            base.wrap_object(PD(histogram));
            base.wrap_object(PD(length_histogram));
        }
        
        draw_footer(base);
    }
    
    /**
     * -----------------------------
     * DISPLAY INFO TEXT WINDOW
     * -----------------------------
     */
    if(_info_visible) {
        static Timer info_timer;
        if(info_timer.elapsed() > 5 || PD(info).txt().empty()) {
            PD(info).set_txt(info(false));
            info_timer.reset();
        }
        
        auto screen = screen_dimensions();
        PD(info).set_clickable(false);
        PD(info).set_origin(Vec2(0.5));
        PD(info).set_pos(screen * 0.5);
        PD(info).set_scale(base.scale().reciprocal());
        base.wrap_object(PD(info));
    }
    
    static Textfield* ptr = NULL;
    
    if(ptr)
        base.wrap_object(*ptr);
    
    /**
     * -----------------------------
     * DISPLAY LOADING BAR if needed
     * -----------------------------
     */
    base.section("loading", [](DrawStructure& base, auto section) {
        GUI::work().update(base, section);
    });
}

void GUI::draw_menu(gui::DrawStructure &base) {
    DrawMenu::draw();
}

void GUI::removed_frames(Frame_t including) {
    std::lock_guard<std::recursive_mutex> gguard(gui().lock());
    if(heatmapController)
        heatmapController->frames_deleted_from(including);
}

void GUI::reanalyse_from(Frame_t frame, bool in_thread) {
    if(!instance())
        return;
    
    auto fn = [gui = instance(), frame](){
        auto before = GUI::analysis()->is_paused();
        if(!before)
            GUI::analysis()->set_paused(true).get();
        
        {
            Tracker::instance()->wait();
            
            std::lock_guard<std::recursive_mutex> gguard(GUI::gui().lock());
            Tracker::LockGuard guard("reanalyse_from");
            
            if(frame <= Tracker::end_frame()) {
                Tracker::instance()->_remove_frames(frame);
                gui->removed_frames(frame);
                
                Output::Library::clear_cache();
                PD(timeline)->reset_events(frame);
                
            } else {
                Except("The requested frame %d is not part of the video, and certainly beyond end_frame (%d).", frame, Tracker::end_frame());
            }
        }
        
        if(!before)
            GUI::analysis()->set_paused(false).get();
    };
    
    if(in_thread)
        GUI::work().add_queue("calculating", fn);
    else
        fn();
}

void GUI::draw_export_options(gui::DrawStructure &base) {
    static Entangled parent(Bounds(100, 100, 200, 500));
    
    struct Item {
        std::string _name;
        uint32_t _count = 0;
        Font _font = Font(0.5, Align::Left);
        Color _color = White;
        std::set<std::string> _sources;

        Font font() const {
            return _font;
        }

        Color color() const {
            return _color;
        }

        std::string tooltip() const {
            std::string append;
            for (auto s : _sources) {
                s = utils::lowercase(s);
                if (utils::endsWith(s, "centroid")) {
                    append.push_back(s[0]);
                    append += "centroid";
                }
                else
                    append += s;
            }
            return append;
        }

        operator std::string() const {
            return _name + (_count ? " (" + Meta::toStr(_count) + ")" : "");
        }

        bool operator!=(const Item& other) const {
            return _name != other._name || _font != other._font || _count != other._count;
        }
    };

    static Button close("x", Bounds(Vec2(parent.width() - 3, 5), Size2(25, 25)));
    static Textfield search("", Bounds(Vec2(5, close.height() + 15), Size2(parent.width() - 10, 30)));
    static ScrollableList<Item> export_options(Bounds(
        search.pos() + Vec2(0, search.height() + 10), 
        Size2(search.width(), parent.height() - (search.pos().y + search.height() + 20))
    ), {}, Font(0.5, Align::Left));

    if (parent.empty()) {
        close.set_font(Font(0.5, Align::Center));
        search.set_placeholder("Type to search...");
        parent.update([&](Entangled& e) {
            e.advance_wrap(export_options);
            e.advance_wrap(search);
            e.advance_wrap(close);
        });
        parent.set_background(Black.alpha(50), Red.alpha(50));

        export_options.on_select([&](auto idx, const std::string&) {
            auto graphs = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
            auto& item = export_options.items().at(idx);
            Debug("Removing '%S'", &item.value()._name);

            for (auto it = graphs.begin(); it != graphs.end(); ++it) {
                if (it->first == item.value()._name) {
                    graphs.erase(it);
                    SETTING(output_graphs) = graphs;
                    return;
                }
            }

            graphs.push_back({ item.value()._name, {} });
            SETTING(output_graphs) = graphs;
        });
    }
    
    auto graphs = SETTING(output_graphs).value<std::vector<std::pair<std::string, std::vector<std::string>>>>();
    auto graphs_map = [&graphs]() {
        std::map<std::string, std::set<std::string>> result;
        for(auto &g : graphs) {
            static const std::set<Output::Modifiers::Class> wanted{
                Output::Modifiers::CENTROID, 
                Output::Modifiers::POSTURE_CENTROID, 
                Output::Modifiers::WEIGHTED_CENTROID
            };
            OptionsList<Output::Modifiers::Class> modifiers;
            for (auto& e : g.second) {
                Output::Library::parse_modifiers(e, modifiers);
            }
            for (auto& e : modifiers.values()) {
                if (wanted.contains(e))
                    result[g.first].insert(e.name());
            }
        }
        return result;
    }();
    static decltype(graphs_map) previous_graphs;
    
    base.wrap_object(parent);
    parent.set_scale(base.scale().reciprocal());
    
    static std::string previous_text;
    //
    if(previous_graphs != graphs_map || search.text() != previous_text) {
        previous_graphs = graphs_map;
        previous_text = search.text();

        auto functions = Output::Library::functions();
        std::vector<Item> items;

        for (auto& f : functions) {
            if (search.text().empty() || utils::contains(utils::lowercase(f), utils::lowercase(search.text()))) {
                uint32_t count = 0;
                std::set<std::string> append;
                auto it = graphs_map.find(f);
                if (it != graphs_map.end()) {
                    count = it->second.size();
                    append = it->second;
                }

                items.push_back(Item{
                    ._name = f,
                    ._count = count,
                    ._font = Font(0.5, count ? Style::Bold : Style::Regular, Align::Left),
                    ._sources = append
                    });
            }
        }

        export_options.set_items(items);
        Debug("Filtering for: '%S'", &search.text());
    }
    
    static bool first = true;
    
    if(first) {        
        parent.set_draggable();
        
        close.set_fill_clr(Red.exposure(0.5));
        close.on_click([](auto) {
            SETTING(gui_show_export_options) = false;
        });
        close.set_origin(Vec2(1, 0));
        
        first = false;
    }
}

void GUI::draw_grid(gui::DrawStructure &base) {
    const auto& grid_points = SETTING(grid_points).value<std::vector<Vec2>>();
    
    //! Draw grid circles
    if(grid_points.empty() || GUI_SETTINGS(gui_show_recognition_bounds))
        return;
    
    static Entangled sign;
    static bool first = true;
    if(first) {
        _static_pointers.insert(_static_pointers.end(), {
            &sign
        });
    }
    
    struct GridPoint {
        size_t _i;
        const Vec2* _point;
        GridPoint(size_t i, const Vec2* point) : _i(i), _point(point) {}
        void convert(std::shared_ptr<Circle> circle) const {
            circle->set_pos(*_point);
            if(circle->hovered())
                circle->set_fill_clr(Red.alpha(250));
            else
                circle->set_fill_clr(Red.alpha(150));
            circle->set_radius(5);
            //circle->set_color(Red);
            
            circle->set_clickable(true);
            
            circle->add_custom_data("gridpoint", (void*)this);
            void * custom = circle->custom_data("grid");
            if(custom == NULL || (Vec2*)custom != _point) {
                custom = (void*)_point;
                circle->add_custom_data("grid", custom);
                
                circle->clear_event_handlers();
                circle->on_click([](auto){
                    //Debug("Clicked (%f,%f).", ((Vec2*)custom)->x, ((Vec2*)custom)->y);
                });
                circle->on_hover([circle](auto) {
                    //Debug("Hover (%f,%f)", ((Vec2*)custom)->x, ((Vec2*)custom)->y);
                    circle->set_dirty();
                });
            }
        }
    };
    static std::vector<std::shared_ptr<Circle>> circles;
    std::vector<std::shared_ptr<GridPoint>> points;
    
    for (size_t i=0; i<grid_points.size(); ++i)
        points.push_back(std::make_shared<GridPoint>(i, &grid_points.at(i)));
    
    update_vector_elements(circles, points);
    
    for(auto circle : circles) {
        base.wrap_object(*circle);
        
        if(circle->hovered()) {
            auto custom = (GridPoint*)circle->custom_data("gridpoint");
            sign.set_background(Black.alpha(50));
            Font font(0.6);
            sign.set_pos(circle->pos() - Vec2(0, Base::default_line_spacing(font)));
            
            sign.update([custom, &font](Entangled& base){
                std::string str = "grid"+Meta::toStr(*custom->_point);
                base.advance(new Text(str, Vec2(5,5), White, font));
            });
            
            base.wrap_object(sign);
            sign.auto_size({10,5});
        }
    }
}

void GUI::debug_optical_flow(DrawStructure &base, Frame_t frameIndex) {
    if(size_t(frameIndex.get()) >= PD(video_source).length())
        return;
    
    auto gen_ov = [this](Frame_t frameIndex, cv::Mat& image) -> std::vector<std::pair<std::vector<HorizontalLine>, std::vector<uchar>>>{
        if(size_t(frameIndex.get()) >= PD(video_source).length() || !frameIndex.valid())
            return {};
        
        //image = cv::Mat::zeros(_average_image.rows, _average_image.cols, CV_8UC1);
        _average_image.get().copyTo(image);
        
        pv::File *file = PDP(video_source);
        
        pv::Frame frame;
        file->read_frame(frame, frameIndex.get());
        
        std::vector<std::pair<std::vector<HorizontalLine>, std::vector<uchar>>> lines;
        
        for (int i=0; i<frame.n(); i++) {
            auto &mask = frame.mask().at(i);
            auto &pixels = frame.pixels().at(i);
            
            size_t recount = 0;
            size_t pos = 0;
            
            for (auto &l : *mask) {
                for (int x=l.x0; x<=l.x1; x++) {
                    int m = pixels->empty() ? 255 : pixels->at(pos++);
                    
                    //if((int)PD(video_source).average().at<uchar>(l.y, x) - (int)m >= threshold)
                    {
                        image.at<uchar>(l.y, x) = m;
                        recount++;
                    }
                }
            }
            
            lines.push_back({*mask, *pixels});
        }
        
        return lines;
    };
    
    auto draw_flow = [&gen_ov](Frame_t frameIndex, cv::Mat& image){
        Tracker::LockGuard guard("draw_flow");
        
        cv::Mat current_, prev_;
        gen_ov(frameIndex > PD(tracker).start_frame() ? frameIndex - 1_f : PD(tracker).start_frame(), prev_);
        auto lines = gen_ov(frameIndex, current_);
        
        gpuMat current, prev;
        
        current_.copyTo(current);
        prev_.copyTo(prev);
        
        float scale = 0.8;
        resize_image(current, scale);
        resize_image(prev, scale);
        
        gpuMat flow_;
        gpuMat cflow;
        cv::Mat flow;
        
        cv::calcOpticalFlowFarneback(prev, current, flow_, 0.5, 3, 15, 3, 5, 1.2, 0);
        flow_.copyTo(flow);
        cv::cvtColor(current, cflow, cv::COLOR_GRAY2BGR);
        
        resize_image(flow, 1.0/scale);
        resize_image(cflow, 1.0/scale);
        cv::cvtColor(cflow, image, cv::COLOR_BGR2RGBA);
        
        if(sizeof(Float2_t) == sizeof(double))
            flow.convertTo(flow, CV_64FC2);
        drawOptFlowMap(flow, image);
    };
    
    if(PD(flow_frame) != frameIndex) {
        if(PD(next_frame) == frameIndex) {
            PD(cflow_next).copyTo(PD(cflow));
        } else {
            draw_flow(frameIndex, PD(cflow));
        }
        PD(flow_frame) = frameIndex;
        
    } else if(PD(next_frame) != frameIndex + 1_f) {
        draw_flow(frameIndex + 1_f, PD(cflow_next));
        PD(next_frame) = frameIndex + 1_f;
    }
    
    base.image(Vec2(0, 0), Image::Make(PD(cflow)));
}

void GUI::set_redraw() {
    std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
    PD(cache).set_redraw();
    PD(gui).set_dirty(GUI::instance()->base());
    //animating = true;
    
    /*auto cache = PD(gui).root().cached(_base);
    if(cache)
        cache->set_changed(true);
    else
        PD(gui).root().insert_cache(_base, new CacheObject);*/
}

void GUI::set_mode(gui::mode_t::Class mode) {
    if(mode != GUI_SETTINGS(gui_mode)) {
        SETTING(gui_mode) = mode;
        PD(cache).set_mode(mode);
    }
}

void GUI::draw_posture(DrawStructure &base, Individual *fish, Frame_t frameNr) {
    static Timing timing("posture draw", 0.1);
    TakeTiming take(timing);
    
    if(!fish)
        return;
    
    Tracker::LockGuard guard("GUI::draw_posture");
    auto midline = fish->midline(frameNr);
    if(midline) {
        // Draw the fish posture with circles
        if(midline) {
            auto && [bg_offset, max_w] = Timeline::timeline_offsets();
            max_w /= PD(gui).scale().x;
            PD(posture_window).set_scale(base.scale().reciprocal());
            auto pos = Vec2(max_w - 10 - bg_offset.x  * PD(posture_window).scale().x,
                            (PD(timeline)->bar() ? (PD(timeline)->bar()->global_bounds().y + PD(timeline)->bar()->global_bounds().height) : 100) + 10 * PD(posture_window).scale().y);
            PD(posture_window).set_pos(pos);
            PD(posture_window).set_origin(Vec2(1, 0));
            PD(posture_window).set_fish(fish);
            PD(posture_window).set_frameIndex(frameNr);
            //PD(posture_window).set_draggable();
            base.wrap_object(PD(posture_window));
            
            //field.show();
        }
    }
}
    
Base * GUI::best_base() {
    if(!instance())
        return nullptr;
#if WITH_MHD
    return instance()->_base ? instance()->_base : (instance()->_http_gui ? &instance()->_http_gui->base() : nullptr);
#else
    return instance()->_base;
#endif
}

Size2 GUI::screen_dimensions() {
    if(!instance())
        return Size2(1);
    
    auto base = best_base();
    auto gui_scale = PD(gui).scale();
    if(gui_scale.x == 0)
        gui_scale = Vec2(1);
    auto window_dimensions = base
        ? base->window_dimensions().div(gui_scale) * gui::interface_scale()
        : instance()->_average_image.dimensions();
    return window_dimensions;
}
    
std::tuple<Vec2, Vec2> GUI::gui_scale_with_boundary(Bounds& boundary, Section* section, bool singular_boundary)
{
    //static Timer timer;
    static Rect temporary;
    static Vec2 target_scale(1);
    static Vec2 target_pos(0,0);
    static Size2 target_size(_average_image.dimensions());
    static bool lost = true;
    static float time_lost = 0;
    
    auto && [offset, max_w] = Timeline::timeline_offsets();
    
    Size2 screen_dimensions = this->screen_dimensions();
    Size2 screen_center = screen_dimensions * 0.5;
    
    if(screen_dimensions.max() <= 0)
        return {Vec2(), Vec2()};
    //if(_base)
    //    offset = Vec2((_base->window_dimensions().width / PD(gui).scale().x * gui::interface_scale() - _average_image.cols) * 0.5, 0);
    
    
    /**
     * Automatically zoom in on the group.
     */
    if(singular_boundary) {//SETTING(gui_auto_scale) && (singular_boundary || !SETTING(gui_auto_scale_focus_one))) {
        if(lost) {
            PD(cache).set_animating(&temporary, false);
        }
        
        if(boundary.x != FLT_MAX) {
            Size2 minimal_size = SETTING(gui_zoom_limit).value<Size2>();
            //Size2(_average_image) * 0.15;
            
            if(boundary.width < minimal_size.width) {
                boundary.x -= (minimal_size.width - boundary.width) * 0.5;
                boundary.width = minimal_size.width;
            }
            if(boundary.height < minimal_size.height) {
                boundary.y -= (minimal_size.height - boundary.height) * 0.5;
                boundary.height = minimal_size.height;
            }
            
            Vec2 scales(boundary.width / max_w,
                        boundary.height / screen_dimensions.height);
            
            float scale = 1.f / scales.max() * 0.8;
            
            //Vec2 topleft(Size2(max_w / PD(gui).scale().x, _average_image.rows) * 0.5 - offset / PD(gui).scale().x - boundary.size() * scale * 0.5);
            
            //boundary.pos() -= offset.div(scale);
            
            target_scale = Vec2(scale);
            Size2 image_center = boundary.pos() + boundary.size() * 0.5;
            
            offset = screen_center - image_center * scale;
            target_pos = offset;
            
            target_size = boundary.size();
            
            lost = false;
        }
        
    } else {
        static Timer lost_timer;
        if(!lost) {
            lost = true;
            time_lost = PD(cache).gui_time();
            lost_timer.reset();
            PD(cache).set_animating(&temporary, true);
        }
        
        if((PD(recording) && PD(cache).gui_time() - time_lost >= 0.5)
           || (!PD(recording) && lost_timer.elapsed() >= 0.5))
        {
            target_scale = Vec2(1);
            //target_pos = offset;//Vec2(0, 0);
            target_size = Size2(_average_image.cols, _average_image.rows);
            target_pos = screen_center - target_size * 0.5;
            PD(cache).set_animating(&temporary, false);
        }
    }
    
    Float2_t mw = _average_image.cols;
    Float2_t mh = _average_image.rows;
    if(target_pos.x / target_scale.x < -mw * 0.95) {
#ifndef NDEBUG
        Debug("target_pos.x = %f target_scale.x = %f", target_pos.x, target_scale.x);
#endif
        target_pos.x = -mw * target_scale.x * 0.95f;
    }
    if(target_pos.y / target_scale.y < -mh * 0.95f)
        target_pos.y = -mh * target_scale.y * 0.95f;
    
    if(target_pos.x / target_scale.x > mw * 0.95f) {
#ifndef NDEBUG
        Debug("target_pos.x = %f target_scale.x = %f screen_center.x = %f screen_dimensions.x = %f window_dimensions.x = %f", target_pos.x, target_scale.x, screen_center.width, screen_dimensions.width, base()->window_dimensions().width);
#endif
        target_pos.x = mw * target_scale.x * 0.95f;
    }
    if(target_pos.y / target_scale.y > mh * 0.95f)
        target_pos.y = mh * target_scale.y * 0.95f;
    
    PD(cache).set_zoom_level(target_scale.x);
    
    static Timer timer;
    auto e = PD(recording) ? PD(cache).dt() : timer.elapsed(); //PD(recording) ? (1 / float(FAST_SETTINGS(frame_rate))) : timer.elapsed();
    //e = PD(cache).dt();
    
    e = min(0.1, e);
    e *= 3;
    
    auto check_target = [](const Vec2& start, const Vec2& target, Float2_t e) {
        Vec2 direction = (target - start) * e;
        Float2_t speed = direction.length();
        auto epsilon = max(target.abs().max(), start.abs().max()) * 0.000001;

        if(speed <= epsilon)
            return target;
        
        if(speed > 0)
            direction /= speed;
        
        auto scale = start + direction * speed;
        
        if((direction.x > 0 && scale.x > target.x)
           || (direction.x < 0 && scale.x < target.x))
        {
            scale.x = target.x;
        }
        if((direction.y > 0 && scale.y > target.y)
           || (direction.y < 0 && scale.y < target.y))
        {
            scale.y = target.y;
        }
        
        return scale;
    };
    
    
    target_pos.x = round(target_pos.x);
    target_pos.y = round(target_pos.y);
    
    if(!section->scale().Equals(target_scale)
       || !section->pos().Equals(target_pos))
    {
        PD(cache).set_animating(section, true);
        
        auto playback_factor = max(1, sqrt(SETTING(gui_playback_speed).value<float>()));
        auto scale = check_target(section->scale(), target_scale, e * playback_factor);
        
        section->set_scale(scale);
        
        auto next_pos = check_target(section->pos(), target_pos, e * playback_factor);
        auto next_size = check_target(section->size(), target_size, e * playback_factor);
        
        section->set_bounds(Bounds(next_pos, next_size));
        
    } else {
        PD(cache).set_animating(section, false);
        
        section->set_scale(target_scale);
        section->set_bounds(Bounds(target_pos, target_size));
    }
    
    timer.reset();
    
    return {Vec2(), Vec2()};
}

void GUI::draw_tracking(DrawStructure& base, Frame_t frameNr, bool draw_graph) {
    static Timing timing("draw_tracking", 10);
    
    auto props = PD(tracker).properties(frameNr);
    if(props) {
        TakeTiming take(timing);
        
        if(SETTING(gui_show_heatmap)) {
            base.section("heatmap", [&](auto & , Section *s){
                auto ptr = PD(gui).find("fishbowl");
                Vec2 ptr_scale(1), ptr_pos(0);
                if(ptr) {
                    ptr_scale = static_cast<Section*>(ptr)->scale();
                    ptr_pos = static_cast<Section*>(ptr)->pos();
                }
                
                if(ptr && (PD(cache).is_animating(ptr) || PD(cache).is_tracking_dirty())) {
                    assert(dynamic_cast<Section*>(ptr));
                    s->set_scale(ptr_scale);
                    s->set_pos(ptr_pos);
                }
                
                if(!heatmapController)
                    heatmapController = std::make_unique<gui::heatmap::HeatmapController>();
                heatmapController->set_frame(frame());
                base.wrap_object(*heatmapController);
            });
        }
        
        base.section("tracking", [&](auto&, Section* s) {
            auto ptr = PD(gui).find("fishbowl");
            Vec2 ptr_scale(1), ptr_pos(0);
            if(ptr) {
                ptr_scale = static_cast<Section*>(ptr)->scale();
                ptr_pos = static_cast<Section*>(ptr)->pos();
            }
            
            if(ptr && (PD(cache).is_animating(ptr) || PD(cache).is_tracking_dirty())) {
                assert(dynamic_cast<Section*>(ptr));
                s->set_scale(ptr_scale);
                s->set_pos(ptr_pos);
            }
            
            if(!PD(cache).is_tracking_dirty() && !PD(cache).is_animating(s) && !PD(cache).is_animating(ptr)
               && !s->is_dirty()) {
                s->reuse_objects();
                return;
            }
            
            PD(cache).updated_tracking();
            
            std::map<long_t, Color> colors;
            for(auto fish : PD(cache).active)
                colors[fish->identity().ID()] = fish->identity().color();
            
            EventAnalysis::EventsContainer *container = NULL;
            container = EventAnalysis::events();
            if(FAST_SETTINGS(calculate_posture) && !container->map().empty() && GUI_SETTINGS(gui_show_histograms))
            {
                std::vector<std::map<long_t, size_t>> data;
                std::vector<std::vector<float>> hist;
                std::vector<float> energies;
                std::vector<Color> ordered_colors;
                
                for(auto &c : container->map()) {
                    ordered_colors.push_back(c.first->identity().color());
                    data.push_back(c.second.lengths);
                    
                    energies.clear();
                    for(auto &e : c.second.events) {
                        energies.push_back(e.second.energy);
                    }
                    if(!energies.empty())
                        hist.push_back(energies);
                }
                
                PD(length_histogram).set_data(data, ordered_colors);
                PD(histogram).set_data(hist, ordered_colors);
            }
            
            {
                const EventAnalysis::EventMap *empty_map = NULL;
                static std::mutex fish_mutex;
                
                Vec2 scale(1);
                if(ptr) {
                    scale = ptr->scale().reciprocal().mul(Vec2(1.5));
                }
                
                Tracker::set_of_individuals_t source;
                if(FAST_SETTINGS(track_max_individuals) && GUI_SETTINGS(gui_show_inactive_individuals)) {
                    for(auto id : FAST_SETTINGS(manual_identities)) {
                        auto it = PD(cache).individuals.find(id);
                        if(it != PD(cache).individuals.end())
                            source.insert(it->second);
                    }
                    
                    for(auto fish : PD(cache).active) {
                        source.insert(fish);
                    }
                }
                
                
                if(PD(cache).has_selection() && SETTING(gui_show_visualfield)) {
                    for(auto id : PD(cache).selected) {
                        auto fish = PD(cache).individuals.at(id);
                        
                        VisualField* ptr = (VisualField*)fish->custom_data(frameNr, VisualField::custom_id);
                        if(!ptr && fish->head(frameNr)) {
                            ptr = new VisualField(id, frameNr, fish->basic_stuff(frameNr), fish->posture_stuff(frameNr), true);
                            fish->add_custom_data(frameNr, VisualField::custom_id, ptr, [](void* ptr) {
                                if(GUI::instance()) {
                                    std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
                                    delete (VisualField*)ptr;
                                } else {
                                    delete (VisualField*)ptr;
                                }
                            });
                        }
                        
                        if(ptr)
                            ptr->show(base);
                    }
                }
                
                {

                    static Entangled _bowl;
                    _bowl.update([&](auto& e) {
                        for (auto& fish : (source.empty() ? PD(cache).active : source)) {
                            if (fish->start_frame() > frameNr || fish->empty())
                                continue;

                            auto segment = fish->segment_for(frameNr);
                            if (!GUI_SETTINGS(gui_show_inactive_individuals)
                                && (!segment || (segment->end() != Tracker::end_frame()
                                    && segment->length() < (long_t)GUI_SETTINGS(output_min_frames))))
                            {
                                continue;
                            }

                            auto it = container->map().find(fish);
                            if (it != container->map().end())
                                empty_map = &it->second;
                            else
                                empty_map = NULL;

                            if (PD(cache)._fish_map.find(fish) == PD(cache)._fish_map.end()) {
                                PD(cache)._fish_map[fish] = std::make_unique<gui::Fish>(*fish);
                                fish->register_delete_callback(PD(cache)._fish_map[fish].get(), [](Individual* f) {
                                    //std::lock_guard<std::mutex> lock(_individuals_frame._mutex);
                                    if (!GUI::instance())
                                        return;

                                    std::lock_guard<std::recursive_mutex> guard(GUI::instance()->gui().lock());

                                    auto it = PD(cache)._fish_map.find(f);
                                    if (it != PD(cache)._fish_map.end()) {
                                        PD(cache)._fish_map.erase(f);
                                    }
                                    });
                            }

                            PD(cache)._fish_map[fish]->set_data(frameNr, props->time, PD(cache).processed_frame, empty_map);

                            {
                                std::unique_lock guard(Categorize::DataStore::cache_mutex());
                                PD(cache)._fish_map[fish]->update(e, base);
                            }
                            //base.wrap_object(*PD(cache)._fish_map[fish]);
                            PD(cache)._fish_map[fish]->label(ptr, e);
                        }
                    });

                    _bowl.set_bounds(average().bounds());
                    //_bowl.set_scale(Vec2(1));
                    //_bowl.set_pos(Vec2(0);
                    base.wrap_object(_bowl);
                }
                
                if(GUI_SETTINGS(gui_show_midline_histogram)) {
                    static Frame_t end_frame;
                    if(FAST_SETTINGS(calculate_posture) && end_frame != PD(cache).tracked_frames.end) {
                        end_frame = PD(cache).tracked_frames.end;
                        
                        Tracker::LockGuard guard("gui_show_midline_histogram");
                        
                        std::vector<std::vector<float>> all;
                        std::vector<float> lengths;
                        
                        std::map<track::Idx_t, Individual*> search;
                        
                        if(FAST_SETTINGS(manual_identities).empty()) {
                            for(auto fish : PD(cache).active) {
                                lengths.clear();
                                for (auto && stuff : fish->posture_stuff()) {
                                    if(stuff->midline_length != Individual::PostureStuff::infinity)
                                        lengths.push_back(stuff->midline_length * FAST_SETTINGS(cm_per_pixel));
                                }
                                all.push_back(lengths);
                                Debug("%d midline samples for %S", lengths.size(), &fish->identity().raw_name());
                            }
                        } else {
                            for(auto id : FAST_SETTINGS(manual_identities)) {
                                auto it = PD(cache).individuals.find(id);
                                if(it != PD(cache).individuals.end()) {
                                    auto fish = it->second;
                                    lengths.clear();
                                    for (auto && stuff : fish->posture_stuff()) {
                                        if(stuff->midline_length != Individual::PostureStuff::infinity)
                                            lengths.push_back(stuff->midline_length * FAST_SETTINGS(cm_per_pixel));
                                    }
                                    all.push_back(lengths);
                                    Debug("%d midline samples for %S", lengths.size(), &fish->identity().raw_name());
                                }
                            }
                        }
                        
                        PD(midline_histogram).set_data(all);
                    }
                }
                
                for(auto it = PD(cache)._fish_map.cbegin(); it != PD(cache)._fish_map.cend();) {
                    if(it->second->idx() != frameNr) {
                        it->first->unregister_delete_callback(it->second.get());
                        PD(cache)._fish_map.erase(it++);
                    } else
                        it++;
                }
            }
            
            delete container;
            
            
            if(!PD(cache).connectivity_matrix.empty()) {
                base.section("connectivity", [frameIndex = frameNr](DrawStructure& base, auto s) {
                    if(PD(cache).connectivity_last_frame == frameIndex && !PD(cache).connectivity_reload) {
                        s->reuse_objects();
                        return;
                    }
                    
                    PD(cache).connectivity_reload = false;
                    PD(cache).connectivity_last_frame = frameIndex;
                    
                    const auto number_fish = FAST_SETTINGS(track_max_individuals);
                    for (uint32_t i=0; i<number_fish; ++i) {
                        if(!PD(cache).individuals.count(Idx_t(i))) {
                            Except("Individuals seem to be named differently than 0-%d. Cannot find %d.", FAST_SETTINGS(track_max_individuals), i);
                            continue;
                        }
                        
                        auto fish0 = PD(cache).individuals.at(Idx_t(i));
                        Vec2 p0(gui::Graph::invalid());
                        
                        if(!fish0->has(frameIndex)) {
                            auto c = PD(cache).processed_frame.cached(fish0->identity().ID());
                            if(c)
                                p0 = c->estimated_px;
                        } else
                            p0 = fish0->centroid_weighted(frameIndex)->pos(Units::PX_AND_SECONDS);
                        
                        if(Graph::is_invalid(p0.x))
                            continue;
                        
                        for(uint32_t j=i+1; j<number_fish; ++j) {
                            if(!PD(cache).individuals.count(Idx_t(j))) {
                                Except("Individuals seem to be named differently than 0-%d. Cannot find %d.", FAST_SETTINGS(track_max_individuals), j);
                                continue;
                            }
                            
                            auto fish1 = PD(cache).individuals.at(Idx_t(j));
                            Vec2 p1(infinity<Float2_t>());
                            
                            if(!fish1->has(frameIndex)) {
                                auto c = PD(cache).processed_frame.cached(fish1->identity().ID());
                                if(c)
                                    p1 = c->estimated_px;
                            } else
                                p1 = fish1->centroid_weighted(frameIndex)->pos(Units::PX_AND_SECONDS);
                            
                            if(Graph::is_invalid(p1.x))
                                continue;
                            
                            auto value = PD(cache).connectivity_matrix.at(FAST_SETTINGS(track_max_individuals) * i + j);
                            
                            base.line(p0, p1, 1 + 5 * value, Viridis::value(value).alpha((value * 0.6) * 255));
                        }
                    }
                });
            }
            
            draw_grid(base);
        });
        
        if(PD(cache).has_selection() && SETTING(gui_show_visualfield_ts)) {
            auto outline = PD(cache).primary_selection()->outline(frameNr);
            if(outline) {
                base.section("visual_field", [&](auto&, Section* s) {
                    s->set_scale(base.scale().reciprocal());
                    VisualField::show_ts(base, frameNr, PD(cache).primary_selection());
                });
            }
        }
        
        if(SETTING(gui_show_graph) && draw_graph) {
            if (PD(cache).has_selection()) {
                size_t i = 0;
                auto window = Frame_t(SETTING(output_frame_window).value<long_t>());
                
                for(auto id : PD(cache).selected) {
                    PD(fish_graphs)[i]->setup_graph(frameNr.get(), Rangel((frameNr - window).get(), (frameNr + window).get()), PD(cache).individuals.at(id), nullptr);
                    PD(fish_graphs)[i]->graph().set_scale(base.scale().reciprocal());
                    PD(fish_graphs)[i]->draw(base);
                    
                    if(++i >= PD(fish_graphs).size())
                        break;
                }
            }
        }
        
        if(SETTING(gui_show_number_individuals)) {
            static Graph individuals_graph(Bounds(50, 100, 500, 300), "#individuals");
            if(individuals_graph.x_range().end == FLT_MAX || individuals_graph.x_range().end != PD(cache).tracked_frames.end.get()) {
                individuals_graph.set_ranges(Rangef(PD(cache).tracked_frames.start.get(), PD(cache).tracked_frames.end.get()), Rangef(0, PD(cache).individuals.size()));
                if(individuals_graph.empty()) {
                    individuals_graph.add_function(Graph::Function("", Graph::Type::DISCRETE, [&](float x) -> float {
                        auto it = PD(cache)._statistics.find(Frame_t(x));
                        if(it != PD(cache)._statistics.end()) {
                            return it->second.number_fish;
                        }
                        return gui::Graph::invalid();
                    }));
                }
                individuals_graph.set_draggable();
            }
            individuals_graph.set_zero(frameNr.get());
            base.wrap_object(individuals_graph);
            individuals_graph.set_scale(base.scale().reciprocal());
        }
        
        if(SETTING(gui_show_uniqueness)) {
            static Graph graph(Bounds(50, 100, 800, 400), "uniqueness");
            static std::mutex mutex;
            static std::map<Frame_t, float> estimated_uniqueness;
            static std::vector<Vec2> uniquenesses;
            static bool running = false;
            
            if(estimated_uniqueness.empty() && Recognition::recognition_enabled()
               && Tracker::instance()->recognition()->has_loaded_weights())
            {
                std::lock_guard<std::mutex> guard(mutex);
                if(!running) {
                    running = true;
                    
                    work().add_queue("generate images", [&]()
                    {
                        auto && [data, images, image_map] = Accumulation::generate_discrimination_data();
                        auto && [u, umap, uq] = Accumulation::calculate_uniqueness(false, images, image_map);
                        
                        std::lock_guard<std::mutex> guard(mutex);
                        estimated_uniqueness = umap;
                        
                        uniquenesses.clear();
                        for(auto && [frame, q] :umap) {
                            uniquenesses.push_back(Vec2(frame.get(), q));
                        }
                        
                        running = false;
                    });
                }
            }
            
            std::lock_guard<std::mutex> guard(mutex);
            if(!estimated_uniqueness.empty()) {
                if(graph.x_range().end == FLT_MAX || graph.x_range().end != PD(cache).tracked_frames.end.get()) {
                    
                    static std::map<long_t, float> smooth_points;
                    long_t L = (long_t)uniquenesses.size();
                    for (long_t i=0; i<L; ++i) {
                        long_t offset = 1;
                        float factor = 0.5;
                        
                        smooth_points[i] = 0;
                        
                        for(; offset < max(1, uniquenesses.size() * 0.15); ++offset) {
                            long_t idx_1 = i-offset;
                            long_t idx1 = i+offset;
                            
                            smooth_points[i] += uniquenesses[idx_1 >= 0 ? idx_1 : 0].y * factor + uniquenesses[idx1 < L ? idx1 : L-1].y * factor;
                            factor *= factor;
                        }
                        
                        smooth_points[i] = (smooth_points[i] + uniquenesses[i].y) * 0.5;
                    }
                    
                    graph.set_ranges(Rangef(PD(cache).tracked_frames.start.get(), PD(cache).tracked_frames.end.get()), Rangef(0, 1));
                    if(graph.empty()) {
                        graph.add_function(Graph::Function("raw", Graph::Type::DISCRETE, [uq = &estimated_uniqueness](float x) -> float {
                            std::lock_guard<std::mutex> guard(mutex);
                            auto it = uq->upper_bound(Frame_t(narrow_cast<long_t>(x)));
                            if(!uq->empty() && it != uq->begin())
                                --it;
                            if(it != uq->end() && it->second <= x) {
                                return it->second;
                            }
                            return gui::Graph::invalid();
                        }, Cyan));
                        /*graph.add_function(Graph::Function("smooth", Graph::Type::DISCRETE, [uq = &smooth_points](float x) -> float {
                            std::lock_guard<std::mutex> guard(mutex);
                            auto it = uq->upper_bound(x);
                            if(!uq->empty() && it != uq->begin())
                                --it;
                            if(it != uq->end() && it->second <= x) {
                                return it->second;
                            }
                            return gui::Graph::invalid();
                        }));*/
                        graph.add_points("", uniquenesses);
                    }
                    graph.set_draggable();
                }
                
                graph.set_zero(frameNr.get());
                base.wrap_object(graph);
                graph.set_scale(base.scale().reciprocal());
            }
        }
        
        ConfirmedCrossings::draw(base, frameNr);
        
        // Draw the fish posture with circles
        if(PD(cache).has_selection()) {
            if(SETTING(gui_show_posture)) {
                draw_posture(base, PD(cache).primary_selection(), frameNr);
            }
        }
        
        if(SETTING(gui_show_dataset) /*&& Recognition::recognition_enabled()*/ && PD(timeline)->visible()) {
            if(!PD(dataset)) {
                PD(dataset) = std::make_shared<DrawDataset>();
                auto screen = screen_dimensions();
                PD(dataset)->set_pos(screen * 0.5 - PD(dataset)->size());
            }
            base.wrap_object(*PD(dataset));
        }
        
        if(SETTING(gui_show_recognition_summary) && Recognition::recognition_enabled()) {
            static RecognitionSummary recognition_summary;
            recognition_summary.update(base);
        }
        
    } else
        PD(cache).updated_tracking();
    
    /*Color clr = Red;
    auto section = PD(gui).find("fishbowl");
    if(section) {
        Vec2 mouse_position = PD(gui).mouse_position();
        mouse_position = (mouse_position - section->pos()).div(section->scale());
        
        if(Tracker::instance()->border().in_recognition_bounds(mouse_position))
            clr = Green;
        base.circle(gui().mouse_position(), 5, clr);
    }*/
    
}

void GUI::selected_setting(long_t index, const std::string& name, Textfield& textfield, Dropdown& settings_dropdown, Layout& layout, DrawStructure& base) {
    Debug("choosing '%S'", &name);
    if(index != -1) {
        //auto name = settings_dropdown.items().at(index);
        auto val = GlobalSettings::get(name);
        if(val.get().is_enum() || val.is_type<bool>()) {
            auto options = val.get().is_enum() ? val.get().enum_values()() : std::vector<std::string>{ "true", "false" };
            auto index = val.get().is_enum() ? val.get().enum_index()() : (val ? 0 : 1);
            
            std::vector<std::shared_ptr<List::Item>> items;
            std::map<std::string, bool> selected_option;
            for(size_t i=0; i<options.size(); ++i) {
                selected_option[options[i]] = i == index;
                items.push_back(std::make_shared<TextItem>(options[i]));
                items.back()->set_selected(i == index);
            }
            
            auto str = Meta::toStr(selected_option);
            Debug("options: %S", &str);
            
            _settings_choice = std::make_shared<List>(Bounds(0, PD(gui).height() / PD(gui).scale().y, 150, textfield.height()), "", items, [&textfield](List*, const List::Item& item){
                Debug("Clicked on item %d", item.ID());
                textfield.set_text(item);
                textfield.enter();
                _settings_choice->set_folded(true);
            });
            
            _settings_choice->set_display_selection(true);
            _settings_choice->set_selected(index, true);
            _settings_choice->set_folded(false);
            _settings_choice->set_foldable(true);
            _settings_choice->set_toggle(false);
            _settings_choice->set_accent_color(Color(80, 80, 80, 200));
            //_settings_choice->set_origin(Vec2(0, 1));
            
        } else {
            _settings_choice = nullptr;
            
            if(val.is_type<std::string>()) {
                textfield.set_text(val.value<std::string>());
            } else if(val.is_type<file::Path>()) {
                textfield.set_text(val.value<file::Path>().str());
            } else
                textfield.set_text(val.get().valueString());
        }
        
        if(!_settings_choice) {
            textfield.set_read_only(GlobalSettings::access_level(name) > AccessLevelType::PUBLIC);
            
            layout.add_child(layout.children().size(), &textfield);
            base.select(&textfield);
        } else {
            _settings_choice->set_pos(textfield.pos());
            layout.add_child(layout.children().size(), _settings_choice.get());
            base.select(_settings_choice.get());
            
            if(contains(layout.children(), &textfield))
                layout.remove_child(&textfield);
        }
        
    } else {
        //! CHEAT CODES
        if(settings_dropdown.text() == "datasetquality") {
            Tracker::LockGuard guard("settings_dropdown.text() datasetquality");
            Tracker::recognition()->dataset_quality()->print_info();
        }
        else if(settings_dropdown.text() == "trainingdata_stats") {
            //TrainingData::print_pointer_stats();
        }
        else if(utils::beginsWith(settings_dropdown.text(), "$ ")) {
            auto code = settings_dropdown.text().substr(2);
            Debug("Code: '%S'", &code);
            code = utils::find_replace(code, "\\n", "\n");
            code = utils::find_replace(code, "\\t", "\t");
            PythonIntegration::async_python_function([code]() -> bool {
                try {
                    PythonIntegration::execute(code);
                } catch(const SoftException& e) {
                    Except("Python runtime exception: '%s'", e.what());
                }
                return true;
            });
        }
        else if(settings_dropdown.text() == "dont panic") {
            if(GlobalSettings::has("panic_button")
               && SETTING(panic_button).value<int>() != 0)
            {
                if(SETTING(panic_button).value<int>() == 1) {
                    SETTING(panic_button) = int(2);
                } else {
                    SETTING(panic_button) = int(0);
                }
            } else
                SETTING(panic_button) = int(1);
        }
        else if(settings_dropdown.text() == "consecutive") {
            Tracker::LockGuard guard("settings_dropdown.text() consecutive");
            auto consec = std::set<Range<Frame_t>>(Tracker::instance()->consecutive().begin(), Tracker::instance()->consecutive().end());
            auto str = Meta::toStr(consec);
            Debug("consecutive frames: %S", &str);
            
        }
        else if(settings_dropdown.text() == "results info") {
            using namespace Output;
            auto filename = TrackingResults::expected_filename();
            Debug("Trying to open results '%S'", &filename.str());
            if(file::Path(filename).exists()) {
                ResultsFormat file(filename, NULL);
                file.start_reading();
                
                if(file.header().version >= ResultsFormat::V_14) {
                    Debug("Settings:\n%S", &file.header().settings);
                } else
                    Except("Cannot load settings from results file < V_14");
            } else
                Except("File '%S' does not exist.", &filename.str());
        }
        else if(settings_dropdown.text() == "free_fish") {
            std::set<long_t> free_fish, inactive;
            for(auto && [fdx, fish] : PD(cache).individuals) {
                if(PD(cache).fish_selected_blobs.find(fdx) == PD(cache).fish_selected_blobs.end() || PD(cache).fish_selected_blobs.at(fdx) == -1) {
                    free_fish.insert(fdx);
                }
                if(PD(cache).active_ids.find(fdx) == PD(cache).active_ids.end())
                    inactive.insert(fdx);
            }
            auto str = Meta::toStr(free_fish);
            Debug("All free fish in frame %d: %S", frame(), &str);
            
            str = Meta::toStr(inactive);
            Debug("All inactive fish: %S", &str);
        }
        else if(settings_dropdown.text() == "print_uniqueness") {
            work().add_queue("discrimination", [](){
                auto && [data, images, map] = Accumulation::generate_discrimination_data();
                auto && [unique, unique_map, up] = Accumulation::calculate_uniqueness(false, images, map);
                auto coverage = data->draw_coverage(unique_map);
                
                auto path = pv::DataLocation::parse("output", "uniqueness"+(std::string)video_source()->filename().filename()+".png");
                
                Debug("Uniqueness: %f (output to '%S')", unique, &path.str());
                cv::imwrite(path.str(), coverage->get());
            });
        }
        else if(settings_dropdown.text() == "print_memory") {
            mem::IndividualMemoryStats overall;
            for(auto && [fdx, fish] : PD(cache).individuals) {
                mem::IndividualMemoryStats stats(fish);
                stats.print();
                overall += stats;
            }
        
            overall.print();
            
            mem::TrackerMemoryStats stats;
            stats.print();
            
            mem::OutputLibraryMemoryStats ol;
            ol.print();
            
        } else if(settings_dropdown.text() == "heatmap") {
            this->work().add_queue("generating heatmap", [](){
                Tracker::LockGuard guard("settings_dropdown.text() heatmap");
                
                cv::Mat map(PD(video_source).header().resolution.height, PD(video_source).header().resolution.width, CV_8UC4);
                
                const uint32_t width = 30;
                std::vector<double> grid;
                grid.resize(SQR(width + 1));
                Vec2 indexing(ceil(PD(video_source).header().resolution.width / float(width)),
                              ceil(PD(video_source).header().resolution.height / float(width)));
                
                size_t count = 0;
                for(auto && [id, fish] : Tracker::instance()->individuals()) {
                    for(auto && stuff : fish->basic_stuff()) {
                        auto blob = stuff->blob.unpack();
                        for (auto &h : blob->hor_lines()) {
                            for (ushort x = h.x0; x<=h.x1; ++x) {
                                uint32_t index = round(x / indexing.x) + round(h.y / indexing.y) * width;
                                grid.at(index) += 1;
                            }
                        }
                    }
                    
                    ++count;
                    work().set_percent(count / float(Tracker::instance()->individuals().size()));
                }
                
                auto mval = *std::max_element(grid.begin(), grid.end());
                
                for (uint32_t x=0; x<width; x++) {
                    for (uint32_t y=0; y<width; y++) {
                        float val = grid.at(x + y * width) / mval;
                        cv::rectangle(map, Vec2(x, y).mul(indexing), Vec2(width, width).mul(indexing), Viridis::value(val), -1);
                        //cv::rectangle(map, Vec2(x, y).mul(indexing), Vec2(width, width).mul(indexing), cv::Scalar(1));
                        //cv::putText(map, std::to_string(x)+","+std::to_string(y), Vec2(x, y).mul(indexing) + Vec2(10), CV_FONT_HERSHEY_PLAIN, 0.5, gui::White);
                    }
                }
                
                cv::cvtColor(map, map, cv::COLOR_RGBA2BGR);
                resize_image(map, 0.25);
                tf::imshow("heatmap", map);
            });
            
        } else if(settings_dropdown.text() == "pixels") {
            Tracker::LockGuard guard("settings_dropdown.text() pixels");
            Debug("Calculating...");
            
            std::map<std::string, size_t> average_pixels;
            std::map<std::string, size_t> samples;
            PPFrame frame;
            
            for(auto idx = PD(tracker).start_frame() + 1_f; idx <= PD(tracker).end_frame() && idx <= PD(tracker).start_frame() + 10000_f; ++idx)
            {
                if(!PD(tracker).properties(idx))
                    continue;
                
                PD(video_source).read_frame(frame.frame(), idx.get());
                auto active = PD(tracker).active_individuals(idx - 1_f);
                
                {
                    std::lock_guard<std::mutex> guard(_blob_thread_pool_mutex);
                    Tracker::instance()->preprocess_frame(frame, active, &_blob_thread_pool);
                }
                
                for(auto fish : active) {
                    auto loaded_blob = fish->compressed_blob(idx);
                    auto blob = frame.find_bdx(loaded_blob->blob_id());
                    if(loaded_blob && blob) {
                        if(blob->split())
                            continue;
                        
                        auto thresholded = blob->threshold(FAST_SETTINGS(track_threshold), *PD(tracker).background());
                        
                        average_pixels[fish->identity().name()] += thresholded->pixels()->size();
                        samples[fish->identity().name()] ++;
                    }
                }
            }
            
            float sum = 0;
            for(auto && [name, value] : average_pixels) {
                value /= samples.at(name);
                sum += value;
            }
            sum /= float(average_pixels.size());
            
            auto str = Meta::toStr(average_pixels);
            Debug("Average pixels:\n%S\n(overall: %f)", &str, sum);
            
        } else if(settings_dropdown.text() == "time_deltas") {
            Graph graph(Bounds(0, 0, 1024, 400), "time_deltas");
            
            float max_val = 0, min_val = FLT_MAX;
            pv::Frame frame;
            PD(video_source).read_frame(frame, 0);
            
            std::vector<double> values {
                frame.timestamp() / 1000.0 / 1000.0
            };
            for(size_t i = 1; i<PD(video_source).length(); ++i) {
                PD(video_source).read_frame(frame, i);
                auto t = frame.timestamp() / 1000.0 / 1000.0;
                values[i - 1] = t - values[i - 1];
                values.push_back(t);
                
                max_val = max(max_val, values[i-1]);
                min_val = min(min_val, values[i-1]);
                
                if(i % int(PD(video_source).length() * 0.1) == 0) {
                    Debug("%d/%d", i, PD(video_source).length());
                }
            }
            
            graph.add_function(Graph::Function("dt", Graph::Type::DISCRETE, [&](float x) ->float {
                if(x > 0 && x < values.size())
                    return values.at(x);
                return gui::Graph::invalid();
            }, Red, "ms"));
            
            Debug("%f-%f %d", min_val, max_val, values.size());
            graph.set_ranges(Rangef(0, values.size()-1), Rangef(min_val * 0.5, max_val * 1.5));
            
            cv::Mat bg = cv::Mat::zeros(graph.height(), graph.width(), CV_8UC4);
            CVBase cvbase(bg);
            DrawStructure window(graph.width(), graph.height());
            window.wrap_object(graph);
            cvbase.paint(window);
            cvbase.display();
        } else if(settings_dropdown.text() == "blob_info") {
            Debug("Preprocessed frame %d:", PD(cache).frame_idx);
            auto str = Meta::toStr(PD(cache).processed_frame.noise());
            Debug("Filtered out: %S", &str);
            str = Meta::toStr(PD(cache).processed_frame.blobs());
            Debug("Blobs: %S", &str);
        }
        
        layout.remove_child(&textfield);
    }
}

std::string& additional_status_text() {
    static std::string _text = "";
    return _text;
}

void GUI::set_status(const std::string& text) {
    if(!instance())
        return;
    
    std::lock_guard guard(instance()->gui().lock());
    additional_status_text() = text;
}

void GUI::draw_footer(DrawStructure& base) {
    static bool first = true;
    auto && [bg_offset, max_w] = Timeline::timeline_offsets();
    
    static HorizontalLayout status_layout({}, Vec2(), Bounds(10,0,0,0));
    static Text gpu_status("", Vec2(), White, Font(0.7)), python_status("", Vec2(), Red, Font(0.7));
    static Text additional_status("", Vec2(), White, Font(0.7));
    static Text mouse_status("", Vec2(), White.alpha(200), Font(0.7));
    
#define SITEM(NAME) DirectSettingsItem<globals::Cache::Variables:: NAME>
    static List options_dropdown(Bounds(0, 0, 150, 33 + 2), "display", {
        std::make_shared<SITEM(gui_show_blobs)>("blobs"),
        std::make_shared<SITEM(gui_show_paths)>("paths"),
        std::make_shared<SITEM(gui_show_texts)>("texts"),
        std::make_shared<SITEM(gui_show_selections)>("selections"),
        std::make_shared<SITEM(gui_show_inactive_individuals)>("inactive"),
        std::make_shared<SITEM(gui_show_outline)>("outlines"),
        std::make_shared<SITEM(gui_show_midline)>("midlines"),
        std::make_shared<SITEM(gui_show_posture)>("posture"),
        std::make_shared<SITEM(gui_show_heatmap)>("heatmap"),
        std::make_shared<SITEM(gui_show_number_individuals)>("#individuals"),
        std::make_shared<SITEM(gui_show_dataset)>("dataset"),
        std::make_shared<SITEM(gui_show_recognition_summary)>("confusion"),
        std::make_shared<SITEM(gui_show_recognition_bounds)>("recognition"),
        std::make_shared<SITEM(gui_show_visualfield)>("visual field"),
        std::make_shared<SITEM(gui_show_visualfield_ts)>("visual field ts"),
        std::make_shared<SITEM(gui_auto_scale)>("auto zoom"),
        std::make_shared<SITEM(gui_auto_scale_focus_one)>("zoom on selected"),
        std::make_shared<SITEM(gui_show_export_options)>("export options")
    });
#undef SITEM
    
    static Dropdown settings_dropdown(Bounds(0, 0, 200, 33), GlobalSettings::map().keys());
    static Textfield textfield("", Bounds(0, 0, 300, settings_dropdown.height()));
    static Tooltip tooltip(&settings_dropdown, 400);
    
    std::vector<Layout::Ptr> objects = { &options_dropdown, &settings_dropdown};
    static HorizontalLayout layout(objects, Vec2());
    
    auto h = screen_dimensions().height;
    layout.set_pos(Vec2(20, h - 10) - bg_offset / base.scale().x);
    layout.set_scale(1.1f * base.scale().reciprocal());
    
    auto layout_scale = layout.scale().x;
    auto stretch_w = status_layout.global_bounds().pos().x - 20 - textfield.global_bounds().pos().x;
    if(textfield.selected())
        textfield.set_size(Size2(max(300, stretch_w / layout_scale), textfield.height()));
    else
        textfield.set_size(Size2(300, textfield.height()));
    
#ifndef NDEBUG
    static FlowMenu pie( min(_average_image.cols, _average_image.rows) * 0.25f, [](size_t , const std::string& item){
        SETTING(enable_pie_chart) = false;
    });
    
    pie.set_scale(base.scale().reciprocal() * gui::interface_scale());
    
    if(SETTING(enable_pie_chart))
        base.wrap_object(pie);
#endif
    
    if(first) {
        _static_pointers.insert(_static_pointers.end(), {
#ifndef NDEBUG
            &pie,
#endif
            &textfield,
            &options_dropdown,
            &layout,
            &settings_dropdown,
            &tooltip
        });
        
        PD(clicked_background) = [&](const Vec2& pos, bool v, std::string key = "") {
            const std::string chosen = settings_dropdown.selected_id() > -1 ? settings_dropdown.items().at(settings_dropdown.selected_id()).name() : "";
            if (key.empty())
                key = chosen;
            _clicked_blob_id = -1;
            
            bool is_bounds = GlobalSettings::get(key).is_type<std::vector<Bounds>>();
            bool is_vec_of_vec = GlobalSettings::get(key).is_type<std::vector< std::vector<Vec2> >>();
            bool is_vectors = GlobalSettings::get(key).is_type<std::vector<Vec2>>();
            
            PD(selected_setting_type) = is_vectors 
                    ? SelectedSettingType::POINTS 
                        : (is_vec_of_vec ? SelectedSettingType::ARRAY_OF_VECTORS 
                            : (is_bounds ? SelectedSettingType::ARRAY_OF_BOUNDS 
                                : SelectedSettingType::NONE));
            PD(selected_setting_name) = key;
            
            if(PD(selected_setting_type) == SelectedSettingType::NONE && v) {
                if(PD(current_boundary).size() == 1 && PD(current_boundary).front().size() == 2) {
                    static NumericTextfield<double> text(1.0, Bounds(0, 0, 200,30), arange<double>{0, infinity<double>()});
                    text.set_postfix("cm");
                    text.set_fill_color(DarkGray.alpha(50));
                    text.set_text_color(White);
                    
                    derived_ptr<Entangled> e = std::make_shared<Entangled>();
                    e->update([&](Entangled& e) {
                        e.advance_wrap(text);
                    });
                    e->auto_size(Margin{0, 0});
                    
                    auto bound = PD(current_boundary).front();
                    auto S = bound.front();
                    auto E = bound.back();
                    auto D = euclidean_distance(S, E);
                    
                    PD(gui).dialog([D](Dialog::Result r) {
                        if(r == Dialog::OKAY) {
                            try {
                                auto value = Meta::fromStr<float>(text.text());
                                Debug("Value is: %f", value);
                                
                                if(value > 0) {
                                    SETTING(cm_per_pixel) = float(value / D);
                                    
                                    PD(gui).dialog("Successfully set <ref>cm_per_pixel</ref> to <nr>"+Meta::toStr(SETTING(cm_per_pixel).value<float>())+"</nr>.");
                                    
                                    return true;
                                }
                                
                            } catch(const std::exception& e) { }
                            
                            return false;
                        }
                        
                        return true;
                        
                    }, "Please enter the equivalent length in centimeters for the selected distance (<nr>"+Meta::toStr(D)+"</nr>px) below. <ref>cm_per_pixel</ref> will then be recalculated based on the given value, affecting parameters such as <ref>track_max_speed</ref>, and <ref>blob_size_ranges</ref>, and tracking results.", "Calibrate with known length", "Okay", "Abort")->set_custom_element(std::move(e));
                }
            }
            
            if(v) {
                if(is_bounds) {
                    if(!PD(current_boundary).empty() && PD(current_boundary).back().size() >= 3) {
                        Bounds bds(FLT_MAX, FLT_MAX, 0, 0);
                        for(auto &pt : PD(current_boundary).back()) {
                            if(pt.x < bds.x) bds.x = pt.x;
                            if(pt.y < bds.y) bds.y = pt.y;
                            if(pt.x > bds.width) bds.width = pt.x;
                            if(pt.y > bds.height) bds.height = pt.y;
                        }
                        bds << bds.size() - bds.pos();
                        
                        try {
                            auto array = GlobalSettings::get(key).value<std::vector<Bounds>>();
                            
                            // if textfield text has been modified, use that one rather than the actual setting value
                            auto tmp = Meta::toStr(array);
                            if(key == chosen && tmp != textfield.text())
                                array = Meta::fromStr<std::vector<Bounds>>(textfield.text());
                            array.push_back(bds);
                            if(key == chosen)
                                textfield.set_text(Meta::toStr(array));
                            GlobalSettings::get(key) = array;
                            
                        } catch(...) {}
                    }
                    
                } else if(is_vec_of_vec) {
                    if(!PD(current_boundary).empty() && PD(current_boundary).back().size() >= 3) {
                        try {
                            auto array = GlobalSettings::get(key).value<std::vector<std::vector<Vec2>>>();
                            
                            // if textfield text has been modified, use that one rather than the actual setting value
                            auto tmp = Meta::toStr(array);
                            if(key == chosen && tmp != textfield.text())
                                array = Meta::fromStr< std::vector<std::vector<Vec2>>>(textfield.text());
                            
                            array.push_back(PD(current_boundary).back());
                            if(key == chosen)
                                textfield.set_text(Meta::toStr(array));
                            GlobalSettings::get(key) = array;
                            
                        } catch(...) {}
                        
                    } else {
                        Error("Cannot create a convex polygon from %d points.", PD(current_boundary).back().size());
                    }
                } else if(is_vectors) {
                    try {
                        auto array = GlobalSettings::get(key).value<std::vector<Vec2>>();
                        
                        // if textfield text has been modified, use that one rather than the actual setting value
                        auto tmp = Meta::toStr(array);
                        if(key == chosen && tmp != textfield.text())
                            array = Meta::fromStr<std::vector<Vec2>>(textfield.text());
                        
                        for(auto &boundary : PD(current_boundary)) {
                            for(auto &pt : boundary)
                                array.push_back(pt);
                        }
                        if(key == chosen)
                            textfield.set_text(Meta::toStr(array));
                        GlobalSettings::get(key) = array;
                        
                    } catch(...) {}
                    
                } else {
                    
                }
                
                Debug("Selected boundary:");
                for(auto & boundary : PD(current_boundary)) {
                    auto str = Meta::toStr(boundary);
                    Debug("\t%S", &str);
                }
                
                PD(current_boundary).clear();
                
            } else {
#ifdef __APPLE__
                if(!PD(gui).is_key_pressed(Codes::LSystem)) {
#else
                if(!PD(gui).is_key_pressed(Codes::LControl)) {
#endif
                    if(PD(current_boundary).empty())
                        PD(current_boundary) = {{pos}};
                    else
                        PD(current_boundary).clear();
                    
                } else {
                    if(PD(current_boundary).empty())
                        PD(current_boundary).push_back({});
                    
                    if(is_vectors)
                        PD(current_boundary).push_back({pos});
                    else
                        PD(current_boundary).back().push_back(pos);
                }
            }
            
            PD(cache).set_tracking_dirty();
            PD(cache).set_raw_blobs_dirty();
            PD(cache).set_redraw();
        };
        
        options_dropdown.set_toggle(true);
        options_dropdown.set_multi_select(true);
        options_dropdown.set_accent_color(Color(80, 80, 80, 200));
        
        layout.set_origin(Vec2(0, 1));
            
#ifndef NDEBUG
        auto base_idx = pie.add_layer(FlowMenu::Layer("menu", {"view", "load", "save", "identity", "quit"}));
        auto view_idx = pie.add_layer(FlowMenu::Layer("view", {"back", "posture", "ai stuff", "confusion", "outlines", "texts", "paths"}));
        auto save_idx = pie.add_layer(FlowMenu::Layer("save", {"back", "state", "config", "csv", "npz"}));
        auto load_idx = pie.add_layer(FlowMenu::Layer("load", {"back", "state", "network"}));
        
        pie.link(base_idx, "view", view_idx);
        pie.link(base_idx, "save", save_idx);
        pie.link(base_idx, "load", load_idx);
        
        pie.link(view_idx, "back", base_idx);
        pie.link(save_idx, "back", base_idx);
        pie.link(load_idx, "back", base_idx);
#endif
            
        settings_dropdown.on_select([&](long_t index, const std::string& name) {
            this->selected_setting(index, name, textfield, settings_dropdown, layout, base);
        });
        textfield.on_enter([&](){
            try {
                auto key = settings_dropdown.items().at(settings_dropdown.selected_id()).name();
                if(GlobalSettings::access_level(key) == AccessLevelType::PUBLIC) {
                    GlobalSettings::get(key).get().set_value_from_string(textfield.text());
                    if(GlobalSettings::get(key).is_type<Color>())
                        this->selected_setting(settings_dropdown.selected_id(), key, textfield, settings_dropdown, layout, base);
                    if((std::string)key == "auto_apply" || (std::string)key == "auto_train")
                    {
                        SETTING(auto_train_on_startup) = false;
                    }
                    
                } else
                    Error("User cannot write setting '%S' (%s).", &key, GlobalSettings::access_level(key).name());
            } catch(const std::logic_error&) {
                //Except("Cannot set '%S' to value '%S' (invalid).", &settings_dropdown.items().at(settings_dropdown.selected_id()), &textfield.text());
            } catch(const UtilsException&) {
                //Except("Cannot set '%S' to value '%S' (invalid).", &settings_dropdown.items().at(settings_dropdown.selected_id()), &textfield.text());
            }
        });
        
        first = false;
    }
    PD(gui).wrap_object(layout);
    PD(gui).wrap_object(status_layout);
    
    if(settings_dropdown.hovered()) {
        auto name = settings_dropdown.hovered_item().name();
        if(name.empty())
            name = settings_dropdown.selected_item().name();
        if(!name.empty()) {
            auto str = "<h3>"+name+"</h3>";
            auto access = GlobalSettings::access_level(name);
            if(access > AccessLevelType::PUBLIC) {
                str += " <i>("+std::string(access.name());
                if(!GlobalSettings::defaults().has(name))
                    str += ", non-default";
                str += ")</i>\n";
                
            } else if(!GlobalSettings::defaults().has(name))
                str += "<i>(non-default)</i>\n";
            
            auto ref = GlobalSettings::get(name);
            str += "type: " +settings::htmlify(ref.get().type_name()) + "\n";
            if(GlobalSettings::defaults().has(name)) {
                auto ref = GlobalSettings::defaults().operator[](name);
                str += "default: " +settings::htmlify(ref.get().valueString()) + "\n";
            }
            if(GlobalSettings::has_doc(name))
                str += "\n" + settings::htmlify(GlobalSettings::doc(name));
            
            tooltip.set_scale(base.scale().reciprocal());
            tooltip.set_text(str);
            PD(gui).wrap_object(tooltip);
        }
    }
    
    if (Recognition::python_available()) {
        static Timer status_timer;
        static Recognition::Detail::Info last_status;
        auto current_status = PD(tracker).recognition() ? PD(tracker).recognition()->detail().info() : Recognition::Detail::Info();
        if (PythonIntegration::python_initialized() && (last_status != current_status || gpu_status.txt().empty() || status_timer.elapsed() > 1)) {
            last_status = current_status;
            status_timer.reset();

            std::string txt;
            if(PythonIntegration::python_gpu_initialized())
                txt += "["+std::string(PythonIntegration::python_uses_gpu() ? PythonIntegration::python_gpu_name() : "CPU")+"]";

            if (SETTING(recognition_enable)) {
                if (current_status.percent == 1)
                    txt += " finished.";
                else if (current_status.percent > 0 || current_status.added > current_status.processed)
                    txt += " processed " + Meta::toStr(size_t(current_status.percent * 100)) + "% of known frames" + (current_status.failed_blobs ? (" " + Meta::toStr(current_status.failed_blobs) + " failed blobs") : "");
            }

            //txt += " " + Meta::toStr(PD(cache).tracked_frames.length()) + " " + Meta::toStr(current_status.N) + " " + Meta::toStr(current_status.processed) + " " + Meta::toStr(current_status.added);

            //txt += " " + Meta::toStr(current_status.N / float(PD(cache).tracked_frames.length()));
            //txt += " " + Meta::toStr((float(current_status.processed) / float(current_status.added)));

            static Timer print_timer;
            if (print_timer.elapsed() > 1) {
                if (txt != gpu_status.txt())
                    Debug("%S", &txt);
                print_timer.reset();
            }
            gpu_status.set_txt(txt);
        } else
            gpu_status.set_txt("");

        if (PythonIntegration::python_initializing()) {
            python_status.set_txt("[Python] initializing...");
            python_status.set_color(Yellow);
        }
        else if (PythonIntegration::python_initialized()) {
            python_status.set_txt("[Python " + Meta::toStr(PythonIntegration::python_major_version().load()) + "." + Meta::toStr(PythonIntegration::python_minor_version().load()) + "]");
            python_status.set_color(Green);
        }
        else if (python_status.txt().empty() || (!PythonIntegration::python_init_error().empty() && !PythonIntegration::python_initialized() && !PythonIntegration::python_initializing())) {
            python_status.set_txt("[Python] " + PythonIntegration::python_init_error());
            python_status.set_color(Red);
        } else {
            python_status.set_txt("[Python] Initializes when required.");
            python_status.set_color(White);
        }
    } else {
        python_status.set_txt("[Python] Not available.");
        python_status.set_color(White);
    }
    
    auto section = PD(gui).find("fishbowl");
    if(section) {
        Vec2 mouse_position = PD(gui).mouse_position();
        mouse_position = (mouse_position - section->pos()).div(section->scale());
        mouse_status.set_txt(Meta::toStr(std::vector<int>{static_cast<int>(mouse_position.x), static_cast<int>(mouse_position.y)})+" "+Meta::toStr(PD(cache).display_blobs.size())+" blobs "+Meta::toStr(PD(cache)._current_pixels)+"px"); //"+Meta::toStr(PD(cache)._average_pixels)+"px/blob pp:"+Meta::toStr(PD(cache).processed_frame.num_pixels())+"px");
    }
        
    additional_status.set_txt(additional_status_text());
    
    status_layout.set_origin(Vec2(1, 0.5));
    status_layout.set_scale(1.1 * base.scale().reciprocal());
    
    status_layout.set_pos(Vec2(max_w / base.scale().x - 30, layout.pos().y - layout.local_bounds().height * 0.5) - bg_offset / base.scale().x);

    if(status_layout.children().empty())
        status_layout.set_children({&python_status, &gpu_status, &additional_status, &mouse_status});
    status_layout.set_policy(HorizontalLayout::Policy::CENTER);
}

void GUI::update_recognition_rect() {
    const float max_w = Tracker::average().cols;
    const float max_h = Tracker::average().rows;
    
    if((PD(recognition_image).source()->cols != max_w || PD(recognition_image).source()->rows != max_h) && Tracker::instance()->border().type() != Border::Type::none) {
        auto border_distance = Image::Make(max_h, max_w, 4);
        border_distance->set_to(0);
        
        auto worker = [&border_distance, max_h](ushort x) {
            for (ushort y = 0; y < max_h; ++y) {
                if(Tracker::instance()->border().in_recognition_bounds(Vec2(x, y)))
                    border_distance->set_pixel(x, y, DarkCyan.alpha(15));
            }
        };
        
        {
            Debug("Calculating border...");
            
            std::lock_guard<std::mutex> guard(blob_thread_pool_mutex());
            for(ushort x = 0; x < max_w; ++x) {
                blob_thread_pool().enqueue(worker, x);
            }
            blob_thread_pool().wait();
        }
        
        PD(recognition_image).set_source(std::move(border_distance));
        PD(cache).set_tracking_dirty();
        PD(cache).set_blobs_dirty();
        PD(cache).set_raw_blobs_dirty();
        PD(cache).set_redraw();
    }
    
    if(!FAST_SETTINGS(track_include).empty())
    {
        auto keys = extract_keys(PD(include_shapes));
        
        for(auto &rect : FAST_SETTINGS(track_include)) {
            auto it = PD(include_shapes).find(rect);
            if(it == PD(include_shapes).end()) {
                if(rect.size() == 2) {
                    auto ptr = std::make_shared<Rect>(Bounds(rect[0], rect[1] - rect[0]), Green.alpha(25), Green.alpha(100));
                    //ptr->set_clickable(true);
                    PD(include_shapes)[rect] = ptr;
                    
                } else if(rect.size() > 2) {
                    //auto r = std::make_shared<std::vector<Vec2>>(rect);
                    auto r = poly_convex_hull(&rect); // force a convex polygon for these shapes, as thats the only thing that the in/out polygon test works with
                    auto ptr = std::make_shared<gui::Polygon>(r);
                    ptr->set_fill_clr(Green.alpha(25));
                    ptr->set_border_clr(Green.alpha(100));
                    //ptr->set_clickable(true);
                    PD(include_shapes)[rect] = ptr;
                }
            }
            keys.erase(rect);
        }
        
        for(auto &key : keys) {
            PD(include_shapes).erase(key);
        }
        
        PD(cache).set_raw_blobs_dirty();
        
    } else if(FAST_SETTINGS(track_include).empty() && !PD(include_shapes).empty()) {
        PD(include_shapes).clear();
        PD(cache).set_raw_blobs_dirty();
    }
    
    if(!FAST_SETTINGS(track_ignore).empty())
    {
        auto keys = extract_keys(PD(ignore_shapes));
        
        for(auto &rect : FAST_SETTINGS(track_ignore)) {
            auto it = PD(ignore_shapes).find(rect);
            if(it == PD(ignore_shapes).end()) {
                if(rect.size() == 2) {
                    auto ptr = std::make_shared<Rect>(Bounds(rect[0], rect[1] - rect[0]), Red.alpha(25), Red.alpha(100));
                    //ptr->set_clickable(true);
                    PD(ignore_shapes)[rect] = ptr;
                    
                } else if(rect.size() > 2) {
                    //auto r = std::make_shared<std::vector<Vec2>>(rect);
                    auto r = poly_convex_hull(&rect); // force convex polygon
                    auto ptr = std::make_shared<gui::Polygon>(r);
                    ptr->set_fill_clr(Red.alpha(25));
                    ptr->set_border_clr(Red.alpha(100));
                    //ptr->set_clickable(true);
                    PD(ignore_shapes)[rect] = ptr;
                }
            }
            keys.erase(rect);
        }
        
        for(auto &key : keys) {
            PD(ignore_shapes).erase(key);
        }
        
        PD(cache).set_raw_blobs_dirty();
        
    } else if(FAST_SETTINGS(track_ignore).empty() && !PD(ignore_shapes).empty()) {
        PD(ignore_shapes).clear();
        PD(cache).set_raw_blobs_dirty();
    }
}

Frame_t GUI::frame() {
    return GUI_SETTINGS(gui_frame);
}

gui::mode_t::Class GUI::mode() const {
    return GUI_SETTINGS(gui_mode);
}

void GUI::update_display_blobs(bool draw_blobs, Section* ) {
    if((PD(cache).raw_blobs_dirty() || PD(cache).display_blobs.size() != PD(cache).raw_blobs.size()) && draw_blobs)
    {
        static std::mutex vector_mutex;
        auto screen_bounds = Bounds(Vec2(), screen_dimensions());
        //auto copy = PD(cache).display_blobs;
        size_t gpixels = 0;
        double gaverage_pixels = 0, gsamples = 0;
        
        distribute_vector([&](auto, auto start, auto end, auto){
            std::unordered_map<pv::Blob*, SimpleBlob*> map;
            //std::vector<std::unique_ptr<gui::ExternalImage>> vector;
            
            const bool gui_show_only_unassigned = SETTING(gui_show_only_unassigned).value<bool>();
            size_t pixels = 0;
            double average_pixels = 0, samples = 0;
            
            for(auto it = start; it != end; ++it) {
                if(!*it)
                    continue;
                
                //bool found = copy.count((*it)->blob.get());
                //if(!found) {
                    //auto bds = bowl.transformRect((*it)->blob->bounds());
                    //if(bds.overlaps(screen_bounds))
                    //{
                if(!gui_show_only_unassigned || !contains(PD(cache).active_blobs, (*it)->blob->blob_id())) {
                    (*it)->convert();
                    //vector.push_back((*it)->convert());
                    map[(*it)->blob.get()] = it->get();
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
            PD(cache).display_blobs.insert(map.begin(), map.end());
            //std::move(vector.begin(), vector.end(), std::back_inserter(PD(cache).display_blobs_list));
            //PD(cache).display_blobs_list.insert(PD(cache).display_blobs_list.end(), vector.begin(), vector.end());
            
        }, _blob_thread_pool, PD(cache).raw_blobs.begin(), PD(cache).raw_blobs.end());
        
        PD(cache)._current_pixels = gpixels;
        PD(cache)._average_pixels = gsamples > 0 ? gaverage_pixels / gsamples : 0;
        
        //Debug("Displaying %lu blobs (list), %lu raw, %lu available", PD(cache).display_blobs.size(), PD(cache).raw_blobs.size(), PD(cache).available_blobs_list.size());
    }
}

void GUI::draw_raw(gui::DrawStructure &base, Frame_t) {
    Section* fishbowl;
    
    static auto collection = std::make_unique<ExternalImage>(Image::Make(Tracker::average().rows, Tracker::average().cols, 4), Vec2());
    const auto mode = GUI_SETTINGS(gui_mode);
    const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
    //const double coverage = double(PD(cache)._num_pixels) / double(collection->source()->rows * collection->source()->cols);
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && TREX_METAL_AVAILABLE
    const bool draw_blobs_separately =
    (!GUI_SETTINGS(gui_blur_enabled) || !std::is_same<MetalImpl, default_impl_t>::value || GUI_SETTINGS(gui_mode) != gui::mode_t::blobs) && coverage < 0.002 && draw_blobs;
#else
    const bool draw_blobs_separately = false;//coverage < 0.002 && draw_blobs;
#endif
    bool redraw_blobs = PD(cache).raw_blobs_dirty();
    
    //Debug("Coverage: %f (%d)", coverage, draw_blobs_separately);
    
    base.section("fishbowl", [&](auto &base, Section* section) {
        fishbowl = section;
        bool shift = PD(gui).is_key_pressed(gui::LShift) && (!PD(gui).selected_object() || !dynamic_cast<Textfield*>(PD(gui).selected_object()));
        gui_scale_with_boundary(PD(cache).boundary, section, !shift && (GUI_SETTINGS(gui_auto_scale) || (GUI_SETTINGS(gui_auto_scale_focus_one) && PD(cache).has_selection())));
        
        //if(((PD(cache).mode() == Mode::DEBUG && !PD(cache).blobs_dirty()) || (PD(cache).mode() == Mode::DEFAULT && !PD(cache).is_tracking_dirty()))
        if(!PD(cache).raw_blobs_dirty() && !PD(cache).is_animating(section) //!PD(cache).is_animating(_setting_animation.display.get()))
           //&& !_setting_animation.display
           )
        {
            redraw_blobs = false;
            section->reuse_objects();
            return;
        }
        
        
        if(Recognition::recognition_enabled() && GUI_SETTINGS(gui_show_recognition_bounds)) {
            if(!PD(recognition_image).source()->empty()) {
                base.wrap_object(PD(recognition_image));
            }
            Tracker::instance()->border().draw(base);
        }
        
        if(PD(timeline)->visible()) {
            for(auto && [rect, ptr] : PD(include_shapes)) {
                base.wrap_object(*ptr);
                
                if(ptr->hovered()) {
                    const Font font(0.85 / (1 - ((1 - PD(cache).zoom_level()) * 0.5)), Align::VerticalCenter);
                    
                    base.add_object(new Text("allowing "+Meta::toStr(rect), ptr->pos() + Vec2(5, Base::default_line_spacing(font) + 5), White, font, PD(gui).scale().reciprocal()));
                }
            }
            
            for(auto && [rect, ptr] : PD(ignore_shapes)) {
                base.wrap_object(*ptr);
                
                if(ptr->hovered()) {
                    const Font font(0.85 / (1 - ((1 - PD(cache).zoom_level()) * 0.5)), Align::VerticalCenter);
                    
                    base.add_object(new Text("excluding "+Meta::toStr(rect), ptr->pos() + Vec2(5, Base::default_line_spacing(font) + 5), White, font, PD(gui).scale().reciprocal()));
                }
            }
        }
        
        update_display_blobs(draw_blobs, fishbowl);
        PD(cache).updated_raw_blobs();
        
        if(draw_blobs_separately) {
            if(GUI_SETTINGS(gui_mode) == gui::mode_t::tracking && PD(cache).tracked_frames.contains(frame())) {
                for(auto &&[k,fish] : PD(cache)._fish_map) {
                    fish->shadow(base);
                }
            }
            
            if(GUI_SETTINGS(gui_mode) != gui::mode_t::blobs) {
                /*std::unordered_map<uint32_t, Idx_t> blob_fish;
                for(auto &[fid, bid] : PD(cache).fish_selected_blobs) {
                    bool found = false;
                    for(auto & [b, ptr] : PD(cache).display_blobs) {
                        if(b->blob_id() == bid) {
                            found = true;
                            blob_fish[b->blob_id()] = fid;
                            break;
                        }
                    }
                }*/
                
                for(auto & [b, ptr] : PD(cache).display_blobs) {
                    //if(blob_fish.find(b->blob_id()) == blob_fish.end())
                    {
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && TREX_METAL_AVAILABLE
                        if(GUI_SETTINGS(gui_blur_enabled) && std::is_same<MetalImpl, default_impl_t>::value)
                        {
                            ptr->tag(Effects::blur);
                        }
#endif
#endif
                        base.wrap_object(*(ptr->ptr));
                    }
                }
                
            } else {
                for(auto &[b, ptr] : PD(cache).display_blobs) {
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && TREX_METAL_AVAILABLE
                    if(GUI_SETTINGS(gui_blur_enabled) && std::is_same<MetalImpl, default_impl_t>::value)
                    {
                        e->untag(Effects::blur);
                    }
#endif
#endif
                    base.wrap_object(*(ptr->ptr));
                }
            }
            
        } else if(draw_blobs && GUI_SETTINGS(gui_mode) == gui::mode_t::tracking && PD(cache).tracked_frames.contains(frame())) {
            for(auto &&[k,fish] : PD(cache)._fish_map) {
                fish->shadow(base);
            }
        }
    });
    
    if(!draw_blobs_separately && draw_blobs) {
        if(redraw_blobs) {
            auto mat = collection->source()->get();
            //std::fill((int*)collection->source()->data(), (int*)collection->source()->data() + collection->source()->cols * collection->source()->rows, 0);
            
            distribute_vector([](auto, auto start, auto end, auto) {
                std::fill(start, end, 0);
                
            }, _blob_thread_pool, (int*)collection->source()->data(), (int*)collection->source()->data() + collection->source()->cols * collection->source()->rows);
            
            distribute_vector([&mat](auto, auto start, auto end, auto){
                for(auto it = start; it != end; ++it) {
                    auto& e = *it;
                    auto input = e.second->ptr->source()->get();
                    auto &bounds = e.second->ptr->bounds();
                    if(bounds.x >= 0 && bounds.y >= 0 && bounds.x + bounds.width < mat.cols && bounds.y + bounds.height < mat.rows) {
                        assert(input.channels() == 2);
                        assert(mat.channels() == 4);
                        
                        for (int y = bounds.y; y < bounds.y + bounds.height; ++y) {
                            for (int x = bounds.x; x < bounds.x + bounds.width; ++x) {
                                auto inp = Color(input.template at<cv::Vec2b>(y - bounds.y, x - bounds.x));
                                if(inp.a > 0)
                                    mat.at<cv::Vec4b>(y, x) = inp;
                                //Color::blend(Color(out), Color(input.template at<cv::Vec2b>(y - pos.y, x - pos.x)));
                            }
                        }
                    }
                }
                
            }, _blob_thread_pool, PD(cache).display_blobs.begin(), PD(cache).display_blobs.end());
            
            collection->set_dirty();
        }
        
        collection->set_scale(fishbowl->scale());
        collection->set_pos(fishbowl->pos());
        base.wrap_object(*collection);
    }
    
#ifndef NDEBUG
    if(draw_blobs_separately)
    {
        base.rect(Bounds(0, 0, 100, 100), Red);
    }
#endif

    static std::unique_ptr<Entangled> combine = std::make_unique<Entangled>();
    static std::shared_ptr<Button> button = nullptr;
    static std::shared_ptr<Dropdown> dropdown = nullptr;
    
    base.section("boundary", [&](auto &base, Section*s) {
        if(!PD(current_boundary).empty()) {
            s->set_scale(fishbowl->scale());
            s->set_pos(fishbowl->pos());
            
            const Font font(0.75);
            Vec2 sca = base.scale().reciprocal().mul(s->scale().reciprocal());

            Vec2 top_left(FLT_MAX, FLT_MAX);
            Vec2 bottom_right(0, 0);
            float a = 0;
            
            for(auto &boundary : PD(current_boundary)) {
                if(boundary.size() > 2) {
                    static gui::Polygon polygon(nullptr);
                    
                    //! need to force a convex hull here
                    auto v = poly_convex_hull(&boundary);
                    polygon.set_vertices(*v);
                    polygon.set_border_clr(Cyan.alpha(125));
                    polygon.set_fill_clr(Cyan.alpha(50));
                    base.wrap_object(polygon);
                    
                } else if(boundary.size() == 2) {
                    base.line(boundary[0], boundary[1], 1, Cyan.alpha(125));
                    
                    Vec2 v;
                    if(boundary[1].x > boundary[0].x)
                        v = boundary[1] - boundary[0];
                    else
                        v = boundary[0] - boundary[1];
                    
                    auto D = v.length();
                    v = v.normalize();
                    
                    a = atan2(v);
                    Text *text = new Text(Meta::toStr(D)+" px", Vec2(boundary[1] - boundary[0]) * 0.5 + boundary[0] + v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525), Cyan.alpha(200), font, sca);
                    text->set_rotation(a);
                    text->set_origin(Vec2(0.5));
                    base.add_object(text);
                    
                    text = new Text(Meta::toStr(D * SETTING(cm_per_pixel).value<float>())+" cm", Vec2(boundary[1] - boundary[0]) * 0.5 + boundary[0] - v.perp().mul(sca) * (Base::default_line_spacing(font) * 0.525), Cyan.alpha(200), font, sca);
                    text->set_rotation(a);
                    text->set_origin(Vec2(0.5));
                    base.add_object(text);
                }
                
                Font f = font;
                f.align = Align::Left;
                for(auto &pt : boundary) {
                    base.circle(pt, 5, Cyan.alpha(125))->set_scale(sca);
                    //base.text(Meta::toStr(pt), pt + Vec2(7 * f.size, 0), White.alpha(200), f, sca);
                    
                    if(pt.x < top_left.x) top_left.x = pt.x;
                    if(pt.y < top_left.y) top_left.y = pt.y;
                    if(pt.x > bottom_right.x) bottom_right.x = pt.x;
                    if(pt.y > bottom_right.y) bottom_right.y = pt.y;
                }
            }
            
            if(top_left.x != FLT_MAX) {
                Bounds bds(Vec2((top_left + bottom_right) * 0.5) + Vec2(0, Base::default_line_spacing(Font(0.85)) + 10).mul(sca), Size2(0, 35));
                std::string name = "";
                
                if(PD(selected_setting_type) == SelectedSettingType::NONE) {
                    if(PD(current_boundary).size() == 1 && PD(current_boundary).front().size() == 2)
                        name = "use known length to calibrate";
                    else
                        name = "print vectors";
                    
                } else {
                    if(PD(selected_setting_type) == SelectedSettingType::ARRAY_OF_VECTORS) {
                        if(PD(current_boundary).size() >= 1 && PD(current_boundary).back().size() >= 3)
                            name = "append shape to "+PD(selected_setting_name);
                        else
                            name = "delete invalid shape";
                        
                    } else if(PD(selected_setting_type) == SelectedSettingType::ARRAY_OF_BOUNDS) {
                        if(PD(current_boundary).size() >= 1 && PD(current_boundary).back().size() >= 2)
                            name = "append bounds to "+PD(selected_setting_name);
                        else
                            name = "delete invalid bounds";
                    } else
                        name = "append points to "+PD(selected_setting_name);
                }
                
                auto text_bounds = _base ? _base->text_bounds(name, NULL, Font(0.85)) : Base::default_text_bounds(name, NULL, Font(0.85));
                bds.width = text_bounds.width + 10;
                
                if(!button) {
                    button = std::make_shared<Button>(name, Bounds(Vec2(), bds.size()));
                    button->on_click([](auto){
                        PD(clicked_background)(Vec2(), true, "");
                    });
                    
                } else {
                    button->set_bounds(Bounds(Vec2(), bds.size()));
                    button->set_txt(name);
                }
                
                if(!dropdown) {
                    dropdown = std::make_shared<Dropdown>(Bounds(Vec2(0, button->local_bounds().height), bds.size()), std::vector<std::string>{
                        "track_ignore",
                        "track_include",
                        "recognition_shapes"
                    });
                    dropdown->on_select([](long_t, const Dropdown::TextItem & item){
                        PD(clicked_background)(Vec2(), true, item.name());
                    });
                    dropdown->textfield()->set_placeholder("append to...");
                    
                } else
                    dropdown->set_bounds(Bounds(Vec2(0, button->local_bounds().height), bds.size()));
                
                combine->update([&](auto&e) {
                    if(PD(current_boundary).size() != 1 || PD(current_boundary).front().size() > 2)
                        e.advance_wrap(*dropdown);
                    e.advance_wrap(*button);
                });
                
                combine->set_scale(sca);
                combine->auto_size(Margin{0, 0});
                combine->set_pos(Vec2(top_left.x, top_left.y + (bottom_right.y - top_left.y) * 0.5) - Vec2(20, 0).mul(sca));
                combine->set_origin(Vec2(1, 0));
                //combine->set_z_index(1);
                
                base.wrap_object(*combine);
            }
        }
    });
}

/*std::unique_ptr<ExternalImage> generate_outer(const pv::BlobPtr& blob) {
    Vec2 offset;
    Image::UPtr image, greyscale;
    Vec2 image_pos;
    
    auto &percentiles = PD(cache).pixel_value_percentiles;
    if(PD(cache)._equalize_histograms && !percentiles.empty()) {
        auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), FAST_SETTINGS(track_threshold), percentiles.front(), percentiles.back());
        image_pos = pos;
        greyscale = std::move(img);
    } else {
        auto && [pos, img] = blob->luminance_alpha_image(*Tracker::instance()->background(), FAST_SETTINGS(track_threshold));
        image_pos = pos;
        greyscale = std::move(img);
    }
    
    if(PD(cache)._equalize_histograms && !percentiles.empty()) {
        auto && [pos, img] = blob->equalized_luminance_alpha_image(*Tracker::instance()->background(), 0, percentiles.front(), percentiles.front());
        offset = pos;
        image = std::move(img);
    } else {
        auto && [pos, img] = blob->luminance_alpha_image(*Tracker::instance()->background(), 0);
        offset = pos;
        image = std::move(img);
    }
    
    cv::Mat outer = image->get();
    
    cv::Mat inner;
    if(greyscale->bounds().size() != image->bounds().size())
        ::pad_image(greyscale->get(), inner, image->bounds().size());
    else
        greyscale->get().copyTo(inner);
    
    cv::Mat tmp = outer - inner;
    
    auto gimage = OuterBlobs(Image::Make(tmp), nullptr, offset, blob->blob_id()).convert();
    gimage->add_custom_data("blob_id", (void*)(uint64_t)blob->blob_id());
    return gimage;
}*/

void GUI::draw_raw_mode(DrawStructure &base, Frame_t frameIndex) {
    pv::File *file = PDP(video_source);
    if(file && file->length() > size_t(frameIndex.get())) {
        struct Outer {
            Image::UPtr image;
            Vec2 off;
            pv::BlobPtr blob;
            
            Outer(Image::UPtr&& image = nullptr, const Vec2& off = Vec2(), pv::BlobPtr blob = nullptr)
            : image(std::move(image)), off(off), blob(blob)
            {}
        };
        
        //static std::vector<Outer> outers;
        static std::vector<std::unique_ptr<ExternalImage>> outer_images;
        auto ptr = PD(gui).find("fishbowl");
        Vec2 ptr_scale(1), ptr_pos(0);
        //Transform transform;
        auto dim = screen_dimensions(); // / gui::interface_scale()
        //auto dim = _base ? _base->window_dimensions().div(PD(gui).scale()) : Size2(_average_image);
        Transform transform;
        
        //PD(gui).rect(Bounds(Vec2(10), dim - 20), Transparent, Green);
        
        if(ptr) {
            assert(dynamic_cast<Section*>(ptr));
            ptr_scale = ptr->scale();
            ptr_pos = ptr->pos();
            transform = ptr->global_transform();
        }
        
        static std::unordered_set<pv::bid> shown_ids;
        std::unordered_set<pv::bid> to_show_ids;
        
        for(auto &blob : PD(cache).processed_frame.original_blobs()) {
            auto bounds = transform.transformRect(blob->bounds());
            
            if(!Bounds(100, 100, dim.width-100, dim.height-100).overlaps(bounds))
                continue;
            
            to_show_ids.insert(blob->blob_id());
        }
        
        //PD(gui).text("Showing "+Meta::toStr(to_show_ids), Vec2(10, 100));
        if(PD(cache).blobs_dirty()) {
            shown_ids.clear();
            outer_images.clear();
        }
        
        if(shown_ids != to_show_ids) {
            PD(cache).set_blobs_dirty();
            //std::vector<Outer> outers;
            std::mutex sync;
            std::atomic<size_t> added_items = 0;
            auto copy = shown_ids;
            
            distribute_vector([&added_items, &sync, &copy](auto, auto start, auto end, auto) {
                std::unordered_set<pv::bid> added_ids;
                
                for(auto it = start; it != end; ++it) {
                    if(copy.find(*it) == copy.end()) {
                        //auto& blob = PD(cache).processed_frame.bdx_to_ptr(*it);
                        //auto image = generate_outer(blob);
                        //outer_images.emplace_back(std::move(image));
                        added_ids.insert(*it);
                    }
                }
                
                added_items += added_ids.size();
                
                std::lock_guard guard(sync);
                shown_ids.insert(added_ids.begin(), added_ids.end());
                
            }, _blob_thread_pool, shown_ids.begin(), shown_ids.end());
            
            std::set<pv::bid> deleted;
            for(auto id : shown_ids) {
                if(to_show_ids.find(id) == to_show_ids.end()) {
                    deleted.insert(id);
                    
                    for(auto it = outer_images.begin(); it != outer_images.end(); ++it) {
                        if((uint64_t)(*it)->custom_data("blob_id") == (uint64_t)id) {
                            outer_images.erase(it);
                            break;
                        }
                    }
                }
            }
            
            for(auto id : deleted)
                shown_ids.erase(id);
            
            /*std::vector<std::shared_ptr<OuterBlobs>> outer_simple;
            for(auto &o : outers) {
                outer_simple.push_back(std::make_shared<OuterBlobs>(std::move(o.image), o.off, o.blob->blob_id()));
            }*/
            
            //update_vector_elements(outer_images, outer_simple);
        }
        
        base.section("blob_outers", [&](auto&base, auto s) {
            if(ptr && (PD(cache).is_animating(ptr) || PD(cache).blobs_dirty())) {
                s->set_scale(ptr_scale);
                s->set_pos(ptr_pos);
            }
            
            if(!PD(cache).blobs_dirty()) {
                s->reuse_objects();
                return;
            }
            
            if(!SETTING(gui_show_pixel_grid)) {
                PD(cache).updated_blobs(); // if show_pixel_grid is active, it will set the cache to "updated"
            }
            
            //for(auto &image : outer_images)
            //    base.wrap_object(*image);
            
            //if(_timeline.visible())
            {
                auto bowl = ptr ? ptr->global_transform() : gui::Transform();
                auto screen_bounds = Bounds(Vec2(-50), screen_dimensions() + 100);
                
                constexpr size_t maximum_number_texts = 1000;
                if(PD(cache).processed_frame.blobs().size() >= maximum_number_texts) {
                    Vec2 pos(10, PD(timeline)->bar()->global_bounds().height + PD(timeline)->bar()->global_bounds().y + 10);
                    auto text = "Hiding some blob texts because of too many blobs ("+Meta::toStr(PD(cache).processed_frame.blobs().size())+").";
                    
                    Rect *rect = new Rect(Bounds(pos, Base::text_dimensions(text, s, Font(0.5)) + Vec2(2, 2)), Black.alpha(125));
                    rect->set_scale(base.scale().reciprocal());
                    base.add_object(rect);
                    
                    Text *t = new Text(text, pos + Vec2(2, 2), White, Font(0.5));
                    t->set_scale(base.scale().reciprocal());
                    base.add_object(t);
                }
                
                static std::unordered_map<pv::bid, std::tuple<bool, std::unique_ptr<Circle>, std::unique_ptr<Label>>> _blob_labels;
                static std::vector<decltype(_blob_labels)::mapped_type> _unused_labels;
                
                for(auto & [id, tup] : _blob_labels)
                    std::get<0>(tup) = false;
                
                std::map<pv::Blob*, float> distances;
                std::set<std::tuple<float, pv::BlobPtr, bool>, std::greater<>> draw_order;
                Transform section_transform = s->global_transform();
                auto mp = section_transform.transformPoint(PD(gui).mouse_position());
                
                for (size_t i=0; i<PD(cache).processed_frame.noise().size(); i++) {
                    //if(PD(cache).processed_frame.noise().at(i)->recount(FAST_SETTINGS(track_threshold), *Tracker::instance()->background()) < FAST_SETTINGS(blob_size_ranges).max_range().start * 0.01)
                       // continue;
                    
                    auto id = PD(cache).processed_frame.noise().at(i)->blob_id();
                    auto d = sqdistance(mp, PD(cache).processed_frame.noise().at(i)->bounds().pos());
                    draw_order.insert({d, PD(cache).processed_frame.noise().at(i), false});
                    
                    if(_blob_labels.count(id))
                        std::get<0>(_blob_labels.at(id)) = true;
                }
                
                if(!SETTING(gui_draw_only_filtered_out)) {
                    for (size_t i=0; i<PD(cache).processed_frame.blobs().size(); i++) {
                        auto id = PD(cache).processed_frame.blobs().at(i)->blob_id();
                        auto d = sqdistance(mp, PD(cache).processed_frame.blobs().at(i)->bounds().pos());
                        draw_order.insert({d, PD(cache).processed_frame.blobs().at(i), true});
                        
                        if(_blob_labels.count(id))
                            std::get<0>(_blob_labels.at(id)) = true;
                    }
                }
                
                Vec2 sca = base.scale().reciprocal().mul(s->scale().reciprocal());
                auto mpos = (PD(gui).mouse_position() - ptr_pos).mul(ptr_scale.reciprocal());
                const float max_distance = sqrtf(SQR((_average_image.cols * 0.25) / ptr_scale.x) + SQR((_average_image.rows * 0.25) / ptr_scale.y));
                size_t displayed = 0;
                
                // move unused elements to unused list
                for(auto it = _blob_labels.begin(); it != _blob_labels.end(); ) {
                    if(!std::get<0>(it->second)) {
                        _unused_labels.emplace_back(std::move(it->second));
                        it = _blob_labels.erase(it);
                    } else
                        ++it;
                }
                
                auto cats = FAST_SETTINGS(categories_ordered);
                
                auto draw_blob = [&](Entangled&e, const pv::BlobPtr& blob, float real_size, bool active){
                    if(displayed >= maximum_number_texts && !active)
                        return;
                    
                    if(!bowl.transformRect(blob->bounds()).overlaps(screen_bounds)) {
                        return;
                    }
                    
                    auto d = euclidean_distance(blob->bounds().pos() + blob->bounds().size() * 0.5, mpos);
                    if(d <= max_distance * 2 && d > max_distance) {
                        d = (d - max_distance) / max_distance;
                        d = SQR(d);
                    } else if(d <= max_distance * 0.5 && d > max_distance * 0.1) {
                        d = (d - max_distance * 0.1) / (max_distance * 0.4);
                        d = 1 - SQR(d);
                    }
                    else if(d > max_distance)
                        d = 1;
                    else if(d > max_distance * 0.5)
                        d = 0;
                    else d = 1;
                    
                    std::stringstream ss;
                    if(!active)
                        ss << "<ref>";
                    ss << blob->name() << " ";
                    if (active)
                        ss << "<a>";
                    ss << "size: " << real_size << (blob->split() ? " split" : "");
                    if(blob->tried_to_split())
                        ss << " tried";
                    if (!active)
                        ss << "</ref>";
                    else
                        ss << "</a>";
                    
                    {
                        //auto label = Categorize::DataStore::ranged_label(Frame_t(PD(cache).frame_idx), blob->blob_id());
                        auto it = PD(cache)._ranged_blob_labels.find(blob->blob_id());
                        if(it != PD(cache)._ranged_blob_labels.end()) {
                            ss << " <nr>" << cats.at(it->second) << "</nr>";
                        }
                        /*if(blob->parent_id().valid() && (label = Categorize::DataStore::ranged_label(Frame_t(PD(cache).frame_idx), blob->parent_id()))) {
                            ss << " parent:<str>" << label->name << "</str>";
                        }
                         */
                    }
                    
                    decltype(_blob_labels)::iterator it = _blob_labels.find(blob->blob_id());
                    if(it == _blob_labels.end()) {
                        if(!_unused_labels.empty()) {
                            auto [k, success] = _blob_labels.try_emplace(blob->blob_id(), std::move(_unused_labels.back()));
                            _unused_labels.resize(_unused_labels.size()-1);
                            
                            it = k;
                            std::get<2>(it->second)->set_data(ss.str(), blob->bounds(), blob->center());
                            
                        } else {
                            auto [k, success] = _blob_labels.insert_or_assign(blob->blob_id(), decltype(_blob_labels)::mapped_type{ true, std::make_unique<Circle>(), std::make_unique<Label>(ss.str(), blob->bounds(), blob->center()) });
                            it = k;
                        }
                        
                        //auto & [visited, circ, label] = _blob_labels[blob->blob_id()];
                        auto circ = std::get<1>(it->second).get();
                        circ->set_clickable(true);
                        circ->set_radius(8);
                        //circ->clear_event_handlers();
                        circ->on_click([id = blob->blob_id()](auto) mutable {
                            PD(current_boundary).clear();
                            GUI::instance()->set_clicked_blob_id(id);
                            GUI::instance()->set_clicked_blob_frame(GUI::frame());
                            PD(cache).set_blobs_dirty();
                        });
                    }
                    
                    auto & [visited, circ, label] = it->second;
                    //e.set_scale(sca);
                    
                    if(circ->hovered())
                        circ->set_fill_clr(White.alpha(205 * d));
                    else
                        circ->set_fill_clr(White.alpha(150 * d));
                    circ->set_line_clr(White.alpha(50));
                    circ->set_pos(blob->center());
                    
                    e.advance(new Rect(blob->bounds(), Transparent, White.alpha(100)));
                    e.advance_wrap(*circ);
                    
                    if(d > 0 && real_size > 0) {
                        label->update(ptr, e, d, !active);
                        ++displayed;
                    }
                };
                
                static Entangled _collection;
                _collection.update([&](auto& e) {
                    displayed = 0;
                    for (auto&& [d, blob, active] : draw_order) {
                        draw_blob(e, blob, blob->recount(-1), active);
                    }
                    });

                _collection.set_bounds(average().bounds());
                //_collection.set_scale(Vec2(1));
                //_collection.set_pos(Vec2());
                base.wrap_object(_collection);
                
                _unused_labels.clear();
            }
        });
        
        static pv::bid last_blob_id;
        if(_clicked_blob_id.load().valid() && _clicked_blob_frame == frameIndex) {
            static std::shared_ptr<Entangled> popup;
            static std::shared_ptr<Dropdown> list;
            if(popup == nullptr) {
                popup = std::make_shared<Entangled>();
                list = std::make_shared<Dropdown>(Bounds(0, 0, 200, 35));
                list->on_open([this, list=list.get()](bool opened) {
                    if(!opened) {
                        //list->set_items({});
                        _clicked_blob_id = -1;
                        GUI::set_redraw();
                    }
                });
                list->on_select([this](long_t, auto& item) {
                    pv::bid clicked_blob_id { (uint32_t)int64_t(item.custom()) };
                    if(item.ID() == 0) /* SPLIT */ {
                        auto copy = FAST_SETTINGS(manual_splits);
                        if(!contains(copy[frame().get()], clicked_blob_id)) {
                            copy[frame().get()].insert(clicked_blob_id);
                        }
                        work().add_queue("", [copy](){
                            SETTING(manual_splits) = copy;
                        });
                    } else {
                        auto it = PD(cache).individuals.find(Idx_t(item.ID() - 1));
                        if(it != PD(cache).individuals.end()) {
                            auto fish = it->second;
                            auto id = it->first;
                            
                            for(auto&& [fdx, bdx] : PD(cache).fish_selected_blobs) {
                                if(bdx == clicked_blob_id) {
                                    if(fdx != id) {
                                        if(PD(cache).is_selected(fdx)) {
                                            PD(cache).deselect(fdx);
                                            PD(cache).do_select(id);
                                        }
                                        break;
                                    }
                                }
                            }
                            
                            auto name = fish->identity().name();
                            Debug("Assigning blob %d to fish %S", clicked_blob_id, &name);
                            this->add_manual_match(this->frame(), id, clicked_blob_id);
                            SETTING(gui_mode) = gui::mode_t::tracking;
                        } else
                            Warning("Cannot find individual with ID %d.", item.ID()-1);
                    }
                    
                    _clicked_blob_id = pv::bid::invalid;
                    GUI::set_redraw();
                    PD(cache).set_raw_blobs_dirty();
                });
                //list->set_background(Black.alpha(125), Black.alpha(230));
                //popup->set_size(Size2(200, 400));
            }
            
            Vec2 blob_pos(FLT_MAX);
            bool found = false;
            for(auto &blob : PD(cache).raw_blobs) {
                if(blob->blob->blob_id() == _clicked_blob_id.load()) {
                    blob_pos = blob->blob->bounds().pos() + blob->blob->bounds().size() * 0.5;
                    popup->set_pos(blob_pos.mul(ptr_scale) + ptr_pos);
                    found = true;
                    break;
                }
            }
            
            if(found) {
                std::set<std::tuple<float, Dropdown::TextItem>> items;
                for(auto id : FAST_SETTINGS(manual_identities)) {
                    if(PD(cache).individuals.count(id) && (!PD(cache).fish_selected_blobs.count(id) ||PD(cache).fish_selected_blobs.at(id) != _clicked_blob_id)) {
                        float d = FLT_MAX;
                        auto c = PD(cache).processed_frame.cached(id);
                        if(frameIndex > Tracker::start_frame() && c) {
                            d = (c->estimated_px - blob_pos).length();
                        }
                        items.insert({d, Dropdown::TextItem(PD(cache).individuals.at(id)->identity().name() + (d != FLT_MAX ? (" ("+Meta::toStr(d * FAST_SETTINGS(cm_per_pixel))+"cm)") : ""), id + 1, PD(cache).individuals.at(id)->identity().name(), (void*)uint64_t(_clicked_blob_id.load()))});
                    }
                }
                
                std::vector<Dropdown::TextItem> sorted_items;
                sorted_items.push_back(Dropdown::TextItem("Split", 0, "", (void*)uint64_t(_clicked_blob_id.load())));
                for(auto && [d, item] : items)
                    sorted_items.push_back(item);
                
                list->set_items(sorted_items);
                list->set_clickable(true);
                
                if(_clicked_blob_id.load() != last_blob_id) {
                    list->set_opened(true);
                    list->select_textfield();
                    list->clear_textfield();
                }
                
                popup->set_scale(base.scale().reciprocal());
                popup->auto_size(Margin{0, 0});
                popup->update([&](Entangled &base){
                    base.advance_wrap(*list);
                });
                
                base.wrap_object(*popup);
                
            } else {
                Warning("Cannot find clicked blob id %d.", _clicked_blob_id.load());
                _clicked_blob_id = -1;
            }
            
        } else if(_clicked_blob_id.load().valid())
            _clicked_blob_id = pv::bid::invalid;
        
        last_blob_id = _clicked_blob_id;
        
        if(SETTING(gui_show_pixel_grid)) {
            base.section("collision_model", [&](auto&, auto s) {
                if(ptr && (PD(cache).is_animating(ptr) || PD(cache).blobs_dirty())) {
                    s->set_scale(ptr_scale);
                    s->set_pos(ptr_pos);
                }
                
                if(!PD(cache).blobs_dirty()) {
                    s->reuse_objects();
                    return;
                }
                
                PD(cache).updated_blobs();
                
                std::unordered_map<pv::bid, Color> colors;
                ColorWheel wheel;
                for(auto &b : PD(cache).processed_frame.original_blobs()) {
                    colors[b->blob_id()] = wheel.next().alpha(200);
                }
                
                auto &grid = PD(cache).processed_frame.blob_grid().get_grid();
                for(auto &set : grid) {
                    for(auto &pixel : set) {
                        base.circle(Vec2(pixel.x, pixel.y), 1, Transparent, colors.find(pixel.v) != colors.end() ? colors.at(pixel.v) : Color(255, 0, 255, 255));
                    }
                }
            });
        }
    }
}

void GUI::local_event(const gui::Event &event) {
    if (event.type == gui::KEY) {
        std::unique_lock<std::recursive_mutex> guard(gui().lock());
        key_event(event);
        PD(cache).set_redraw();
        
    } else {
        std::unique_lock<std::recursive_mutex> guard(gui().lock());
        if(event.type == gui::MBUTTON) {
            if(event.mbutton.pressed)
                PD(gui).mouse_down(event.mbutton.button == 0);
            else
                PD(gui).mouse_up(event.mbutton.button == 0);
        }
        else if(event.type == gui::WINDOW_RESIZED) {
            const float interface_scale = gui::interface_scale();
            Size2 size(event.size.width * interface_scale, event.size.height * interface_scale);
            
            float scale = min(size.width / float(_average_image.cols),
                              size.height / float(_average_image.rows));
            
            PD(gui).set_scale(scale);
            
            Vec2 real_size(_average_image.cols * scale,
                           _average_image.rows * scale);
            
            PD(cache).set_tracking_dirty();
            GUI::set_redraw();
        }
        else if(event.type == gui::MMOVE) {
            _frameinfo.mx = PD(gui).mouse_position().x;
            //PD(cache).set_tracking_dirty();
            //PD(cache).set_blobs_dirty();
            PD(cache).set_redraw();
        }
    }
}

void GUI::toggle_fullscreen() {
    if(base()) {
        auto e = _base->toggle_fullscreen(PD(gui));
        this->event(e);
    }
}

void GUI::confirm_terminate() {
    static bool terminate_visible = false;
    if(terminate_visible)
        return;
    
    terminate_visible = true;
    
    work().add_queue("", [ptr = &terminate_visible](){
        std::lock_guard<std::recursive_mutex> lock_guard(PD(gui).lock());
        PD(gui).dialog([ptr = ptr](Dialog::Result result) {
            if(result == Dialog::Result::OKAY) {
                SETTING(terminate) = true;
            }
            
            *ptr = false;
            
        }, "Are you sure you want to quit?", "Terminate application", "Yes", "Cancel");
    });
}

void GUI::update_backups() {
    // every five minutes
    if(PD(gui_last_backup).elapsed() > 60 * 5) {
        start_backup();
        PD(gui_last_backup).reset();
    }
}

void GUI::start_backup() {
    work().add_queue("", [](){
        Debug("Writing backup of settings...");
        GUI::write_config(true, TEXT, ".backup");
    });
}

void GUI::open_docs() {
    std::string filename("https://trex.run/docs");
    Debug("Opening '%S' in browser...", &filename);
#if __linux__
    auto pid = fork();
    if (pid == 0) {
        execl("/usr/bin/xdg-open", "xdg-open", filename.c_str(), (char *)0);
        exit(0);
    }
#elif __APPLE__
    auto pid = fork();
    if (pid == 0) {
        execl("/usr/bin/open", "open", filename.c_str(), (char *)0);
        exit(0);
    }
#else
    ShellExecute(0, 0, filename.c_str(), 0, 0 , SW_SHOW );
#endif
}

void GUI::key_event(const gui::Event &event) {
    auto &key = event.key;
    if(!key.pressed)
        return;
    
    if(key.code >= Codes::Num0 && key.code <= Codes::Num9) {
        std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
        Identity id(Idx_t(narrow_cast<uint32_t>(key.code - Codes::Num0)));
        PD(cache).deselect_all_select(id.ID());
        GUI::set_redraw();
        return;
    }
    
    auto next_crossing = [&](){
        if(ConfirmedCrossings::next(PD(cache)._current_foi)) {
            std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
            
            SETTING(gui_frame) = Frame_t(PD(cache)._current_foi.foi.frames().start - 1_f);
            
            auto &cache = PD(cache);
            if(!cache._current_foi.foi.fdx().empty()) {
                cache.deselect_all();
                for(auto id : cache._current_foi.foi.fdx()) {
                    if(!cache.is_selected(Idx_t(id.id)))
                        cache.do_select(Idx_t(id.id));
                }
            }
        }
    };
    
    switch (key.code) {
        case Codes::F1:
            open_docs();
            break;
#if !defined(__APPLE__)
        case Codes::F11:
            if(_base) {
#else
        case Codes::F:
            if(PD(gui).is_key_pressed(Codes::LSystem) && _base) {
#endif
                toggle_fullscreen();
            }
            break;
        case Codes::LSystem:
        case Codes::RSystem:
            break;
            
        case Codes::Escape:
            confirm_terminate();
            break;
            
        case Codes::Return: {
            if(ConfirmedCrossings::started()) {
                ConfirmedCrossings::set_confirmed();
                next_crossing();
            }
            break;
        }
        
        case Codes::Right: {
            if(!run()) {
                std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
                
                direction_change() = play_direction() != 1;
                if (direction_change()) {
                    PD(last_direction_change).reset();
                    PD(last_increase_timer).reset();
                }
                
                if(PD(last_increase_timer).elapsed() >= 0.15)
                    PD(last_direction_change).reset();
                
                float percent = min(1, PD(last_direction_change).elapsed() / 2.f);
                percent *= percent;
                
                int inc = !direction_change() && PD(last_increase_timer).elapsed() < 0.15 ? ceil(PD(last_increase_timer).elapsed() * max(2, FAST_SETTINGS(frame_rate) * 4) * percent) : 1;
                
                //Debug("%d %f", inc, PD(last_increase_timer).elapsed());
                
                play_direction() = 1;
                
                Frame_t new_frame = min(Frame_t(PD(video_source).length()-1), frame() + Frame_t(inc));
                SETTING(gui_frame) = new_frame;
                
                PD(last_increase_timer).reset();
                
                //Tracker::LockGuard guard;
                //Tracker::find_next_problem(*PD(video_source), frame_ref());
            }
            break;
        }
            
        case Codes::Space: {
            run(!run());
            
            std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
            direction_change() = play_direction() != 1;
            play_direction() = 1;
            
            break;
        }
        
        case Codes::BackSpace: {
            if(ConfirmedCrossings::started()) {
                ConfirmedCrossings::set_wrong();
                next_crossing();
            }
            break;
        }
        
        case Codes::Left: {
            if(!run()) {
                std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
                
                direction_change() = play_direction() != -1;
                if (direction_change()) {
                    PD(last_direction_change).reset();
                    PD(last_increase_timer).reset();
                }
                
                if(PD(last_increase_timer).elapsed() >= 0.15)
                    PD(last_direction_change).reset();
                
                float percent = min(1, PD(last_direction_change).elapsed() / 2.f);
                percent *= percent;
                
                int inc = !direction_change() && PD(last_increase_timer).elapsed() < 0.15 ? ceil(PD(last_increase_timer).elapsed() * max(2, FAST_SETTINGS(frame_rate) * 4) * percent) : 1;
                
                //Debug("%d %f", inc, PD(last_increase_timer).elapsed());
                
                play_direction() = -1;
                
                auto new_frame = max(0_f, frame() - Frame_t(inc));
                if(!frame().valid())
                    new_frame = 0_f;
                SETTING(gui_frame) = new_frame;
                
                PD(last_increase_timer).reset();
                
                //Tracker::LockGuard guard;
                //Tracker::find_next_problem(*PD(video_source), frame_ref());
            }
            
            break;
        }
            
        case Codes::Comma: {
            auto fn = []() {
                if(!PD(analysis).paused())
                    PD(tracker).wait();
                PD(analysis).set_paused(!PD(analysis).paused());
            };
            
            work().add_queue(PD(analysis).paused() ? "Unpausing..." : "Pausing...", fn);
            break;
        }
            
        case Codes::B: {
            // make properties window visible/hidden
            SETTING(gui_show_posture) = !SETTING(gui_show_posture);
            break;
        }
            
        case Codes::C:
            Output::Library::clear_cache();
            break;
            
        case Codes::D:
            set_mode(mode() == gui::mode_t::blobs ? gui::mode_t::tracking : gui::mode_t::blobs);
            GUI::set_redraw();
            break;
            
        case Codes::G: {
            // make graph window visible/hidden
            SETTING(gui_show_graph) = !SETTING(gui_show_graph);
            break;
        }
            
        case Codes::P: {
            std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
            Identity id;
            
            if(PD(cache).has_selection() && !PD(cache).active_ids.empty()) {
                auto it = PD(cache).active_ids.find(PD(cache).selected.front());
                if(it != PD(cache).active_ids.end()) {
                    if(++it == PD(cache).active_ids.end())
                        it = PD(cache).active_ids.begin();
                } else
                    it = PD(cache).active_ids.begin();
                
                id = Identity(*it);
                
            } else if(!PD(cache).active_ids.empty()) {
                id = Identity(*PD(cache).active_ids.begin());
            } else
                break;
            
            PD(cache).deselect_all_select(id.ID());
            break;
        }
            
        case Codes::O: {
            std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
            Identity id;
            
            if(PD(cache).has_selection() && !PD(cache).active_ids.empty()) {
                auto it = PD(cache).active_ids.find(PD(cache).selected.front());
                if(it != PD(cache).active_ids.end()) {
                    if(it == PD(cache).active_ids.begin())
                        it = PD(cache).active_ids.end();
                    --it;
                } else
                    it = --PD(cache).active_ids.end();
                
                id = Identity(*it);
                
            } else if(!PD(cache).active_ids.empty()) {
                id = Identity(*(--PD(cache).active_ids.end()));
            } else
                break;
            
            PD(cache).deselect_all_select(id.ID());
            break;
        }
            
        case Codes::R:
            if(PD(recording))
                stop_recording();
            else
                start_recording();
            
            break;
            
        case Codes::S:
            work().add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { export_tracks(); });
            break;
            
        case Codes::T:
            // make timeline visible/hidden
            PD(timeline)->set_visible(!PD(timeline)->visible());
            break;
        case Codes::H:
        {
            if(PD(cache).has_selection()) {
                auto fish = PD(cache).primary_selection();
                PD(timeline)->prev_poi(fish->identity().ID());
            }
            break;
        }
            
        case Codes::J:
        {
            if(PD(cache).has_selection()) {
                auto fish = PD(cache).primary_selection();
                PD(timeline)->next_poi(fish->identity().ID());
            }
            break;
        }
        case Codes::M:
        {
            std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
            if (ConfirmedCrossings::started()) {
                next_crossing();
                
            } else
                PD(timeline)->next_poi();
            break;
        }
            
        case Codes::N:
        {
            std::lock_guard<std::recursive_mutex> lock(PD(gui).lock());
            if (ConfirmedCrossings::started()) {
                if(ConfirmedCrossings::previous(PD(cache)._current_foi)) {
                    SETTING(gui_frame) = Frame_t(PD(cache)._current_foi.foi.frames().start - 1_f);
                    
                    auto &cache = PD(cache);
                    if(!cache._current_foi.foi.fdx().empty()) {
                        cache.deselect_all();
                        for(auto id : cache._current_foi.foi.fdx()) {
                            if(!cache.is_selected(Idx_t(id.id)))
                                cache.do_select(Idx_t(id.id));
                        }
                    }
                }
                
            } else
                PD(timeline)->prev_poi();
            break;
        }
        case Codes::Z: {
            // save tracking results
            save_state();
            break;
        }
        case Codes::L: {
            // load tracking results
            load_state(GUIType::GRAPHICAL);
            break;
        }
            
        case Codes::K: {
            work().add_queue("", [this](){
                bool before = PD(analysis).is_paused();
                PD(analysis).set_paused(true).get();
                
                /*auto per_frame = Tracker::find_next_problem(*PD(video_source), frame());
                if(per_frame.empty()) {
                    Warning("per_frame is empty.");
                    return;
                }
                
                try {
                    this->generate_training_data(GUIType::GRAPHICAL, false, per_frame);
                } catch(const UtilsException& ex) {
                    Warning("Aborting training data because an exception was thrown.");
                }*/
                
                Tracker::instance()->check_segments_identities(false, [](auto){}, [this](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                    this->work().add_queue(t, fn, b);
                }, frame());
                
                if(!before)
                    SETTING(analysis_paused) = false;
            });
            break;
        }
            
        case Codes::I: {
            // save events
            auto fn = [&]() {
                bool before = PD(analysis).is_paused();
                PD(analysis).set_paused(true).get();
                
                Tracker::LockGuard guard("Codes::I");
                PD(tracker).wait();
                
                Results results(PD(tracker));
                
                file::Path fishdata = pv::DataLocation::parse("output", SETTING(fishdata_dir).value<file::Path>());
                
                if(!fishdata.exists())
                    if(!fishdata.create_folder())
                        U_EXCEPTION("Cannot create folder '%S' for saving fishdata.", &fishdata.str());
                
                try {
                    results.save_events((fishdata / SETTING(filename).value<file::Path>().filename()).str() + "_events", [](float percent) { work().set_percent(percent); });
                } catch(const UtilsException& e) {
                    
                }
                
                //PD(analysis).reset_PD(cache);
                Output::Library::clear_cache();
                if(!before)
                    PD(analysis).set_paused(false).get();
            };
            
            work().add_queue("Saving events...", fn);
            
            break;
        }
            
        case Codes::LShift:
        case Codes::RShift:
            break;
            
        default:
#ifndef NDEBUG
            if(key.code != -1)
                Warning("Unknown key code %d.", key.code);
#endif
            break;
    }
    
    GUI::set_redraw();
}

void GUI::auto_correct(GUI::GUIType type, bool force_correct) {
    //work().add_queue("checking identities...", [this](){
    if(!Tracker::instance())
        return;
    if(!Recognition::recognition_enabled()) {
        Warning("No identity network loaded and training internally is disabled. Restart with -use_network true");
        return;
    }
    
    if(type == GUIType::GRAPHICAL) {
        PD(gui).dialog([this](gui::Dialog::Result r) {
            this->work().add_queue("checking identities...", [this, r](){
                Tracker::instance()->check_segments_identities(r == Dialog::OKAY, [](float x) { work().set_percent(x); }, [this](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                    this->work().add_queue(t, fn, b);
                });
                
                std::lock_guard<std::recursive_mutex> lock_guard(PD(gui).lock());
                PD(cache).recognition_updated = false;
                PD(cache).set_tracking_dirty();
            });
            
        }, "Do you wish to overwrite <key>manual_matches</key> and reanalyse the video from the beginning with automatic corrections enabled? You will probably want to click <b>Yes</b> after training the visual identification network.\n<b>No</b> will only generate averages and does not change any tracked trajectories.", "Auto-correct", "Yes", "No");
    } else {
        this->work().add_queue("checking identities...", [this, force_correct](){
            Tracker::instance()->check_segments_identities(force_correct, [](float x) { work().set_percent(x); }, [this](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                this->work().add_queue(t, fn, b);
            });
            PD(cache).recognition_updated = false;
            PD(cache).set_tracking_dirty();
            
            if(!force_correct)
                Debug("Automatic correct has not been performed (only averages have been calculated). In order to do so, add the keyword 'force' after the command.");
        });
    }
    //});
}

void GUI::save_state(GUI::GUIType type, bool force_overwrite) {
    std::shared_ptr<file::Path> file = std::make_shared<file::Path>(Output::TrackingResults::expected_filename());
    static bool save_state_visible = false;
    if(save_state_visible)
        return;
    
    save_state_visible = true;
    
    auto fn = [file, ptr = &save_state_visible]() {
        bool before = PD(analysis).is_paused();
        PD(analysis).set_paused(true).get();
        
        Tracker::LockGuard guard("GUI::save_state");
        PD(tracker).wait();
        
        try {
            Output::TrackingResults results(PD(tracker));
            results.save([](const std::string& title, float x, const std::string& description){ work().set_progress(title, x, description); }, *file);
        } catch(const UtilsException&e) {
            work().add_queue("", [e](){
                GUI::instance()->gui().dialog([](Dialog::Result){}, "Something went wrong saving the program state. Maybe no write permissions? Check out this message, too:\n<i>"+std::string(e.what())+"</i>", "Error");
            });
            
            Except("Something went wrong saving program state. Maybe no write permissions?"); }
        
        if(!before)
            PD(analysis).set_paused(false).get();
        
        *ptr = false;
    };
    
    if(file->exists() && !force_overwrite) {
        if(type != GUIType::GRAPHICAL) {
            Debug("The file '%S' already exists. To overwrite this setting, add the keyword 'force'.", &file->str());
            save_state_visible = false;
        } else {
            this->work().add_queue("", [file, fn, ptr = &save_state_visible](){
                PD(gui).dialog([file, fn, ptr = ptr](Dialog::Result result) {
                    if(result == Dialog::Result::OKAY) {
                        work().add_queue("Saving results...", fn);
                    } else if(result == Dialog::Result::SECOND) {
                        do {
                            if(file->remove_filename().empty()) {
                                *file = file::Path("backup_" + file->str());
                            } else
                                *file = file->remove_filename() / ("backup_" + (std::string)file->filename());
                        } while(file->exists());
                        
                        auto expected = Output::TrackingResults::expected_filename();
                        if(expected.move_to(*file)) {
                            *file = expected;
                            work().add_queue("Saving backup...", fn);
                        //if(std::rename(expected.str().c_str(), file->str().c_str()) == 0) {
//                          *file = expected;
//                            work().add_queue("Saving backup...", fn);
                        } else {
                            Except("Cannot rename '%S' to '%S'.", &expected.str(), &file->str());
                            *ptr = false;
                        }
                    } else
                        *ptr = false;
                    
                }, "Overwrite tracking previous results at <i>"+file->str()+"</i>?", "Overwrite", "Yes", "Cancel", "Backup old one");
            });
        }
        
    } else
        work().add_queue("Saving results...", fn);
}

void GUI::auto_quit() {
    Warning("Saving and quitting...");
                        
    std::lock_guard<std::recursive_mutex> lock(instance()->gui().lock());
    Tracker::LockGuard guard("saving and quitting");
    PD(cache).deselect_all();
    instance()->write_config(true);
    
    if(!SETTING(auto_no_results)) {
        Output::TrackingResults results(PD(tracker));
        results.save();
    } else {
        file::Path path = Output::TrackingResults::expected_filename();
        path = path.add_extension("meta");
        
        Debug("Writing '%S' meta file instead of .results", &path.str());
        
        auto f = fopen(path.str().c_str(), "wb");
        if(f) {
            auto str = SETTING(cmd_line).value<std::string>()+"\n";
            fwrite(str.data(), sizeof(uchar), str.length(), f);
            fclose(f);
        } else
            Warning("Cannot write '%S' meta file.", &path.str());
    }
    
    try {
        instance()->export_tracks();
    } catch(const UtilsException&) {
        SETTING(error_terminate) = true;
    }
    
    SETTING(auto_quit) = false;
    if(!SETTING(terminate))
        SETTING(terminate) = true;
}
    
void GUI::auto_categorize() {
    Categorize::Work::set_state(Categorize::Work::State::LOAD);
    Categorize::Work::set_state(Categorize::Work::State::APPLY);
}

void GUI::auto_train() {
    SETTING(auto_train) = false;
    if(!instance())
        return;
    
    auto rec = Tracker::recognition();
    if(rec) {
        rec->detail().register_finished_callback([&](){
            Debug("Finished.");
            
            Tracker::recognition()->check_last_prediction_accuracy();
            
            std::lock_guard<std::recursive_mutex> lock(instance()->gui().lock());
            instance()->auto_correct(GUI::GUIType::TEXT, true);
        });
        Debug("Registering finished callback.");
    }
    
    std::lock_guard<std::recursive_mutex> lock(instance()->gui().lock());
    instance()->training_data_dialog(GUI::GUIType::TEXT, false /* retrain */);
}

void GUI::auto_apply() {
    SETTING(auto_apply) = false;
    if(!instance())
        return;
    
    auto rec = Tracker::recognition();
    if(rec) {
        rec->detail().register_finished_callback([&](){
            Debug("Finished.");
            
            Tracker::recognition()->check_last_prediction_accuracy();
            
            std::lock_guard<std::recursive_mutex> lock(instance()->gui().lock());
            instance()->auto_correct(GUI::GUIType::TEXT, true);
        });
    }
    
    std::lock_guard<std::recursive_mutex> lock(instance()->gui().lock());
    instance()->training_data_dialog(GUI::GUIType::TEXT, true);
}

void GUI::load_state(GUI::GUIType type, file::Path from) {
    static bool state_visible = false;
    if(state_visible)
        return;
    
    state_visible = true;
    auto fn = [&, ptr = &state_visible, from = from]() {
        bool before = PD(analysis).is_paused();
        PD(analysis).set_paused(true).get();
        
        Tracker::LockGuard guard("GUI::load_state");
        PD(tracker).wait();
        
        Output::TrackingResults results(PD(tracker));
        
        PD(timeline)->reset_events();
        //PD(analysis).clear();
        
        try {
            results.load([](const std::string& title, float value, const std::string& desc) {
                if(GUI::instance()) {
                    work().set_progress(title, value, desc);
                    //work().set_item_abortable(true);
                }
            }, from);
        } catch(const UtilsException& e) {
            Except("Cannot load results. Crashed with exception: %s", e.what());
            
            if(GUI::instance()) {
                work().add_queue("", [e, from](){
                    GUI::instance()->gui().dialog([](Dialog::Result){}, "Cannot load results from '"+from.str()+"'. Loading crashed with this message:\n<i>"+std::string(e.what())+"</i>", "Error");
                });
            
                auto start = Tracker::start_frame();
                Tracker::instance()->_remove_frames(start);
                removed_frames(start);
            }
        }
        
        //PD(analysis).reset_PD(cache);
        Output::Library::clear_cache();
        
        auto range = PD(tracker).analysis_range();
        bool finished = PD(tracker).end_frame() >= range.end;
        if(finished && SETTING(auto_categorize)) {
            auto_categorize();
        } else if(finished && SETTING(auto_train)) {
            auto_train();
        }
        else if(finished && SETTING(auto_apply)) {
            auto_apply();
        }
        else if(finished && SETTING(auto_quit)) {
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
        
        if(!before || (!finished && SETTING(auto_quit)))
            PD(analysis).set_paused(false).get();
        
        *ptr = false;
    };
    
    if(type == GRAPHICAL) {
        PD(gui).dialog([ptr = &state_visible, fn](Dialog::Result result) {
            if(result == Dialog::Result::OKAY) {
                work().add_queue("Loading results...", fn, PD(video_source).filename().str());
            } else {
                *ptr = false;
            }
            
        }, "Are you sure you want to load results?\nThis will discard any unsaved changes.", "Load results", "Yes", "Cancel");
    } else {
        work().add_queue("Loading results...", fn, PD(video_source).filename().str());
    }
}

void GUI::save_visual_fields() {
    bool before = PD(analysis).is_paused();
    PD(analysis).set_paused(true).get();
    
    Tracker::LockGuard guard("GUI::save_visual_fields");
    PD(tracker).wait();
    
    Individual *selected = PD(cache).primary_selection();
    
    auto fishdata_dir = SETTING(fishdata_dir).value<file::Path>();
    auto fishdata = pv::DataLocation::parse("output", fishdata_dir);
    if(!fishdata.exists())
        if(!fishdata.create_folder())
            U_EXCEPTION("Cannot create folder '%S' for saving fishdata.", &fishdata.str());
    auto filename = (std::string)SETTING(filename).value<file::Path>().filename();
    
    if(selected) {
        auto path = fishdata / (filename + "_visual_field_"+selected->identity().name());
        work().set_progress("generating visual field", 0, path.str());
        selected->save_visual_field(path.str(), Range<Frame_t>({},{}), [](float percent, const std::string& title){ GUI::work().set_progress(title, percent); }, false);
        
    } else {
        std::atomic_size_t counter = 0;
        work().set_percent(0);
        auto &individuals = Tracker::individuals();
        
        auto worker = [&counter, fishdata, filename, &individuals](Individual* fish){
            auto path = fishdata / (filename + "_visual_field_"+fish->identity().name());
            work().set_progress("generating visual fields", -1, path.str());
            
            fish->save_visual_field(path.str(), Range<Frame_t>({},{}), [&](float, const std::string& title){ GUI::work().set_progress(title, (counter + 0) / float(individuals.size())); }, false);
            
            ++counter;
        };
        
        std::lock_guard<std::mutex> guard(blob_thread_pool_mutex());
        for(auto && [id, fish] : Tracker::individuals()) {
            while(blob_thread_pool().queue_length() >= cmn::hardware_concurrency())
                blob_thread_pool().wait_one();
            blob_thread_pool().enqueue(worker, fish);
        }
        blob_thread_pool().wait();
    }
    
    SETTING(analysis_paused) = before;
}

Vec2 GUI::pad_image(cv::Mat& padded, Size2 output_size) {
    Vec2 offset;
    int left = 0, right = 0, top = 0, bottom = 0;
    if(padded.cols < output_size.width) {
        left = roundf(output_size.width - padded.cols);
        right = left / 2;
        left -= right;
    }
    
    if(padded.rows < output_size.height) {
        top = roundf(output_size.height - padded.rows);
        bottom = top / 2;
        top -= bottom;
    }
    
    if(left || right || top || bottom) {
        offset.x -= left;
        offset.y -= top;
        
        cv::copyMakeBorder(padded, padded, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
    }
    
    assert(padded.cols >= output_size.width && padded.rows >= output_size.height);
    if(padded.cols > output_size.width || padded.rows > output_size.height) {
        left = padded.cols - output_size.width;
        right = left / 2;
        left -= right;
        
        top = padded.rows - output_size.height;
        bottom = top / 2;
        top -= bottom;
        
        offset.x += left;
        offset.y += top;
        
        padded(Bounds(left, top, padded.cols - left - right, padded.rows - top - bottom)).copyTo(padded);
    }
    return offset;
}

void GUI::export_tracks(const file::Path& , long_t fdx, Range<Frame_t> range) {
    bool before = GUI::analysis()->is_paused();
    GUI::analysis()->set_paused(true).get();
    
    track::export_data(PD(tracker), fdx, range);
    
    if(!before)
        GUI::analysis()->set_paused(false).get();
}

ConnectedTasks* GUI::analysis() {
    return PDP(analysis);
}

void GUI::set_analysis(ConnectedTasks* ptr) {
    PDP(analysis) = ptr;
}

file::Path GUI::frame_output_dir() {
    return pv::DataLocation::parse("output", file::Path("frames") / (std::string)SETTING(filename).value<file::Path>().filename());
}

std::string GUI::info(bool escape) {
    assert(instance);
    
    auto pv = PDP(video_source);
    auto str = std::string("<h1>File</h1>");
    if(escape)
        str += escape_html(pv->get_info());
    else
        str += pv->get_info_rich_text();
    
    str.append("\n\n<h1>Tracking</h1>");
    //str.append("\n<b>frames where the number of individuals changed</b>: "+std::to_string(PD(tracker).changed_frames().size()-1));
    
    str.append("<b>midline-errors:</b> "+std::to_string(Tracker::overall_midline_errors()));
    str.append("\n<b>max-curvature:</b> "+std::to_string(Outline::max_curvature()));
    str.append("\n<b>average max-curvature:</b> "+std::to_string(Outline::average_curvature()));
    
    auto consec = instance()->frameinfo().global_segment_order.empty() ? Range<Frame_t>({},{}) : instance()->frameinfo().global_segment_order.front();
    std::stringstream number;
    number << consec.start.toStr() << "-" << consec.end.toStr() << " (" << (consec.end - consec.start).toStr() << ")";
    str.append("\n<b>consecutive frames:</b> "+number.str());
    
#if WITH_SFML
    if(instance()->_base) {
        PD(gui).lock().lock();
        str.append("\n<b>GUI stats:</b> obj:"+std::to_string(instance()->_base->last_draw_objects())+" paint:"+std::to_string(instance()->_base->last_draw_repaint()));
        PD(gui).lock().unlock();
    }
#endif
    
    return str;
}

void GUI::write_config(bool overwrite, GUI::GUIType type, const std::string& suffix) {
    auto filename = file::Path(pv::DataLocation::parse("output_settings").str() + suffix);
    auto text = default_config::generate_delta_config();
    
    if(filename.exists() && !overwrite) {
        if(type == GUIType::GRAPHICAL) {
            PD(gui).dialog([str = text, filename](Dialog::Result r) {
                if(r == Dialog::OKAY) {
                    if(!filename.remove_filename().exists())
                        filename.remove_filename().create_folder();
                    
                    FILE *f = fopen(filename.str().c_str(), "wb");
                    if(f) {
                        Warning("Overwriting file '%S'.", &filename.str());
                        fwrite(str.data(), 1, str.length(), f);
                        fclose(f);
                    } else {
                        Except("Dont have write permissions for file '%S'.", &filename.str());
                    }
                }
                
            }, "Overwrite file <i>"+filename/*.filename()*/.str()+"</i> ?", "Write configuration", "Yes", "No");
        } else
            Warning("Settings file '%S' already exists. To overwrite, please add the keyword 'force'.", &filename.str());
        
    } else {
        if(!filename.remove_filename().exists())
            filename.remove_filename().create_folder();
        
        FILE *f = fopen(filename.str().c_str(), "wb");
        if(f) {
            fwrite(text.data(), 1, text.length(), f);
            fclose(f);
            DebugCallback("Saved '%S'.", &filename.str());
        } else {
            Except("Cannot write file '%S'.", &filename.str());
        }
    }
}

void GUI::training_data_dialog(GUIType type, bool force_load, std::function<void()> callback) {
    if(!Recognition::recognition_enabled() || !Recognition::python_available()) {
        auto message = Recognition::python_available() ? "Recognition is not enabled." : "Python is not available. Check your configuration.";
        if(SETTING(auto_train_on_startup))
            U_EXCEPTION(message);
        
        Warning(message);
        return;
    }
    
    if(FAST_SETTINGS(track_max_individuals) == 1) {
        Warning("Are you sure you want to train on only one individual?");
        //callback();
        //return;
    }
    
    this->work().add_queue("initializing python...", [this, type, force_load, callback]()
    {
        auto task = std::async(std::launch::async, [](){
            cmn::set_thread_name("async::ensure_started");
            auto f = PythonIntegration::ensure_started();
            if(!f.get()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                PD(gui).close_dialogs();
                
                std::string text;
#if defined(__APPLE__) && defined(__aarch64__)
                text = "Initializing Python failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow). Since you are using an ARM64 Mac, you may need to install additional libraries.";
#else
                text = "Initializing Python failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow).";
#endif
                
                auto message = text + "Python says: '"+PythonIntegration::python_init_error()+"'.";
                Except(message.c_str());
                
                if(!SETTING(nowindow)) {
#if defined(__APPLE__) && defined(__aarch64__)
                    std::string command = "pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl";
                    
                    text += "\n<i>"+escape_html(PythonIntegration::python_init_error())+"</i>";
                    text += "\n\nYou can run <i>"+command+"</i> automatically in the current environment by clicking the button below.";
                    
                    PD(gui).dialog([command](Dialog::Result r) {
                        if(r == Dialog::ABORT) {
                            // install
                            system(command.c_str());
                        }
                        
                    }, text, "Python initialization failure", "Do nothing", "Install macos-tensorflow");
#else
                    PD(gui).dialog(text, "Error");
#endif
                }
            } else
                Debug("Initialization success.");
        });
        //PythonIntegration::instance();
        
        bool before = PD(analysis).is_paused();
        PD(analysis).set_paused(true).get();
        
        {
            Tracker::LockGuard guard("GUI::training_data_dialog");
            if(Tracker::recognition() && Tracker::recognition()->dataset_quality())
                Tracker::recognition()->dataset_quality()->update(guard);
        }
        
        try {
            generate_training_data(type, force_load);
        } catch(const SoftException& ex) {
            if(SETTING(auto_train_on_startup)) {
                U_EXCEPTION("Aborting training data because an exception was thrown ('%s').", ex.what());
            } else
                Warning("Aborting training data because an exception was thrown ('%s').", ex.what());
        }
        
        if(!before)
            SETTING(analysis_paused) = false;
        
        callback();
    });
}
    
    

void GUI::generate_training_data(GUI::GUIType type, bool force_load) {
    /*-------------------------/
     SAVE METADATA
     -------------------------*/
    
    auto fn = [](TrainingMode::Class load) -> bool {
        std::vector<Rangel> trained;
        
        work().set_progress("training network", 0);
        work().set_item_abortable(true);
        
        try {
            Accumulation acc(load);
            auto ret = acc.start();
            if(ret && SETTING(auto_train_dont_apply)) {
                GUI::auto_quit();
            }
            
            return ret;
            
        } catch(const SoftException& error) {
            if(SETTING(auto_train_on_startup))
                U_EXCEPTION("The training process failed. Please check whether you are in the right python environment and check previous error messages.");
            
            if(!SETTING(nowindow))
                GUI::instance()->gui().dialog("The training process failed. Please check whether you are in the right python environment and check out this error message:\n\n<i>"+escape_html(error.what())+"</i>", "Error");
            Error("The training process failed. Please check whether you are in the right python environment and check previous error messages.");
            return false;
        }
    };
    
    if(Recognition::network_weights_available()) {
        //auto acc = PD(tracker).recognition()->available_weights_accuracy(data);
        //Debug("The prediction accuracy for the selected segment was: %.2f%%", acc * 100);
        
        //float full_random = 1 / FAST_SETTINGS(track_max_individuals);
        //if(acc <= full_random) {
        //    Warning("Calculated accuracy is lower than or equal to completely random assignment (%.2f%%).", full_random * 100);
        //}
        
        if(type == GUIType::GRAPHICAL) {
            PD(gui).dialog([fn](Dialog::Result result){
                work().add_queue("training network", [fn, result](){
                    try {
                        TrainingMode::Class mode;
                        switch(result) {
                            case gui::Dialog::OKAY:
                                mode = TrainingMode::Continue;
                                break;
                            case gui::Dialog::SECOND:
                                mode = TrainingMode::Apply;
                                break;
                            case gui::Dialog::THIRD:
                                mode = TrainingMode::Restart;
                                break;
                            case gui::Dialog::FOURTH:
                                mode = TrainingMode::LoadWeights;
                                break;
                            case gui::Dialog::ABORT:
                                return;
                                
                            default:
                                SOFT_EXCEPTION("Unknown mode %d in generate_training_data.", (int)result);
                                return;
                        }
                        
                        fn(mode);
                        
                    } catch(const SoftException& error) {
                        if(SETTING(auto_train_on_startup))
                            U_EXCEPTION("Initialization of the training process failed. Please check whether you are in the right python environment and check previous error messages.");
                        if(!SETTING(nowindow))
                            GUI::instance()->gui().dialog("Initialization of the training process failed. Please check whether you are in the right python environment and check out this error message:\n\n<i>"+escape_html(error.what())+"<i/>", "Error");
                        Error("Initialization of the training process failed. Please check whether you are in the right python environment and check previous error messages.");
                    }
                });
                
            }, "<b>Weights from a previous training are available.</b>\nData from a previous training session exists. You can either load it and <i>continue</i> training, just load it and <i>apply</i> the network to the whole video, or simply restart training from scratch. If available, you may also load the weights without any further actions.\n\nNone of these options automatically corrects the tracking data. However, you may first review the prospective identity assignments after training and then manually select to <i>auto correct</i> from the menu.", "Training mode", "Continue", "Cancel", "Apply", "Restart", Recognition::network_weights_available() ? "Load weights" : "");
            
        } else {
            auto mode = TrainingMode::Restart;
            if(force_load)
                mode = TrainingMode::Apply;
            if(!fn(mode)) {
                if(SETTING(auto_train_on_startup))
                    U_EXCEPTION("Using the network returned a bad code (false). See previous errors.");
            }
            if(!force_load)
                Warning("Weights will not be loaded. In order to load weights add 'load' keyword after the command.");
        }
        
    } else {
        if(force_load)
            Warning("Cannot load weights, as no previous weights exist.");
        
        work().add_queue("training network", [fn](){
            if(!fn(TrainingMode::Restart)) {
                if(SETTING(auto_train_on_startup))
                    U_EXCEPTION("Using the network returned a bad code (false). See previous errors.");
            }
        });
    }
}

void GUI::generate_training_data_faces(const file::Path& path) {
    Tracker::LockGuard guard("GUI::generate_training_data_faces");
    work().set_item("Generating data...");
    
    auto ranges = frameinfo().global_segment_order;
    auto range = ranges.empty() ? Range<Frame_t>({},{}) : ranges.front();
    
    if(!path.exists()) {
        if(path.create_folder())
            Debug("Created folder '%S'.", &path.str());
        else {
            Except("Cannot create folder '%S'. Check permissions.", &path.str());
            return;
        }
    }
    
    DebugCallback("Generating training dataset [%d-%d] in folder '%S'.", range.start, range.end, &path.str());
    
    PPFrame frame;
    
    std::vector<uchar> images;
    std::vector<float> heads;
    
    std::vector<uchar> unassigned_blobs;
    size_t num_unassigned_blobs = 0;
    
    Size2 output_size(200,200);
    
    if(!FAST_SETTINGS(calculate_posture))
        Warning("Cannot normalize samples if no posture has been calculated.");
    
    size_t num_images = 0;
    
    for(auto i = range.start; i <= range.end; ++i)
    {
        if(!i.valid() || (size_t)i.get() >= PD(video_source).length()) {
            Except("Frame %d out of range.", i);
            continue;
        }
        
        work().set_percent((i - range.start).get() / (float)(range.end - range.start).get());
        
        auto active = i == PD(tracker).start_frame() ? Tracker::set_of_individuals_t() : Tracker::active_individuals(i - 1_f);
        PD(video_source).read_frame(frame.frame(), i.get());
        Tracker::instance()->preprocess_frame(frame, active, NULL);
        
        cv::Mat image, padded, mask;
        for(auto &blob : frame.blobs()) {
            if(!PD(tracker).border().in_recognition_bounds(blob->bounds().pos() + blob->bounds().size() * 0.5)) {
                Debug("Skipping %d@%d because its out of bounds.", blob->blob_id(), i);
                continue;
            }
            
            auto recount = blob->recount(FAST_SETTINGS(track_threshold), *PD(tracker).background());
            if(recount < FAST_SETTINGS(blob_size_ranges).max_range().start) 
            {
                continue;
            }
            
            imageFromLines(blob->hor_lines(), &mask, NULL, &image, blob->pixels().get(), 0, &Tracker::average(), 0);
            
            auto b = blob->bounds();
            b << output_size;
            
            Vec2 offset = (Size2(padded) - Size2(image)) * 0.5;
            
            offset.x = round(offset.x);
            offset.y = round(offset.y);
            
            b << b.pos() - offset;
            
            padded = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
            b.restrict_to(_average_image.bounds());
            
            //_average_image(b).copyTo(padded);//image(dims), mask(dims));
            b = blob->bounds();
            
            b.restrict_to(_average_image.bounds());
            
            Bounds p(blob->bounds());
            p << Size2(mask);
            p << Vec2(offset);
            
            p.restrict_to(Bounds(padded));
            
            auto rest = [](Bounds& p, const Bounds& b){
                if(p.width > b.width) {
                    float o = (p.width - b.width) * 0.5;
                    p.x += round(o);
                    p.width = b.width;
                }
                
                if(p.height > b.height) {
                    float o = (p.height - b.height) * 0.5;
                    p.y += round(o);
                    p.height = b.height;
                }
            };
            
            rest(p, b);
            rest(b, p);
            
            if(image.cols <= output_size.width && image.rows <= output_size.height && image.cols > 0 && image.rows > 0) {
                ::pad_image(image, padded, output_size, -1, false, mask);
                
                if(!padded.isContinuous())
                    U_EXCEPTION("Padded is not continous.");
                
                PhysicalProperties *found_head = NULL;
                
                for(auto fish : active) {
                    auto fish_blob = fish->blob(i);
                    auto head = fish->head(i);
                    
                    if(fish_blob && fish_blob->blob_id() == blob->blob_id() && head) {
                        found_head = head;
                        break;
                    }
                }
                
                if(found_head) {
                    images.insert(images.end(), padded.data, padded.data + padded.cols * padded.rows);
                    
                    cv::circle(padded, found_head->pos(Units::PX_AND_SECONDS) - b.pos() + offset, 2, cv::Scalar(255));
                    tf::imshow("padded", padded);
                    
                    heads.push_back(found_head->pos(Units::PX_AND_SECONDS).x - b.x + offset.x);
                    heads.push_back(found_head->pos(Units::PX_AND_SECONDS).y - b.y + offset.y);
                    ++num_images;
                } else if(num_unassigned_blobs < 1000) {
                    tf::imshow("unlabelled", padded);
                    unassigned_blobs.insert(unassigned_blobs.end(), padded.data, padded.data + padded.cols * padded.rows);
                    ++num_unassigned_blobs;
                }
                
            } else {
                auto prefix = SETTING(individual_prefix).value<std::string>();
                tf::imshow("too big", image);
                Warning("%S image too big (%dx%d)", &prefix, image.cols, image.rows);
            }
        }
    }
    
    /*-------------------------/
     SAVE METADATA
     -------------------------*/
    
    try {
        file::Path npz_path = path / "data.npz";
        cmn::npz_save(npz_path.str(), "range", std::vector<long_t>{ range.start.get(), range.end.get() });
        Debug("Saving %d positions...", num_images);
        cmn::npz_save(npz_path.str(), "positions", heads.data(), {num_images, 2}, "a");
        cmn::npz_save(npz_path.str(), "images", images.data(), {num_images, (size_t)output_size.height, (size_t)output_size.width}, "a");
        /*if(num_unassigned_blobs > 0) {
            Debug("Saving %d unsorted images...", num_unassigned_blobs);
            cmn::npz_save(npz_path.str(), "unsorted_images", unassigned_blobs.data(), {num_unassigned_blobs, (size_t)output_size.height, (size_t)output_size.width}, "a");
        }*/
        
        Debug("Saved %d unsorted and %d sorted images to '%S'.", num_unassigned_blobs, num_images);
    } catch(const std::runtime_error& e) {
        Except("Runtime error while saving to '%S' (%s).", &path.str(), e.what());
    } catch(...) {
        U_EXCEPTION("Unknown error while saving to '%S'", &path.str());
    }
}

void GUI::add_manual_match(Frame_t frameIndex, Idx_t fish_id, pv::bid blob_id) {
    Debug("Requesting change of fish %d to blob %u in frame %d", fish_id, blob_id, frameIndex);
    
    auto matches = FAST_SETTINGS(manual_matches);
    auto &current = matches[frameinfo().frameIndex.load().get()];
    for(auto &it : current) {
        if(it.first != fish_id && it.second == blob_id) {
            current.erase(it.first);
            DebugCallback("Deleting old assignment for blob %d", blob_id);
            break;
        }
    }
    
    current[fish_id] = blob_id;
    SETTING(manual_matches) = matches;
}
