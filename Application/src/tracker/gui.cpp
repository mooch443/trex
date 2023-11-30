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
#include <tracking/DatasetQuality.h>
#include <tracking/VisualField.h>
#include <tracking/Export.h>
#include <tracking/Accumulation.h>
#include <tracking/Categorize.h>
#include <python/GPURecognition.h>

#include <misc/ConnectedTasks.h>
#include <misc/default_settings.h>
#include <misc/Output.h>
#include <misc/MemoryStats.h>
#include <processing/PadImage.h>
#include <misc/Results.h>

#include <gui/DrawBlobView.h>
#include <gui/DrawTrackingView.h>
#include <gui/DrawExportOptions.h>
#include <gui/ScreenRecorder.h>
#include <misc/IdentifiedTag.h>
#include <gui/DrawPreviewImage.h>
#include <gui/AnimatedBackground.h>

#include <pv.h>
#include <file/DataLocation.h>
#include <gui/types/SettingsTooltip.h>
#include <tracking/IndividualManager.h>
#include <grabber/misc/default_config.h>

#include <gui/Coordinates.h>

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
#include "DrawMenu.h"
#include "DrawDataset.h"
#include <gui/RecognitionSummary.h>
#include <gui/DrawFish.h>
#include <gui/Label.h>
#include <gui/InfoCard.h>
#ifndef NDEBUG
#include <gui/FlowMenu.h>
#endif

#include <tracking/PythonWrapper.h>

using namespace gui;
namespace py = Python;

struct PrivateData {
    Frame_t _flow_frame;
    Frame_t _next_frame;
    cv::Mat _cflow, _cflow_next;

    float _tdelta_gui;
    Frame_t _gui_last_frame;
    Timer _gui_last_frame_timer;
    Timer _gui_last_backup;
    
    struct TrackingView {
        Entangled _bowl;
        
        gui::ExternalImage _recognition_image;
        std::map<std::vector<Vec2>, std::shared_ptr<gui::Drawable>> _ignore_shapes, _include_shapes;
        
        gui::Histogram<float, gui::Hist::Options::NORMED> _histogram{ "Event energy", Bounds(200, 450, 800, 300), Filter::FixedBins(40), Display::LimitY(0.45) };
        gui::Histogram<float, gui::Hist::Options::NORMED> _midline_histogram{ "Midline length", Bounds(0, 0, 800, 300), Filter::FixedBins(0.6, 1.4, 50) };
        gui::Histogram<size_t, gui::Hist::Options::NORMED, std::vector<std::map<long_t, size_t>>> _length_histogram{ "Event length", Bounds(1, 800, 800, 300), Filter::FixedBins(0, 100, 50), Display::LimitY(0.45) };
        
    } _tracking;
    
    //! A window that displays export options and allows to change them
    //! by adding/removing from output_graphs
    DrawExportOptions _export_options;

    //! A pointer to the video file (lazy to avoid asking the tracker first?)
    pv::File *_video_source = nullptr;
    
    //! contains blobs in raw_blobs
    std::unique_ptr<ExternalImage> _collection;
    
    //! contains the background image-image and potentially a mask
    std::unique_ptr<ExternalImage> _gui_mask;
    std::unique_ptr<AnimatedBackground> _background;
    
    //! info card shown when an individual is selected
    InfoCard _info_card;
    
    //! For recording the screen
    ScreenRecorder _recorder;
    
    //! The heatmap controller.
    std::unique_ptr<gui::heatmap::HeatmapController> _heatmapController;

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
    gui::DrawStructure* _gui;

    GUICache _cache;
    
    struct Footer {
        Dropdown _settings_dropdown = Dropdown(Box(0, 0, 200, 33), GlobalSettings::map().keys());
        Textfield _value_input = Textfield(Box(0, 0, 300, 33));
        
        Footer()
        {
            auto keys = GlobalSettings::map().keys();
            keys.push_back("$");
            _settings_dropdown.set_items(std::vector<gui::Dropdown::TextItem>(keys.begin(), keys.end()));
        }
    } _footer;

    std::mutex _analyis_mutex;
    ConnectedTasks* _analysis = nullptr;

    Timer _last_frame_change;

    std::queue<std::function<void()>> _tracking_callbacks;
    
    gui::StaticText _info;

    std::function<void(const Vec2&, bool, std::string)> _clicked_background;

    PrivateData(pv::File& video, DrawStructure* gui)
        : _video_source(& video ),
        _info_card([](Frame_t frame) {
            GUI::reanalyse_from(frame);
        }),
        _gui(gui),
        _cache( _gui, _video_source )
    {
        FindCoord::set_video(video.header().resolution);
    }
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
    return PD(recorder).recording();
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
    
    std::string toStr() const {
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

GUI::GUI(DrawStructure* graph, pv::File& video_source, const Image& average, Tracker& tracker)
  :
    _average_image(average),
    _direction_change(false), _play_direction(1),
    _base(NULL),
    _blob_thread_pool(cmn::hardware_concurrency(), "GUI::blob_thread_pool", [](std::exception_ptr e) {
        WorkProgress::add_queue("", [e](){
            try {
                std::rethrow_exception(e);
            } catch(const std::exception& ex) {
                if(GUI::instance())
                    GUI::instance()->gui().dialog("An error occurred in the blob thread pool:\n<i>"+std::string(ex.what())+"</i>", "Error");
                else
                    FormatExcept("An error occurred in the blob thread pool: ", std::string(ex.what()));
            }
        });
    }),
    _properties_visible(false),
    _private_data(new PrivateData{ video_source, graph }),
#if WITH_MHD
    _http_gui(NULL),
#endif
    _info_visible(false)
{
    GUI::_instance = this;

    PDP(tracker) = &tracker;
    PD(posture_window).set_bounds(Bounds(average.cols - 550 - 10, 100, 550, 400));
    PD(gui).set_size(Size2(average.cols, average.rows));

    PD(collection) = std::make_unique<ExternalImage>(Image::Make(average.rows, average.cols, 4), Vec2());
    
    PDP(timeline) = std::make_shared<Timeline>(best_base(), [](bool b) {
            if (!GUI::instance())
                return;
            GUI::instance()->set_info_visible(b);
        }, []() {
            if(GUI::instance()) {
                auto guard = GUI_LOCK(GUI::instance()->gui().lock());
                GUI::instance()->update_recognition_rect();
            }
        }, _frameinfo);

    PD(gui).root().insert_cache(_base, std::make_unique<CacheObject>());
    
    PD(info).set_pos(Vec2(_average_image.cols * 0.5, _average_image.rows * 0.5));
    PD(info).set_max_size(Size2(min(_average_image.cols * 0.75, 700), min(_average_image.rows * 0.75, 700)));
    PD(info).set_origin(Vec2(0.5, 0.5));
    PD(info).set_background(Color(50, 50, 50, 150), Black.alpha(150));


    PD(tracking)._histogram.set_origin(Vec2(0.5, 0.5));
    PD(tracking)._midline_histogram.set_origin(Vec2(0.5, 0.5));
    PD(tracking)._length_histogram.set_origin(Vec2(0.5, 0.5));
    
    for(size_t i=0; i<2; ++i)
        PD(fish_graphs).push_back(new PropertiesGraph(PD(tracker), PD(gui).mouse_position()));
    
    auto changed = [this](std::string_view name) {
        // ignore gui frame
        if(name == "gui_frame") {
            return;
        }
        if(name == "auto_train") {
            print("Changing");
        }
        if(!GUI::instance())
            return;
        
        WorkProgress::add_queue("", [this, name, &value = GlobalSettings::map()[name].get()](){
            if(!GUI::instance())
                return;
            
            auto lock_guard = GUI_LOCK(this->gui().lock());
            
            /*if(name == "track_max_speed") {
                _setting_animation.name = name;
                _setting_animation.display = nullptr;
            }*/
            
            if(is_in(name, "app_name", "output_prefix")) {
                if(_base)
                    _base->set_title(window_title());
            } //else if(name == "gui_run")
                //globals::_settings.gui_run = value.value<bool>();
            //else if(name == "nowindow")
                //globals::_settings.nowindow = value.value<bool>();
            
            if(is_in(name, "output_graphs", "limit", "event_min_peak_offset", "output_normalize_midline_data"))
            {
                Output::Library::clear_cache();
                for(auto &graph : PD(fish_graphs))
                    graph->reset();
                GUI::set_redraw();
            }
            
            if(name == "exec") {
                if(!SETTING(exec).value<file::Path>().empty()) {
                    file::Path settings_file = file::DataLocation::parse("settings", SETTING(exec).value<file::Path>());
                    default_config::execute_settings_file(settings_file, AccessLevelType::PUBLIC);
                    SETTING(exec) = file::Path();
                }
            }
            
            if(name == "gui_connectivity_matrix_file") {
                try {
                    this->load_connectivity_matrix();
                } catch(const UtilsException&) { }
                GUI::set_redraw();
            }
        
            if(is_in(name, "track_threshold", "grid_points", "recognition_shapes", "grid_points_scaling", "recognition_border_shrink_percent", "recognition_border", "recognition_coeff", "recognition_border_size_rescale") && Tracker::instance())
            {
                WorkProgress::add_queue("updating border", [this, name](){
                    if(is_in(name, "recognition_coeff", "recognition_border_shrink_percent", "recognition_border_size_rescale", "recognition_border"))
                    {
                        PD(tracker).border().clear();
                    }
                    PD(tracker).border().update(PD(video_source));
                    
                    {
                        FilterCache::clear();
                        
                        LockGuard guard(w_t{}, "setting_changed_"+std::string(name));
                        auto start = Tracker::start_frame();
                        DatasetQuality::remove_frames(start);
                    }
                    
                    auto lock_guard = GUI_LOCK(this->gui().lock());
                    PD(tracking)._recognition_image.set_source(Image::Make());
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
                "gui_show_detailed_probabilities",
                "visual_field_eye_separation",
                "visual_field_eye_offset",
                "visual_field_history_smoothing"
            };
            
            if(name == "gui_equalize_blob_histograms") {
                PD(cache).set_tracking_dirty();
                PD(cache).set_blobs_dirty();
                PD(cache).set_raw_blobs_dirty();
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
            
            if(name == "gui_background_color" && _base) {
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
                
                if(matches != compare || first_run) {
                    if(first_run)
                        first_run = false;
                    compare = matches;
                    
                    WorkProgress::add_queue("updating with new manual matches...", [matches](){
                        //LockGuard tracker_lock;
                        auto first_change = Tracker::instance()->update_with_manual_matches(matches);
                        reanalyse_from(first_change, true);
                        
                        auto guard = GUI_LOCK(PD(gui).lock());
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
                            if(frame == this->frame()) {
                                PD(cache).last_threshold = -1;
                                PD(cache).set_tracking_dirty();
                            }
                            //reanalyse_from(frame);
                            break;
                        }
                    }
                    
                    old = next;
                    
                } else
                    print("Nothing changed.");
            }
        });
    };
    
    static CallbackCollection _callback;
    _callback = GlobalSettings::map().register_callbacks({
        "app_name",
        "blacklist",
        "event_min_peak_offset",
        "exec",
        "grid_points",
        "grid_points_scaling",
        "gui_auto_scale",
        "gui_auto_scale_focus_one",
        "gui_background_color",
        "gui_connectivity_matrix_file",
        "gui_equalize_blob_histograms",
        "gui_focus_group",
        "gui_foi_name",
        "gui_interface_scale",
        "gui_mode",
        "gui_show_detailed_probabilities",
        "gui_show_midline",
        "gui_show_outline",
        "gui_show_paths",
        "gui_show_posture",
        "gui_show_recognition_bounds",
        "gui_show_recognition_summary",
        "gui_show_selections",
        "gui_show_texts",
        "gui_show_visualfield",
        "gui_show_visualfield_ts",
        "gui_zoom_limit",
        "limit",
        "manual_matches",
        "manual_splits",
        "output_graphs",
        "output_normalize_midline_data",
        "output_prefix",
        "recognition_border",
        "recognition_border_shrink_percent",
        "recognition_border_size_rescale",
        "recognition_coeff",
        "recognition_shapes",
        "track_threshold",
        "visual_field_eye_offset",
        "visual_field_eye_separation",
        "visual_field_history_smoothing",
        "whitelist"
    }, changed);
    
    GlobalSettings::map().register_shutdown_callback([](auto) {
        
    });
    
#if WITH_MHD
    _http_gui = new HttpGui(_gui);
#endif
    
    { // do this in order to trigger calculating pixel percentages
        auto range = Tracker::analysis_range();
        LockGuard guard(ro_t{}, "GUI::update_data(-1)");
        PD(cache).update_data(range.start());
    }
    
    while(!PD(timeline)->update_thread_updated_once()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

#if !COMMONS_NO_PYTHON
    //static bool did_init_map = false;
    if (py::python_available()) {
        track::PythonIntegration::set_display_function([](const std::string& name, const cv::Mat& image)
        {
            WorkProgress::set_image(name, Image::Make(image));
        });
    }
#endif
}

GUI::~GUI() {
    DrawMenu::close();
    
#if WITH_MHD
    if(_http_gui)
        delete _http_gui;
#endif
    
    _private_data->_timeline = nullptr;
    set_base(nullptr);
    
    //! cannot use PD(gui) below because GUI::instance(), which is used in PD, is
    //! not available anymore at this point!
    {
        auto lock = GUI_LOCK(_private_data->_gui->lock());
        GUI::_instance = NULL;
    }
    
    delete _private_data->_work_progress;
        
    {
        auto lock = GUI_LOCK(_private_data->_gui->lock());
        for(auto d : _static_pointers) {
            d->clear_parent_dont_check();
        }
    }
    
    if(_private_data->_recorder.recording()) {
        auto guard = GUI_LOCK(_private_data->_gui->lock());
        _private_data->_recorder.stop_recording(nullptr, nullptr);
    }

    delete _private_data;
}

void GUI::set_base(gui::Base* base) {
    auto guard = GUI_LOCK(PD(gui).lock());
    _base = base;
    if (PDP(timeline))
        PDP(timeline)->set_base(base);
        
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
    auto lock = GUI_LOCK(PD(gui).lock());
    return instance()->_frameinfo;
}

void GUI::run(bool r) {
    if(r != GUI_SETTINGS(gui_run))
        SETTING(gui_run) = r;
}

void GUI::load_connectivity_matrix() {
    print("Updating connectivity matrix...");
    auto path = SETTING(gui_connectivity_matrix_file).value<file::Path>();
    path = file::DataLocation::parse("input", path);
    
    if(not path.exists() || not path.is_regular())
        throw U_EXCEPTION("Cannot find connectivity matrix file ",path.str(),".");
    
    auto contents = utils::read_file(path.str());
    auto rows = utils::split(contents, '\n');
    size_t expected_number = 1 + SQR(FAST_SETTING(track_max_individuals));
    std::map<long_t, std::vector<float>> matrix;
    std::vector<float> array;
    
    float maximum = 0;
    Frame_t::number_t min_frame = std::numeric_limits<Frame_t::number_t>::max(), max_frame = 0;
    for(size_t index = 0; index < rows.size(); ++index) {
        auto values = utils::split(rows[index], ',');
        if(values.size() == expected_number) {
            auto frame = Meta::fromStr<Frame_t::number_t>((std::string)values[0]);
            array.resize(values.size()-1);
            
            for(size_t i=1; i<values.size(); ++i) {
                array[i-1] = cmn::abs(Meta::fromStr<float>((std::string)values[i]));
                if(array[i-1] > maximum)
                    maximum = array[i-1];
            }
            
            matrix[frame] = array;
            
            if(frame < min_frame)
                min_frame = frame;
            if(frame > max_frame)
                max_frame = frame;
            
        } else {
            FormatWarning("Row ",index," doesnt have enough columns (",values.size()," / ",expected_number,"), skipping.");
        }
    }
    
    if(maximum > 0) {
        for(auto && [frame, array] : matrix)
            for(auto &v : array)
                v /= maximum;
    }
    
    print(matrix.size()," frames read (",min_frame,"-",max_frame,")");
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
        if(!GUI_SETTINGS(nowindow) && (/*PD(cache).is_animating() &&*/  redraw_timer.elapsed() >= 0.2)) {
            redraw_timer.reset();
            //set_redraw();
            GUI::gui().set_dirty(base);
            PD(cache).set_raw_blobs_dirty();
            PD(cache).set_blobs_dirty();
            //is_automatic = true;
            
        } else if((!GUI_SETTINGS(nowindow) && redraw_timer.elapsed() >= 0.30) || recording()) {
            redraw_timer.reset();
            //set_redraw();
            GUI::gui().set_dirty(base);
            PD(cache).set_raw_blobs_dirty();
            PD(cache).set_blobs_dirty();
            is_automatic = true;
        }
        
    } else if (image_index.valid() && !recording()) {
        const float frame_rate = 1.f / (float(GUI_SETTINGS(frame_rate)) * GUI_SETTINGS(gui_playback_speed));
        Frame_t inc = Frame_t(sign_cast<Frame_t::number_t>(t / frame_rate));
        bool is_required = false;
        
        if(inc >= 1_f) {
            auto before = image_index;
            if (PD(tracker).start_frame().valid()) {
                image_index = cmn::max(0_f, PD(tracker).start_frame(), min(PD(tracker).end_frame(), image_index + inc));
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
            PD(cache).set_raw_blobs_dirty();
            PD(cache).set_blobs_dirty();
            
            if(!is_required)
                is_automatic = true;
        }
        
        /*if (image_index > PD(tracker).end_frame()) {
            image_index = PD(tracker).end_frame();
        }*/
        
    } else if(!image_index.valid())
        image_index = PD(tracker).start_frame();
    
    if(recording()) {
        //! playback_speed can only make it faster
        if(GUI_SETTINGS(gui_playback_speed) != uint32_t(GUI_SETTINGS(gui_playback_speed))) {
            static std::once_flag flag{};
            std::call_once(flag, [](){
                FormatWarning("Recording only supports integer gui_playback_speed values. Given: ", GUI_SETTINGS(gui_playback_speed));
            });
        }
        const uint32_t frames_per_second = max(1u, (uint32_t)GUI_SETTINGS(gui_playback_speed));
        image_index += Frame_t(frames_per_second);
        
        if (PD(tracker).end_frame().valid() && image_index > PD(tracker).end_frame()) {
            image_index = PD(tracker).end_frame();
            GUI::stop_recording();
        }
    }
    else if(GUI::run() && PD(tracker).end_frame().valid() && image_index >= PD(tracker).end_frame()) {
        GUI::run(false);
    }
    
    const bool animating = PD(cache).is_animating();
    const bool changed = (base && (!GUI::gui().root().cached(base) || GUI::gui().root().cached(base)->changed())) || PD(cache).must_redraw() || status == LoopStatus::UPDATED;

    PD(real_update) = changed && (!is_automatic || GUI::run() || recording() || animating);
    
    if(changed) { //|| PD(last_frame_change).elapsed() < 5.5) {
        if(changed) {
            auto ptr = PD(gui).root().cached(base);
            if(base && !ptr) {
                ptr = PD(gui).root().insert_cache(base, std::make_unique<CacheObject>()).get();
            }
            if(ptr)
                ptr->set_changed(false);
        }
        
        if(GUI::frame() != image_index) {
            SETTING(gui_frame) = image_index;
            GUICache::instance().request_frame_change_to(image_index);
            if(best_base())
                FindCoord::set_screen_size(PD(gui), *best_base());
        }
        
        //print(image_index, " ", animating, " ", GUICache::instance().animators());
        
        //std::vector<std::string> changed_objects_str;
        size_t changed_objects = 0;
        if(!is_automatic && changed) {
            auto o = PD(gui).collect();
            for(auto obj : o) {
                if(obj->type() == Type::SINGLETON) {
                    obj = static_cast<SingletonObject*>(obj)->ptr();
                }
                if(base && obj->cached(base) && obj->cached(base)->changed() && obj->was_visible()) {
                    ++changed_objects;
                    //changed_objects_str.push_back(Meta::toStr(obj->type()) + " / " + obj->name() + " " + Meta::toStr((size_t)obj));
                }
            }
        }
        
        GUI::frameinfo().frameIndex = GUI::frame();
        
        static Timer last_redraw;
        if(!recording())
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
                if(base && obj->cached(base) && obj->cached(base)->changed() && obj->was_visible()) {
                    ++changed_objects;
                    //changed_objects_str.push_back(Meta::toStr(obj->type()) + " / " + obj->name() + " " + Meta::toStr((size_t)obj));
                }
            }
        }
        
        if(changed_objects) {
            PD(last_frame_change).reset();
        }
        
    }
    
    if(recording())
        PD(recorder).set_frame(image_index);
    
    GUI::update_backups();
}

void GUI::do_recording() {
    //PD(recorder).update_recording(instance()->base(), frame(), PD(cache).tracked_frames.end);
}

bool GUI::is_recording() const {
    return recording();
}

void GUI::start_recording() {
    PD(recorder).start_recording(instance()->base(), frame());
}

void GUI::stop_recording() {
    PD(recorder).stop_recording(instance()->base(), &gui());
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
    static std::once_flag flag;
    auto lock = GUI_LOCK(PD(gui).lock());
    
    std::call_once(flag, [&]() {
        gpuMat bg;
        PD(video_source).average().copyTo(bg);
        PD(video_source).processImage(bg, bg, false);
        cv::Mat original;
        bg.copyTo(original);
        
        PD(background) = std::make_unique<AnimatedBackground>(Image::Make(original));
        
        PD(background)->add_event_handler(EventType::MBUTTON, [](Event e){
            if(e.mbutton.pressed) {
                PD(clicked_background)(Vec2(e.mbutton.x, e.mbutton.y).map<round>(), e.mbutton.button == 1, "");
            }
        });
        
        if(PD(video_source).has_mask()) {
            cv::Mat mask = PD(video_source).mask().mul(cv::Scalar(255));
            mask.convertTo(mask, CV_8UC1);
            
            PD(gui_mask) = std::make_unique<ExternalImage>(Image::Make(mask), Vec2(0, 0), Vec2(1), Color(255, 255, 255, 125));
        }
    });
    
    auto alpha = SETTING(gui_background_color).value<Color>().a;
    PD(background)->set_color(Color(255, 255, 255, alpha ? alpha : 1));
    
    if(alpha > 0) {
        PD(gui).wrap_object(*PD(background));
        if(PD(gui_mask)) {
            PD(gui_mask)->set_color(PD(background)->color().alpha(PD(background)->color().a * 0.5));
            PD(gui).wrap_object(*PD(gui_mask));
        }
    }
    
    //const Mode mode = (Mode)VALUE(mode).value<int>();
    auto ptr = PD(gui).find("fishbowl");
    if(ptr && (PD(cache).is_animating() || PD(cache).blobs_dirty() || PD(cache).is_tracking_dirty())) {
        assert(dynamic_cast<Section*>(ptr));
        
        auto pos = static_cast<Section*>(ptr)->pos();
        PD(background)->set_scale(static_cast<Section*>(ptr)->scale());
        PD(background)->set_pos(pos);
        
        if(PD(gui_mask)) {
            PD(gui_mask)->set_scale(PD(background)->scale());
            PD(gui_mask)->set_pos(PD(background)->pos());
        }
    }
    
    instance()->draw(PD(gui));
}

void GUI::draw(DrawStructure &base) {
    const auto mode = GUI_SETTINGS(gui_mode);
    
    if(not PD(gui_last_frame).valid()
       || PD(gui_last_frame) != frame())
    {
        if(PD(gui_last_frame).valid()
           && PD(gui_last_frame) < frame())
            PD(tdelta_gui) = PD(gui_last_frame_timer).elapsed() / (frame() - PD(gui_last_frame)).get();
        else
            PD(tdelta_gui) = 0u;
        PD(gui_last_frame) = frame();
        PD(gui_last_frame_timer).reset();
    }
    
    PD(gui).section("show", [this, mode](DrawStructure &base, auto* section) {
        if (!PD(real_update) && not PD(cache).is_animating()) {
            section->reuse_objects();
            return;
        }

        LockGuard guard(ro_t{}, "show()", 100);

        if (!guard.locked()) {
            section->reuse_objects();

        }
        else {
            auto loaded = PD(cache).update_data(this->frame());
            if(not loaded.valid()) {
                //section->reuse_objects();
            } else {
            }
            
            if(not loaded.valid()
               || loaded == this->frame()) {
                this->draw_raw(base, this->frame());
                PD(cache).set_mode(mode);
                
                if (mode == gui::mode_t::tracking)
                    this->draw_tracking(base, this->frame());
                else if (mode == gui::mode_t::blobs)
                    this->draw_raw_mode(base, this->frame());
                
                PD(cache).updated_blobs();
            } else {
                section->reuse_objects();
            }
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
            PD(info_card).update(base, this->frame());
            base.wrap_object(PD(info_card));
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
            PD(export_options).draw(base);
        draw_menu();
        
        auto& tracking = PD(tracking);
        if(FAST_SETTING(calculate_posture) && GUI_SETTINGS(gui_show_midline_histogram)) {
            PD(tracking)._midline_histogram.set_bounds(Bounds(_average_image.cols * 0.5, _average_image.rows * 0.5, 800, 300));
            tracking._midline_histogram.set_scale(base.scale().reciprocal());
            base.wrap_object(tracking._midline_histogram);
        }
        
        if(FAST_SETTING(calculate_posture) && GUI_SETTINGS(gui_show_histograms)) {
            tracking._histogram.set_scale(base.scale().reciprocal());
            tracking._length_histogram.set_scale(base.scale().reciprocal());
            
            PD(tracking)._histogram.set_bounds(Bounds(_average_image.cols * 0.5, 450, 800, 300));
            PD(tracking)._length_histogram.set_bounds(Bounds(_average_image.cols * 0.5, 800, 800, 300));
            
            Size2 window_size(_average_image.cols, _average_image.rows);
            Vec2 pos = window_size * 0.5 - Vec2(0, (tracking._histogram.global_bounds().height + tracking._length_histogram.global_bounds().height + 10) * 0.5);
            tracking._histogram.set_pos(pos);
            tracking._length_histogram.set_pos(pos + Vec2(0, tracking._histogram.global_bounds().height + 5));
            
            base.wrap_object(tracking._histogram);
            base.wrap_object(tracking._length_histogram);
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
        if(info_timer.elapsed() > 5 || PD(info).text().empty()) {
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
    
    /**
     * -----------------------------
     * DISPLAY LOADING BAR if needed
     * -----------------------------
     */
    base.section("loading", [](DrawStructure& base, auto section) {
        WorkProgress::update((IMGUIBase*)best_base(), base, section, screen_dimensions());
    });
}

void GUI::draw_menu() {
    DrawMenu::draw();
}

void GUI::removed_frames(Frame_t including) {
    auto gguard = GUI_LOCK(gui().lock());
    if(PD(heatmapController))
        PD(heatmapController)->frames_deleted_from(including);
}

void GUI::reanalyse_from(Frame_t frame, bool in_thread) {
    if(!instance() || !GUI::analysis())
        return;
    
    if(not frame.valid())
        frame = 0_f;
    
    auto fn = [gui = instance(), frame](){
        auto before = GUI::analysis()->is_paused();
        if(!before)
            GUI::analysis()->set_paused(true).get();
        
        {
            auto gguard = GUI_LOCK(GUI::gui().lock());
            LockGuard guard(w_t{}, "reanalyse_from");
            
            if(Tracker::end_frame().valid()
               && frame <= Tracker::end_frame())
            {
                Tracker::instance()->_remove_frames(frame);
                gui->removed_frames(frame);
                
                Output::Library::clear_cache();
                PD(timeline)->reset_events(frame);
                
            } else if(Tracker::end_frame().valid()) {
                FormatExcept("The requested frame ", frame," is not part of the video, and certainly beyond end_frame (", Tracker::end_frame(),").");
            }
        }
        
        if(!before)
            GUI::analysis()->set_paused(false).get();
    };
    
    if(in_thread)
        WorkProgress::add_queue("calculating", fn);
    else
        fn();
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
        void convert(const std::unique_ptr<Circle>& circle) const {
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
                });
                circle->on_hover([circle = circle.get()](auto) {
                    circle->set_dirty();
                });
            }
        }
    };
    static std::vector<std::unique_ptr<Circle>> circles;
    std::vector<std::unique_ptr<GridPoint>> points;
    
    for (size_t i=0; i<grid_points.size(); ++i)
        points.emplace_back(std::make_unique<GridPoint>(i, &grid_points.at(i)));
    
    update_vector_elements(circles, points);
    
    for(auto& circle : circles) {
        base.wrap_object(*circle);
        
        if(circle->hovered()) {
            auto custom = (GridPoint*)circle->custom_data("gridpoint");
            sign.set_background(Black.alpha(50));
            Font font(0.6);
            sign.set_pos(circle->pos() - Vec2(0, Base::default_line_spacing(font)));
            
            sign.update([custom, &font](Entangled& base){
                std::string str = "grid"+Meta::toStr(*custom->_point);
                base.add<Text>(Str(str), Loc(5,5), font);
            });
            
            base.wrap_object(sign);
            sign.auto_size({10,5});
        }
    }
}

void GUI::debug_optical_flow(DrawStructure &base, Frame_t frameIndex) {
    if(frameIndex >= PD(video_source).length())
        return;
    
    auto gen_ov = [this](Frame_t frameIndex, cv::Mat& image) -> std::vector<std::pair<std::vector<HorizontalLine>, std::vector<uchar>>>{
        if(not frameIndex.valid()
           || frameIndex >= PD(video_source).length())
            return {};
        
        //image = cv::Mat::zeros(_average_image.rows, _average_image.cols, CV_8UC1);
        _average_image.get().copyTo(image);
        
        pv::File *file = PDP(video_source);
        
        pv::Frame frame;
        file->read_frame(frame, frameIndex);
        
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
        LockGuard guard(ro_t{}, "draw_flow");
        
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
    auto lock = GUI_LOCK(PD(gui).lock());
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
    
    LockGuard guard(ro_t{}, "GUI::draw_posture");
    auto midline = fish->midline(frameNr);
    if(midline) {
        // Draw the fish posture with circles
        if(midline) {
            auto && [bg_offset, max_w] = Timeline::timeline_offsets(best_base());
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
    constexpr const char* animator = "scale-boundary-gui.cpp";
    static Vec2 target_scale(1);
    static Vec2 target_pos(0,0);
    static Size2 target_size(_average_image.dimensions());
    static bool lost = true;
    static float time_lost = 0;
    
    auto && [offset, max_w] = Timeline::timeline_offsets(best_base());
    
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
            //PD(cache).set_animating(animator, false);
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
            PD(cache).set_animating(animator, true);
        }
        
        if((recording() && PD(cache).gui_time() - time_lost >= 0.5)
           || (!recording() && lost_timer.elapsed() >= 0.5))
        {
            target_scale = Vec2(1);
            //target_pos = offset;//Vec2(0, 0);
            target_size = Size2(_average_image.cols, _average_image.rows);
            target_pos = screen_center - target_size * 0.5;
            if(PD(cache).is_animating(animator))
                PD(cache).set_animating(animator, false);
        }
        else if(lost_timer.elapsed() < 0.5 || time_lost < 0.5) {
            PD(cache).set_animating(animator, true);
        }
        else {
            if (PD(cache).is_animating(animator))
                PD(cache).set_animating(animator, false);
        }
    }
    
    Float2_t mw = _average_image.cols;
    Float2_t mh = _average_image.rows;
    if(target_pos.x / target_scale.x < -mw * 0.95) {
#ifndef NDEBUG
        print("target_pos.x = ", target_pos.x," target_scale.x = ",target_scale.x);
#endif
        target_pos.x = -mw * target_scale.x * 0.95f;
    }
    if(target_pos.y / target_scale.y < -mh * 0.95f)
        target_pos.y = -mh * target_scale.y * 0.95f;
    
    if(target_pos.x / target_scale.x > mw * 0.95f) {
#ifndef NDEBUG
        print("target_pos.x = ",target_pos.x," target_scale.x = ",target_scale.x," screen_center.x = ",screen_center.width," screen_dimensions.x = ",screen_dimensions.width," window_dimensions.x = ",base()->window_dimensions().width);
#endif
        target_pos.x = mw * target_scale.x * 0.95f;
    }
    if(target_pos.y / target_scale.y > mh * 0.95f)
        target_pos.y = mh * target_scale.y * 0.95f;
    
    PD(cache).set_zoom_level(target_scale.x);
    
    static Timer timer;
    auto e = recording() ? PD(cache).dt() : timer.elapsed(); //PD(recording) ? (1 / float(FAST_SETTING(frame_rate))) : timer.elapsed();
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
    
    constexpr const char * zoom_animator = "zoom-animator";
    if(!section->scale().Equals(target_scale)
       || !section->pos().Equals(target_pos))
    {
        PD(cache).set_animating(zoom_animator, true);
        
        auto playback_factor = max(1, sqrt(SETTING(gui_playback_speed).value<float>()));
        auto scale = check_target(section->scale(), target_scale, e * playback_factor);
        
        section->set_scale(scale);
        
        auto next_pos = check_target(section->pos(), target_pos, e * playback_factor);
        auto next_size = check_target(section->size(), target_size, e * playback_factor);
        
        section->set_bounds(Bounds(next_pos, next_size));
        
    } else {
        PD(cache).set_animating(zoom_animator, false);
        
        section->set_scale(target_scale);
        section->set_bounds(Bounds(target_pos, target_size));
    }
    
    FindCoord::set_bowl_transform(section->global_transform());
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
                
                if(ptr && (PD(cache).is_animating() || PD(cache).is_tracking_dirty())) {
                    assert(dynamic_cast<Section*>(ptr));
                    s->set_scale(ptr_scale);
                    s->set_pos(ptr_pos);
                }
                
                if(!PD(heatmapController))
                    PD(heatmapController) = std::make_unique<gui::heatmap::HeatmapController>();
                PD(heatmapController)->set_frame(frame());
                base.wrap_object(*PD(heatmapController));
            });
        }
        
        base.section("tracking", [&](auto&, Section* s) {
            auto ptr = PD(gui).find("fishbowl");
            Vec2 ptr_scale(1), ptr_pos(0);
            if(ptr) {
                ptr_scale = static_cast<Section*>(ptr)->scale();
                ptr_pos = static_cast<Section*>(ptr)->pos();
            }
            
            if(ptr && (PD(cache).is_animating() || PD(cache).is_tracking_dirty())) {
                assert(dynamic_cast<Section*>(ptr));
                s->set_scale(ptr_scale);
                s->set_pos(ptr_pos);
            }
            
            if(!PD(cache).is_tracking_dirty() && !PD(cache).is_animating() //&& !PD(cache).is_animating(ptr)
               && !s->is_dirty()) {
                s->reuse_objects();
                return;
            }
            
            PD(cache).updated_tracking();
            
            std::map<Idx_t, Color> colors;
            for(auto fish : PD(cache).active)
                colors[fish->identity().ID()] = fish->identity().color();
            
            EventAnalysis::EventsContainer *container = NULL;
            container = EventAnalysis::events();
            if(FAST_SETTING(calculate_posture) && !container->map().empty() && GUI_SETTINGS(gui_show_histograms))
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
                
                PD(tracking)._length_histogram.set_data(data, ordered_colors);
                PD(tracking)._histogram.set_data(hist, ordered_colors);
            }
            
            {
                const EventAnalysis::EventMap *empty_map = NULL;
                static std::mutex fish_mutex;
                
                Vec2 scale(1);
                if(ptr) {
                    scale = ptr->scale().reciprocal().mul(Vec2(1.5));
                }
                
                set_of_individuals_t source;
                if(Tracker::has_identities() && GUI_SETTINGS(gui_show_inactive_individuals))
                {
                    for(auto [id, fish] : PD(cache).individuals)
                        source.insert(fish);
                    //! TODO: Tracker::identities().count(id) ?
                    
                } else {
                    for(auto fish : PD(cache).active)
                        source.insert(fish);
                }
                
                if(PD(cache).has_selection() && SETTING(gui_show_visualfield)) {
                    for(auto id : PD(cache).selected) {
                        auto fish = PD(cache).individuals.at(id);
                        
                        VisualField* ptr = (VisualField*)fish->custom_data(frameNr, VisualField::custom_id);
                        if(!ptr && fish->head(frameNr)) {
                            ptr = new VisualField(id, frameNr, *fish->basic_stuff(frameNr), fish->posture_stuff(frameNr), true);
                            fish->add_custom_data(frameNr, VisualField::custom_id, ptr, [](void* ptr) {
                                if(GUI::instance()) {
                                    auto lock = GUI_LOCK(PD(gui).lock());
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
                    PD(tracking._bowl).update([&](auto& e) {
                        auto coord = FindCoord::get();
                        for (auto& fish : (source.empty() ? PD(cache).active : source)) {
                            if (fish->empty()
                                || fish->start_frame() > frameNr)
                                continue;

                            auto segment = fish->segment_for(frameNr);
                            if (!GUI_SETTINGS(gui_show_inactive_individuals)
                                && (!segment || (segment->end() != Tracker::end_frame()
                                    && segment->length().get() < (long_t)GUI_SETTINGS(output_min_frames))))
                            {
                                continue;
                            }

                            auto it = container->map().find(fish);
                            if (it != container->map().end())
                                empty_map = &it->second;
                            else
                                empty_map = NULL;

                            auto id = fish->identity().ID();
                            if (PD(cache)._fish_map.find(id) == PD(cache)._fish_map.end()) {
                                PD(cache)._fish_map[id] = std::make_unique<gui::Fish>(*fish);
                                fish->register_delete_callback(PD(cache)._fish_map[id].get(), [](Individual* f) {
                                    //std::lock_guard<std::mutex> lock(_individuals_frame._mutex);
                                    if (!GUI::instance())
                                        return;

                                    auto guard = GUI_LOCK(GUI::instance()->gui().lock());

                                    auto it = PD(cache)._fish_map.find(f->identity().ID());
                                    if (it != PD(cache)._fish_map.end()) {
                                        PD(cache)._fish_map.erase(f->identity().ID());
                                    }
                                    PD(cache).set_tracking_dirty();
                                });
                            }

                            PD(cache)._fish_map[id]->set_data(*fish, frameNr, props->time, empty_map);

                            {
                                std::unique_lock guard(Categorize::DataStore::cache_mutex());
                                PD(cache)._fish_map[id]->update(coord, e, base);
                            }
                            //base.wrap_object(*PD(cache)._fish_map[fish]);
                            //PD(cache)._fish_map[fish]->label(ptr, e);
                        }
                    });

                    PD(tracking._bowl).set_bounds(average().bounds());
                    //_bowl.set_scale(Vec2(1));
                    //_bowl.set_pos(Vec2(0);
                    base.wrap_object(PD(tracking._bowl));
                }
                
                if(GUI_SETTINGS(gui_show_midline_histogram)) {
                    static Frame_t end_frame;
                    if(FAST_SETTING(calculate_posture)
                       && (not end_frame.valid()
                           || end_frame != PD(cache).tracked_frames.end))
                    {
                        end_frame = PD(cache).tracked_frames.end;
                        
                        LockGuard guard(ro_t{}, "gui_show_midline_histogram");
                        
                        std::vector<std::vector<float>> all;
                        std::vector<float> lengths;
                        
                        std::map<track::Idx_t, Individual*> search;
                        
                        if(!Tracker::has_identities()) {
                            for(auto fish : PD(cache).active) {
                                lengths.clear();
                                for (auto && stuff : fish->posture_stuff()) {
                                    if(stuff->midline_length != PostureStuff::infinity)
                                        lengths.push_back(stuff->midline_length * FAST_SETTING(cm_per_pixel));
                                }
                                all.push_back(lengths);
                                print(lengths.size()," midline samples for ",fish->identity().raw_name().c_str());
                            }
                            
                        } else {
                            for(auto &[id, fish] : PD(cache).individuals) {
                                //! TODO: Tracker::identities().count(id) ?
                                lengths.clear();
                                for (auto && stuff : fish->posture_stuff()) {
                                    if(stuff->midline_length != PostureStuff::infinity)
                                        lengths.push_back(stuff->midline_length * FAST_SETTING(cm_per_pixel));
                                }
                                all.push_back(lengths);
                                print(lengths.size()," midline samples for ",fish->identity().raw_name().c_str());
                            }
                        }
                        
                        PD(tracking)._midline_histogram.set_data(all);
                    }
                }
                
                for(auto it = PD(cache)._fish_map.cbegin(); it != PD(cache)._fish_map.cend();) {
                    if(it->second->frame() != frameNr) {
                        auto fish = PD(cache).individuals.find(it->first);
                        if(fish != PD(cache).individuals.end())
                            fish->second->unregister_delete_callback(it->second.get());
                        else
                            FormatError("Cannot find individual ", it->first, " in cache map.");
                        it = PD(cache)._fish_map.erase(it);
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
                    
                    const auto number_fish = FAST_SETTING(track_max_individuals);
                    for (uint32_t i=0; i<number_fish; ++i) {
                        if(!PD(cache).individuals.count(Idx_t(i))) {
                            FormatExcept("Individuals seem to be named differently than 0-", FAST_SETTING(track_max_individuals),". Cannot find ", i,".");
                            continue;
                        }
                        
                        auto fish0 = PD(cache).individuals.at(Idx_t(i));
                        Vec2 p0(gui::Graph::invalid());
                        
                        if(!fish0->has(frameIndex)) {
                            auto c = PD(cache).processed_frame().cached(fish0->identity().ID());
                            if(c)
                                p0 = c->estimated_px;
                        } else
                            p0 = fish0->centroid_weighted(frameIndex)->pos<Units::PX_AND_SECONDS>();
                        
                        if(Graph::is_invalid(p0.x))
                            continue;
                        
                        for(uint32_t j=i+1; j<number_fish; ++j) {
                            if(!PD(cache).individuals.count(Idx_t(j))) {
                                FormatExcept("Individuals seem to be named differently than 0-", FAST_SETTING(track_max_individuals),". Cannot find ", j,".");
                                continue;
                            }
                            
                            auto fish1 = PD(cache).individuals.at(Idx_t(j));
                            Vec2 p1(infinity<Float2_t>());
                            
                            if(!fish1->has(frameIndex)) {
                                auto c = PD(cache).processed_frame().cached(fish1->identity().ID());
                                if(c)
                                    p1 = c->estimated_px;
                            } else
                                p1 = fish1->centroid_weighted(frameIndex)->pos<Units::PX_AND_SECONDS>();
                            
                            if(Graph::is_invalid(p1.x))
                                continue;
                            
                            auto value = PD(cache).connectivity_matrix.at(FAST_SETTING(track_max_individuals) * i + j);
                            
                            base.line(Line::Point_t{ p0 }, Line::Point_t{ p1 }, LineClr{ Viridis::value(value).alpha((value * 0.6) * 255) }, Line::Thickness_t{ 1 + 5 * value });
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
                auto window = Frame_t(SETTING(output_frame_window).value<uint32_t>());
                
                for(auto id : PD(cache).selected) {
                    PD(fish_graphs)[i]->setup_graph(frameNr.get(), Rangel((frameNr.try_sub( window)).get(), (frameNr + window).get()), PD(cache).individuals.at(id), nullptr);
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
                        auto it = PD(cache)._statistics.find(Frame_t(uint32_t(x)));
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
        
        if(SETTING(gui_show_processing_time)) {
            static Graph individuals_graph(Bounds(50, 100, 500, 300), "Processing time");
            if(individuals_graph.x_range().end == FLT_MAX || individuals_graph.x_range().end != PD(cache).tracked_frames.end.get()) {
                //const auto track_max_individuals = FAST_SETTING(track_max_individuals);
                const float ymax = Tracker::max_individuals() * Tracker::average_seconds_per_individual() * 1000 * 3;
                individuals_graph.set_ranges(Rangef(PD(cache).tracked_frames.start.get(), PD(cache).tracked_frames.end.get()), Rangef(0, ymax));
                if(individuals_graph.empty()) {
                    individuals_graph.add_function(Graph::Function("ms/frame", Graph::Type::DISCRETE, [&](float x) -> float {
                        auto it = PD(cache)._statistics.find(Frame_t(uint32_t(x)));
                        if(it != PD(cache)._statistics.end()) {
                            return it->second.adding_seconds * 1000;
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
        
        DrawPreviewImage::draw(Tracker::average(), PD(cache).processed_frame(), frameNr, base);
        
#if !COMMONS_NO_PYTHON
        if(SETTING(gui_show_uniqueness)) {
            static Graph graph(Bounds(50, 100, 800, 400), "uniqueness");
            static std::mutex mutex;
            static std::map<Frame_t, float> estimated_uniqueness;
            static std::vector<Vec2> uniquenesses;
            static bool running = false;
            
            if(estimated_uniqueness.empty()
               && py::VINetwork::status().weights_valid)
            {
                std::lock_guard<std::mutex> guard(mutex);
                if(!running) {
                    running = true;
                    
                    WorkProgress::add_queue("generate images", [&]()
                    {
                        auto && [data, images, image_map] = Accumulation::generate_discrimination_data(*video_source());
                        auto && [u, umap, uq] = Accumulation::calculate_uniqueness(false, images, image_map);
                        
                        estimated_uniqueness.clear();

                        std::lock_guard<std::mutex> guard(mutex);
                        for(auto &[k,v] : umap)
                            estimated_uniqueness[k] = v;
                        
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
                            auto it = uq->upper_bound(Frame_t(sign_cast<uint32_t>(x)));
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
#endif
        
        ConfirmedCrossings::draw(base, frameNr);
        
        // Draw the fish posture with circles
        if(PD(cache).has_selection()) {
            if(SETTING(gui_show_posture)) {
                draw_posture(base, PD(cache).primary_selection(), frameNr);
            }
        }
        
        if(SETTING(gui_show_dataset)
           && PD(timeline)->visible())
        {
            if(!PD(dataset)) {
                PD(dataset) = std::make_shared<DrawDataset>();
                auto screen = screen_dimensions();
                PD(dataset)->set_pos(screen * 0.5 - PD(dataset)->size());
            }
            base.wrap_object(*PD(dataset));
        }
        
        if(SETTING(gui_show_recognition_summary)) {
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
    print("choosing ",name);
    if(index != -1) {
        //auto name = settings_dropdown.items().at(index);
        auto val = GlobalSettings::get(name);
        if(not val.get().valid()) {
            FormatExcept("Cannot find property: ", name);
            _settings_choice = nullptr;
            textfield.set_text("");
            
        } else if(val.get().is_enum() || val.is_type<bool>()) {
            auto options = val.get().is_enum() ? val.get().enum_values()() : std::vector<std::string>{ "true", "false" };
            auto index = val.get().is_enum() ? val.get().enum_index()() : (val ? 0 : 1);
            
            std::vector<std::shared_ptr<List::Item>> items;
            std::map<std::string, bool> selected_option;
            for(size_t i=0; i<options.size(); ++i) {
                selected_option[options[i]] = i == index;
                items.push_back(std::make_shared<TextItem>(options[i]));
                items.back()->set_selected(i == index);
            }
            
            print("options: ", selected_option);
            
            _settings_choice = std::make_shared<List>(Bounds(0, PD(gui).height() / PD(gui).scale().y, 150, textfield.height()), "", items, [&textfield](List*, const List::Item& item){
                print("Clicked on item ", item.ID());
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
            LockGuard guard(ro_t{}, "settings_dropdown.text() datasetquality");
            DatasetQuality::print_info();
        }
        else if(settings_dropdown.text() == "trainingdata_stats") {
            //TrainingData::print_pointer_stats();
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
            LockGuard guard(ro_t{}, "settings_dropdown.text() consecutive");
            auto consec = std::set<Range<Frame_t>>(Tracker::instance()->consecutive().begin(), Tracker::instance()->consecutive().end());
            print("consecutive frames: ", consec);
            
        }
        else if(settings_dropdown.text() == "results info") {
            using namespace Output;
            auto filename = TrackingResults::expected_filename();
            print("Trying to open results ",filename.str());
            if(file::Path(filename).exists()) {
                ResultsFormat file(filename, NULL);
                file.start_reading();
                
                if(file.header().version >= ResultsFormat::V_14) {
                    print("Settings:\n", file.header().settings);
                } else
                    FormatExcept("Cannot load settings from results file < V_14");
            } else
                FormatExcept("File ",filename.str()," does not exist.");
        }
        else if(settings_dropdown.text() == "free_fish") {
            std::set<Idx_t> free_fish, inactive;
            for(auto && [fdx, fish] : PD(cache).individuals) {
                if(!PD(cache).fish_selected_blobs.at(fdx).valid()
                   || PD(cache).fish_selected_blobs.find(fdx) == PD(cache).fish_selected_blobs.end())
                {
                    free_fish.insert(fdx);
                }
                if(PD(cache).active_ids.find(fdx) == PD(cache).active_ids.end())
                    inactive.insert(fdx);
            }
            print("All free fish in frame ", frame(),": ", free_fish);
            print("All inactive fish: ", inactive);
        }
#if !COMMONS_NO_PYTHON
        else if(settings_dropdown.text() == "print_uniqueness") {
            WorkProgress::add_queue("discrimination", [](){
                auto && [data, images, map] = Accumulation::generate_discrimination_data(*video_source());
                auto && [unique, unique_map, up] = Accumulation::calculate_uniqueness(false, images, map);
                
                std::map<Frame_t, float> tmp;
                for(auto&[k,v] : unique_map)
                    tmp[k] = v;
                auto coverage = data->draw_coverage(tmp);
                
                auto path = file::DataLocation::parse("output", "uniqueness"+(std::string)video_source()->filename().filename()+".png");
                
                print("Uniqueness: ", unique," (output to ",path.str(),")");
                cv::imwrite(path.str(), coverage->get());
            });
        }
#endif
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
            WorkProgress::add_queue("generating heatmap", [](){
                LockGuard guard(ro_t{}, "settings_dropdown.text() heatmap");
                
                cv::Mat map(PD(video_source).header().resolution.height, PD(video_source).header().resolution.width, CV_8UC4);
                
                const uint32_t width = 30;
                std::vector<double> grid;
                grid.resize(SQR(width + 1));
                Vec2 indexing(ceil(PD(video_source).header().resolution.width / float(width)),
                              ceil(PD(video_source).header().resolution.height / float(width)));
                
                size_t count = 0;
                IndividualManager::transform_all([&, N = IndividualManager::num_individuals()](auto, auto fish){
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
                    WorkProgress::set_percent(count / float(N));
                });
                
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
            LockGuard guard(ro_t{}, "settings_dropdown.text() pixels");
            print("Calculating...");
            
            std::map<std::string, size_t> average_pixels;
            std::map<std::string, size_t> samples;
            PPFrame pp;
            pv::Frame frame;
            
            for(auto idx = PD(tracker).start_frame() + 1_f; idx <= PD(tracker).end_frame() && idx <= PD(tracker).start_frame() + 10000_f; ++idx)
            {
                if(!PD(tracker).properties(idx))
                    continue;
                
                PD(video_source).read_frame(frame, idx);
                {
                    std::lock_guard<std::mutex> guard(_blob_thread_pool_mutex);
                    Tracker::instance()->preprocess_frame(std::move(frame), pp, &_blob_thread_pool, PPFrame::NeedGrid::NoNeed, video_source()->header().resolution);
                }
                
                IndividualManager::transform_ids(pp.previously_active_identities(), [&](Idx_t fdx, Individual* fish) -> void {
                    auto loaded_blob = fish->compressed_blob(idx);
                    auto blob = pp.bdx_to_ptr(loaded_blob->blob_id());
                    if(loaded_blob && blob) {
                        if(blob->split())
                            return;
                        
                        auto thresholded = blob->threshold(FAST_SETTING(track_threshold), *PD(tracker).background());
                        
                        average_pixels[fish->identity().name()] += thresholded->pixels()->size();
                        samples[fish->identity().name()] ++;
                    }
                });
            }
            
            float sum = 0;
            for(auto && [name, value] : average_pixels) {
                value /= samples.at(name);
                sum += value;
            }
            sum /= float(average_pixels.size());
            
            print("Average pixels:\n", average_pixels,"\n(overall: ",sum,")");
            
        } else if(settings_dropdown.text() == "time_deltas") {
            Graph graph(Bounds(0, 0, 1024, 400), "time_deltas");
            
            float max_val = 0, min_val = FLT_MAX;
            pv::Frame frame;
            PD(video_source).read_frame(frame, 0_f);
            
            std::vector<double> values {
                double(frame.timestamp()) / 1000.0 / 1000.0
            };
            for(size_t i = 1; i<PD(video_source).length().get(); ++i) {
                PD(video_source).read_frame(frame, Frame_t(i));
                auto t = double(frame.timestamp()) / 1000.0 / 1000.0;
                values[i - 1] = t - values[i - 1];
                values.push_back(t);
                
                max_val = max(max_val, values[i-1]);
                min_val = min(min_val, values[i-1]);
                
                if(i % int(PD(video_source).length().get() * 0.1) == 0) {
                    print(i,"/",PD(video_source).length());
                }
            }
            
            graph.add_function(Graph::Function("dt", Graph::Type::DISCRETE, [&](float x) ->float {
                if(x > 0 && x < values.size())
                    return values.at(x);
                return gui::Graph::invalid();
            }, Red, "ms"));
            
            print(min_val,"-",max_val," ",values.size());
            graph.set_ranges(Rangef(0, values.size()-1), Rangef(min_val * 0.5, max_val * 1.5));
            
            cv::Mat bg = cv::Mat::zeros(graph.height(), graph.width(), CV_8UC4);
            CVBase cvbase(bg);
            DrawStructure window(graph.width(), graph.height());
            window.wrap_object(graph);
            cvbase.paint(window);
            cvbase.display();
        } else if(settings_dropdown.text() == "blob_info") {
            print("Preprocessed frame ", PD(cache).frame_idx,":");
            //print("Filtered out: ", PD(cache).processed_frame().noise().size());
            //print("Blobs: ", PD(cache).processed_frame().blobs().size());
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
    
    auto guard = GUI_LOCK(instance()->gui().lock());
    additional_status_text() = text;
}

void GUI::draw_footer(DrawStructure& base) {
    static bool first = true;
    auto && [bg_offset, max_w] = Timeline::timeline_offsets(best_base());
    
    static HorizontalLayout status_layout(Box(10,0,0,0));
    static Text gpu_status(Font(0.7)), python_status(TextClr{Red}, Font(0.7));
    static Text additional_status(Font(0.7));
    static Text mouse_status(TextClr{White.alpha(200)}, Font(0.7));
    
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
    
    static SettingsTooltip tooltip(&PD(footer)._settings_dropdown);
    
    std::vector<Layout::Ptr> objects = { &options_dropdown, &PD(footer)._settings_dropdown};
    static HorizontalLayout layout(objects);
    
    auto h = screen_dimensions().height;
    layout.set_pos(Vec2(20, h - 10) - bg_offset / base.scale().x);
    layout.set_scale(1.1f * base.scale().reciprocal());
    
    auto layout_scale = layout.scale().x;
    auto stretch_w = status_layout.global_bounds().pos().x - 20 - PD(footer)._value_input.global_bounds().pos().x;
    if(PD(footer)._value_input.selected())
        PD(footer)._value_input.set_size(Size2(max(300, stretch_w / layout_scale), PD(footer)._value_input.height()));
    else
        PD(footer)._value_input.set_size(Size2(300, PD(footer)._value_input.height()));
    
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
            &PD(footer)._value_input,
            &options_dropdown,
            &layout,
            &PD(footer)._settings_dropdown,
            &tooltip
        });
        
        PD(clicked_background) = [&](const Vec2& pos, bool v, std::string key) {
            tracker::gui::clicked_background(PD(gui), PD(cache), pos, v, key); //PD(footer)._settings_dropdown, PD(footer)._value_input);
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
            
        PD(footer)._settings_dropdown.on_select([&](auto index, const std::string& name) {
            this->selected_setting(index.value, name, PD(footer)._value_input, PD(footer)._settings_dropdown, layout, base);
        });
        PD(footer)._value_input.on_enter([&](){
            try {
                auto key = PD(footer)._settings_dropdown.selected_item().name();
#if !COMMONS_NO_PYTHON
            if(key == "$") {
                auto code = utils::trim(PD(footer)._value_input.text());
                print("Code: ",code);
                code = utils::find_replace(code, "\\n", "\n");
                code = utils::find_replace(code, "\\t", "\t");
                
                py::schedule(py::PackagedTask{
                    ._task = py::PromisedTask([code]() -> void {
                        using py = PythonIntegration;
                        try {
                            py::execute(code);
                        } catch(const SoftExceptionImpl& e) {
                            FormatExcept("Python runtime exception: ", e.what());
                        }
                    }),
                    ._can_run_before_init = true
                });
            
            } else
#endif
            if(GlobalSettings::access_level(key) == AccessLevelType::PUBLIC) {
                    GlobalSettings::get(key).get().set_value_from_string(PD(footer)._value_input.text());
                    if(GlobalSettings::get(key).is_type<Color>())
                        this->selected_setting(PD(footer)._settings_dropdown.selected_item().ID(), key, PD(footer)._value_input, PD(footer)._settings_dropdown, layout, base);
                    if((std::string)key == "auto_apply" || (std::string)key == "auto_train")
                    {
                        SETTING(auto_train_on_startup) = false;
                    }
                    if(key == "auto_tags") {
                        SETTING(auto_tags_on_startup) = false;
                    }
                    
                } else
                   FormatError("User cannot write setting ", key," (",GlobalSettings::access_level(key).name(),").");
            } catch(const std::logic_error&) {
                //FormatExcept("Cannot set ",settings_dropdown.items().at(settings_dropdown.selected_id())," to value ",textfield.text()," (invalid).");
            } catch(const UtilsException&) {
                //FormatExcept("Cannot set ",settings_dropdown.items().at(settings_dropdown.selected_id())," to value ",textfield.text()," (invalid).");
            }
        });
        
        first = false;
    }
    PD(gui).wrap_object(layout);
    PD(gui).wrap_object(status_layout);
    
    if(PD(footer)._settings_dropdown.hovered()) {
        auto name = PD(footer)._settings_dropdown.hovered_item().name();
        if(name.empty())
            name = PD(footer)._settings_dropdown.selected_item().name();
        if(!name.empty()) {
            tooltip.set_parameter(name);
            PD(gui).wrap_object(tooltip);
        }
    }
    
#if !COMMONS_NO_PYTHON
    if (py::python_available()) {
        namespace py = Python;
        
        static Timer status_timer;
        static Accumulation::Status last_status;
        auto current_status = Accumulation::status();
        
        if (py::python_initialized()
            && (last_status != current_status
                || gpu_status.txt().empty()
                || status_timer.elapsed() > 1))
        {
            last_status = current_status;
            status_timer.reset();

            std::string txt;
            if(python_gpu_initialized())
                txt += "["+std::string(python_uses_gpu() ? python_gpu_name() : "CPU")+"]";

            if (!current_status.busy && current_status.percent == 1)
                txt += " Finished.";
            else if (current_status.busy)
                txt += " Applied " + Meta::toStr(size_t(current_status.percent * 100)) + "%" + (current_status.failed_blobs ? (" " + Meta::toStr(current_status.failed_blobs) + " failed blobs") : "");
            else txt += " Idle.";

            static Timer print_timer;
            if (print_timer.elapsed() > 1) {
                if (txt != gpu_status.txt())
                    print(txt.c_str());
                print_timer.reset();
            }
            gpu_status.set_txt(txt);
        } else if(!py::python_initialized())
            gpu_status.set_txt("");

        if (py::python_initializing()) {
            python_status.set_txt("[Python] initializing...");
            python_status.set_color(Yellow);
        }
        else if (py::python_initialized()) {
            python_status.set_txt("[Python " + Meta::toStr(python_major_version().load()) + "." + Meta::toStr(python_minor_version().load()) + "]");
            python_status.set_color(Green);
        }
        else if (python_status.txt().empty() || (!python_init_error().empty() && !py::python_initialized() && !py::python_initializing())) {
            python_status.set_txt("[Python] " + python_init_error());
            python_status.set_color(Red);
        } else {
            python_status.set_txt("[Python] Initializes when required.");
            python_status.set_color(White);
        }
    } else {
        python_status.set_txt("[Python] Not available.");
        python_status.set_color(White);
    }
#endif
    
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
    //! TODO: Thread-safety?
    const float max_w = Tracker::average().cols;
    const float max_h = Tracker::average().rows;
    
    if((PD(tracking)._recognition_image.source()->cols != max_w || PD(tracking)._recognition_image.source()->rows != max_h) && Tracker::instance()->border().type() != Border::Type::none) {
        auto border_distance = Image::Make(max_h, max_w, 4);
        border_distance->set_to(0);
        
        auto worker = [&border_distance, max_h](ushort x) {
            for (ushort y = 0; y < max_h; ++y) {
                if(Tracker::instance()->border().in_recognition_bounds(Vec2(x, y)))
                    border_distance->set_pixel(x, y, DarkCyan.alpha(15));
            }
        };
        
        {
            print("Calculating border...");
            
            std::lock_guard<std::mutex> guard(blob_thread_pool_mutex());
            try {
                for(ushort x = 0; x < max_w; ++x) {
                    blob_thread_pool().enqueue(worker, x);
                }
            } catch(...) {
                FormatExcept("blob_thread_pool error when enqueuing worker to calculate border.");
            }
            blob_thread_pool().wait();
        }
        
        PD(tracking)._recognition_image.set_source(std::move(border_distance));
        PD(cache).set_tracking_dirty();
        PD(cache).set_blobs_dirty();
        PD(cache).set_raw_blobs_dirty();
        PD(cache).set_redraw();
    }
    
    if(!FAST_SETTING(track_include).empty())
    {
        auto keys = extract_keys(PD(tracking)._include_shapes);
        
        for(auto &rect : FAST_SETTING(track_include)) {
            auto it = PD(tracking)._include_shapes.find(rect);
            if(it == PD(tracking)._include_shapes.end()) {
                if(rect.size() == 2) {
                    auto ptr = std::make_shared<Rect>(Box(rect[0], rect[1] - rect[0]), FillClr{Green.alpha(25)}, LineClr{Green.alpha(100)});
                    //ptr->set_clickable(true);
                    PD(tracking)._include_shapes[rect] = ptr;
                    
                } else if(rect.size() > 2) {
                    //auto r = std::make_shared<std::vector<Vec2>>(rect);
                    auto r = poly_convex_hull(&rect); // force a convex polygon for these shapes, as thats the only thing that the in/out polygon test works with
                    auto ptr = std::make_shared<gui::Polygon>(r);
                    ptr->set_fill_clr(Green.alpha(25));
                    ptr->set_border_clr(Green.alpha(100));
                    //ptr->set_clickable(true);
                    PD(tracking)._include_shapes[rect] = ptr;
                }
            }
            keys.erase(rect);
        }
        
        for(auto &key : keys) {
            PD(tracking)._include_shapes.erase(key);
        }
        
        PD(cache).set_raw_blobs_dirty();
        
    } else if(FAST_SETTING(track_include).empty() && !PD(tracking)._include_shapes.empty()) {
        PD(tracking)._include_shapes.clear();
        PD(cache).set_raw_blobs_dirty();
    }
    
    if(!FAST_SETTING(track_ignore).empty())
    {
        auto keys = extract_keys(PD(tracking)._ignore_shapes);
        
        for(auto &rect : FAST_SETTING(track_ignore)) {
            auto it = PD(tracking)._ignore_shapes.find(rect);
            if(it == PD(tracking)._ignore_shapes.end()) {
                if(rect.size() == 2) {
                    auto ptr = std::make_shared<Rect>(Box(rect[0], rect[1] - rect[0]), FillClr{Red.alpha(25)}, LineClr{Red.alpha(100)});
                    //ptr->set_clickable(true);
                    PD(tracking)._ignore_shapes[rect] = ptr;
                    
                } else if(rect.size() > 2) {
                    //auto r = std::make_shared<std::vector<Vec2>>(rect);
                    auto r = poly_convex_hull(&rect); // force convex polygon
                    auto ptr = std::make_shared<gui::Polygon>(r);
                    ptr->set_fill_clr(Red.alpha(25));
                    ptr->set_border_clr(Red.alpha(100));
                    //ptr->set_clickable(true);
                    PD(tracking)._ignore_shapes[rect] = ptr;
                }
            }
            keys.erase(rect);
        }
        
        for(auto &key : keys) {
            PD(tracking)._ignore_shapes.erase(key);
        }
        
        PD(cache).set_raw_blobs_dirty();
        
    } else if(FAST_SETTING(track_ignore).empty() && !PD(tracking)._ignore_shapes.empty()) {
        PD(tracking)._ignore_shapes.clear();
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
    if((/*PD(cache).raw_blobs_dirty() ||*/ PD(cache).display_blobs.size() != PD(cache).raw_blobs.size()) && draw_blobs)
    {
        static std::mutex vector_mutex;
        auto screen_bounds = Bounds(Vec2(), screen_dimensions());
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
                   (!PD(cache).display_blobs.contains((*it)->blob->blob_id()) && !contains(PD(cache).active_blobs, (*it)->blob->blob_id())))
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
            PD(cache).display_blobs.insert(map.begin(), map.end());
            //std::move(vector.begin(), vector.end(), std::back_inserter(PD(cache).display_blobs_list));
            //PD(cache).display_blobs_list.insert(PD(cache).display_blobs_list.end(), vector.begin(), vector.end());
            
        }, _blob_thread_pool, PD(cache).raw_blobs.begin(), PD(cache).raw_blobs.end());
        
        PD(cache)._current_pixels = gpixels;
        PD(cache)._average_pixels = gsamples > 0 ? gaverage_pixels / gsamples : 0;
        
    }
}

void GUI::draw_raw(gui::DrawStructure &base, Frame_t) {
    Section* fishbowl;
    
    const auto mode = GUI_SETTINGS(gui_mode);
    const auto draw_blobs = GUI_SETTINGS(gui_show_blobs) || mode != gui::mode_t::tracking;
    //const double coverage = double(PD(cache)._num_pixels) / double(PD(collection)->source()->rows * PD(collection)->source()->cols);
#if defined(TREX_ENABLE_EXPERIMENTAL_BLUR) && defined(__APPLE__) && COMMONS_METAL_AVAILABLE
    const bool draw_blobs_separately = SETTING(gui_draw_blobs_separately) || GUI_SETTINGS(gui_macos_blur);//(GUI_SETTINGS(gui_macos_blur) || coverage < 0.002);
    //(!GUI_SETTINGS(gui_macos_blur) || !std::is_same<MetalImpl, default_impl_t>::value || GUI_SETTINGS(gui_mode) != gui::mode_t::blobs) && coverage < 0.002 && draw_blobs;
#else
    const bool draw_blobs_separately = false;//coverage < 0.002 && draw_blobs;
#endif
    bool redraw_blobs = true;//PD(cache).raw_blobs_dirty();
    
    /*struct FPS {
        double fps = 0, samples = 0, ratio = 0;
    };
    static std::map<bool, FPS> playback;
    auto &play = playback[draw_blobs_separately];
    play.fps += PD(tdelta_gui);
    ++play.samples;
    play.ratio += coverage;
    
    SETTING(gui_draw_blobs_separately) = !SETTING(gui_draw_blobs_separately).value<bool>();
    
    if(int(play.samples) % 500 == 0)
    {
        print("playback(together): ", 1.0 / (playback[false].fps / playback[false].samples), " at average coverage ", playback[false].ratio / playback[false].samples);
        print("playback(separate): ", 1.0 / (playback[true].fps / playback[true].samples) , " at average coverage ", playback[true].ratio / playback[true].samples);
        print("---");
        playback[false].fps = playback[false].samples = playback[true].fps = playback[true].samples = 0;
    }*/
    
    
    base.section("fishbowl", [&](DrawStructure &base, Section* section) {
        fishbowl = section;
        bool shift = PD(gui).is_key_pressed(gui::LShift) && (!PD(gui).selected_object() || !dynamic_cast<Textfield*>(PD(gui).selected_object()));
        gui_scale_with_boundary(PD(cache).boundary, section, !shift && (GUI_SETTINGS(gui_auto_scale) || (GUI_SETTINGS(gui_auto_scale_focus_one) && PD(cache).has_selection())));
        
        //if(((PD(cache).mode() == Mode::DEBUG && !PD(cache).blobs_dirty()) || (PD(cache).mode() == Mode::DEFAULT && !PD(cache).is_tracking_dirty()))
        if(!PD(cache).raw_blobs_dirty() && !PD(cache).is_animating() //!PD(cache).is_animating(_setting_animation.display.get()))
           //&& !_setting_animation.display
           )
        {
            redraw_blobs = false;
            section->reuse_objects();
            return;
        }
        
        
        if(GUI_SETTINGS(gui_show_recognition_bounds)) {
            if(!PD(tracking)._recognition_image.source()->empty()) {
                base.wrap_object(PD(tracking)._recognition_image);
            }
            Tracker::instance()->border().draw(base);
        }
        
        if(PD(timeline)->visible()) {
            Scale scale{PD(gui).scale().reciprocal()};
            
            for(auto && [rect, ptr] : PD(tracking)._include_shapes) {
                base.wrap_object(*ptr);
                
                if(ptr->hovered()) {
                    const Font font(0.85 / (1 - ((1 - PD(cache).zoom_level()) * 0.5)), Align::VerticalCenter);
                    base.text(Str("allowing "+Meta::toStr(rect)), Loc(ptr->pos() + Vec2(5, Base::default_line_spacing(font) + 5)), font, scale);
                }
            }
            
            for(auto && [rect, ptr] : PD(tracking)._ignore_shapes) {
                base.wrap_object(*ptr);
                
                if(ptr->hovered()) {
                    const Font font(0.85 / (1 - ((1 - PD(cache).zoom_level()) * 0.5)), Align::VerticalCenter);
                    base.text(Str("excluding "+Meta::toStr(rect)), Loc(ptr->pos() + Vec2(5, Base::default_line_spacing(font) + 5)), font, scale);
                }
            }
        }
        
        //update_display_blobs(draw_blobs, fishbowl);
        //PD(cache).updated_raw_blobs();
        
        if(draw_blobs_separately) {
            if(GUI_SETTINGS(gui_mode) == gui::mode_t::tracking && PD(cache).tracked_frames.contains(frame())) {
                for(auto &&[k,fish] : PD(cache)._fish_map) {
                    auto obj = fish->shadow();
                    if(obj)
                        base.wrap_object(*obj);
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
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                        if(GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
                        {
                            ptr->ptr->tag(Effects::blur);
                        }
#endif
#endif
                        base.wrap_object(*(ptr->ptr));
                    }
                }
                
            } else {
                for(auto &[b, ptr] : PD(cache).display_blobs) {
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
#if defined(__APPLE__) && COMMONS_METAL_AVAILABLE
                    if(GUI_SETTINGS(gui_macos_blur) && std::is_same<MetalImpl, default_impl_t>::value)
                    {
                        ptr->ptr->untag(Effects::blur);
                    }
#endif
#endif
                    base.wrap_object(*(ptr->ptr));
                }
            }
            
        } else if(draw_blobs && GUI_SETTINGS(gui_mode) == gui::mode_t::tracking && PD(cache).tracked_frames.contains(frame())) {
            for(auto &&[k,fish] : PD(cache)._fish_map) {
                auto obj = fish->shadow();
                if(obj)
                    base.wrap_object(*obj);
            }
        }
    });
    
    /**
     * Drawing all blobs separately is deemed inefficient.
     * (if most of the screen is covered with individuals,
     * it's usually better to draw them on one image and
     * then draw that instead).
     */
    if(!draw_blobs_separately && draw_blobs) {
        if(redraw_blobs) {
            auto mat = PD(collection)->source()->get();
            //std::fill((int*)collection->source()->data(), (int*)collection->source()->data() + collection->source()->cols * collection->source()->rows, 0);
            
            distribute_indexes([](auto, auto start, auto end, auto) {
                std::fill(start, end, 0);
                
            }, _blob_thread_pool, (int*)PD(collection)->source()->data(), (int*)PD(collection)->source()->data() + PD(collection)->source()->cols * PD(collection)->source()->rows);
            
            const auto image = Bounds(Size2(mat));
            
            static std::mutex _mutex;
            distribute_indexes([&mat, &image](auto, auto start, auto end, auto){
                using namespace grab::default_config;
                auto apply = [&]<grab::default_config::meta_encoding_t::Class encoding>() {
                    Color inp;
                    std::unique_lock guard(_mutex, std::defer_lock);
                    for(auto it = start; it != end; ++it) {
                        auto& e = *it;
                        auto input = e.second->ptr->source()->get();
                        auto &bounds = e.second->ptr->bounds();
                        
                        if(image.contains(bounds)) {
                            for (int y = bounds.y; y < bounds.y + bounds.height && y - image.y < image.height; ++y) {
                                for (int x = bounds.x; x < bounds.x + bounds.width && x - image.x < image.width; ++x) {
                                    if constexpr(encoding == meta_encoding_t::r3g3b2)
                                        inp = Color(input.template at<cv::Vec4b>(y - bounds.y, x - bounds.x));
                                    else {
                                        inp = Color(input.template at<cv::Vec2b>(y - bounds.y, x - bounds.x));
                                    }
                                    
                                    if(inp.a > 0) {
                                        guard.lock();
                                        mat.at<cv::Vec4b>(y, x) = inp;
                                        guard.unlock();
                                    }
                                }
                            }
                        }
                    }
                };
                
                
                        //assert(input.channels() == 2);
                        //assert(mat.channels() == 4);
                if(SETTING(meta_encoding).value<meta_encoding_t::Class>() == meta_encoding_t::r3g3b2)
                        apply.template operator()<meta_encoding_t::r3g3b2>();
                else
                    apply.template operator()<meta_encoding_t::gray>();
                
            }, _blob_thread_pool, PD(cache).display_blobs.begin(), PD(cache).display_blobs.end());
            
            PD(collection)->set_dirty();
        }
        
        PD(collection)->set_scale(fishbowl->scale());
        PD(collection)->set_pos(fishbowl->pos());
        base.wrap_object(*PD(collection));
    }
    
#ifndef NDEBUG
    if(draw_blobs_separately)
    {
        base.rect(Box(0, 0, 100, 100), FillClr{Red});
    }
#endif
    
    tracker::gui::draw_boundary_selection(gui(), this->base(), PD(cache), fishbowl);//, PD(footer)._settings_dropdown, PD(footer)._value_input);
}

void GUI::draw_raw_mode(DrawStructure &base, Frame_t frameIndex) {
    pv::File *file = PDP(video_source);
    if(file && file->length() > frameIndex) {
        auto ptr = PD(gui).find("fishbowl");
        Vec2 ptr_scale(1), ptr_pos(0);
        auto dim = screen_dimensions();
        Transform transform;
        
        if(ptr) {
            assert(dynamic_cast<Section*>(ptr));
            ptr_scale = ptr->scale();
            ptr_pos = ptr->pos();
            transform = ptr->global_transform();
        }
        
        tracker::gui::draw_blob_view({
            .graph = base,
            .cache = PD(cache),
            .coord = FindCoord::get()
        });
    }
}

void GUI::local_event(const gui::Event &event) {
    if (event.type == gui::KEY) {
        auto guard = GUI_LOCK(gui().lock());
        key_event(event);
        PD(cache).set_redraw();
        
    } else {
        auto guard = GUI_LOCK(gui().lock());
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
        this->event(PD(gui), e);
    }
}

void GUI::confirm_terminate() {
    static bool terminate_visible = false;
    if(terminate_visible)
        return;
    
    terminate_visible = true;
    
    WorkProgress::add_queue("", [ptr = &terminate_visible](){
        auto lock_guard = GUI_LOCK(PD(gui).lock());
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
    WorkProgress::add_queue("", [](){
        print("Writing backup of settings...");
        GUI::write_config(true, TEXT, "backup");
    });
}

void GUI::open_docs() {
    std::string filename("https://trex.run/docs");
    print("Opening ",filename," in browser...");
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
#elif !defined(__EMSCRIPTEN__)
    ShellExecute(0, 0, filename.c_str(), 0, 0 , SW_SHOW );
#endif
}

void GUI::key_event(const gui::Event &event) {
    auto &key = event.key;
    if(!key.pressed)
        return;
    
    if(key.code >= Codes::Num0 && key.code <= Codes::Num9) {
        auto lock = GUI_LOCK(PD(gui).lock());
        Identity id(Idx_t(narrow_cast<uint32_t>(key.code - Codes::Num0)));
        PD(cache).deselect_all_select(id.ID());
        GUI::set_redraw();
        return;
    }
    
    auto next_crossing = [&](){
        if(ConfirmedCrossings::next(PD(cache)._current_foi)) {
            auto lock = GUI_LOCK(PD(gui).lock());
            
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
                auto lock = GUI_LOCK(PD(gui).lock());
                
                direction_change() = play_direction() != 1;
                if (direction_change()) {
                    PD(last_direction_change).reset();
                    PD(last_increase_timer).reset();
                }
                
                if(PD(last_increase_timer).elapsed() >= 0.15)
                    PD(last_direction_change).reset();
                
                float percent = min(1, PD(last_direction_change).elapsed() / 2.f);
                percent *= percent;
                
                uint32_t inc = !direction_change() && PD(last_increase_timer).elapsed() < 0.15 ? ceil(PD(last_increase_timer).elapsed() * max(2u, FAST_SETTING(frame_rate) * 4u) * percent) : 1;
                
                
                play_direction() = 1;
                
                Frame_t new_frame = min(PD(video_source).length()-1_f, frame() + Frame_t(inc));
                SETTING(gui_frame) = new_frame;
                
                PD(last_increase_timer).reset();
                
                //LockGuard guard;
                //Tracker::find_next_problem(*PD(video_source), frame_ref());
            }
            break;
        }
            
        case Codes::Space: {
            run(!run());
            
            auto lock = GUI_LOCK(PD(gui).lock());
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
                auto lock = GUI_LOCK(PD(gui).lock());
                
                direction_change() = play_direction() != -1;
                if (direction_change()) {
                    PD(last_direction_change).reset();
                    PD(last_increase_timer).reset();
                }
                
                if(PD(last_increase_timer).elapsed() >= 0.15)
                    PD(last_direction_change).reset();
                
                float percent = min(1, PD(last_direction_change).elapsed() / 2.f);
                percent *= percent;
                
                uint32_t inc = !direction_change() && PD(last_increase_timer).elapsed() < 0.15 ? ceil(PD(last_increase_timer).elapsed() * max(2u, FAST_SETTING(frame_rate) * 4u) * percent) : 1;
                
                
                play_direction() = -1;
                
                auto new_frame = frame().try_sub(Frame_t(inc));
                SETTING(gui_frame) = new_frame;
                
                PD(last_increase_timer).reset();
                
                //LockGuard guard;
                //Tracker::find_next_problem(*PD(video_source), frame_ref());
            }
            
            break;
        }
            
        case Codes::Comma: {
            auto fn = []() {
                PD(analysis).set_paused(!PD(analysis).paused()).get();
            };
            
            WorkProgress::add_queue(PD(analysis).paused() ? "Unpausing..." : "Pausing...", fn);
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
            auto lock = GUI_LOCK(PD(gui).lock());
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
            auto lock = GUI_LOCK(PD(gui).lock());
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
            if(recording())
                stop_recording();
            else
                start_recording();
            
            break;
            
        case Codes::S:
            WorkProgress::add_queue("Saving to "+(std::string)GUI_SETTINGS(output_format).name()+" ...", [this]() { export_tracks(); });
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
            auto lock = GUI_LOCK(PD(gui).lock());
            if (ConfirmedCrossings::started()) {
                next_crossing();
                
            } else
                PD(timeline)->next_poi();
            break;
        }
            
        case Codes::N:
        {
            auto lock = GUI_LOCK(PD(gui).lock());
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
            WorkProgress::add_queue("", [](){
                bool before = PD(analysis).is_paused();
                PD(analysis).set_paused(true).get();
                
                /*auto per_frame = Tracker::find_next_problem(*PD(video_source), frame());
                if(per_frame.empty()) {
                    FormatWarning("per_frame is empty.");
                    return;
                }
                
                try {
                    this->generate_training_data(GUIType::GRAPHICAL, false, per_frame);
                } catch(const UtilsException& ex) {
                    FormatWarning("Aborting training data because an exception was thrown.");
                }*/
                
                Tracker::instance()->check_segments_identities(false, IdentitySource::VisualIdent, [](auto){}, [](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                    WorkProgress::add_queue(t, fn, b);
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
                
                LockGuard guard(w_t{}, "Codes::I");
                Results results(PD(tracker));
                
                file::Path fishdata = file::DataLocation::parse("output", SETTING(fishdata_dir).value<file::Path>());
                
                if(!fishdata.exists())
                    if(!fishdata.create_folder())
                        throw U_EXCEPTION("Cannot create folder ",fishdata.str()," for saving fishdata.");
                
                try {
                    results.save_events((fishdata / SETTING(filename).value<file::Path>().filename()).str() + "_events", [](float percent) { WorkProgress::set_percent(percent); });
                } catch(const UtilsException& e) {
                    
                }
                
                //PD(analysis).reset_PD(cache);
                Output::Library::clear_cache();
                if(!before)
                    PD(analysis).set_paused(false).get();
            };
            
            WorkProgress::add_queue("Saving events...", fn);
            
            break;
        }
            
        case Codes::LShift:
        case Codes::RShift:
            break;
            
        default:
#ifndef NDEBUG
            if(key.code != -1)
                print("Unknown key code ",key.code,".");
#endif
            break;
    }
    
    GUI::set_redraw();
}

void GUI::auto_correct(GUI::GUIType type, bool force_correct) {
    //work().add_queue("checking identities...", [this](){
    if(!Tracker::instance())
        return;
    
    if(type == GUIType::GRAPHICAL) {
        const char* message_only_ml = "Automatic correction uses machine learning based predictions to correct potential tracking mistakes. Make sure that you have trained the visual identification network prior to using auto-correct.\n<i>Apply and retrack</i> will overwrite your <key>manual_matches</key> and replace any previous automatic matches based on new predictions made by the visual identification network. If you just want to see averages for the predictions without changing your tracks, click the <i>review</i> button.";
        const char* message_both = "Automatic correction uses machine learning based predictions to correct potential tracking mistakes (visual identification, or physical tag data). Make sure that you have trained the visual identification network prior to using auto-correct, or that tag information is available.\n<i>Visual identification</i> and <i>Tags</i> will overwrite your <key>manual_matches</key> and replace any previous automatic matches based on new predictions made by the visual identification network/the tag data. If you just want to see averages for the visual identification predictions without changing your tracks, click the <i>Review VI</i> button.";
        const bool tags_available = tags::available();

        PD(gui).dialog([this, tags_available](gui::Dialog::Result r) {
            if(r == Dialog::ABORT)
                return;
            
            WorkProgress::add_queue("checking identities...", [this, r, tags_available](){
                if(r == Dialog::OKAY) {
                    auto lock_guard = GUI_LOCK(PD(gui).lock());
                    PD(tracking_callbacks).push([](){
                        instance()->auto_correct(GUI::GUIType::TEXT, false);
                    });
                }
                
                Tracker::instance()->check_segments_identities(r != Dialog::SECOND, tags_available && r == Dialog::THIRD ? IdentitySource::QRCodes : IdentitySource::VisualIdent, [](float x) { WorkProgress::set_percent(x); }, [this](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                    WorkProgress::add_queue(t, fn, b);
                });
                
                auto lock_guard = GUI_LOCK(PD(gui).lock());
                PD(cache).recognition_updated = false;
                PD(cache).set_tracking_dirty();
            });
            
        }, tags_available ? message_both : message_only_ml, "Auto-correct", tags_available ? "Apply visual identification" : "Apply and retrack", "Cancel", "Review VI", tags_available ? "Apply tags" : "");
    } else {
        WorkProgress::add_queue("checking identities...", [this, force_correct](){
            Tracker::instance()->check_segments_identities(force_correct, IdentitySource::VisualIdent, [](float x) { WorkProgress::set_percent(x); }, [this](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                WorkProgress::add_queue(t, [fn](){
                    {
                        auto lock = GUI_LOCK(instance()->gui().lock());
                        PD(tracking_callbacks).push([](){
                            instance()->auto_correct(GUI::GUIType::TEXT, false);
                        });
                    }
                    
                    fn();
                }, b);
            });
            
            auto lock = GUI_LOCK(instance()->gui().lock());
            PD(cache).recognition_updated = false;
            PD(cache).set_tracking_dirty();
            
            //if(!force_correct)
            //    print("Automatic correct has not been performed (only averages have been calculated). In order to do so, add the keyword 'force' after the command.");
        });
    }
    //});
}

void GUI::save_state(GUI::GUIType type, bool force_overwrite) {
    static bool save_state_visible = false;
    if(save_state_visible)
        return;
    
    save_state_visible = true;
    static file::Path file;
    file = Output::TrackingResults::expected_filename();
    
    auto fn = []() {
        bool before = PD(analysis).is_paused();
        PD(analysis).set_paused(true).get();
        
        LockGuard guard(w_t{}, "GUI::save_state");
        try {
            Output::TrackingResults results(PD(tracker));
            results.save([](const std::string& title, float x, const std::string& description){ WorkProgress::set_progress(title, x, description); }, file);
        } catch(const UtilsException&e) {
            auto what = std::string(e.what());
            WorkProgress::add_queue("", [what]() {
                GUI::instance()->gui().dialog([](Dialog::Result){}, "Something went wrong saving the program state. Maybe no write permissions? Check out this message, too:\n<i>"+what+"</i>", "Error");
            });
            
            FormatExcept("Something went wrong saving program state. Maybe no write permissions?"); }
        
        if(!before)
            PD(analysis).set_paused(false).get();
        
        save_state_visible = false;
    };
    
    if(file.exists() && !force_overwrite) {
        if(type != GUIType::GRAPHICAL) {
            print("The file ",file.str()," already exists. To overwrite this setting, add the keyword 'force'.");
            save_state_visible = false;
        } else {
            WorkProgress::add_queue("", [fn](){
                PD(gui).dialog([fn](Dialog::Result result) {
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
        }
        
    } else
        WorkProgress::add_queue("Saving results...", fn);
}

void GUI::auto_quit() {
    FormatWarning("Saving and quitting...");
                        
    auto lock = GUI_LOCK(instance()->gui().lock());
    LockGuard guard(w_t{}, "saving and quitting");
    PD(cache).deselect_all();
    instance()->write_config(true);
    
    if(!SETTING(auto_no_results)) {
        Output::TrackingResults results(PD(tracker));
        results.save();
    } else {
        file::Path path = Output::TrackingResults::expected_filename();
        path = path.add_extension("meta");
        
        print("Writing ",path.str()," meta file instead of .results");
        
        auto f = fopen(path.str().c_str(), "wb");
        if(f) {
            auto str = SETTING(cmd_line).value<std::string>()+"\n";
            fwrite(str.data(), sizeof(uchar), str.length(), f);
            fclose(f);
        } else
            print("Cannot write ",path.str()," meta file.");
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
    
void GUI::tracking_finished() {
    {
        auto lock = GUI_LOCK(instance()->gui().lock());
        while(!PD(tracking_callbacks).empty()) {
            auto &item = PD(tracking_callbacks).front();
            item();
            PD(tracking_callbacks).pop();
        }
    }

#if !COMMONS_NO_PYTHON
    
    if(SETTING(auto_categorize)) {
        GUI::auto_categorize();
    } else if(SETTING(auto_train)) {
        GUI::auto_train();
    } else if(SETTING(auto_apply)) {
        GUI::auto_apply();
    } else if(SETTING(auto_tags)) {
        auto message = "Can currently only use auto_tags in combination with '-load', when loading from a results file generated by TGrabs (where the tag information is stored). Please append '-load' to the command-line, for example, to load an existing results file.\nOtherwise please open a ticket at https://github.com/mooch443/trex, if you have a specific application for this kind of function (where TRex, not TGrabs, applies a network model to existing tag images).";
        if(SETTING(auto_tags_on_startup))
            throw U_EXCEPTION(message);
        else
            FormatWarning(message);
        
        //GUI::auto_tags();
    }
    
    // check if results should be saved and the app should quit
    // automatically after analysis is done.
    else
#endif
        if(SETTING(auto_quit))
    {
        GUI::auto_quit();
    }
}
    
#if !COMMONS_NO_PYTHON
void GUI::auto_categorize() {
    Categorize::Work::set_state(video_source(), Categorize::Work::State::LOAD);
    Categorize::Work::set_state(video_source(), Categorize::Work::State::APPLY);
}

void GUI::auto_train() {
    SETTING(auto_train) = false;
    if(!instance())
        return;
    
    Accumulation::register_apply_callback([&](){
        print("Finished.");
        
        // TODO: MISSING
        //Tracker::recognition()->check_last_prediction_accuracy();
        
        auto lock = GUI_LOCK(instance()->gui().lock());
        instance()->auto_correct(GUI::GUIType::TEXT, true);
    });
    
    print("Registering finished callback.");
    
    auto lock = GUI_LOCK(instance()->gui().lock());
    instance()->training_data_dialog(GUI::GUIType::TEXT, false /* retrain */);
}
    
    void GUI::auto_tags() {
        SETTING(auto_tags) = false;
        
        if(!tags::available()) {
            auto message = "Cannot perform automatic correction based on tags, since no tag information is available.";
            if(SETTING(auto_tags_on_startup)) {
                throw U_EXCEPTION(message);
            } else
                FormatWarning(message);
            
            return;
        }
        
        auto lock = GUI_LOCK(instance()->gui().lock());
        WorkProgress::add_queue("checking identities...", [](){
            Tracker::instance()->check_segments_identities(
                true,
                IdentitySource::QRCodes,
                [](float x) { WorkProgress::set_percent(x); },
                [](const std::string&t, const std::function<void()>& fn, const std::string&b) {
                    WorkProgress::add_queue(t, fn, b);
                }
            );
            
            PD(cache).recognition_updated = false;
            PD(cache).set_tracking_dirty();
        });
    }

void GUI::auto_apply() {
    SETTING(auto_apply) = false;
    if(!instance())
        return;
    
    Accumulation::register_apply_callback([&](){
        print("Finished.");
        
        // TODO: MISSING
        //Tracker::recognition()->check_last_prediction_accuracy();
        
        auto lock = GUI_LOCK(instance()->gui().lock());
        instance()->auto_correct(GUI::GUIType::TEXT, true);
    });
    
    auto lock = GUI_LOCK(instance()->gui().lock());
    instance()->training_data_dialog(GUI::GUIType::TEXT, true);
}
#endif

void GUI::load_state(GUI::GUIType type, file::Path from) {
    static bool state_visible = false;
    if(state_visible)
        return;
    
    state_visible = true;

    auto fn = [this, from]() {
        bool before = PD(analysis).is_paused();
        PD(analysis).set_paused(true).get();
        
        Categorize::DataStore::clear();
        
        LockGuard guard(w_t{}, "GUI::load_state");
        Output::TrackingResults results(PD(tracker));
        
        PD(timeline)->reset_events();
        //PD(analysis).clear();
        
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
                    GUI::instance()->video_source()->read_frame(f, k);
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
                                GUI::instance()->video_source()->read_frame(f, k);
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
                config.set_do_print(false);
                
                default_config::get(config, docs, NULL);
                try {
                    default_config::load_string_with_deprecations(from.str(), header.settings, config, AccessLevelType::STARTUP, {}, true);
                    
                } catch(const cmn::illegal_syntax& e) {
                    print("Illegal syntax in .results settings (",e.what(),").");
                }
                
                std::vector<Idx_t> focus_group;
                if(config.has("gui_focus_group"))
                    focus_group = config["gui_focus_group"].value<std::vector<Idx_t>>();
                
                if(GUI::instance() && !gui_frame_on_startup().frame.valid()) {
                    WorkProgress::add_queue("", [f = Frame_t(header.gui_frame)](){
                        SETTING(gui_frame) = f;
                    });
                }
                
                if(GUI::instance() && !gui_frame_on_startup().focus_group.has_value()) {
                    WorkProgress::add_queue("", [focus_group](){
                        SETTING(gui_focus_group) = focus_group;
                    });
                }
                
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
            
            if(GUI::instance()) {
                auto what = std::string(e.what());
                WorkProgress::add_queue("", [what, from]() {
                    GUI::instance()->gui().dialog([](Dialog::Result){}, "Cannot load results from '"+from.str()+"'. Loading crashed with this message:\n<i>"+what+"</i>", "Error");
                });
            
                auto start = Tracker::start_frame();
                Tracker::instance()->_remove_frames(start);
                removed_frames(start);
            }
        }
        
        //PD(analysis).reset_PD(cache);
        Output::Library::clear_cache();
        
        auto range = PD(tracker).analysis_range();
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
            PD(analysis).set_paused(false).get();
        
        state_visible = false;
    };
    
    if(type == GRAPHICAL) {
        PD(gui).dialog([fn](Dialog::Result result) {
            if(result == Dialog::Result::OKAY) {
                WorkProgress::add_queue("Loading results...", fn, PD(video_source).filename().str());
            } else {
                state_visible = false;
            }
            
        }, "Are you sure you want to load results?\nThis will discard any unsaved changes.", "Load results", "Yes", "Cancel");
    } else {
        WorkProgress::add_queue("Loading results...", fn, PD(video_source).filename().str());
    }
}

void GUI::save_visual_fields() {
    bool before = PD(analysis).is_paused();
    PD(analysis).set_paused(true).get();
    
    LockGuard guard(w_t{}, "GUI::save_visual_fields");
    Individual *selected = PD(cache).primary_selection();
    
    auto fishdata_dir = SETTING(fishdata_dir).value<file::Path>();
    auto fishdata = file::DataLocation::parse("output", fishdata_dir);
    if(!fishdata.exists())
        if(!fishdata.create_folder())
            throw U_EXCEPTION("Cannot create folder ",fishdata.str()," for saving fishdata.");
    auto filename = (std::string)SETTING(filename).value<file::Path>().filename();
    
    if(selected) {
        auto path = fishdata / (filename + "_visual_field_"+selected->identity().name());
        WorkProgress::set_progress("generating visual field", 0, path.str());
        selected->save_visual_field(path.str(), Range<Frame_t>({},{}), [](float percent, const std::string& title){ WorkProgress::set_progress(title, percent); }, false);
        
    } else {
        std::atomic_size_t counter = 0;
        WorkProgress::set_percent(0);
        static GenericThreadPool visual_field_thread_pool(cmn::hardware_concurrency(), "visual_fields");
        IndividualManager::transform_parallel(visual_field_thread_pool, [&, N = float(IndividualManager::num_individuals())](auto, auto fish)
        {
            auto path = fishdata / (filename + "_visual_field_"+fish->identity().name());
            WorkProgress::set_progress("generating visual fields", -1, path.str());
            
            fish->save_visual_field(path.str(), Range<Frame_t>({},{}), [&](float, const std::string& title){
                WorkProgress::set_progress(title, (counter + 0) / N);
            }, false);
            
            ++counter;
        });
    }
    
    SETTING(analysis_paused) = before;
}



void GUI::export_tracks(const file::Path& , Idx_t fdx, Range<Frame_t> range) {
    bool before = GUI::analysis()->is_paused();
    GUI::analysis()->set_paused(true).get();
    
    track::export_data(*video_source(), PD(tracker), fdx, range);
    
    if(!before)
        GUI::analysis()->set_paused(false).get();
}

ConnectedTasks* GUI::analysis() {
    std::unique_lock guard(PD(analyis_mutex));
    return PDP(analysis);
}

void GUI::set_analysis(ConnectedTasks* ptr) {
    std::unique_lock guard(PD(analyis_mutex));
    PDP(analysis) = ptr;
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
    
    str.append("\n<b>max-curvature:</b> "+std::to_string(Outline::max_curvature()));
    str.append("\n<b>average max-curvature:</b> "+std::to_string(Outline::average_curvature()));
    
    auto consec = instance()->frameinfo().global_segment_order.empty() ? Range<Frame_t>({},{}) : instance()->frameinfo().global_segment_order.front();
    std::stringstream number;
    number << consec.start.toStr() << "-" << consec.end.toStr() << " (" << (consec.start.valid() ? consec.end - consec.start : Frame_t{}).toStr() << ")";
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
    auto filename = file::DataLocation::parse(suffix == "backup" ? "backup_settings" : "output_settings");
    auto text = default_config::generate_delta_config().to_settings();
    
    if(filename.exists() && !overwrite) {
        if(type == GUIType::GRAPHICAL) {
            PD(gui).dialog([str = text, filename](Dialog::Result r) {
                if(r == Dialog::OKAY) {
                    if(!filename.remove_filename().exists())
                        filename.remove_filename().create_folder();
                    
                    FILE *f = fopen(filename.str().c_str(), "wb");
                    if(f) {
                        print("Overwriting file ",filename.str(),".");
                        fwrite(str.data(), 1, str.length(), f);
                        fclose(f);
                    } else {
                        FormatExcept("Dont have write permissions for file ",filename.str(),".");
                    }
                }
                
            }, "Overwrite file <i>"+filename/*.filename()*/.str()+"</i> ?", "Write configuration", "Yes", "No");
        } else
            print("Settings file ",filename.str()," already exists. To overwrite, please add the keyword 'force'.");
        
    } else {
        if(!filename.remove_filename().exists())
            filename.remove_filename().create_folder();
        
        FILE *f = fopen(filename.str().c_str(), "wb");
        if(f) {
            fwrite(text.data(), 1, text.length(), f);
            fclose(f);
            DebugCallback("Saved ", filename, ".");
        } else {
            FormatExcept("Cannot write file ",filename,".");
        }
    }
}

#if !COMMONS_NO_PYTHON
void GUI::training_data_dialog(GUIType type, bool force_load, std::function<void()> callback) {
    if(!py::python_available()) {
        auto message = py::python_available() ? "Recognition is not enabled." : "Python is not available. Check your configuration.";
        if(SETTING(auto_train_on_startup))
            throw U_EXCEPTION(message);
        
        FormatWarning(message);
        return;
    }
    
    if(FAST_SETTING(track_max_individuals) == 1) {
        FormatWarning("Are you sure you want to train on only one individual?");
        //callback();
        //return;
    }
    
    WorkProgress::add_queue("initializing python...", [this, type, force_load, callback]()
    {
        auto task = std::async(std::launch::async, [](){
            cmn::set_thread_name("async::ensure_started");
            try {
                //py::init().get();
                print("Initialization success.");
                
            } catch(...) {
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
                PD(gui).close_dialogs();
                
                std::string text;
#if defined(__APPLE__) && defined(__aarch64__)
                text = "Initializing Python failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow). Since you are using an ARM64 Mac, you may need to install additional libraries.";
#else
                text = "Initializing Python failed. Most likely one of the required libraries is missing from the current python environment (check for keras and tensorflow).";
#endif
                
                auto message = text + "Python says: "+python_init_error()+".";
                FormatExcept(message.c_str());
                
                if(!SETTING(nowindow)) {
#if defined(__APPLE__) && defined(__aarch64__)
                    std::string command = "pip install --upgrade --force --no-dependencies https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl https://github.com/apple/tensorflow_macos/releases/download/v0.1alpha3/tensorflow_addons_macos-0.1a3-cp38-cp38-macosx_11_0_arm64.whl";
                    
                    text += "\n<i>"+escape_html(python_init_error())+"</i>";
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
            }
        });
        //PythonIntegration::instance();
        
        bool before = PD(analysis).is_paused();
        PD(analysis).set_paused(true).get();
        
        DatasetQuality::update();
        
        try {
            generate_training_data(std::move(task), type, force_load);
        } catch(const SoftExceptionImpl& ex) {
            if(SETTING(auto_train_on_startup)) {
                throw U_EXCEPTION("Aborting training data because an exception was thrown (",std::string(ex.what()),").");
            } else
                print("Aborting training data because an exception was thrown (",std::string(ex.what()),").");
        }
        
        if(!before)
            SETTING(analysis_paused) = false;
        
        callback();
    });
}

void GUI::generate_training_data(std::future<void>&& initialized, GUI::GUIType type, bool force_load) {
    /*-------------------------/
     SAVE METADATA
     -------------------------*/
    static std::future<void> current;
    current = std::move(initialized);
    //! TODO: Dont do this.

    auto fn = [&](TrainingMode::Class load) -> bool {
        std::vector<Rangel> trained;

        WorkProgress::set_progress("training network", 0);
        WorkProgress::set_item_abortable(true);

        try {
            Accumulation acc(video_source(), (IMGUIBase*) best_base(), load);
            if(current.valid())
                current.get();

            auto ret = acc.start();
            if (ret && SETTING(auto_train_dont_apply)) {
                GUI::auto_quit();
            }

            return ret;

        }
        catch (const SoftExceptionImpl& error) {
            if (SETTING(auto_train_on_startup))
                throw U_EXCEPTION("The training process failed. Please check whether you are in the right python environment and check previous error messages.");

            if (!SETTING(nowindow) && GUI::instance())
                GUI::instance()->gui().dialog("The training process failed. Please check whether you are in the right python environment and check out this error message:\n\n<i>" + escape_html(error.what()) + "</i>", "Error");
            FormatError("The training process failed. Please check whether you are in the right python environment and check previous error messages.");
            return false;
        }
    };
    
    static constexpr const char message_concern[] = "Note that once visual identification succeeds, the entire video will be retracked and any previous <i>manual_matches</i> overwritten - you should save them by clicking <i>save config</i> (in the menu) prior to this. Further information is available at <i>trex.run/docs</i>.\n\nKeep in mind that automatically generated results should always be manually validated (at least in samples). Bad results are often indicated by long training times or by ending on uniqueness values below chance.";
    
    static constexpr const char message_no_weights[] = "<b>Training will start from scratch.</b>\nMake sure all of your individuals are properly tracked first, by setting parameters like <i>track_threshold</i>, <i>track_max_speed</i> and <i>blob_size_ranges</i> first. Always try to achieve a decent number of consecutively tracked frames for all individuals (at the same time), but avoid misassignments due to too wide parameter ranges. You may then click on <i>Start</i> below to start the process.";
    
    static constexpr const char message_weights_available[] = "<b>A network from a previous session is available.</b>\nYou can either <i>Continue</i> training (trying to improve training results further), simply <i>Apply</i> it to the video, or <i>Restart</i> training from scratch (this deletes the previous network).";
    
    const auto avail = py::VINetwork::weights_available();
    const std::string message = (avail ?
                std::string(message_weights_available)
            :   std::string(message_no_weights))
        + "\n\n" + std::string(message_concern);
    
    if(type == GUIType::GRAPHICAL) {
        PD(gui).dialog([fn, avail](Dialog::Result result) {
            WorkProgress::add_queue("training network", [fn, result, avail = avail]() {
                try {
                    TrainingMode::Class mode;
                    if(avail) {
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
                                throw SoftException("Unknown mode ",result," in generate_training_data.");
                                return;
                        }
                            
                    } else {
                        switch(result) {
                            case gui::Dialog::OKAY:
                                mode = TrainingMode::Restart;
                                break;
                            case gui::Dialog::ABORT:
                                return;
                                    
                            default:
                                throw SoftException("Unknown mode ",result," in generate_training_data.");
                                return;
                        }
                    }
                        
                    if(is_in(mode, TrainingMode::Continue, TrainingMode::Restart, TrainingMode::Apply))
                    {
                        print("Registering auto correct callback.");
                            
                        Accumulation::register_apply_callback([&](){
                            print("Finished. Auto correcting...");
                                
                            // TODO: MISSING
                            //Tracker::recognition()->check_last_prediction_accuracy();
                            if(!instance())
                                return;
                            
                            auto lock = GUI_LOCK(instance()->gui().lock());
                            instance()->auto_correct(GUI::GUIType::TEXT, true);
                        });
                    }
                        
                    fn(mode);
                        
                } catch(const SoftExceptionImpl& error) {
                    if(SETTING(auto_train_on_startup))
                        throw U_EXCEPTION("Initialization of the training process failed. Please check whether you are in the right python environment and check previous error messages.");
                    if(!SETTING(nowindow))
                        GUI::instance()->gui().dialog("Initialization of the training process failed. Please check whether you are in the right python environment and check out this error message:\n\n<i>"+escape_html(error.what())+"<i/>", "Error");
                    FormatError("Initialization of the training process failed. Please check whether you are in the right python environment and check previous error messages.");
                }
            });
                
        }, message, "Training mode", avail ? "Continue" : "Start", "Cancel", avail ? "Apply" : "", avail ? "Restart" : "", avail ? "Load weights" : "");
            
    } else {
        auto mode = TrainingMode::Restart;
        if(force_load)
            mode = TrainingMode::Apply;
        if(!fn(mode)) {
            if(SETTING(auto_train_on_startup))
                throw U_EXCEPTION("Using the network returned a bad code (false). See previous errors.");
        }
        if(!force_load)
            FormatWarning("Weights will not be loaded. In order to load weights add 'load' keyword after the command.");
    }
        
    /*} else {
        if(force_load)
            FormatWarning("Cannot load weights, as no previous weights exist.");
        
        work().add_queue("training network", [fn](){
            if(!fn(TrainingMode::Restart)) {
                if(SETTING(auto_train_on_startup))
                    throw U_EXCEPTION("Using the network returned a bad code (false). See previous errors.");
            }
        });
    }*/
}

void GUI::generate_training_data_faces(const file::Path& path) {
    LockGuard guard(ro_t{}, "GUI::generate_training_data_faces");
    WorkProgress::set_item("Generating data...");
    
    auto ranges = frameinfo().global_segment_order;
    auto range = ranges.empty() ? Range<Frame_t>({},{}) : ranges.front();
    
    if(!path.exists()) {
        if(path.create_folder())
            print("Created folder ", path.str(),".");
        else {
            FormatExcept("Cannot create folder ",path.str(),". Check permissions.");
            return;
        }
    }
    
    DebugCallback("Generating training dataset ", range," in folder ", path, ".");
    
    PPFrame pp;
    pv::Frame frame;
    
    std::vector<uchar> images;
    std::vector<float> heads;
    
    std::vector<uchar> unassigned_blobs;
    size_t num_unassigned_blobs = 0;
    
    Size2 output_size(200,200);
    
    if(!FAST_SETTING(calculate_posture))
        FormatWarning("Cannot normalize samples if no posture has been calculated.");
    
    size_t num_images = 0;
    
    for(auto i = range.start; i <= range.end; ++i)
    {
        if(not i.valid() || i >= PD(video_source).length()) {
            FormatExcept("Frame ", i," out of range.");
            continue;
        }
        
        WorkProgress::set_percent(i.try_sub(range.start).get() / (float)(range.end - range.start).get());
        
        PD(video_source).read_frame(frame, i);
        Tracker::instance()->preprocess_frame(std::move(frame), pp, nullptr, PPFrame::NeedGrid::NoNeed, video_source()->header().resolution);
        
        cv::Mat image, padded, mask;
        pp.transform_blobs([&](pv::Blob& blob){
            if(!PD(tracker).border().in_recognition_bounds(blob.center() * 0.5)) {
                print("Skipping ", blob.blob_id(),"@",i," because its out of bounds.");
                return;
            }
            
            auto recount = blob.recount(FAST_SETTING(track_threshold), *PD(tracker).background());
            if(recount < FAST_SETTING(blob_size_ranges).max_range().start) 
            {
                return;
            }
            
            imageFromLines(blob.hor_lines(), &mask, NULL, &image, blob.pixels().get(), 0, &Tracker::average(), 0);
            
            auto b = blob.bounds();
            b << output_size;
            
            Vec2 offset = (Size2(padded) - Size2(image)) * 0.5;
            
            offset.x = round(offset.x);
            offset.y = round(offset.y);
            
            b << b.pos() - offset;
            
            padded = cv::Mat::zeros(output_size.height, output_size.width, CV_8UC1);
            b.restrict_to(_average_image.bounds());
            
            //_average_image(b).copyTo(padded);//image(dims), mask(dims));
            b = blob.bounds();
            
            b.restrict_to(_average_image.bounds());
            
            Bounds p(blob.bounds());
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
                    throw U_EXCEPTION("Padded is not continous.");
                
                MotionRecord *found_head = NULL;
                
                IndividualManager::transform_ids(pp.previously_active_identities(), [&](auto, auto fish) {
                    auto fish_blob = fish->blob(i);
                    auto head = fish->head(i);
                    
                    if(fish_blob && fish_blob->blob_id() == blob.blob_id() && head) {
                        found_head = head;
                        return false;
                    }
                    
                    return true;
                });
                
                if(found_head) {
                    images.insert(images.end(), padded.data, padded.data + padded.cols * padded.rows);
                    
                    cv::circle(padded, found_head->pos<Units::PX_AND_SECONDS>() - b.pos() + offset, 2, cv::Scalar(255));
                    tf::imshow("padded", padded);
                    
                    heads.push_back(found_head->pos<Units::PX_AND_SECONDS>().x - b.x + offset.x);
                    heads.push_back(found_head->pos<Units::PX_AND_SECONDS>().y - b.y + offset.y);
                    ++num_images;
                } else if(num_unassigned_blobs < 1000) {
                    tf::imshow("unlabelled", padded);
                    unassigned_blobs.insert(unassigned_blobs.end(), padded.data, padded.data + padded.cols * padded.rows);
                    ++num_unassigned_blobs;
                }
                
            } else {
                auto prefix = SETTING(individual_prefix).value<std::string>();
                tf::imshow("too big", image);
                FormatWarning(prefix.c_str()," image too big (",image.cols,"x",image.rows,")");
            }
        });
    }
    
    /*-------------------------/
     SAVE METADATA
     -------------------------*/
    
    try {
        file::Path npz_path = path / "data.npz";
        cmn::npz_save(npz_path.str(), "range", std::vector<long_t>{ (long_t)range.start.get(), (long_t)range.end.get() });
        print("Saving ", num_images," positions...");
        cmn::npz_save(npz_path.str(), "positions", heads.data(), {num_images, 2}, "a");
        cmn::npz_save(npz_path.str(), "images", images.data(), {num_images, (size_t)output_size.height, (size_t)output_size.width}, "a");
        /*if(num_unassigned_blobs > 0) {
            print("Saving ", num_unassigned_blobs," unsorted images...");
            cmn::npz_save(npz_path.str(), "unsorted_images", unassigned_blobs.data(), {num_unassigned_blobs, (size_t)output_size.height, (size_t)output_size.width}, "a");
        }*/
        
        print("Saved ",num_unassigned_blobs," unsorted and "," sorted images to '",num_images,"'.");
    } catch(const std::runtime_error& e) {
        FormatExcept("Runtime error while saving to ",path.str()," (", e.what(),").");
    } catch(...) {
        throw U_EXCEPTION("Unknown error while saving to ",path.str());
    }
}
#endif

void GUI::add_manual_match(Frame_t frameIndex, Idx_t fish_id, pv::bid blob_id) {
    print("Requesting change of fish ", fish_id," to blob ", blob_id," in frame ",frameIndex);
    
    auto matches = FAST_SETTING(manual_matches);
    auto &current = matches[frameinfo().frameIndex.load()];
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
