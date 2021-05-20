#ifndef _GUI_H
#define _GUI_H

#include <types.h>
#include <tracking/Posture.h>
#include <tracking/Tracker.h>
#include <gui/Graph.h>
#include <gui/DrawBase.h>
#include <gui/DrawSFBase.h>
#include <gui/DrawStructure.h>
#include <gui/DrawHTMLBase.h>
#include <gui/Timeline.h> 
#include <gui/types/Button.h>
#include <gui/types/List.h>
#include <gui/types/StaticText.h>
#include <gui/DrawGraph.h>
#include <gui/DrawPosture.h>
#include <gui/types/Histogram.h>
#include <gui/HttpGui.h>
#include <gui/types/Dropdown.h>
#include <gui/types/Layout.h>
#include <misc/ConnectedTasks.h>
#include <gui/GUICache.h>
#include <misc/default_config.h>

using namespace cmn;
using namespace track;

namespace gui {
namespace globals {

CREATE_STRUCT(Cache,
    (bool, gui_run),
    (gui::mode_t::Class, gui_mode),
    (bool, nowindow),
    (bool, auto_train),
    (bool, auto_apply),
    (bool, auto_quit),
    (long_t, gui_frame),
    (bool, terminate),
    (bool, gui_show_blobs),
    (bool, gui_show_paths),
    //(bool, gui_show_manual_matches),
    (bool, gui_show_texts),
    (bool, gui_show_selections),
    (bool, gui_show_inactive_individuals),
    (bool, gui_show_outline),
    (bool, gui_show_midline),
    (bool, gui_show_posture),
    (bool, gui_show_heatmap),
    (bool, gui_show_number_individuals),
    (bool, gui_show_dataset),
    (bool, gui_show_recognition_summary),
    (bool, gui_show_visualfield),
    (bool, gui_show_visualfield_ts),
    (bool, gui_show_export_options),
    (bool, gui_show_recognition_bounds),
    (bool, gui_show_midline_histogram),
    (bool, gui_show_histograms),
    (bool, gui_auto_scale),
    (bool, gui_auto_scale_focus_one),
    (size_t, output_min_frames),
    (Color, gui_background_color),
    (bool, gui_equalize_blob_histograms),
    (float, gui_playback_speed),
    (int, frame_rate),
    (float, gui_interface_scale),
    (default_config::output_format_t::Class, output_format),
    (uchar, gui_timeline_alpha)
#ifdef TREX_ENABLE_EXPERIMENTAL_BLUR
    ,(bool, gui_blur_enabled)
#endif
)

}
}

#define GUI_SETTINGS(NAME) gui::globals::Cache::copy< gui::globals::Cache:: NAME >()

namespace pv {
    class Frame;
}

namespace gui {
    class DrawDataset;
    class WorkProgress;
}

class GUI {
public:
    //! For executing commands from console as well
    enum GUIType {
        GRAPHICAL,
        TEXT
    };
    
    static GUI *_instance;
    static GUI* instance();
    static Vec2 pad_image(cv::Mat& padded, Size2 output_size);
    static std::vector<gui::Drawable*>& static_pointers();

private:
    //! Saved reference to the average image.
    Image _average_image;
    
    GETTER_NCONST(gui::DrawStructure, gui)
    
    cv::VideoWriter *_recording_capture;
    cv::Size _recording_size;
    file::Path _recording_path;
    default_config::gui_recording_format_t::Class _recording_format;
    GETTER(bool, recording)
    
    gui::GUICache _cache;
    long_t _recording_start;
    long_t _recording_frame;
    long_t _last_recording_frame;
    
    long_t _flow_frame;
    long_t _next_frame;
    cv::Mat _cflow, _cflow_next;
    
    float _tdelta_gui;
    long_t _gui_last_frame;
    Timer _gui_last_frame_timer;
    
    Timer _gui_last_backup;
    
    //! Reference to the Tracker
    Tracker &_tracker;
    
    GETTER_SETTER_PTR(ConnectedTasks*, analysis)
    
    //! Timer for determining speed of increase by keypress
    Timer last_increase_timer, last_direction_change;
    
    GETTER_NCONST(bool, direction_change)
    GETTER_NCONST(int, play_direction)

    GETTER_PTR(pv::File*, video_source)
    bool _real_update;
    
    GETTER_PTR(gui::Base*, base)
    GETTER_NCONST(GenericThreadPool, blob_thread_pool)
    GETTER_NCONST(std::mutex, blob_thread_pool_mutex)
    
    GETTER(bool, properties_visible)
    std::shared_ptr<gui::DrawDataset> _dataset;
    
    GETTER_NCONST(gui::FrameInfo, frameinfo)
#if WITH_MHD
    gui::HttpGui* _http_gui;
#endif
    
    std::shared_ptr<gui::Timeline> _timeline;
public:
    const gui::Timeline& timeline() const {
        assert(_timeline);
        return *_timeline;
    }
protected:
    std::vector<gui::PropertiesGraph*> _fish_graphs;
    gui::Posture _posture_window;
    gui::Histogram<float, gui::Hist::Options::NORMED> _histogram;
    gui::Histogram<float, gui::Hist::Options::NORMED> _midline_histogram;
    gui::Histogram<size_t, gui::Hist::Options::NORMED, std::vector<std::map<long_t, size_t>>> _length_histogram;
    
    gui::StaticText _info;
    
    /*struct SettingAnimation {
        std::shared_ptr<gui::Entangled> display;
        std::string name;
        Timer timer;
        Vec2 position;
        
        SettingAnimation() {}
        
    } _setting_animation;*/
    
    gui::WorkProgress *_work_progress;
    
    gui::ExternalImage _recognition_image;
    std::map<std::vector<Vec2>, std::shared_ptr<gui::Drawable>> _ignore_shapes, _include_shapes;
    
    GETTER_SETTER(bool, info_visible)
    
    Timer last_frame_change;
    
    GETTER(std::atomic<long_t>, clicked_blob_id)
    GETTER(std::atomic<long_t>, clicked_blob_frame)
    
public:
    void set_clicked_blob_id(long_t v) { _clicked_blob_id = v; }
    void set_clicked_blob_frame(long_t v) { _clicked_blob_frame = v; }
    
public:
    GUI(pv::File &video_source, const Image& average, Tracker& tracker);
    ~GUI();
    
    static gui::WorkProgress& work();
    
    static inline void event(const gui::Event& event) {
        instance()->local_event(event);
    }
    static inline const Image& background_image() {
        return instance()->_average_image;
    }
    
    void set_redraw();
    void set_mode(gui::mode_t::Class mode);
    gui::mode_t::Class mode() const;
    
    bool has_window() { return _base != NULL; }
    
    bool run() const;
    void run(bool r);
    
    void set_base(gui::Base* base);
    
    static long_t frame();
    //static inline sprite::Property<long_t>& frame_ref() { return GUI::current_frame; }
    
    static bool execute_settings(file::Path, AccessLevelType::Class);
    static void reanalyse_from(long_t frame, bool in_thread = true);
    
    static void trigger_redraw();
    
    static std::string info(bool escape);
    static void auto_apply();
    static void auto_train();
    static void auto_quit();
    
    static gui::GUICache& cache();
    
    void write_config(bool overwrite, GUIType type = GUIType::GRAPHICAL, const std::string& suffix = "");
    
    void run_loop(gui::LoopStatus);
    void export_tracks(const file::Path& prefix = "", long_t fdx = -1, Rangel range = Rangel());
    void save_visual_fields();
    void load_connectivity_matrix();
    
    void toggle_fullscreen();
    void open_docs();
    
    std::string window_title() const;
    
private:
    void draw_raw(gui::DrawStructure&, long_t frameIndex);
    void draw_tracking(gui::DrawStructure& main_base, long_t frameNr, bool draw_graph = true);
    void draw(gui::DrawStructure& main_base);
    void draw_footer(gui::DrawStructure& base);
    void draw_posture(gui::DrawStructure &base, Individual* fish, long_t frameNr);
    void draw_menu(gui::DrawStructure& base);
    void draw_export_options(gui::DrawStructure& base);
    void draw_grid(gui::DrawStructure& base);
    
    void removed_frames(long_t including);
    
    void debug_binary(gui::DrawStructure& main_base, long_t frameIndex);
    void debug_optical_flow(gui::DrawStructure& base, long_t frameIndex);
    void redraw();
    
    void key_event(const gui::Event& event);
    void local_event(const gui::Event& event);
    
    void generate_training_data(GUIType type = GUIType::GRAPHICAL, bool force_load = false);
    std::map<long_t, std::set<long_t>> generate_individuals_per_frame(const Rangel& range, TrainingData& data);
    
    void generate_training_data_faces(const file::Path& path);
    std::map<long_t, long_t> check_additional_range(const Rangel& range, TrainingData& data);
    
public:
    void add_manual_match(long_t frameIndex, Idx_t fish_id, long_t blob_id);
    
private:
    void selected_setting(long_t index, const std::string& name, gui::Textfield& textfield, gui::Dropdown& settings_dropdown, gui::Layout& layout, gui::DrawStructure& base);
    
    void start_recording();
public:
    void do_recording();
    bool is_recording() const;
private:
    void stop_recording();
    
public:
    void training_data_dialog(GUIType type = GUIType::GRAPHICAL, bool force_load = false, std::function<void()> = [](){});
    
    void confirm_terminate();
    
    void update_backups();
    void start_backup();
    
    void load_state(GUIType type, file::Path from = "");
    void save_state(GUIType type = GUIType::GRAPHICAL, bool force_overwrite = false);
    
    void auto_correct(GUIType type = GUIType::GRAPHICAL, bool force_correct = false);
    
    file::Path frame_output_dir() const;
    
    void update_recognition_rect();
    static Size2 screen_dimensions();
    static gui::Base* best_base();
    static void set_status(const std::string& text);
    
private:
    std::tuple<Vec2, Vec2> gui_scale_with_boundary(Bounds& bounds, gui::Section* section, bool singular);
    std::function<void(const Vec2&, bool, std::string)> _clicked_background;
    std::vector<std::vector<Vec2>> _current_boundary;
    enum class SelectedSettingType {
        ARRAY_OF_BOUNDS,
        ARRAY_OF_VECTORS,
        POINTS,
        NONE
    } _selected_setting_type;
    std::string _selected_setting_name;
    
    friend class DrawMenuPrivate;
    
    void update_display_blobs(bool draw_blobs, gui::Section*);
};

#endif
