#ifndef _GUI_H
#define _GUI_H

/*#include <types.h>
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
#include <misc/default_config.h>*/

#include <misc/defines.h>
#include <misc/create_struct.h>
#include <misc/default_config.h>
#include <gui/Timeline.h>
#include <gui/colors.h>
#include <gui/DrawStructure.h>
#include <misc/PVBlob.h>
#include <misc/ThreadPool.h>
#include <gui/DrawBase.h>

using namespace cmn;
using namespace track;

namespace cmn {
    class ConnectedTasks;
}

namespace gui {
    class GUICache;
    class Textfield;
    class Dropdown;
    class PropertiesGraph;

namespace globals {

CREATE_STRUCT(Cache,
    (bool, gui_run),
    (gui::mode_t::Class, gui_mode),
    (bool, nowindow),
    (bool, auto_train),
    (bool, auto_apply),
    (bool, auto_quit),
    (Frame_t, gui_frame),
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
    (uint16_t, output_min_frames),
    (Color, gui_background_color),
    (bool, gui_equalize_blob_histograms),
    (float, gui_playback_speed),
    (int, frame_rate),
    (float, gui_interface_scale),
    (default_config::output_format_t::Class, output_format),
    (uchar, gui_timeline_alpha),
    (bool, gui_happy_mode),
    (bool, auto_categorize)
)
//(bool, gui_blur_enabled)

}
}

#define GUI_SETTINGS(NAME) gui::globals::Cache::copy< gui::globals::Cache:: NAME >()

namespace pv {
    class Frame;
    class File;
}

namespace track {
    class Tracker;
    class Individual;
    class TrainingData;
}

namespace gui {
    class DrawDataset;
    class WorkProgress;
    class Drawable;
}

struct PrivateData;

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
    
    GETTER_NCONST(bool, direction_change)
    GETTER_NCONST(int, play_direction)

    GETTER_PTR(gui::Base*, base)
    GETTER_NCONST(GenericThreadPool, blob_thread_pool)
    GETTER_NCONST(std::mutex, blob_thread_pool_mutex)

    GETTER(bool, properties_visible)
    PrivateData* _private_data = nullptr;

    gui::FrameInfo _frameinfo;
#if WITH_MHD
    gui::HttpGui* _http_gui;
#endif

public:
    static const gui::Timeline& timeline();
    static pv::File* video_source();

protected:    
    GETTER_SETTER(bool, info_visible)
    
public:
    GUI(pv::File &video_source, const Image& average, Tracker& tracker);
    ~GUI();

    static bool recording();
    static gui::WorkProgress& work();
    static ConnectedTasks* analysis();
    static void set_analysis(ConnectedTasks*);
    static gui::FrameInfo& frameinfo();
    
    static inline void event(const gui::Event& event) {
        instance()->local_event(event);
    }
    static inline const Image& background_image() {
        return instance()->_average_image;
    }
    static inline const Image& average() { return background_image(); }

    static void set_redraw();
    static void set_mode(gui::mode_t::Class mode);
    gui::mode_t::Class mode() const;
    
    bool has_window() { return _base != NULL; }
    
    static bool run();
    static void run(bool r);
    static void run_loop(gui::LoopStatus status);

    static gui::DrawStructure& gui();
    
    void set_base(gui::Base* base);
    
    static Frame_t frame();
    //static inline sprite::Property<long_t>& frame_ref() { return GUI::current_frame; }
    
    static bool execute_settings(file::Path, AccessLevelType::Class);
    static void reanalyse_from(Frame_t frame, bool in_thread = true);
    
    static void trigger_redraw();
    
    static std::string info(bool escape);
    static void auto_apply();
    static void auto_train();
    static void auto_quit();
    static void auto_categorize();
    
    static gui::GUICache& cache();
    
    static void write_config(bool overwrite, GUIType type = GUIType::GRAPHICAL, const std::string& suffix = "");
    
    void export_tracks(const file::Path& prefix = "", long_t fdx = -1, Range<Frame_t> range = Range<Frame_t>({}, {}));
    void save_visual_fields();
    void load_connectivity_matrix();
    
    void toggle_fullscreen();
    void open_docs();
    
    std::string window_title() const;
    
private:
    void draw_raw(gui::DrawStructure&, Frame_t frameIndex);
    void draw_tracking(gui::DrawStructure& main_base, Frame_t frameNr, bool draw_graph = true);
    void draw(gui::DrawStructure& main_base);
    void draw_footer(gui::DrawStructure& base);
    void draw_posture(gui::DrawStructure &base, Individual* fish, Frame_t frameNr);
    void draw_menu();
    void draw_export_options(gui::DrawStructure& base);
    void draw_grid(gui::DrawStructure& base);
    
    void removed_frames(Frame_t including);
    
    void draw_raw_mode(gui::DrawStructure& main_base, Frame_t frameIndex);
    void debug_optical_flow(gui::DrawStructure& base, Frame_t frameIndex);
    
    void key_event(const gui::Event& event);
    void local_event(const gui::Event& event);
    
    void generate_training_data(GUIType type = GUIType::GRAPHICAL, bool force_load = false);
    std::map<Frame_t, std::set<long_t>> generate_individuals_per_frame(const Rangel& range, TrainingData& data);
    
    void generate_training_data_faces(const file::Path& path);
    std::map<Frame_t, long_t> check_additional_range(const Range<Frame_t>& range, TrainingData& data);
    
public:
    void add_manual_match(Frame_t frameIndex, Idx_t fish_id, pv::bid blob_id);
    static void redraw();

private:
    void selected_setting(long_t index, const std::string& name, gui::Textfield& textfield, gui::Dropdown& settings_dropdown, gui::Layout& layout, gui::DrawStructure& base);

public:
    void do_recording();
    bool is_recording() const;
    static void stop_recording();
    static void start_recording();

public:
    void training_data_dialog(GUIType type = GUIType::GRAPHICAL, bool force_load = false, std::function<void()> = [](){});
    void confirm_terminate();
    
    static void update_backups();
    static void start_backup();
    
    void load_state(GUIType type, file::Path from = "");
    void save_state(GUIType type = GUIType::GRAPHICAL, bool force_overwrite = false);
    
    void auto_correct(GUIType type = GUIType::GRAPHICAL, bool force_correct = false);
    
    static file::Path frame_output_dir();
    
    void update_recognition_rect();
    static Size2 screen_dimensions();
    static gui::Base* best_base();
    static void set_status(const std::string& text);
    
private:
    std::tuple<Vec2, Vec2> gui_scale_with_boundary(Bounds& bounds, gui::Section* section, bool singular);
    
    friend class DrawMenuPrivate;
    
    void update_display_blobs(bool draw_blobs, gui::Section*);
};

#endif
