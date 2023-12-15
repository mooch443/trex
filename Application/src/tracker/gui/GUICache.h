#pragma once

#include <commons.pc.h>
#include <gui/GuiTypes.h>
#include <tracking/ConfirmedCrossings.h>
#include <gui/FramePreloader.h>
#include <misc/Buffers.h>
#include <tracker/misc/default_config.h>
#include <pv.h>
#include <tracking/TrackingSettings.h>
#include <misc/ThreadPool.h>

class Timer;
namespace track {
class Individual;
class PPFrame;
}

namespace gui {
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
        (bool, gui_show_timeline),
        (uint16_t, output_min_frames),
        (gui::Color, gui_background_color),
        (bool, gui_equalize_blob_histograms),
        (float, gui_playback_speed),
        (uint32_t, frame_rate),
        (float, gui_interface_scale),
        (default_config::output_format_t::Class, output_format),
        (uchar, gui_timeline_alpha),
        (bool, gui_happy_mode),
        (bool, auto_categorize),
        (bool, gui_macos_blur),
        (Size2, gui_zoom_limit),
        (blob::Pose::Skeleton, meta_skeleton)
    )
}

#define GUI_SETTINGS(NAME) ::gui::globals::Cache::copy< ::gui::globals::Cache:: NAME >()

    struct SimpleBlob {
        pv::BlobWeakPtr blob;
        int threshold;
        std::unique_ptr<ExternalImage> ptr;
        Vec2 image_pos;
        Frame_t frame;
        
        SimpleBlob(std::unique_ptr<ExternalImage>&& available, pv::BlobWeakPtr b, int t);
        void convert();
    };
    
    class Fish;
    
    using namespace track;
    
    class GUICache {
        GenericThreadPool _pool;

        struct PPFrameMaker {
            std::unique_ptr<PPFrame> operator()() const;
        };
        
        std::unique_ptr<PPFrame> _current_processed_frame;
        Buffers< std::unique_ptr<PPFrame>, PPFrameMaker > buffers;
        pv::File* _video{ nullptr };
        gui::DrawStructure* _graph{ nullptr };
        using FramePtr = std::unique_ptr<PPFrame>;
        FramePreloader<FramePtr> _preloader;
        Timer _last_success;
        std::unique_ptr<PPFrame> _next_processed_frame;
        
        LOGGED_MUTEX_VAR(vector_mutex, "GUICache::vector_mutex");

    public:
        Size2 _video_resolution;
        int last_threshold = -1;
        Bounds boundary;
        std::vector<Idx_t> previous_active_fish;
        std::set<pv::bid> previous_active_blobs, active_blobs, selected_blobs;
        Vec2 previous_mouse_position;
        bool _dirty = true;
        FOIStatus _current_foi;
        size_t _num_pixels = 0;

        Frame_t frame_idx;
        
        bool _frame_contained{false};
        
        std::vector<float> pixel_value_percentiles;
        bool _equalize_histograms = true;
        
        GETTER(std::vector<Range<Frame_t>>, global_segment_order);
        GETTER_I(bool, blobs_dirty, false);
        GETTER_I(bool, raw_blobs_dirty, false);
        GETTER_I(bool, fish_dirty, false);
        GETTER_I(mode_t::Class, mode, mode_t::tracking);
        GETTER_I(double, gui_time, 0);
        GETTER_SETTER_I(float, zoom_level, 1);
        GETTER_I(float, dt, 0);
        std::atomic_bool _tracking_dirty = false;
        
        std::unordered_map<std::string_view, gui::Drawable*> _animator_map;
        std::unordered_map<gui::Drawable*, Drawable::delete_function_handle_t> _delete_handles;
        GETTER(std::set<std::string_view>, animators);
        
    public:
        bool recognition_updated = false;

        static GUICache& instance();
        static bool exists();
        Range<Frame_t> tracked_frames;
        std::atomic_bool connectivity_reload;
        
        std::unordered_map<Idx_t, Individual*> individuals;
        std::set<Idx_t> active_ids;
        std::set<Idx_t> inactive_ids;
        std::set<Idx_t> recognized_ids;
        std::map<Idx_t, std::shared_ptr<gui::Circle>> recognition_circles;
        std::map<Idx_t, Timer> recognition_timer;
        
        set_of_individuals_t _registered_callback;
        
        std::map<Idx_t, pv::bid> fish_selected_blobs;
        set_of_individuals_t active;
        //std::vector<std::shared_ptr<gui::ExternalImage>> blob_images;
        std::vector<std::unique_ptr<SimpleBlob>> raw_blobs;
        std::unordered_map<pv::bid, SimpleBlob*> display_blobs;
        std::vector<std::unique_ptr<SimpleBlob>> available_blobs_list;
        std::vector<Vec2> inactive_estimates;
        
        ska::bytell_hash_map<Idx_t, ska::bytell_hash_map<pv::bid, Probability>> probabilities;
        std::set<Idx_t> checked_probs;
        
    public:
        std::mutex _fish_map_mutex;
        std::unordered_map<Idx_t, std::unique_ptr<gui::Fish>> _fish_map;
        std::map<Frame_t, track::Statistics> _statistics;
        std::unordered_map<pv::bid, int> _ranged_blob_labels;
        
        std::vector<track::Clique> _cliques;
        
        Frame_t connectivity_last_frame;
        std::vector<float> connectivity_matrix;
        
        std::vector<Idx_t> selected;
        cmn::atomic<uint64_t> _current_pixels = 0;
        std::atomic<double> _average_pixels = 0;
        
    protected:
        std::unique_ptr<std::thread> percentile_ptr;
        std::mutex percentile_mutex;
        std::once_flag _percentile_once;
        std::atomic<bool> done_calculating{false};
        
    public:
        bool has_selection() const;
        Individual * primary_selection() const;
        Idx_t primary_selected_id() const;
        void deselect_all();
        bool is_selected(Idx_t id) const;
        void do_select(Idx_t id);
        
        void deselect(Idx_t id);
        void deselect_all_select(Idx_t id);
        
        const ska::bytell_hash_map<pv::bid, Probability>* probs(Idx_t fdx);
        bool has_probs(Idx_t fdx);
        
        void set_tracking_dirty();
        void set_blobs_dirty();
        void set_raw_blobs_dirty();
        void set_redraw();
        void set_animating(std::string_view, bool v, Drawable* = nullptr);
        void clear_animators();
        bool is_animating(std::string_view = {}) const;
        void set_dt(float dt);
        
        void set_mode(const gui::mode_t::Class&);
        
        void updated_tracking() { _tracking_dirty = false; }
        void updated_blobs() { _blobs_dirty = false; }
        void updated_raw_blobs() { _raw_blobs_dirty = false; }
        void on_redraw() { _dirty = false; }
        Frame_t update_data(Frame_t frameIndex);
        
        bool is_tracking_dirty() const { return _tracking_dirty; }
        bool must_redraw() const;
        
        bool something_important_changed(Frame_t) const;
        
        /// We can preload a pv::Frame here already, but not invalidate
        /// any of the actual data.
        void request_frame_change_to(Frame_t);
        
        const PPFrame& processed_frame() const {
            return *_current_processed_frame;
        }
        
        const grid::ProximityGrid& blob_grid();
        
        GUICache(gui::DrawStructure*, pv::File*);
        ~GUICache();
    };
}

STRUCT_META_EXTENSIONS(gui::globals::Cache)

