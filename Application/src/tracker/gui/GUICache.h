#pragma once

#include <commons.pc.h>
//#include <gui/GuiTypes.h>
//#include <gui/ConfirmedCrossings.h>
#include <gui/FramePreloader.h>
#include <misc/Buffers.h>
#include <tracker/misc/default_config.h>
#include <pv.h>
#include <misc/TrackingSettings.h>
#include <misc/ThreadPool.h>
#include <tracking/MotionRecord.h>
#include <processing/Background.h>
#include <misc/Border.h>
#include <tracking/Stuffs.h>
#include <gui/Event.h>
#include <gui/ShadowTracklet.h>
#include <tracking/IndividualCache.h>
#include <gui/BdxAndPred.h>
#include <misc/TimingStatsCollector.h>
#include <misc/DetectionTypes.h>

class Timer;
namespace track {
class Individual;
class PPFrame;
struct TrackletInformation;
namespace constraints {
struct FilterCache;
}
}

namespace cmn::gui {
class ExternalImage;
class DrawStructure;
class Circle;
class Drawable;
class PropertiesGraph;

namespace globals {
    CREATE_STRUCT(CachedGUIOptions,
        (bool, gui_show_outline),
        (bool, gui_show_midline),
        (gui::Color, gui_single_identity_color),
        (std::string, gui_fish_color),
        (bool, gui_show_boundary_crossings),
        (uchar, gui_faded_brightness),
        (bool, gui_show_probabilities),
        (bool, gui_show_shadows),
        (bool, gui_show_selections),
        (bool, gui_show_paths),
        (uint8_t, gui_outline_thickness),
        (bool, gui_show_texts),
        (float, gui_max_path_time),
        (std::string, gui_fish_label),
        (int, panic_button),
        (bool, gui_happy_mode),
        (bool, gui_highlight_categories),
        (bool, gui_show_cliques),
        (bool, gui_show_match_modes),
        (Frame_t, gui_pose_smoothing),
        (track::detect::KeypointNames, detect_keypoint_names)
    )

    #define GUIOPTION(NAME) ::cmn::gui::globals::CachedGUIOptions::copy < ::cmn::gui::globals::CachedGUIOptions :: NAME > ()

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
        (bool, gui_show_uniqueness),
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
        (Float2_t, gui_interface_scale),
        (default_config::output_format_t::Class, output_format),
        (uchar, gui_timeline_alpha),
        (bool, gui_happy_mode),
        (bool, auto_categorize),
        (bool, gui_macos_blur),
        (Size2, gui_zoom_limit),
        (blob::Pose::Skeleton, detect_skeleton),
        (std::vector<Vec2>, gui_zoom_polygon),
        (std::string, gui_foi_name),
        (bool, track_pause)
    )
}

#define GUI_SETTINGS(NAME) ::cmn::gui::globals::Cache::copy< ::cmn::gui::globals::Cache:: NAME >()

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
    class Posture;

    using namespace track;
    
    class GUICache {
        GETTER_NCONST(GenericThreadPool, pool);
        
        mutable std::shared_mutex _next_frame_cache_mutex, _tracklet_cache_mutex;
        std::unordered_map<Idx_t, IndividualCache> _next_frame_caches;
        std::unordered_map<Idx_t, std::tuple<bool, FrameRange>> _processed_tracklet_caches;
        std::unordered_map<Idx_t, std::shared_ptr<track::TrackletInformation>> _tracklet_caches;
        std::optional<ska::bytell_hash_map<pv::bid, std::vector<float>>> _current_predictions;

        struct PPFrameMaker {
            std::unique_ptr<PPFrame> operator()() const;
        };
        
        std::unique_ptr<PPFrame> _current_processed_frame;
        Buffers< std::unique_ptr<PPFrame>, PPFrameMaker > buffers;
        std::weak_ptr<pv::File> _video;
        gui::DrawStructure* _graph{ nullptr };
        std::unique_ptr<gui::Posture> _posture_window;
        using FramePtr = std::unique_ptr<PPFrame>;
        FramePreloader<FramePtr> _preloader;
        Timer _last_success;
        std::unique_ptr<PPFrame> _next_processed_frame;
        GETTER_SETTER(bool, load_frames_blocking){false};
        size_t _mistakes_count{0};
        
        LOGGED_MUTEX_VAR(vector_mutex, "GUICache::vector_mutex");
        
    public:
        Size2 _video_resolution;
        int last_threshold = -1;
        Bounds boundary;
        std::vector<Idx_t> previous_active_fish;
        std::set<pv::bid> previous_active_blobs, active_blobs, selected_blobs;
        Vec2 previous_mouse_position;
        bool _dirty = true;
        //FOIStatus _current_foi;
        size_t _num_pixels = 0;

        Frame_t frame_idx;
        std::optional<std::size_t> _delete_frame_callback;
        
        bool _frame_contained{false};
        std::optional<FrameProperties> _props;
        std::optional<FrameProperties> _next_props;
        
        std::vector<float> pixel_value_percentiles;
        bool _equalize_histograms = true;
        
        GETTER(std::vector<Range<Frame_t>>, global_tracklet_order);
        GETTER_I(bool, blobs_dirty, false);
        GETTER_I(bool, raw_blobs_dirty, false);
        GETTER(Frame_t, do_reload_frame);
        GETTER_SETTER_I(bool, fish_dirty, false);
        GETTER_I(mode_t::Class, mode, mode_t::tracking);
        GETTER_I(double, gui_time, 0);
        GETTER_SETTER_I(float, zoom_level, 1);
        GETTER_I(float, dt, 0);
        std::atomic_bool _tracking_dirty = false;
        
        GETTER_PTR(const Background*, background){nullptr};
        
        Timer _last_consecutive_update;
        std::atomic<bool> _updating_consecutive;
        std::future<std::vector<Range<Frame_t>>> _next_tracklet;
        
    public:
        bool recognition_updated = false;
        
        static GUICache& instance();
        static bool exists();
        Range<Frame_t> tracked_frames;
        std::atomic_bool connectivity_reload;
        
    private:
        mutable std::mutex individuals_mutex;
        std::unordered_map<Idx_t, Individual*> individuals;
        set_of_individuals_t _registered_callback;
        
    public:
        struct LockIndividuals {
            std::unique_lock<std::mutex> guard;
            const std::unordered_map<Idx_t, Individual*>& individuals;
            
            LockIndividuals(std::mutex& individuals_mutex, auto const& individuals)
                : guard(individuals_mutex), individuals(individuals)
            { }
            LockIndividuals(LockIndividuals&&) = default;
        };
        
        auto lock_individuals() const {
            return LockIndividuals{ individuals_mutex, individuals };
        }
        
        std::set<Idx_t> all_ids;
        std::set<Idx_t> active_ids;
        std::set<Idx_t> inactive_ids;
        std::set<Idx_t> recognized_ids;
        std::map<Idx_t, std::shared_ptr<gui::Circle>> recognition_circles;
        std::map<Idx_t, Timer> recognition_timer;
        std::unordered_map<Idx_t, std::vector<ShadowTracklet>> _individual_ranges;
        
        std::unordered_map<pv::bid, Idx_t> blob_selected_fish;
        std::map<Idx_t, BdxAndPred> fish_selected_blobs;
        std::map<Idx_t, Bounds> fish_last_bounds;
        std::map<Idx_t, std::shared_ptr<constraints::FilterCache>> filter_cache;
        set_of_individuals_t active;
        //std::vector<std::shared_ptr<gui::ExternalImage>> blob_images;
        std::vector<std::unique_ptr<SimpleBlob>> raw_blobs;
        std::unordered_map<pv::bid, SimpleBlob*> display_blobs;
        std::vector<std::unique_ptr<SimpleBlob>> available_blobs_list;
        std::vector<Vec2> inactive_estimates;
        
        ska::bytell_hash_map<Idx_t, std::map<Idx_t, float>> vi_predictions;
        ska::bytell_hash_map<Idx_t, ska::bytell_hash_map<pv::bid, DetailProbability>> probabilities;
        std::set<Idx_t> checked_probs;
        
        std::unordered_map<Idx_t, std::unique_ptr<PropertiesGraph>> _displayed_graphs;
        
    public:
        std::mutex _fish_map_mutex;
        std::unordered_map<Idx_t, std::unique_ptr<gui::Fish>> _fish_map;
        std::map<Frame_t, track::Statistics> _statistics;
        
        std::unordered_map<pv::bid, uint16_t> _ranged_blob_labels;
        std::unordered_map<pv::bid, uint16_t> _blob_labels;
        std::unordered_map<Idx_t, uint16_t> _individual_avg_categories;
        
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
        
        GETTER(Border, border){nullptr};
        
    public:
        bool has_selection() const;
        Individual * primary_selection() const;
        Idx_t primary_selected_id() const;
        void deselect_all();
        bool is_selected(Idx_t id) const;
        void do_select(Idx_t id);
        
        void deselect(Idx_t id);
        void deselect_all_select(Idx_t id);
        
        const ska::bytell_hash_map<pv::bid, DetailProbability>* probs(Idx_t fdx);
        bool has_probs(Idx_t fdx);
        
        void set_tracking_dirty();
        void set_blobs_dirty();
        void set_raw_blobs_dirty();
        void set_reload_frame(Frame_t);
        void set_redraw();
        void set_dt(float dt);
        
        void set_mode(const gui::mode_t::Class&);
        
        bool key_down(cmn::gui::Codes code) const;
        void updated_tracking() { _tracking_dirty = false; }
        void updated_blobs() { _blobs_dirty = false; }
        void updated_raw_blobs() { _raw_blobs_dirty = false; }
        void on_redraw() { _dirty = false; }
        Frame_t update_data(const Frame_t frameIndex);
        
        bool is_tracking_dirty() const { return _tracking_dirty; }
        bool must_redraw() const;
        
        bool something_important_changed(Frame_t) const;
        
        std::optional<const IndividualCache*> next_frame_cache(Idx_t) const;
        std::tuple<bool, FrameRange> processed_tracklet_cache(Idx_t) const;
        std::shared_ptr<track::TrackletInformation> tracklet_cache(Idx_t id) const;
        
        /// We can preload a pv::Frame here already, but not invalidate
        /// any of the actual data.
        void request_frame_change_to(Frame_t);
        
        const PPFrame& processed_frame() const {
            if(not _current_processed_frame)
                throw InvalidArgumentException("Cannot access processed_frame() since it is null.");
            return *_current_processed_frame;
        }
        
        const grid::ProximityGrid& blob_grid();
        
        GUICache(gui::DrawStructure*, std::weak_ptr<pv::File>);
        ~GUICache();
        
        std::optional<std::vector<float>> find_prediction(pv::bid) const;
        
        void draw_posture(gui::DrawStructure &base, Frame_t frameNr);
        std::optional<std::vector<Range<Frame_t>>> update_slow_tracker_stuff();
        
        void update_graphs(const Frame_t frameIndex);
    };
}

STRUCT_META_EXTENSIONS(cmn::gui::globals::Cache)

