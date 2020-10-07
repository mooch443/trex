#pragma once

#include <types.h>
#include <gui/GuiTypes.h>
#include <tracking/Individual.h>
#include <tracking/Tracker.h>
#include <tracking/ConfirmedCrossings.h>

class Timer;
namespace track { class Individual; }

namespace gui {
    struct SimpleBlob {
        pv::BlobPtr blob;
        int threshold;
        
        SimpleBlob(pv::BlobPtr b, int t);
        std::unique_ptr<ExternalImage> convert();
    };
    
    class Fish;
    
    using namespace track;
    
    class GUICache {
    public:
        int last_threshold;
        long_t last_frame;
        Bounds boundary;
        std::vector<idx_t> previous_active_fish;
        std::set<idx_t> previous_active_blobs, active_blobs;
        Vec2 previous_mouse_position;
        bool _dirty;
        FOIStatus _current_foi;
        size_t _num_pixels;

        long_t frame_idx;
        
        bool _calculating_pixel_percentiles;
        std::vector<float> pixel_value_percentiles;
        bool _equalize_histograms;
        
        GETTER(bool, blobs_dirty)
        GETTER(bool, raw_blobs_dirty)
        GETTER(mode_t::Class, mode)
        GETTER(double, gui_time)
        GETTER_SETTER(float, zoom_level)
        GETTER(float, dt)
        std::atomic_bool _tracking_dirty;
        
        std::map<Drawable*, Drawable::delete_function_handle_t> _delete_handles;
        std::set<Drawable*> _animators;
        
    public:
        bool recognition_updated;
        
        Rangel tracked_frames;
        std::atomic_bool connectivity_reload;
        
        std::map<uint32_t, long_t> automatic_assignments;
        
        std::unordered_map<idx_t, Individual*> individuals;
        std::set<idx_t> active_ids;
        std::set<idx_t> inactive_ids;
        std::set<idx_t> recognized_ids;
        std::map<idx_t, std::shared_ptr<gui::Circle>> recognition_circles;
        std::map<idx_t, Timer> recognition_timer;
        
        Tracker::set_of_individuals_t _registered_callback;
        
        std::map<idx_t, long_t> fish_selected_blobs;
        Tracker::set_of_individuals_t active;
        //std::vector<std::shared_ptr<gui::ExternalImage>> blob_images;
        std::vector<std::shared_ptr<SimpleBlob>> raw_blobs;
        std::unordered_map<pv::Blob*, gui::ExternalImage*> display_blobs;
        std::vector<std::unique_ptr<gui::ExternalImage>> display_blobs_list;
        std::vector<Vec2> inactive_estimates;
        
    protected:
        std::map<idx_t, std::map<uint32_t, Individual::Probability>> probabilities;
        std::set<idx_t> checked_probs;
        
    public:
        std::map<Individual*, gui::Fish*> _fish_map;
        std::map<long_t, track::Tracker::Statistics> _statistics;
        
        long_t connectivity_last_frame;
        std::vector<float> connectivity_matrix;
        
        PPFrame processed_frame;
        std::vector<idx_t> selected;
        
    public:
        bool has_selection() const;
        Individual * primary_selection() const;
        void deselect_all();
        bool is_selected(idx_t id) const;
        void do_select(idx_t id);
        
        void deselect(idx_t id);
        void deselect_all_select(idx_t id);
        
        const std::map<uint32_t, Individual::Probability>* probs(long_t fdx);
        bool has_probs(long_t fdx);
        
        void set_tracking_dirty();
        void set_blobs_dirty();
        void set_raw_blobs_dirty();
        void set_redraw();
        void set_animating(Drawable* obj, bool v);
        bool is_animating(Drawable* obj = nullptr) const;
        void set_dt(float dt);
        
        void set_mode(const gui::mode_t::Class&);
        
        void updated_tracking() { _tracking_dirty = false; }
        void updated_blobs() { _blobs_dirty = false; }
        void updated_raw_blobs() { _raw_blobs_dirty = false; }
        void on_redraw() { _dirty = false; }
        void update_data(long_t frameIndex);
        
        bool is_tracking_dirty() { return _tracking_dirty; }
        bool must_redraw() const;
    
        GUICache();
    };
}
