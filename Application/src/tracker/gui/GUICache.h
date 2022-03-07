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
        std::unique_ptr<ExternalImage> ptr;
        
        SimpleBlob(std::unique_ptr<ExternalImage>&& available, pv::BlobPtr b, int t);
        void convert();
    };
    
    class Fish;
    
    using namespace track;
    
    class GUICache {
    public:
        int last_threshold = -1;
        Frame_t last_frame;
        Bounds boundary;
        std::vector<Idx_t> previous_active_fish;
        std::set<pv::bid> previous_active_blobs, active_blobs, selected_blobs;
        Vec2 previous_mouse_position;
        bool _dirty = true;
        FOIStatus _current_foi;
        size_t _num_pixels = 0;

        Frame_t frame_idx;
        
        std::vector<float> pixel_value_percentiles;
        bool _equalize_histograms = true;
        
        GETTER_I(bool, blobs_dirty, false)
        GETTER_I(bool, raw_blobs_dirty, false)
        GETTER_I(mode_t::Class, mode, mode_t::tracking)
        GETTER_I(double, gui_time, 0)
        GETTER_SETTER_I(float, zoom_level, 1)
        GETTER_I(float, dt, 0)
        std::atomic_bool _tracking_dirty = false;
        
        std::map<Drawable*, Drawable::delete_function_handle_t> _delete_handles;
        std::set<Drawable*> _animators;
        
    public:
        bool recognition_updated = false;
        
        Range<Frame_t> tracked_frames;
        std::atomic_bool connectivity_reload;
        
        ska::bytell_hash_map<Idx_t, Individual*> individuals;
        std::set<Idx_t> active_ids;
        std::set<Idx_t> inactive_ids;
        std::set<Idx_t> recognized_ids;
        std::map<Idx_t, std::shared_ptr<gui::Circle>> recognition_circles;
        std::map<Idx_t, Timer> recognition_timer;
        
        Tracker::set_of_individuals_t _registered_callback;
        
        std::map<Idx_t, pv::bid> fish_selected_blobs;
        Tracker::set_of_individuals_t active;
        //std::vector<std::shared_ptr<gui::ExternalImage>> blob_images;
        std::vector<std::unique_ptr<SimpleBlob>> raw_blobs;
        std::unordered_map<pv::Blob*, SimpleBlob*> display_blobs;
        std::vector<std::unique_ptr<SimpleBlob>> available_blobs_list;
        std::vector<Vec2> inactive_estimates;
        
    protected:
        ska::bytell_hash_map<Idx_t, ska::bytell_hash_map<pv::bid, Individual::Probability>> probabilities;
        std::set<uint32_t> checked_probs;
        
    public:
        ska::bytell_hash_map<Individual*, std::unique_ptr<gui::Fish>> _fish_map;
        std::map<Frame_t, track::Tracker::Statistics> _statistics;
        std::unordered_map<pv::bid, int> _ranged_blob_labels;
        
        std::vector<Tracker::Clique> _cliques;
        
        Frame_t connectivity_last_frame;
        std::vector<float> connectivity_matrix;
        
        PPFrame processed_frame;
        std::vector<Idx_t> selected;
        std::atomic<uint64_t> _current_pixels = 0;
        std::atomic<double> _average_pixels = 0;
        
    public:
        bool has_selection() const;
        Individual * primary_selection() const;
        void deselect_all();
        bool is_selected(Idx_t id) const;
        void do_select(Idx_t id);
        
        void deselect(Idx_t id);
        void deselect_all_select(Idx_t id);
        
        const ska::bytell_hash_map<pv::bid, Individual::Probability>* probs(Idx_t fdx);
        bool has_probs(Idx_t fdx);
        
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
        void update_data(Frame_t frameIndex);
        
        bool is_tracking_dirty() { return _tracking_dirty; }
        bool must_redraw() const;
    
        ~GUICache();
    };
}
