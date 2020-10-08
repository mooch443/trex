#ifndef _TIMELINE_H
#define _TIMELINE_H

#include <types.h>
#include <gui/DrawObject.h>
#include <tracking/Tracker.h>
#include <gui/DrawCVBase.h>
#include <misc/Results.h>
#include <misc/EventAnalysis.h>
#include <gui/types/Layout.h>
#include <gui/types/Button.h>

class GUI;

namespace gui {
    
    
    struct FrameInfo {
        std::atomic_int current_fps;
        long_t video_length;
        std::atomic_long frameIndex;
        
        std::set<Rangel> training_ranges;
        Rangel analysis_range;
        
        float mx, my;
        
        size_t small_count;
        track::idx_t current_count;
        size_t big_count;
        size_t up_to_this_frame;
        
        size_t tdelta_gui;
        float tdelta;
        
        std::vector<Range<long_t>> global_segment_order;
        std::deque<Range<long_t>> consecutive;
        
        FrameInfo() : current_fps(0), video_length(0), frameIndex(0), mx(0), my(0), small_count(0), current_count(0), big_count(0), up_to_this_frame(0), tdelta_gui(0), tdelta(0) {}
    };
    
    using namespace track;
    
    class Timeline : Object {
        Vec2 pos;
        //Size2 size;
        
        GETTER(std::unique_ptr<ExternalImage>, bar)
        GETTER(std::unique_ptr<ExternalImage>, consecutives)
        
        std::atomic<long_t> tracker_endframe, tracker_startframe;
        float tdelta;
        
        GUI& _gui;
        Tracker& _tracker;
        FrameInfo& _frame_info;
        
        bool _visible;
        GETTER(long_t, mOverFrame)
        
        GETTER(std::atomic_bool, update_thread_updated_once)
        
        std::string _thread_status;
        std::atomic_bool _terminate;
        
        HorizontalLayout _title_layout;
        Text _status_text, _status_text2, _status_text3;
        Text _raw_text, _auto_text;
        Button _pause;
        
        std::shared_ptr<std::thread> _update_events_thread;
        
        struct {
            uint64_t last_change;
            FOI::foi_type::mapped_type changed_frames;
            std::string name;
            Color color;
        } _foi_state;
        
        
    protected:
        //! NeighborDistances drawn out
        struct ProximityBar {
            Size2 _dimensions;
            long_t start, end;
            std::map<long_t, std::set<FOI::fdx_t>> changed_frames;
            std::vector<uint32_t> samples_per_pixel;
            std::mutex mutex;
            
            ProximityBar() : start(-1), end(-1) {}
        } _proximity_bar;
        
        //Image _border_distance;
        
    public:
        Timeline(GUI& gui, FrameInfo& info);
        ~Timeline() {
            _terminate = true;
            _update_events_thread->join();
        }
        void draw(DrawStructure& window) override;
        
        bool visible() const { return _visible; }
        void set_visible(bool v);
        
        void update_thread();
        void reset_events(long_t after_frame = -1);
        //void update_border();
        void next_poi(long_t fdx = -1);
        void prev_poi(long_t fdx = -1);
        static std::tuple<Vec2, float> timeline_offsets();
        
    private:
        void update_fois();
        void update_consecs(float max_w, const Range<long_t>&, const std::vector<Rangel>&, float scale);
        //void update_recognition_rect();
    };
}

#endif
