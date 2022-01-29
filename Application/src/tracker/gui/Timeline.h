#ifndef _TIMELINE_H
#define _TIMELINE_H

#include <misc/defines.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>
#include <misc/vec2.h>

class GUI;

namespace gui {
    using namespace cmn;

    class DrawStructure;
    class ExternalImage;
    
    struct FrameInfo {
        std::atomic_int current_fps;
        uint64_t video_length;
        std::atomic_long frameIndex;
        
        std::set<Rangel> training_ranges;
        Rangel analysis_range;
        
        float mx, my;
        
        size_t small_count;
        uint32_t current_count;
        size_t big_count;
        size_t up_to_this_frame;
        
        size_t tdelta_gui;
        float tdelta;
        
        std::vector<Range<long_t>> global_segment_order;
        std::deque<Range<long_t>> consecutive;
        
        FrameInfo() : current_fps(0), video_length(0), frameIndex(0), mx(0), my(0), small_count(0), current_count(0), big_count(0), up_to_this_frame(0), tdelta_gui(0), tdelta(0) {}
    };
    
    using namespace track;
    
    class Timeline {
        //Size2 size;
        
        GETTER(std::unique_ptr<ExternalImage>, bar)
        GETTER(std::unique_ptr<ExternalImage>, consecutives)
        
        float tdelta;
        
        bool _visible;
        GETTER(long_t, mOverFrame)
        
        GETTER(std::atomic_bool, update_thread_updated_once)
        
        
        
    protected:
        
        //Image _border_distance;
        
    public:
        Timeline(GUI& gui, FrameInfo& info);
        ~Timeline();
        void draw(DrawStructure& window);
        
        bool visible() const { return _visible; }
        void set_visible(bool v);
        
        void update_thread();
        void reset_events(long_t after_frame = -1);
        //void update_border();
        void next_poi(Idx_t fdx = Idx_t());
        void prev_poi(Idx_t fdx = Idx_t());
        static std::tuple<Vec2, float> timeline_offsets();
        
    private:
        friend class Interface;
        void update_fois();
        void update_consecs(float max_w, const Range<long_t>&, const std::vector<Rangel>&, float scale);
        //void update_recognition_rect();
    };
}

#endif
