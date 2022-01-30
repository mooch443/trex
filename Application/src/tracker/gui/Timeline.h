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
        std::atomic_int current_fps{0};
        uint64_t video_length{0};
        std::atomic<Frame_t> frameIndex{Frame_t()};
        
        std::set<Range<Frame_t>> training_ranges;
        Range<Frame_t> analysis_range;
        
        float mx{0}, my{0};
        
        size_t small_count{0};
        uint32_t current_count{0};
        size_t big_count{0};
        size_t up_to_this_frame{0};
        
        size_t tdelta_gui{0};
        float tdelta{0};
        
        std::vector<Range<Frame_t>> global_segment_order;
        std::deque<Range<Frame_t>> consecutive;
    };
    
    using namespace track;
    
    class Timeline {
        //Size2 size;
        
        GETTER(std::unique_ptr<ExternalImage>, bar)
        GETTER(std::unique_ptr<ExternalImage>, consecutives)
        
        float tdelta;
        
        bool _visible;
        GETTER(Frame_t, mOverFrame)
        
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
        void reset_events(Frame_t after_frame = {});
        //void update_border();
        void next_poi(Idx_t fdx = Idx_t());
        void prev_poi(Idx_t fdx = Idx_t());
        static std::tuple<Vec2, float> timeline_offsets();
        
    private:
        friend class Interface;
        void update_fois();
        void update_consecs(float max_w, const Range<Frame_t>&, const std::vector<Range<Frame_t>>&, float scale);
        //void update_recognition_rect();
    };
}

#endif
