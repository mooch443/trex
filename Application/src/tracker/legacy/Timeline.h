#ifndef _TIMELINE_H
#define _TIMELINE_H

#include <commons.pc.h>
#include <misc/ranges.h>
#include <misc/idx_t.h>
#include <misc/Image.h>

class GUI;

namespace cmn::gui {
    

    class DrawStructure;
    class ExternalImage;
    class Base;
    
    struct FrameInfo {
        std::atomic_int current_fps{0};
        uint64_t video_length{0};
        std::atomic<Frame_t> frameIndex{Frame_t()};
        
        std::set<Range<Frame_t>> training_ranges;
        FrameRange analysis_range;
        
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
        static auto& mutex() {
            static auto _mutex = new LOGGED_MUTEX("Timeline::mutex");
            return *_mutex;
        }
        static Timeline& instance();
        std::atomic<bool> foi_update_scheduled{false};
        std::atomic<float> use_scale{1.f};
        std::atomic<Size2> bar_size;
        std::atomic<float> timeline_max_w;
        std::atomic<Vec2> timeline_offset;
        
        std::mutex bar_mutex;
        Image::Ptr bar_image;
        
    public:
        static auto& frame_info_mutex() {
            static auto _frame_info_mutex = new LOGGED_MUTEX("Timeline::_frame_info_mutex");
            return *_frame_info_mutex;
        }
        
    protected:
        GETTER(std::unique_ptr<ExternalImage>, bar);
        GETTER(std::unique_ptr<ExternalImage>, consecutives);
        
        float tdelta;
        
        bool _visible;
        GETTER(Frame_t, mOverFrame);

        GETTER_SETTER_PTR_I(Base*, base, nullptr)
        GETTER(std::atomic_bool, update_thread_updated_once);
        std::function<void()> _updated_recognition_rect;
        std::function<void(bool)> _hover_status_text;
        
    protected:
        
        //Image _border_distance;
        
    public:
        Timeline(Base *, std::function<void(bool)> hover_status, std::function<void()> update_recognition, FrameInfo& info);
        ~Timeline();
        void draw(DrawStructure& window);
        
        static bool visible();
        static void set_visible(bool v);
        
        void update_thread();
        void reset_events(Frame_t after_frame = {});
        //void update_border();
        void next_poi(Idx_t fdx = Idx_t());
        void prev_poi(Idx_t fdx = Idx_t());
        static std::tuple<Vec2, float> timeline_offsets(Base*);
        
    private:
        friend struct Interface;
        void update_fois();
        void update_consecs(float max_w, const Range<Frame_t>&, const std::vector<Range<Frame_t>>&, float scale);
        //void update_recognition_rect();
    };
}

#endif
