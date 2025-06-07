#ifndef _DRAW_POSTURE_H
#define _DRAW_POSTURE_H

#include <commons.pc.h>
#include <pv.h>
#include <gui/types/Entangled.h>
#include <misc/idx_t.h>
#include <tracking/Outline.h>

namespace track {
class Individual;
}

namespace cmn::gui {
    class StaticText;
    class Button;

    class Posture : public Entangled {
        //track::Individual* _fish{nullptr};
        track::Idx_t _fdx;
        Frame_t _frameIndex;
        Vec2 zero;
        //gui::Rect _background;
        bool _valid{false};
        pv::BlobPtr _lines;
        track::Midline::Ptr _midline;
        Float2_t midline_length{0};
        std::vector<Vec2> _outline;
        std::unique_ptr<StaticText> _text;
        std::unique_ptr<Button> _close;
        
        bool _average_active;
        std::map<track::Idx_t, std::deque<float>> _scale;
        
    public:
        Posture(const Bounds& size = Bounds(500,300,550,400));
        ~Posture();
        
        void set_fish(track::Individual* fish, Frame_t frame);
        virtual void update() override;
        bool valid() const;
    };
}

#endif
