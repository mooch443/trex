#ifndef _DRAW_POSTURE_H
#define _DRAW_POSTURE_H

#include <gui/types/Drawable.h>
#include <gui/types/Basic.h>
#include <gui/GuiTypes.h>
#include <gui/DrawStructure.h>
#include <tracking/Individual.h>
#include <tracking/Tracker.h>
#include <gui/Graph.h>
#include <misc/OutputLibrary.h>

namespace gui {
    class Posture : public Entangled {
        track::Individual* _fish;
        Frame_t _frameIndex;
        Vec2 zero;
        //gui::Rect _background;
        
        bool _average_active;
        std::map<uint32_t, std::deque<float>> _scale;
        
    public:
        Posture(const Bounds& size = Bounds());
        
        void set_fish(track::Individual* fish);
        
        void set_frameIndex(Frame_t frameIndex) {
            if(frameIndex == _frameIndex)
                return;
            
            _frameIndex = frameIndex;
            _average_active = true;
            set_content_changed(true);
        }
        
        virtual void update() override;
    };
}

#endif
