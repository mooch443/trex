#pragma once

#include <types.h>
#include "Camera.h"

namespace fg {
    class InteractiveCamera : public Camera {
        Size2 _size;
        
        struct Fish {
            Vec2 boundary;
            Vec2 position;
            Vec2 velocity;
            Vec2 _force;
            
            float width;
            float L;
            //float angle;
            
            void update(float dt, const Vec2& poi, const std::vector<Fish>& individuals);
            void draw(cv::Mat&);
        };
        
        std::vector<Fish> _fishies;
        
    public:
        InteractiveCamera();
        ~InteractiveCamera() {
        }
        
        virtual Size2 size() const override { return _size; }
        virtual bool next(Image& image) override;
        virtual bool open() override { return true; }
        virtual void close() override { }
    };
}
