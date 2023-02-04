#pragma once

#include <commons.pc.h>
#include <gui/types/Entangled.h>
#include <video/VideoSource.h>
#include <misc/frame_t.h>
#include <misc/Timer.h>

namespace gui {

class AnimatedBackground : public Entangled {
    Color _tint{White};
    gpuMat _buffer;
    gpuMat _resized;
    std::unique_ptr<VideoSource> _source;
    float _source_scale{1.f};
    
    ExternalImage _static_image;
    Frame_t _current_frame;
    
    Timer _next_timer;
    std::mutex _next_mutex;
    Image::UPtr _next_image;
    std::future<bool> _next_frame;
    
public:
    AnimatedBackground(Image::UPtr&&);
    AnimatedBackground(VideoSource&&);
    
    AnimatedBackground(const AnimatedBackground&) = delete;
    AnimatedBackground(AnimatedBackground&&) = delete;
    AnimatedBackground& operator=(const AnimatedBackground&) = delete;
    AnimatedBackground& operator=(AnimatedBackground&&) = delete;
    
    void set_color(const Color&);
    const Color& color() const;
    
    void before_draw() override;
    using Entangled::update;
};

}
