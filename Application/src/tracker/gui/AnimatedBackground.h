#pragma once

#include <commons.pc.h>
#include <misc/frame_t.h>
#include <gui/types/Entangled.h>
#include <video/VideoSource.h>
#include <misc/ThreadManager.h>
#include <misc/Timer.h>
#include <pv.h>
#include <gui/FramePreloader.h>
#include <misc/Buffers.h>
#include <gui/GuiTypes.h>

namespace cmn::gui {

class AnimatedBackground : public Entangled {
    Color _tint{White};
    Image _local_buffer;
    gpuMat _buffer;
    gpuMat _resized;

    struct ImageMaker {
        Image::Ptr operator()() const {
            return Image::Make();
        }
    };
    
    ImageBuffers<Image::Ptr, ImageMaker> buffers;
    ImageBuffers<Image::Ptr, ImageMaker> grey_buffers;
    
    std::mutex _source_mutex;
    std::unique_ptr<VideoSource> _source;
    std::atomic<float> _source_scale{1.f};
    
    Image::Ptr _average;
    ExternalImage _static_image, _grey_image;
    Frame_t _current_frame;
    Frame_t _increment{1_f};
    
    double _fade{1.0};
    double _target_fade{1.0};
    Timer _fade_timer;
    
    Timer _next_timer;
    std::mutex _next_mutex;
    //Image::Ptr _next_image;
    std::future<Image::Ptr> _next_frame;
    bool gui_show_video_background{true};
    
    FramePreloader<Image::Ptr> preloader;
    std::atomic<bool> _strict{false};
    
public:
    AnimatedBackground(Image::Ptr&&, const pv::File* = nullptr);
    AnimatedBackground(VideoSource&&);
    
    AnimatedBackground(const AnimatedBackground&) = delete;
    AnimatedBackground(AnimatedBackground&&) = delete;
    AnimatedBackground& operator=(const AnimatedBackground&) = delete;
    AnimatedBackground& operator=(AnimatedBackground&&) = delete;
    
    void set_color(const Color&);
    const Color& color() const;
    
    void before_draw() override;
    using Entangled::update;
    
    void set_strict(bool v) { _strict = v; }
    void set_video_scale(float);
    
    void set_undistortion(std::optional<std::vector<double>> &&cam_matrix,
                          std::optional<std::vector<double>> &&undistort_vector);
    void set_increment(Frame_t inc);
    Frame_t increment() const { return _increment; }
    
    Image::Ptr preload(Frame_t);
};

}
