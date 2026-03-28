#pragma once
#include <commons.pc.h>

#include <ui/Scene.h>
#include <misc/Image.h>
#include <misc/ObjectManager.h>
#include <gui/types/Entangled.h>
#include <gui/dyn/VarProps.h>
#include <misc/Timer.h>
#include <core/TaskPipeline.h>
#include <ui/ImageGeneratorRegistry.h>

namespace cmn {
class VideoSource;
}

namespace cmn::gui {

using Pose = blob::Pose;
class Bowl;
class DrawStructure;
class Rect;
class Circle;
class ExternalImage;

namespace dyn {
struct DynamicGUI;
}

struct VideoFrame {
    Image::Ptr image;
    Frame_t index;
};

class LiveSegmentation : public Scene {
    struct Data;
    
private:
    std::mutex _next_frame_mutex, _generate_mutex;
    std::unique_ptr<VideoSource> _video;
    std::condition_variable _condition;
    std::atomic<bool> _terminated{false};
    
    Frame_t video_length;
    Size2 video_size;
    std::unique_ptr<Bowl> _bowl;
    
    std::atomic<bool> _playback{false};
    
    std::optional<Frame_t> _requested_frame;
    std::optional<VideoFrame> _next_frame;
    std::optional<VideoFrame> _previous_frame;
    std::optional<SegmentationData> _next_data;
    
    VideoFrame _current_frame;
    std::unique_ptr<ExternalImage> _current_image;
    std::optional<SegmentationData> _current_data;
    std::unique_ptr<dyn::DynamicGUI> _gui;
    ImageGeneratorRegistry _image_generators;
    
    Timer _timer;
    std::unique_ptr<std::thread> _fetch_thread;

public:
    // Constructor
    LiveSegmentation(Base& window);
    ~LiveSegmentation();

    // Activation and deactivation
    virtual void activate() override;
    virtual void deactivate() override;

    // Handling global events for video navigation
    virtual bool on_global_event(Event event) override;

private:
    // Custom drawing
    void _draw(DrawStructure&);
    
    std::unique_ptr<Data> _data;
};

} // namespace gui
