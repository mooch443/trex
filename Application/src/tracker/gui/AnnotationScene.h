#pragma once

#include <commons.pc.h>
//#include <gui/DrawBase.h>
#include <gui/Scene.h>
#include <misc/Image.h>
#include <misc/ObjectManager.h>
#include <gui/types/Entangled.h>
#include <gui/dyn/VarProps.h>
#include <misc/Timer.h>

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

enum class AnnotationType {
    BOX,
    POSE,
    SEGMENTATION
};

struct Annotation {
    uint8_t clid;
    AnnotationType type;
    std::vector<blob::Pose::Point> points;
};

class AnnotationView : public Entangled {
    std::vector<derived_ptr<Circle>> _circles;
    std::unique_ptr<Rect> _rect;
    Annotation _a;
    
public:
    AnnotationView() = default;
    
    template<typename... Args>
    AnnotationView(Args... args)
    {
        create(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void create(Args... args) {
        (set(std::forward<Args>(args)), ...);
        init();
    }
    
    using Entangled::set;
    
    void set_annotation(Annotation&&);
    void update() override;
    
private:
    void init();
};

class AnnotationScene : public Scene {
public:
    using Manager = ObjectManager<Annotation>;
private:
    static inline constexpr uint32_t max_cache = 1000;
    
    std::unordered_set<Frame_t> _selected_frames;
    std::unordered_map<Frame_t, Image::Ptr> _loaded_frames;
    std::unordered_map<Frame_t, std::vector<std::shared_ptr<dyn::VarBase_t>>> _gui_annotations;
    std::unordered_map<Frame_t, std::vector<sprite::Map>> _gui_data;
    
    // views for current frame
    Frame_t _view_frame;
    std::unordered_map<Manager::ID, derived_ptr<AnnotationView>> _views;
    
    std::mutex _video_mutex;
    std::unique_ptr<VideoSource> _video;
    std::unordered_map<Frame_t, Manager> annotations; // Frame index to Pose mapping
    Frame_t currentFrameIndex; // Current frame index in the video
    
    std::future<std::unordered_set<Frame_t>> _frame_future;
    
    std::unique_ptr<Rect> _drag_box;
    Frame_t video_length;
    Size2 video_size;
    std::unique_ptr<Bowl> _bowl;
    std::future<Image::Ptr> _next_frame;
    std::unique_ptr<ExternalImage> _current_image;
    std::unique_ptr<dyn::DynamicGUI> _gui;
    
    blob::Pose::Skeleton _skeleton;
    Annotation _pose_in_progress;
    Timer _timer;

public:
    // Constructor
    AnnotationScene(Base& window);

    // Activation and deactivation
    virtual void activate() override;
    virtual void deactivate() override;

    // Handling global events for video navigation
    virtual bool on_global_event(Event event) override;

    // Methods to manage annotations
    Manager::ID addAnnotation(Frame_t frameNumber, Annotation&&);
    void removeAnnotation(Frame_t frameNumber, Manager::ID id);
    const Annotation& getAnnotation(Frame_t frameNumber, Manager::ID id) const;

    // Method to handle frame navigation
    void navigateToFrame(Frame_t frameIndex);

private:
    // Custom drawing
    void _draw(DrawStructure&);
    Image::Ptr retrieveFrame(Frame_t);
    std::future<std::unordered_set<Frame_t>> select_unique_frames();
    std::future<Image::Ptr> retrieve_next_frame();
};

} // namespace gui
