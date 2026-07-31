#pragma once

#include <commons.pc.h>
//#include <gui/DrawBase.h>
#include <ui/Scene.h>
#include <misc/Image.h>
#include <misc/ObjectManager.h>
#include <gui/types/Entangled.h>
#include <gui/dyn/VarProps.h>
#include <misc/Timer.h>
#include <core/DetectAnnotation.h>

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

/// Draws and edits one detect annotation in the annotation scene.
class DetectAnnotationView : public Entangled {
    std::vector<derived_ptr<Circle>> _circles;
    std::unique_ptr<Rect> _rect;
    track::detect::Annotation _a;
    
public:
    DetectAnnotationView() = default;
    
    template<typename... Args>
    DetectAnnotationView(Args... args)
    {
        create(std::forward<Args>(args)...);
    }
    
    template<typename... Args>
    void create(Args... args) {
        (set(std::forward<Args>(args)), ...);
        init();
    }
    
    using Entangled::set;
    
    void set_detect_annotation(track::detect::Annotation&&);
    void update() override;
    
private:
    void init();
};

/// Scene for selecting source frames and manually authoring detect annotations.
class DetectAnnotationScene : public Scene {
public:
    using Manager = ObjectManager<track::detect::Annotation>;
private:
    static inline constexpr uint32_t max_cache = 1000;
    
    std::unordered_set<Frame_t> _selected_frames;
    std::unordered_map<Frame_t, Image::Ptr> _loaded_frames;
    std::unordered_map<Frame_t, std::vector<std::shared_ptr<dyn::VarBase_t>>> _gui_detect_annotations;
    std::unordered_map<Frame_t, std::vector<sprite::Map>> _gui_data;
    
    // views for current frame
    Frame_t _view_frame;
    std::unordered_map<Manager::ID, derived_ptr<DetectAnnotationView>> _views;
    
    std::mutex _video_mutex;
    std::unique_ptr<VideoSource> _video;
    std::unordered_map<Frame_t, Manager> detect_annotations; // Frame index to detect-annotation mapping
    Frame_t currentFrameIndex; // Current frame index in the video
    
    std::future<std::unordered_set<Frame_t>> _frame_future;
    
    std::unique_ptr<Rect> _drag_box;
    Frame_t video_length;
    Size2 video_size;
    std::unique_ptr<Bowl> _bowl;
    std::future<Image::Ptr> _next_frame;
    std::unique_ptr<ExternalImage> _current_image;
    std::unique_ptr<dyn::DynamicGUI> _gui;
    
    std::optional<blob::Pose::Skeletons> _skeleton;
    track::detect::Annotation _pose_in_progress;
    Timer _timer;

public:
    // Constructor
    DetectAnnotationScene(Base& window);
    ~DetectAnnotationScene();

    // Activation and deactivation
    virtual void activate() override;
    virtual void deactivate() override;

    // Handling global events for video navigation
    virtual bool on_global_event(Event event) override;

    /// Adds a detect annotation to `frameNumber` and returns its frame-local ID.
    Manager::ID addDetectAnnotation(Frame_t frameNumber, track::detect::Annotation&&);
    /// Removes the detect annotation identified by the frame-local object ID.
    void removeDetectAnnotation(Frame_t frameNumber, Manager::ID id);
    /// Returns the detect annotation identified by the frame-local object ID.
    const track::detect::Annotation& getDetectAnnotation(Frame_t frameNumber, Manager::ID id) const;

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
