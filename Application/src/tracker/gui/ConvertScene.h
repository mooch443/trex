#pragma once

#include <commons.pc.h>
#include <tracker/Scene.h>
#include <misc/OverlayedVideo.h>
#include <tracking/Detection.h>
#include <gui/DynamicGUI.h>

#include <indicators/progress_bar.hpp>
#include <indicators/progress_spinner.hpp>

#include <tracking/Tracker.h>
#include <Alterface.h>

namespace gui {

using namespace dyn;
std::string window_title();

namespace ind = indicators;
using namespace track;

class ConvertScene : public Scene {
    static sprite::Map fish;
    static sprite::Map _video_info;

    // condition variables and mutexes for thread synchronization
    std::condition_variable _cv_messages, _cv_ready_for_tracking;
    std::mutex _mutex_general, _mutex_current;
    std::atomic<bool> _should_terminate{false};

    // Segmentation data for the next frame
    SegmentationData _next_frame_data;

    // Progress and current data for tracking
    SegmentationData _progress_data, _current_data, _transferred_current_data;

    // External images for background and overlay
    std::shared_ptr<ExternalImage> _background_image = std::make_shared<ExternalImage>(),
                                   _overlay_image = std::make_shared<ExternalImage>();

    // Vectors for object blobs and GUI objects
    std::vector<pv::BlobPtr> _object_blobs, _progress_blobs, _transferred_blobs;
    std::vector<std::shared_ptr<VarBase_t>> _gui_objects;

    // Individual properties for each object
    std::vector<sprite::Map> _individual_properties;

    // Overlayed video with detections and tracker for object tracking
    std::unique_ptr<OverlayedVideo<Detection>> _overlayed_video;
    std::unique_ptr<Tracker> _tracker;

    // File for output
    std::unique_ptr<pv::File> _output_file;

    // Threads for tracking and generation
    std::thread _tracking_thread, _generator_thread;
    std::shared_future<void> _scene_active;
    std::promise<void> _scene_promise;

    // Frame data
    Frame_t _actual_frame, _video_frame;

    // Size of output and start time for timing operations
    GETTER(Size2, output_size)
    std::chrono::time_point<std::chrono::system_clock> _start_time;
    
    Alterface menu;
    
    ind::ProgressBar bar;
    ind::ProgressSpinner spinner;
    
public:
    ConvertScene(Base& window);
    
    ~ConvertScene();
    
private:
    void deactivate() override;
    
    void open_video();

    void open_camera();
    
    void activate() override;

    void setDefaultSettings();

    void printDebugInformation();

    void fetch_new_data();
    
    // Helper function to calculate window dimensions
    Size2 calculateWindowSize(const Size2& output_size, const Size2& window_size);

    // Helper function to draw outlines
    void drawOutlines(DrawStructure& graph, const Size2& scale);
    
    void drawBlobs(const std::vector<std::string>& meta_classes, const Vec2& scale, const std::unordered_map<pv::bid, Identity>& visible_bdx);

    // Main _draw function
    void _draw(DrawStructure& graph);
    
    void generator_thread();
    
    void perform_tracking();
    
    void tracking_thread();
};

} // namespace gui
