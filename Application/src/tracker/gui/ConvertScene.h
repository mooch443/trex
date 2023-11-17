#pragma once

#include <commons.pc.h>
#include <tracker/Scene.h>
#include <tracking/Detection.h>
#include <gui/DynamicGUI.h>

#include <indicators/progress_bar.hpp>
#include <indicators/progress_spinner.hpp>

#include <tracking/Segmenter.h>
#include <Alterface.h>

namespace gui {

using namespace dyn;
std::string window_title();

namespace ind = indicators;
using namespace track;

class Label;

class ConvertScene : public Scene {
    static sprite::Map fish;
    static sprite::Map _video_info;
    
    Timer last_tick;
    Segmenter* _segmenter{nullptr};

    // External images for background and overlay
    std::shared_ptr<ExternalImage> _background_image = std::make_shared<ExternalImage>(),
                                   _overlay_image = std::make_shared<ExternalImage>();

    // Vectors for object blobs and GUI objects
    std::vector<pv::BlobPtr> _object_blobs;
    SegmentationData _current_data;

    // Individual properties for each object
    std::vector<std::shared_ptr<VarBase_t>> _untracked_gui, _tracked_gui, _joint;
    std::map<Idx_t, sprite::Map> _individual_properties;
    std::vector<sprite::Map> _untracked_properties;
    std::vector<sprite::Map*> _tracked_properties;

    std::unordered_map<Idx_t, std::shared_ptr<Label>> _labels;

    std::shared_future<void> _scene_active;
    std::promise<void> _scene_promise;
    
    Size2 window_size;
    Size2 output_size;
    Size2 video_size;
    //Alterface menu;
    
    ind::ProgressBar bar;
    ind::ProgressSpinner spinner;
    
    dyn::DynamicGUI dynGUI;
    double dt = 0;
    
    // Frame data
    GETTER(Frame_t, actual_frame)
    GETTER(Frame_t, video_frame)
    
    std::function<void(ConvertScene&)> _on_activate, _on_deactivate;
    
public:
    ConvertScene(Base& window, std::function<void(ConvertScene&)> on_activate, std::function<void(ConvertScene&)> on_deactivate);
    ~ConvertScene();
    
    auto& segmenter() const {
        if(not _segmenter)
            throw U_EXCEPTION("No segmenter exists.");
        return *_segmenter;
    }
    
    void set_segmenter(Segmenter* seg);
    
    //Size2 output_size() const;
    
private:
    void deactivate() override;
    
    void open_video();

    void open_camera();
    
    void activate() override;

    void fetch_new_data();
    
    // Helper function to calculate window dimensions
    Size2 calculateWindowSize(const Size2& output_size, const Size2& window_size);

    // Helper function to draw outlines
    void drawOutlines(DrawStructure& graph, const Size2& scale, Vec2 offset);
    
    void drawBlobs(const std::vector<std::string>& meta_classes, const Vec2& scale, Vec2 offset, const std::unordered_map<pv::bid, Identity>& visible_bdx);

    // Main _draw function
    void _draw(DrawStructure& graph);
};

} // namespace gui
