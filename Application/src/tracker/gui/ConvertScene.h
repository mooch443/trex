#pragma once

#include <commons.pc.h>
#include <gui/Scene.h>
#include <python/Detection.h>
#include <misc/indicators.h>

//#include <tracking/Segmenter.h>
#include <misc/tomp4.h>

//#include <gui/ScreenRecorder.h>
#include <gui/Skelett.h>
#include <gui/Bowl.h>

#include <misc/idx_t.h>
//#include <misc/Identity.h>

namespace track {
class Segmenter;
}

namespace gui {

std::string window_title();

namespace ind = indicators;
using namespace track;

class Label;

class ConvertScene : public Scene {
    static sprite::Map fish;
    static sprite::Map _video_info;
    
    Timer last_tick;
    struct Data;
    
    std::shared_future<void> _scene_active;
    std::promise<void> _scene_promise;
    
    ind::ProgressBar bar;
    ind::ProgressSpinner spinner;

    
    std::function<void(ConvertScene&)> _on_activate, _on_deactivate;
    
    std::unique_ptr<Data> _data;
    
public:
    ConvertScene(Base& window, std::function<void(ConvertScene&)> on_activate, std::function<void(ConvertScene&)> on_deactivate);
    ~ConvertScene();
    
    Segmenter& segmenter() const;
    
    void set_segmenter(Segmenter* seg);
    
    //Size2 output_size() const;
    bool on_global_event(Event) override;
    
private:
    void deactivate() override;
    
    void open_video();

    void open_camera();
    
    void activate() override;

    
    // Helper function to calculate window dimensions
    Size2 calculateWindowSize(const Size2& output_size, const Size2& window_size);

    
    

    // Main _draw function
    void _draw(DrawStructure& graph);
    
    std::string window_title() const;
    
    SegmentationData& current_data();
    ExternalImage& background_image();
};

} // namespace gui
