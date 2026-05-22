#pragma once

#include <commons.pc.h>
#include <ui/Scene.h>
#include <python/Detection.h>
#include <core/indicators.h>

//#include <tracking/Segmenter.h>
#include <core/tomp4.h>

//#include <ui/ScreenRecorder.h>
#include <ui/Skelett.h>
#include <ui/Bowl.h>

#include <core/idx_t.h>
//#include <core/Identity.h>

namespace track {
class Segmenter;
}

namespace cmn::gui::convert {

struct VideoInfo {
    Frame_t frame;
    Frame_t length;
    Size2 resolution;
    
    glz::json_t to_json() const {
        glz::json_t r;
        r["frame"] = frame.to_json();
        r["length"] = length.to_json();
        r["resolution"] = resolution.to_json();
        return r;
    }
};

}

template <>
struct glz::meta<cmn::gui::convert::VideoInfo> {
    using T = cmn::gui::convert::VideoInfo;
    static constexpr auto value = glz::object(
        "frame", &T::frame,
        "length", &T::length,
        "resolution", &T::resolution
    );
};

namespace cmn::gui {

namespace ind = indicators;
using namespace track;

class Label;
class ExternalImage;

class ConvertScene : public Scene {
    static glz::json_t fish;
    static convert::VideoInfo _video_info;
    std::atomic<Frame_t> _video_length;
    
    Timer last_tick;
    struct Data;
    
    std::shared_future<void> _scene_active;
    std::promise<void> _scene_promise;
    
    std::function<void(ConvertScene&)> _on_activate, _on_deactivate;
    
    std::unique_ptr<Data> _data;

public:
    static read_once<bool> force_start_over;
    
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
    
    SegmentationData& current_data();
    ExternalImage& background_image();
    
private:
    void update_progress_callback();
};

} // namespace gui
