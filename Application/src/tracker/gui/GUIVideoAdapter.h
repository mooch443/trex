#pragma once

#include <commons.pc.h>
#include <gui/ControlsAttributes.h>
#include <misc/BlurryVideoLoop.h>
#include <gui/types/Entangled.h>
#include <gui/GuiTypes.h>
#include <file/PathArray.h>
#include <misc/Image.h>
#include <gui/Dispatcher.h>

using Alpha = cmn::gui::attr::Alpha;

namespace cmn::gui {

class IMGUIBase;

NUMBER_ALIAS(Blur, double);
NUMBER_ALIAS(FrameTime, double);

class GUIVideoAdapter : public Entangled {
private:
    GETTER(BlurryVideoLoop, video_loop);
    ExternalImage _image;
    IMGUIBase *_queue;
    Margins _margins;
    
    double _fade_percent{0.0};
    double _target_alpha{1};
    double _current_alpha{0};
    Timer timer;
    
    Image::Ptr _buffer;
    BlurryVideoLoop::VideoFrame _latest_image;
    
    std::mutex _future_mutex;
    std::future<void> _executed;
    Size2 _video_size;
    GuardedProperty<VideoInfo> _current_info;
    std::function<void(VideoInfo)> _open_callback;
    file::PathArray _array;
    
public:
    GUIVideoAdapter(const file::PathArray& array, IMGUIBase* queue, std::function<void(VideoInfo)> callback);
    
    ~GUIVideoAdapter();
    
    void update() override;
    
    void set_content_changed(bool v) override;
    
    using Entangled::set;
    void set(SizeLimit limit);
    void set(file::PathArray path);
    void set(Blur blur);
    void set(FrameTime time);
    void set(Margins margins);
    void set(Alpha clr);
    Alpha alpha() const;
    void set_scale(const Vec2& scale) override;
};

namespace attr {
CMN_GUI_REGISTER_ATTRIBUTE_MEMBER(GUIVideoAdapter, Blur);
CMN_GUI_REGISTER_ATTRIBUTE_MEMBER(GUIVideoAdapter, FrameTime);
CMN_GUI_REGISTER_ATTRIBUTE_MEMBER(GUIVideoAdapter, Margins);
CMN_GUI_REGISTER_ATTRIBUTE_MEMBER(GUIVideoAdapter, Alpha);
CMN_GUI_REGISTER_ATTRIBUTE_MEMBER(GUIVideoAdapter, file::PathArray);
CMN_GUI_REGISTER_ATTRIBUTE_MEMBER(GUIVideoAdapter, SizeLimit);
}

}
