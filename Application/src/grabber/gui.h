#ifndef _GUI_H
#define _GUI_H

#include <commons.pc.h>
#include <grabber.h>
#include <gui/DrawBase.h>
#include <http/httpd.h>
#include <gui/DrawStructure.h>
#include <gui/DrawHTMLBase.h>
#include <gui/DrawSFBase.h>
#include <gui/ControlsAttributes.h>

namespace cmn::gui {
class ExternalImage;
}

namespace grab {
using namespace ::cmn::gui;

class GUI {
public:
    static const std::map<std::string, std::string> setting_keys;
    
protected:
    FrameGrabber &_grabber;
    GETTER(CropOffsets, crop_offsets);
    GETTER(cv::Size, size);
    GETTER(cv::Size, cropped_size);
    //GETTER(float, window_scale);
    
    bool _redraw;
    float _record_alpha;
    bool _record_direction;
    
    std::mutex _display_queue_lock;
    std::thread *_display_thread;
    
    std::mutex _gui_frame_lock;
    Timer _gui_bytes_timer, _gui_timer;
    long _gui_bytes_count = 0, _gui_bytes_per_second = 0;
    std::vector<uchar> _gui_frame_bytes;
    
    bool _pulse_direction;
    float _pulse = 0;
    Timer pulse_timer;
    
    Base* _sf_base{nullptr};
    DrawStructure* _gui{nullptr};
    std::unique_ptr<pv::Frame> _frame, _noise;
    Image::Ptr _image;
    ExternalImage *background = nullptr, *noise_image = nullptr;
    
    HTMLBase _html_base;
    
public:
    GUI(DrawStructure*, FrameGrabber& grabber);
    ~GUI();
    void event(const Event& e);
    static void static_event(DrawStructure&, const Event& e);
    void key_event(const Event& e);
    
    void set_base(Base*);
    static GUI* instance();
    
#if WITH_MHD
    Httpd::Response render_html();
#endif
    
    bool has_window() const { return _sf_base != NULL; }
    
    void set_redraw();
    bool terminated() const;
    void draw(DrawStructure& base);
    void draw_tracking(DrawStructure& base, const attr::Scale& scale);
    std::string info_text() const;
    void update_loop();
    
    DrawStructure& gui() const {
        if(not _gui)
            throw U_EXCEPTION("DrawStructure pointer not set yet.");
        return *_gui;
    }
    
private:
    void run_loop();
    void update();
};
}

#endif

