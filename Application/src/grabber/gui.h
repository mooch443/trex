#ifndef _GUI_H
#define _GUI_H

#include <types.h>
#include <grabber.h>
#include <gui/DrawBase.h>
#include <http/httpd.h>
#include <gui/DrawStructure.h>
#include <gui/DrawHTMLBase.h>
#include <gui/DrawSFBase.h>

namespace grab {
class GUI {
public:
    static const std::map<std::string, std::string> setting_keys;
    
protected:
    FrameGrabber &_grabber;
    GETTER(CropOffsets, crop_offsets)
    GETTER(cv::Size, size)
    GETTER(cv::Size, cropped_size)
    //GETTER(float, window_scale)
    
    bool _redraw;
    float _record_alpha;
    bool _record_direction;
    
    std::mutex _display_queue_lock;
    std::thread *_display_thread;
    
    std::mutex _gui_frame_lock;
    Timer _gui_bytes_timer, _gui_timer;
    long _gui_bytes_count, _gui_bytes_per_second;
    std::vector<uchar> _gui_frame_bytes;
    
    bool _pulse_direction;
    float _pulse;
    Timer pulse_timer;
    
    GETTER_NCONST(gui::DrawStructure, gui)
    gui::Base* _sf_base;
    std::unique_ptr<pv::Frame> _frame, _noise;
    Image::UPtr _image;

    gui::HTMLBase _html_base;
    
public:
    GUI(FrameGrabber& grabber);
    ~GUI();
    void event(const gui::Event& e);
    static void static_event(const gui::Event& e);
    void key_event(const gui::Event& e);
    
    void set_base(gui::Base*);
    static GUI* instance();
    
#if WITH_MHD
    Httpd::Response render_html();
#endif
    
    bool has_window() const { return _sf_base != NULL; }
    
    void set_redraw();
    bool terminated() const;
    void draw(gui::DrawStructure& base);
    std::string info_text() const;
    void update_loop();
    
private:
    void run_loop();
    void update();
};
}

#endif
