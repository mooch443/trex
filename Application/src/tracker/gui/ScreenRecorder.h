#pragma once
#include <misc/ranges.h>
#include <misc/Image.h>

namespace gui {
class Base;
class DrawStructure;
class WorkProgress;

class ScreenRecorder {
    struct Data;
    Data* _data;
    
public:
    ScreenRecorder();
    ~ScreenRecorder();
    void update_recording(Image::Ptr&&, Base*, cmn::Frame_t frame, cmn::Frame_t max_frame);
    bool recording() const;
    void start_recording(Base*, cmn::Frame_t frame);
    void stop_recording(Base*, DrawStructure*);
    void set_frame(cmn::Frame_t frame);
};

}
