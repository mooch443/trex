#pragma once

#include <misc/defines.h>

#if CMN_WITH_IMGUI_INSTALLED
#include <gui/IMGUIBase.h>
#include "grabber.h"

namespace gui {
    class CropWindow {
        std::shared_ptr<IMGUIBase> _base;
        std::shared_ptr<DrawStructure> _graph;
        std::vector<std::shared_ptr<Circle>> circles;
        std::shared_ptr<Rect> _rect;
        Size2 _video_size;
        
    public:
        CropWindow(FrameGrabber& grabber);
        void update_rectangle();
    };
}
#endif
