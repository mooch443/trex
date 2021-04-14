#pragma once

#include <commons/common/misc/defines.h>
#include <gui/DrawStructure.h>
#include <gui/CrossPlatform.h>
#include <misc/Timer.h>

namespace gui {
    class SFLoop {
        Base* _base;
        DrawStructure& _graph;
        std::function<void(SFLoop&)> _after_display;
        std::function<void(SFLoop&)> _idle_callback;
        GETTER_SETTER(bool, please_end)
        
        std::string _name;
        std::mutex queue_mutex;
        std::queue<std::function<void()>> main_exec_queue;
        GETTER(Timer, time_since_last_update)
        
    public:
        SFLoop(DrawStructure& graph,
               Base*,
               const std::function<void(SFLoop&, gui::LoopStatus)>& custom_loop = nullptr,
               const std::function<void(SFLoop&)>& after_display = nullptr,
               const std::function<void(SFLoop&)>& idle_callback = nullptr);
        
        void add_to_queue(std::function<void()> fn);
        
    private:
        void draw(const std::function<void(DrawStructure&)>& custom_draw);
    };
}
