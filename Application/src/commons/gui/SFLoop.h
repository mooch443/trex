#pragma once

#include <commons/common/misc/defines.h>
#include <gui/DrawStructure.h>
#include <gui/CrossPlatform.h>

namespace gui {
    class SFLoop {
        Base* _base;
        DrawStructure& _graph;
        std::function<void(SFLoop&)> _after_display;
        GETTER_SETTER(bool, please_end)
        
        std::mutex queue_mutex;
        std::queue<std::function<void()>> main_exec_queue;
        
    public:
        SFLoop(DrawStructure& graph,
               Base*,
               const std::function<void(SFLoop&)>& custom_loop = [](SFLoop&){},
               const std::function<void(SFLoop&)>& after_display = [](SFLoop&){});
        
        void add_to_queue(std::function<void()> fn);
        
    private:
        void draw(const std::function<void(DrawStructure&)>& custom_draw);
    };
}
