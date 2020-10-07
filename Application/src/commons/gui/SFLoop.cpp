#include "SFLoop.h"
#include <misc/GlobalSettings.h>
#include <gui/DrawBase.h>

namespace gui {
    SFLoop::SFLoop(DrawStructure& graph, Base* base,
           const std::function<void(SFLoop&)>& custom_loop,
           const std::function<void(SFLoop&)>& after_display)
        : _base(base), _graph(graph), _after_display(after_display), _please_end(false)
    {
        SETTING(terminate) = false;
        bool _do_terminate = false;
        
        GlobalSettings::map().register_callback(this, [&_do_terminate](auto&, const std::string& key, auto& value)
        {
            if(key == "terminate") {
                _do_terminate = value.template value<bool>();
            }
        });
        
        while (!_do_terminate && !_please_end) {
            tf::show();
            
            custom_loop(*this);
            
            if(_base) {
                auto status = _base->update_loop();
                if(status == gui::LoopStatus::END)
                    SETTING(terminate) = true;
                else if(status != gui::LoopStatus::UPDATED)
                    std::this_thread::sleep_for(std::chrono::milliseconds(15));
            } else {
                std::lock_guard<std::recursive_mutex> guard(_graph.lock());
                _graph.before_paint((Base*)nullptr);
            }
                
            {
                //std::unique_lock<std::recursive_mutex> guard(_graph.lock());
                _after_display(*this);
            }
            
            {
                std::lock_guard<std::mutex> guard(queue_mutex);
                while(!main_exec_queue.empty()) {
                    main_exec_queue.front()();
                    main_exec_queue.pop();
                }
            }
        }
        
        GlobalSettings::map().unregister_callback(this);
    }
    
    void SFLoop::add_to_queue(std::function<void ()> fn) {
        std::lock_guard<std::mutex> guard(queue_mutex);
        main_exec_queue.push(fn);
    }
}
