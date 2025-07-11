#pragma once

#include <commons.pc.h>
#include <pv.h>

#include <misc/GlobalSettings.h>
#include <misc/Timer.h>

namespace track {
    class PPFrame;
}

namespace cmn {
    class ConnectedTasks {
    public:
        typedef std::unique_ptr<track::PPFrame> Type;
        
        struct Stage {
            mutable std::mutex mutex;
            std::condition_variable condition;
            std::queue<Type> queue;
            uint32_t id;
            std::vector<bool> paused;
            
            Timer timer;
            float timings;
            float samples;
            float average_time;
        };
        
    private:
        std::atomic_bool _stop;
        
        std::vector<std::function<bool(Type&&, const Stage&)>> _tasks;
        std::vector<std::thread*> _threads;
        //std::vector<bool> _thread_paused;
        std::vector<Stage> _stages;
        
        std::thread *_main_thread{nullptr};
        std::mutex _finish_mutex;
        std::condition_variable _finish_condition;
        
        GETTER(std::atomic_bool,  paused);
        
    public:
        ConnectedTasks(std::vector<std::function<bool(Type&&, const Stage&)>>&&);
        ~ConnectedTasks();
        
        void start(const std::function<void()>& main);
        
        void add(Type&& obj);
        
        void bump() {
            if(not _stages.empty())
                _stages[0].condition.notify_all();
            
            std::unique_lock lock(_finish_mutex);
            _finish_condition.notify_all();
        }
        void terminate();
        //void reset_cache();
        
        bool is_paused() const;
        std::future<void> set_paused(bool pause);
        bool stage_empty(size_t i) {
            std::unique_lock<std::mutex> lock(_stages[i].mutex);
            return _stages[i].queue.empty();
        }
    };
}
