#include "ConnectedTasks.h"
#include <misc/SpriteMap.h>
#include <tracking/PPFrame.h>

//#define DEBUG_THREAD_STATE

namespace cmn {
    ConnectedTasks::ConnectedTasks(std::vector<std::function<bool(Type&&, const Stage&)>>&& tasks)
        : _stop(false), _tasks(std::move(tasks)), _stages(_tasks.size()), _paused(false)
    {
        std::vector<std::unique_lock<std::mutex>> guards;
        for(uint32_t i=0; i<(uint32_t)_tasks.size(); i++) {
            std::unique_lock<std::mutex> lock(_stages.at(i).mutex);
            _stages.at(i).id = i;
            _stages.at(i).timings = 0;
            _stages.at(i).samples = 0;
            
            guards.emplace_back(_stages.at(i).mutex);
            for(int j=0; j<(i == 0 ? 2 : 1); ++j) {
                _stages.at(i).paused.push_back(false);
                _threads.push_back(new std::thread([this](size_t i, size_t j) {
                    auto name = "ConnectedTasks::stage_"+Meta::toStr(i)+"_"+Meta::toStr(j);
                    set_thread_name(name);
                    
                    const auto &task = _tasks.at(i);
                    auto &stage = _stages.at(i);
                    auto *next_stage = i+1 < _stages.size() ? &_stages.at(i+1) : NULL;
                    
                    std::unique_lock<std::mutex> lock(stage.mutex);
                    
                    for(;;) {
#ifdef DEBUG_THREAD_STATE
                        set_thread_name(name+"::idle");
#endif
                        stage.condition.wait_for(lock, std::chrono::seconds(1), [&](){ return (!_paused && !stage.queue.empty()) || _stop || stage.paused.at(j) != _paused; });
                        
                        if(stage.paused.at(j) != _paused) {
                            stage.paused.at(j) = _paused.load();
                        }
                        
                        if(!_paused && !stage.queue.empty()) {
#ifdef DEBUG_THREAD_STATE
                            set_thread_name(name);
#endif
                            stage.timer.reset();
                            
                            auto obj = std::move(stage.queue.front());
                            stage.queue.pop();
                            
                            // free mutex, perform task, regain mutex
                            lock.unlock();
                            auto result = task(std::move(obj), stage);
                            lock.lock();
                            
                            if(next_stage && result) {
                                std::unique_lock<std::mutex> lock(next_stage->mutex);
                                next_stage->queue.emplace(std::move(obj));
                            }
                            
                            if(next_stage && result)
                                next_stage->condition.notify_one();
                            if (!next_stage)
                                _finish_condition.notify_one();
                            
                            if(result) {
                                stage.timings += (float)stage.timer.elapsed();
                                stage.samples ++;
                                stage.average_time = stage.timings / stage.samples;
                            }
                            
                        } else if(_stop)
                            break;
                    }
                    
                }, i, _threads.size()));
            }
        }
    }

    void ConnectedTasks::start(const std::function<void()>& main) {
        std::atomic_bool started(false);
        _main_thread = new std::thread([this, &started](std::function<void()> main){
            std::unique_lock<std::mutex> lock(_finish_mutex);
            set_thread_name("ConnectedTasks::main");
            
            Timer timer;
            started = true;
            
            for(;;) {
                _finish_condition.wait_for(lock, std::chrono::seconds(1));
                if(!_stop) {
                    if(!_paused) {
                        main();
                        
                        if(timer.elapsed() >= 60) {
                            float total_time = 0;
                            std::stringstream ss;
                            ss << "ConnectedTasks["<< _stages.size() <<"] ";
                            
                            for (auto &s : _stages) {
                                std::lock_guard<std::mutex> lock(s.mutex);
                                total_time += s.average_time;
                                ss << "-> " << s.average_time * 1000 << "ms ";
                            }
                            
                            ss << "(total: "<<total_time * 1000<<"ms)";
                            auto str = ss.str();
                            print(str.c_str());
                            
                            timer.reset();
                            
                            for(auto &stage : _stages)
                                stage.condition.notify_all();
                        }
                    }
                    
                } else
                    break;
            }
            
        }, main);
        
        // wait for main thread to start
        while(!started) {}
        
        std::lock_guard<std::mutex> lock(_finish_mutex);
        _finish_condition.notify_one();
        print("Initialized ", _stages.size()," stages");
    }

    ConnectedTasks::~ConnectedTasks() {
        terminate();
    }
    
    /*void ConnectedTasks::reset_cache()  {
        /for(auto &stage: _stages) {
            std::unique_lock<std::mutex> lock(stage.mutex);
            while(!stage.queue.empty()) {
                delete stage.queue.front();
                stage.queue.pop();
            }
        }/
        
        _finish_condition.notify_one();
    }*/
    
    void ConnectedTasks::terminate() {
        if(!_stop) {
            _stop = true;
            
            _finish_condition.notify_all();
            _main_thread->join();
            delete _main_thread;
            
            for(auto &t : _stages)
                t.condition.notify_all();
            
            for(auto &t : _threads) {
                t->join();
                delete t;
            }
        }
    }
    
    bool ConnectedTasks::is_paused() const {
        bool paused = true;
        for(auto& s : _stages) {
            std::unique_lock guard(s.mutex);
            for(auto p : s.paused) {
                if(!p) {
                    paused = false;
                    break;
                }
            }
        }
        return paused;
    }
    
    std::future<void> ConnectedTasks::set_paused(bool pause) {
        if(_paused != pause) {
            _paused = pause;
            SETTING(analysis_paused) = _paused.load();
            
            _finish_condition.notify_all();
            for(auto &s : _stages)
                s.condition.notify_all();
            
            auto task = std::async(std::launch::async, [this](){
                cmn::set_thread_name("ConnectedTasks::set_paused");
                while(is_paused() != _paused)
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                
                _finish_condition.notify_all();
                for(auto &s : _stages)
                    s.condition.notify_all();
            });
            
            return task;
        }
        
        return std::async(std::launch::async, [](){});
    }
}
