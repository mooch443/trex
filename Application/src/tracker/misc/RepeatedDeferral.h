#pragma once

#include <commons.pc.h>
#include <misc/ThreadManager.h>
#include <misc/Timer.h>

namespace cmn {

std::atomic<uint32_t>& thread_index();

template<typename F, typename R = typename cmn::detail::return_type<F>::type>
struct RepeatedDeferral {
    size_t _threads{ 1 }, _minimal_fill { 0 };
    uint32_t _index;
    ThreadGroupId _group_id;
    std::vector<std::tuple<std::string, R>> _next;
    //std::future<R> _next_image;
    F _fn;
    std::string name;
    std::mutex mtiming;
    double _waiting{0}, _samples{0};
    double _since_last{0}, _ssamples{0};
    
    double _runtime{0}, _rsamples{0};
    double _average_fill_state{ 0 }, _as{ 0 };

    std::atomic<double> average_fps{0};

    Timer since_last, since_print;
    Timer runtime;
    
    std::condition_variable /*_message,*/ _new_item;
    mutable std::mutex _mutex;
    //std::unique_ptr<std::thread> _updater{nullptr};
    std::atomic<bool> _terminate{ false }, _allow_dropping{ false };
    std::once_flag _flag;
    
    RepeatedDeferral(size_t threads, size_t minimal_fill, std::string name, F fn,
                     cmn::source_location loc = cmn::source_location::current()) : _threads(threads), _minimal_fill(minimal_fill), _index ( thread_index().fetch_add(1) ), _fn(std::forward<F>(fn)), name(name)
    {
        thread_print("Instance of RepeatedDeferral(", name,") with ID ", _index, " with ", &thread_index());
        _group_id = ThreadManager::getInstance().registerGroup(name, loc);
        ThreadManager::getInstance().addThread(_group_id, name, ManagedThread{
            [this](auto&){ updater(); }
        });
    }
    
    void updater() {
        //set_thread_name(this->name+"_update_thread");
        std::unique_lock guard(_mutex);
        
        //while (not _terminate)
        {
            if (not _allow_dropping
                && _next.size() >= _threads)
            {
                //thread_print("TM ",this->name.c_str(), " #NEXT Sleeping at ", _next.size());
                //_message.wait(guard);
                //thread_print(this->name.c_str(), " #NEXT Wake at ", _next.size());
                return;
            }
            
            
            if (not _terminate)
            {
                _since_last += since_last.elapsed() * 1000;
                _ssamples++;
                
                R r;
                double e;
                
                guard.unlock();
                try {
                    runtime.reset();
                    r = _fn();
                    e = runtime.elapsed();
                }
                catch (...) {
                    //p.set_exception(std::current_exception());
                }
                guard.lock();
                
                if(_next.size() > _threads) {
                    thread_print("TM Fill state of ", this->name.c_str(), " is > ", _threads, ": ", _next.size());
                    if(_allow_dropping) {
                        // we can drop this frame safely
                    } else {
                        // we HAVE to accumulate more memory...
                        // maybe print a warning every now and then?
                        _next.emplace_back(this->name, std::move(r));
                    }
                    
                } else {
                    _next.emplace_back(this->name, std::move(r));
                    //thread_print("TM ", this->name.c_str(), " #NEXT Filled up to ", _next.size());
                }
                _new_item.notify_one();
                
                //std::unique_lock guard(mtiming);
                _runtime += e * 1000;
                _rsamples++;
                
                {
                    //std::unique_lock guard(mtiming);
                    //++predicted;

                    if(int64_t(_rsamples) % 100 == 0)
                        average_fps = (_rsamples / _runtime) * 1000;
                    
                    if (since_print.elapsed() > 30) {
                        std::unique_lock guard(mtiming);
                        //auto total = (_waiting / _samples);

                        thread_print("runtime ", dec<2>(_runtime / _rsamples), "ms; gap:", dec<2>(_since_last / _ssamples), "ms; wait = ",
                                     dec<2>(_waiting / _samples), "ms ", dec<2>(_average_fill_state / _as), "/", _threads, " fill");
                        
                        if (_rsamples > 1000) {
                            _waiting = _samples = 0;
                            _runtime = _rsamples = 0;
                            _since_last = _ssamples = 0;
                            _average_fill_state = _as = 0;
                        }
                        since_print.reset();
                    }
                }
                
                since_last.reset();
            }
        }
        
        //thread_print("Task ",name.c_str(), " ending.");
    }
    
    void notify() {
        ThreadManager::getInstance().notify(_group_id);
        //_message.notify_all();
        _new_item.notify_all();
    }
    
    ~RepeatedDeferral() {
        //if(_next_image.valid())
        //    _next_image.get();
        quit();
    }
    
    void quit() {
        {
            std::unique_lock guard(_mutex);
            if(_terminate)
                return;
            _terminate = true;
            notify();
        }

        //if(_updater && _updater->joinable())
        //    _updater->join();
        ThreadManager::getInstance().terminateGroup(_group_id);
        //_updater = nullptr;
    }
    
    bool has_next() const {
        std::unique_lock guard(_mutex);
        return not _next.empty();
    }
    
    R get_next() {
        Timer timer;
        
        std::unique_lock guard(_mutex);
        if(not _terminate) {
            //thread_print("Restarting thread for ", name);
            ThreadManager::getInstance().startGroup(_group_id);
        }
        
        if(_next.empty())
            _new_item.wait(guard, [this]() {return not _next.empty() or _terminate; });

        if(_terminate)
            throw U_EXCEPTION(name, " already terminated.");
        
        auto e = timer.elapsed();
        {
            std::unique_lock guard(mtiming);
            _waiting += e * 1000;
            _samples++;
            
            _average_fill_state += _next.size();
            ++_as;
        }
        
        auto [from, f] = std::move(_next.front());
        _next.erase(_next.begin());
        
        //thread_print("TM got item<",type_name<R>(),">/", _next.size(), " from ", from);
        if(_next.size() <= _minimal_fill) {
            //_message.notify_one();
            ThreadManager::getInstance().notify(_group_id);
            //thread_print("TM ",this->name.c_str(), " #NEXT Need to update: ", _next.size(), " / ", _threads);
        } //else
            //thread_print("TM ",this->name.c_str(), " #NEXT No need to update: ", _next.size(), " / ", _threads);
        
        return std::move(f);
    }
    
    R next() {
        return get_next();
    }
};

}
