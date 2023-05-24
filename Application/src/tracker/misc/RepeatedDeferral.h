#pragma once

#include <commons.pc.h>

namespace cmn {

template<typename F, typename R = typename cmn::detail::return_type<F>::type>
struct RepeatedDeferral {
    size_t _threads{ 1 }, _minimal_fill { 0 };
    std::vector<R> _next;
    //std::future<R> _next_image;
    F _fn;
    std::string name;
    std::mutex mtiming;
    double _waiting{0}, _samples{0};
    double _since_last{0}, _ssamples{0};
    
    double _runtime{0}, _rsamples{0};
    double _average_fill_state{ 0 }, _as{ 0 };
    Timer since_last, since_print;
    
    std::condition_variable _message, _new_item;
    mutable std::mutex _mutex;
    std::unique_lock<std::mutex> _init_guard{_mutex};
    std::thread _updater;
    std::atomic<bool> _terminate{ false };
    
    RepeatedDeferral(size_t threads, size_t minimal_fill, std::string name, F fn) : _threads(threads), _minimal_fill(minimal_fill), _fn(std::forward<F>(fn)), name(name),
    _updater([this]() {
        set_thread_name(this->name+"_update_thread");
        std::unique_lock guard(_mutex);
        _message.notify_all();
        _message.wait(guard);
        
        Timer runtime;
        
        while (not _terminate) {
            if ((not _next.empty() and _next.size() >= _threads) and not _terminate) {
                //thread_print(this->name.c_str(), " #NEXT Sleeping at ", _next.size());
                _message.wait(guard);
                //thread_print(this->name.c_str(), " #NEXT Wake at ", _next.size());
            }
            
            if (not _terminate and _next.size() < _threads)
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
                
                _next.push_back(std::move(r));
                _new_item.notify_one();
                
                //thread_print(this->name.c_str(), " #NEXT Filled up to ", _next.size());
                
                //std::unique_lock guard(mtiming);
                _runtime += e * 1000;
                _rsamples++;
                
                {
                    //std::unique_lock guard(mtiming);
                    //++predicted;
                    
                    if (since_print.elapsed() > 5) {
                        std::unique_lock guard(mtiming);
                        //auto total = (_waiting / _samples);
                        thread_print("runtime ", (_runtime / _rsamples), "ms; gap:", (_since_last / _ssamples), "ms; wait = ",
                                     (_waiting / _samples), "ms ", dec<2>(_average_fill_state / _as), "/", _threads, " fill");
                        
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
    })
    {
        _message.wait(_init_guard);
        _init_guard = {};
    }
    
    void notify() {
        _message.notify_all();
        _new_item.notify_all();
    }
    
    ~RepeatedDeferral() {
        //if(_next_image.valid())
        //    _next_image.get();
        _terminate = true;
        notify();
        _updater.join();
    }
    
    bool has_next() const {
        std::unique_lock guard(_mutex);
        return not _next.empty();
    }
    
    R get_next() {
        Timer timer;
        
        std::unique_lock guard(_mutex);
        if(_next.empty())
            _new_item.wait(guard, [this]() {return not _next.empty(); });
        
        auto e = timer.elapsed();
        {
            std::unique_lock guard(mtiming);
            _waiting += e * 1000;
            _samples++;
            
            _average_fill_state += _next.size();
            ++_as;
        }
        
        auto f = std::move(_next.front());
        _next.erase(_next.begin());
        if(_next.size() <= _minimal_fill) {
            _message.notify_one();
            //thread_print(this->name.c_str(), " #NEXT Need to update: ", _next.size(), " / ", _threads);
        } //else
          //thread_print(this->name.c_str(), " #NEXT No need to update: ", _next.size(), " / ", _threads);
        
        return f;
    }
    
    R next() {
        return get_next();
    }
};

}
