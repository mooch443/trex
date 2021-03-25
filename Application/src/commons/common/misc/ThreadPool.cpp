#include "ThreadPool.h"
#include <misc/metastring.h>

namespace cmn {
    GenericThreadPool::GenericThreadPool(size_t nthreads, std::function<void(std::exception_ptr)> handle_exceptions, const std::string& thread_prefix, std::function<void()> init)
        : _exception_handler(handle_exceptions), nthreads(0), stop(false), _init(init), _working(0), _thread_prefix(thread_prefix)
    {
        resize(nthreads);
    }

    void GenericThreadPool::resize(size_t num_threads) {
        std::unique_lock<std::mutex> lock(m);
        
        if(num_threads < nthreads) {
            size_t i=(nthreads - num_threads);
            for(auto it = thread_pool.begin() + i; it != thread_pool.end(); ++i) {
                stop_thread.at(i) = true;
                (*it)->join();
                delete *it;
                it = thread_pool.erase(it);
            }
            stop_thread.resize(num_threads);
            
        } else {
            stop_thread.resize(num_threads);
            
            while(thread_pool.size() < num_threads) {
                size_t i = thread_pool.size();
                thread_pool.push_back(new std::thread([this](std::function<void()> init, int idx)
                {
                    set_thread_name(this->thread_prefix()+"::thread_"+Meta::toStr(idx));
                    std::unique_lock<std::mutex> lock(m);
                    
                    init();
                    
                    for(;;) {
                        condition.wait(lock, [&](){ return !q.empty() || stop; });
                        if(!q.empty()) {
                            // set busy!
                            _working++;
                            
                            auto task = std::move(q.front());
                            q.pop();
                            
                            // free mutex, perform task, regain mutex
                            lock.unlock();
                            try {
                                task();
                            } catch(...) {
                                _exception_handler(std::current_exception());
                            }
                            lock.lock();
                            
                            // not busy anymore!
                            _working--;
                            finish_condition.notify_one();
                            
                        } else if(stop || stop_thread.at(idx))
                            break;
                    }
                    
                }, _init, i));
            }
        }
        
        nthreads = num_threads;
    }

    void GenericThreadPool::wait_one() {
        std::unique_lock<std::mutex> lock(m);
        if(_working == 0 || q.empty())
            return;
        auto previous = q.size();
        finish_condition.wait(lock, [&](){ return q.size() < previous || q.empty() || !_working; });
    }
}
