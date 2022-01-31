#pragma once

#include <types.h>
#include <misc/metastring.h>

namespace cmn {
    class GenericThreadPool {
        std::queue< std::function<void()> > q;
        std::mutex m;
        
        std::condition_variable condition;
        std::condition_variable finish_condition;
        std::vector<std::thread*> thread_pool;
        std::vector<bool> stop_thread;
        std::function<void(std::exception_ptr)> _exception_handler;
        size_t nthreads;
        std::atomic_bool stop;
        std::function<void()> _init;
        
        GETTER(std::atomic_int, working)
        GETTER(std::string, thread_prefix)
        
    public:
        GenericThreadPool(size_t nthreads, std::function<void(std::exception_ptr)> handle_exceptions = [](auto e) { std::rethrow_exception(e); }, const std::string& thread_prefix = "GenericThreadPool", std::function<void()> init = [](){});
        
        ~GenericThreadPool() {
            force_stop();
        }
        
        size_t num_threads() const {
            return nthreads;
        }
        
        void resize(size_t num_threads);
        
        size_t queue_length() {
            std::unique_lock<std::mutex> lock(m);
            return q.size();
        }
        
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F, Args...>>
        {
            using return_type = typename std::invoke_result_t<F, Args...>;
            
            auto task = std::make_shared< std::packaged_task<return_type()> >(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            
            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(m);
                
                // don't allow enqueueing after stopping the pool
                if(stop) {
                    Except("enqueue on stopped ThreadPool");
                    return res;
                }
                    
                q.push([task](){ (*task)(); });
            }
            condition.notify_one();
            return res;
        }
        /*void enqueue(const std::function<void()>& fn) {
            {
                std::unique_lock<std::mutex> lock(m);
                q.push(fn);
            }
         
            condition.notify_one();
        }*/
        
        void force_stop() {
            stop = true;
            condition.notify_all();
            
            for (auto &t : thread_pool) {
                t->join();
                delete t;
            }
            
            thread_pool.clear();
        }
        
        void wait() {
            std::unique_lock<std::mutex> lock(m);
            if(q.empty() && (_working == 0))
                return;
            finish_condition.wait(lock, [&](){ return q.empty() && (_working == 0); });
        }
        
        void wait_one();
    };

template<typename F, typename Iterator, typename Pool>
void distribute_vector(F&& fn, Pool& pool, Iterator start, Iterator end, const uint8_t = 5) {
    static const auto threads = cmn::hardware_concurrency();
    int64_t i = 0, N = std::distance(start, end);
    const int64_t per_thread = max(1, N / threads);
    int64_t processed = 0, enqueued = 0;
    std::mutex mutex;
    std::condition_variable variable;
    
    Iterator nex = start;
    
    for(auto it = start; it != end;) {
        auto step = i + per_thread < N ? per_thread : (N - i);
        std::advance(nex, step);
        
        assert(step > 0);
        if(nex == end) {
            fn(i, it, nex, step);
            
        } else {
            ++enqueued;
            
            pool.enqueue([&](auto i, auto it, auto nex, auto step) {
                fn(i, it, nex, step);
                
                std::unique_lock g(mutex);
                ++processed;
                variable.notify_one();
                
            }, i, it, nex, step);
        }
        
        it = nex;
        i += step;
    }
    
    std::unique_lock g(mutex);
    while(processed < enqueued)
        variable.wait(g);
}

    template<typename T>
    class QueueThreadPool {
        std::queue<T> q;
        std::mutex m;
        
        std::condition_variable condition;
        std::condition_variable finish_condition;
        std::vector<std::thread*> thread_pool;
        const size_t nthreads;
        std::atomic_bool stop;
        std::atomic_int _working;
        std::function<void(T&)> work_function;
        
    public:
        QueueThreadPool(size_t nthreads, const std::function<void(T&)>& fn)
            : nthreads(nthreads), stop(false), _working(0), work_function(fn)
        {
            for (size_t i=0; i<nthreads; i++) {
                thread_pool.push_back(new std::thread([&](int idx){
                    std::unique_lock<std::mutex> lock(m);
                    cmn::set_thread_name("QueueThreadPool::thread_"+cmn::Meta::toStr(idx));
                    
                    for(;;) {
                        condition.wait(lock, [&](){ return !q.empty() || stop; });
                        
                        if(q.empty() && stop)
                            break;
                        
                        auto && item = std::move(q.front());
                        q.pop();
                        ++_working;
                        
                        lock.unlock();
                        work_function(item);
                        lock.lock();
                        
                        --_working;
                        finish_condition.notify_one();
                    }
                    
                }, i));
            }
        }
        
        ~QueueThreadPool() {
            stop = true;
            condition.notify_all();
            
            for (auto &t : thread_pool) {
                t->join();
                delete t;
            }
        }

        size_t num_threads() const {
            return nthreads;
        }
        
        void enqueue(T obj) {
            {
                std::unique_lock<std::mutex> lock(m);
                q.push(obj);
            }
            
            condition.notify_one();
        }
        
        void wait() {
            std::unique_lock<std::mutex> lock(m);
            {
                finish_condition.wait(lock, [&](){ return q.empty() && _working.load() == 0; });
            }
        }
    };

    class SimpleThreadPool {
        size_t max_size;
        std::queue<std::thread*> threads;
        
    public:
        SimpleThreadPool(size_t nthreads) : max_size(nthreads)
            {}
        
        void enqueue(std::thread*ptr) {
            while (threads.size() >= max_size) {
                threads.front()->join();
                delete threads.front();
                threads.pop();
            }
            
            threads.push(ptr);
        }
        
        void wait() {
            while (!threads.empty()) {
                threads.front()->join();
                delete threads.front();
                threads.pop();
            }
        }
        
        ~SimpleThreadPool() {
            wait();
        }
    };

    class ThreadPool {
    public:
        ThreadPool(size_t);
        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args) -> std::future<typename std::invoke_result_t<F(Args...)>>;
        ~ThreadPool();
    private:
        // need to keep track of threads so we can join them
        std::vector< std::thread > workers;
        // the task queue
        std::queue< std::function<void()> > tasks;
        
        // synchronization
        std::mutex queue_mutex;
        std::condition_variable condition;
        bool stop;
    };

    // the constructor just launches some amount of workers
    inline ThreadPool::ThreadPool(size_t threads)
    :   stop(false)
    {
        for(size_t i = 0;i<threads;++i)
            workers.emplace_back(
                                 [this]
                                 {
                                     for(;;)
                                     {
                                         std::function<void()> task;
                                         
                                         {
                                             std::unique_lock<std::mutex> lock(this->queue_mutex);
                                             this->condition.wait(lock,
                                                                  [this]{ return this->stop || !this->tasks.empty(); });
                                             if(this->stop && this->tasks.empty())
                                                 return;
                                             task = std::move(this->tasks.front());
                                             this->tasks.pop();
                                         }
                                         
                                         task();
                                     }
                                 }
                                 );
    }

    // add new work item to the pool
    template<class F, class... Args>
    auto ThreadPool::enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result_t<F(Args...)>>
    {
        using return_type = typename std::invoke_result_t<F(Args...)>;
        
        auto task = std::make_shared< std::packaged_task<return_type()> >(
                                                                          std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                                                                          );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            
            // don't allow enqueueing after stopping the pool
            if(stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            
            tasks.emplace([task](){ (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    // the destructor joins all threads
    inline ThreadPool::~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for(std::thread &worker: workers)
            worker.join();
    }

}
