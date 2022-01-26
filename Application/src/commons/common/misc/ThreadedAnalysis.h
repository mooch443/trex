#ifndef _THREADEDANALYSIS_H_
#define _THREADEDANALYSIS_H_

#include <types.h>
#include <misc/GlobalSettings.h>

namespace cmn {
    namespace Queue {
        enum Code {
            ITEM_NEXT,
            ITEM_WAIT,
            ITEM_REMOVE
        };
    }

    namespace ElementLoading {
        enum Type {
            MULTI_THREADED,
            SINGLE_THREADED
        };
    }

    #define MAX_THREADS_CACHE size_t(3)

    /**
     * This class creates two threads for processing data
     * that has to be read from a source and processed in
     * another thread. Data is transported from one thread
     * to another, while keeping future data in a cache
     * to speed up access.
     */
    // Typename T is the type that has to be transported
    template <typename T=unsigned char, size_t _cache_size=48>
    class ThreadedAnalysis {
    public:
        typedef int index_t;
        typedef std::function<bool(const T*, T&)> loading_type; // loads the next element after const T*prev into the next cache element
        typedef std::function<Queue::Code(const T&)> processing_type;
        typedef std::function<T*(void)> create_type;
        typedef std::function<void(T*)> destroy_type;
        typedef std::function<bool(const T*, T&)> prepare_type;
        
        struct Container {
            T* data;
            bool processed;
            bool initialized;
            
            Container(T* ptr) {
                data = ptr;
                processed = false;
                initialized = false;
            }
        };
        
        static const size_t cache_size = _cache_size;
        
    private:
        loading_type _loading;
        processing_type _analysis;
        create_type _create_element;
        destroy_type _destroy_element;
        prepare_type _prepare;
        
        GETTER_PTR(std::thread*, loading_thread)
        GETTER_PTR(std::thread*, analysis_thread)
        
        std::mutex lock;
        
        std::atomic_bool _terminate_threads;
        const ElementLoading::Type _type;
        
        Container** _cache;
        T** _tmp_object;
        
        T* _processed_object;
        long_t _array_index;
        long_t _currently_processed;
        
        GETTER(bool, paused)
        
        enum PausedIndex {
            ANALYSIS_THREAD_PAUSED = 0,
            LOADING_THREAD_PAUSED = 1
        };
        bool _threads_paused[2];
        
    public:
        ThreadedAnalysis(const ElementLoading::Type type,
                         const create_type& create_element,
                         const prepare_type& prepare_element,
                         const loading_type& loading,
                         const processing_type& analysis,
                         const destroy_type& destroy = [](T*obj){ delete obj; });
        ~ThreadedAnalysis();
        
        void reset_cache();
        int fill_state() {
            std::lock_guard<std::mutex> ulock(lock);
            return _array_index;
        }
        
        void terminate() {
            _terminate_threads = true;
            
            if(_loading_thread && _analysis_thread) {
                _loading_thread->join();
                _analysis_thread->join();
                
                delete _loading_thread;
                delete _analysis_thread;
            }
            
            _loading_thread = NULL;
            _analysis_thread = NULL;
        }
        
        std::future<void> set_paused(bool pause) {
            {
                std::lock_guard<std::mutex> guard(lock);
                _paused = pause;
            }
            
            SETTING(analysis_paused) = pause;
            
            //! Wait for the pausing to finish.
            if(std::this_thread::get_id() == _loading_thread->get_id() || std::this_thread::get_id() == _analysis_thread->get_id())
                U_EXCEPTION("Cannot be called from LoadingThread or AnalysisThread (deadlock territory).");
            
            auto task = std::async(std::launch::async, [this](){
                cmn::set_thread_name("set_paused");
                while(is_paused() != _paused)
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
            });
            
            return task;
        }
        
        void pause_from_thread(std::thread::id tid) {
            {
                std::lock_guard<std::mutex> guard(lock);
                _paused = true;
            }
            
            if(tid == _analysis_thread->get_id())
                _threads_paused[PausedIndex::ANALYSIS_THREAD_PAUSED] = true;
            else if(tid == _loading_thread->get_id())
                _threads_paused[PausedIndex::LOADING_THREAD_PAUSED] = true;
            else {
                set_paused(true).wait();
                return;
            }
            
            while(!is_paused())
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            
            SETTING(analysis_paused) = true;
        }
        
        bool is_paused() {
            std::lock_guard<std::mutex> guard(lock);
            return _threads_paused[0] && _threads_paused[1];
        }
        
    private:
        void loading_function();
        void analysis_function();
    };

    #include <misc/ThreadedAnalysis_impl.h>
}

#endif
