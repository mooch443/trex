#pragma once

#include <misc/ThreadedAnalysis.h>
#include "gpuImage.h"

namespace grab {

using namespace cmn;

class ImageThreads {
    std::function<ImagePtr()> _fn_create;
    std::function<bool(long_t, Image_t&)> _fn_prepare;
    std::function<bool(Image_t&)> _fn_load;
    std::function<Queue::Code(Image_t&)> _fn_process;
    
    std::atomic_bool _terminate{false}, _loading_terminated{false};
    std::mutex _image_lock;
    std::condition_variable _condition;
    
    std::thread *_load_thread;
    std::thread *_process_thread;
    
    std::deque<ImagePtr> _used;
    std::deque<ImagePtr> _unused;
    
public:
    ImageThreads(const decltype(_fn_create)& create,
                 const decltype(_fn_prepare)& prepare,
                 const decltype(_fn_load)& load,
                 const decltype(_fn_process)& process);
    
    ~ImageThreads();
    
    void terminate();
    
    const std::thread* loading_thread() const { return _load_thread; }
    const std::thread* analysis_thread() const { return _process_thread; }
    
private:
    void loading();
    void processing();
};

}
