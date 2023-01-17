#include "ImageThreads.h"
#include <misc/ocl.h>

namespace grab {

ImageThreads::ImageThreads(const decltype(_fn_create)& create,
                           const decltype(_fn_prepare)& prepare,
                           const decltype(_fn_load)& load,
                           const decltype(_fn_process)& process)
  : _fn_create(create),
    _fn_prepare(prepare),
    _fn_load(load),
    _fn_process(process),
    _terminate(false),
    _load_thread(NULL),
    _process_thread(NULL)
{
    // create the cache
    std::unique_lock<std::mutex> lock(_image_lock);
    for (int i=0; i<10; i++) {
        _unused.push_front(_fn_create());
    }
    
    _load_thread = new std::thread([this](){loading();});
    _process_thread = new std::thread([this](){ processing();});
}

ImageThreads::~ImageThreads() {
    terminate();

    _load_thread->join();
    _process_thread->join();

    std::unique_lock<std::mutex> lock(_image_lock);

    delete _load_thread;
    delete _process_thread;

    // clear cache
    while (!_unused.empty())
        _unused.pop_front();
    while (!_used.empty())
        _used.pop_front();
}

void ImageThreads::terminate() {
    _terminate = true;
    _condition.notify_all();
}
void ImageThreads::loading() {
    long_t last_loaded = -1;
    cmn::set_thread_name("ImageThreads::loading");
    std::unique_lock guard(_image_lock);

    while (!_terminate) {
        // retrieve images from camera
        if (_unused.empty()) {
            // skip this image. queue is full...
            guard.unlock();
            std::this_thread::sleep_for(std::chrono::microseconds(10));
            guard.lock();

        }
        else {
            auto current = std::move(_unused.front());
            _unused.pop_front();
            guard.unlock();

            if (_fn_prepare(last_loaded, *current)) {
                if (_fn_load(*current)) {
                    last_loaded = current->index();

                    // loading was successful, so push to processing
                    guard.lock();
                    _used.push_front(std::move(current));
                    _condition.notify_one();
                    continue;
                }
            }

            guard.lock();
            _unused.push_front(std::move(current));
        }
    }
    
    _loading_terminated = true;
    print("[load] loading terminated.");
}

void ImageThreads::processing() {
    std::unique_lock<std::mutex> lock(_image_lock);
    ocl::init_ocl();
    cmn::set_thread_name("ImageThreads::processing");
    
    try {
        while(!_loading_terminated || !_used.empty()) {
            // process images and write to file
            _condition.wait_for(lock, std::chrono::milliseconds(1));
            
            while(!_used.empty()) {
                auto current = std::move(_used.back());
                _used.pop_back();
                lock.unlock();
                //print("[proc] processing ", current->index());
                _fn_process(*current);
                
                if(current->image().empty()) {
                    current = _fn_create();
                }
                
                lock.lock();
                assert(!contains(_unused, current));
                _unused.push_back(std::move(current));
            }
        }
    } catch(...) {
        print("Ended.");
    }
    
    print("[proc] processing terminated.");
}

}
