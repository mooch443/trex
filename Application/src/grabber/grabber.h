#ifndef _GRABBER_H
#define _GRABBER_H

#include <types.h>
#include <misc/ThreadedAnalysis.h>
#include <misc/Median.h>

#include <misc/Timer.h>
#include <pv.h>

#include <video/VideoSource.h>

#include <misc/PylonCamera.h>
#include <misc/Webcam.h>
#include <misc/Camera.h>

#include <misc/ThreadPool.h>
#include <processing/LuminanceGrid.h>

#if CV_MAJOR_VERSION >= 3
#include <opencv2/core/ocl.hpp>
#endif

#if WITH_FFMPEG
#include "tomp4.h"
#endif

#include "gpuImage.h"

using namespace cmn;

class ImageThreads {
    std::function<Image_t*()> _fn_create;
    std::function<bool(const Image_t*, Image_t&)> _fn_prepare;
    std::function<bool(Image_t&)> _fn_load;
    std::function<Queue::Code(Image_t&)> _fn_process;
    
    std::atomic_bool _terminate;
    std::mutex _image_lock;
    std::condition_variable _condition;
    
    std::thread *_load_thread;
    std::thread *_process_thread;
    
    std::deque<Image_t*> _used;
    std::deque<Image_t*> _unused;
    
public:
    ImageThreads(const decltype(_fn_create)& create,
                 const decltype(_fn_prepare)& prepare,
                 const decltype(_fn_load)& load,
                 const decltype(_fn_process)& process);
    
    ~ImageThreads() {
        terminate();
        
        _load_thread->join();
        _process_thread->join();
        
        delete _load_thread;
        delete _process_thread;
        
        // clear cache
        while(!_unused.empty()) {
            delete _unused.front();
            _unused.pop_front();
        }
        while(!_used.empty()) {
            delete _used.front();
            _used.pop_front();
        }
    }
    
    void terminate() { _terminate = true; _condition.notify_all(); }
    
    const std::thread* loading_thread() const { return _load_thread; }
    const std::thread* analysis_thread() const { return _process_thread; }
    
private:
    void loading();
    void processing();
};

namespace track {
class Tracker;
}

class FrameGrabber {
public:
    //typedef ThreadedAnalysis<Image, 10> AnalysisType;
    typedef ImageThreads AnalysisType;
    
    static track::Tracker* tracker_instance();
    struct Task {
        std::future<void> _future;
        std::atomic<bool> _complete = false;
        std::atomic<bool> _valid = true;
        
        Task() = default;
        Task(Task&& task)
            : _future(std::move(task._future)), _complete(task._complete.load()), _valid(task._valid.load())
        {}
    };
    
protected:
    Task _task;
    
    GETTER(cv::Size, cam_size)
    GETTER(cv::Size, cropped_size)
    GETTER(Bounds, crop_rect)
    
    std::unique_ptr<GenericThreadPool> _pool;
    
    AnalysisType* _analysis;
    std::shared_ptr<Image> _current_image;
    gpuMat _average;
    GETTER(cv::Mat, original_average)
    cv::Mat _current_average;
    std::atomic<uint64_t> _current_average_timestamp;
    std::atomic<double> _tracking_time, _saving_time;
    
    GETTER(std::atomic_bool, average_finished)
    GETTER(uint32_t, average_samples)
    GETTER(std::atomic_long, last_index)
    
    //std::chrono::time_point<Image::clock_> _start_timing;
    uint64_t _start_timing;
    std::chrono::time_point<std::chrono::system_clock> _real_timing;
	
    GETTER_PTR(VideoSource*, video)
    VideoSource * _video_mask;
    GETTER_PTR(fg::Camera*, camera)
    
	long _current_fps;
	GETTER(std::atomic<float>, fps)
	Timer _fps_timer;
    
	std::mutex _lock;
    std::mutex _camera_lock;
	
	//std::vector<std::thread*> _pool;
    GETTER_NCONST(pv::File, processed)
    std::atomic_bool _paused;
    
    std::queue<const Image_t*> _image_queue;
    
    std::mutex process_image_mutex;
    std::queue<Image_t*> _unused_process_images;
    
    std::mutex _frame_lock;
    std::unique_ptr<pv::Frame> _last_frame;
    std::unique_ptr<pv::Frame> _noise;
    
    uint64_t previous_time;
    
    std::atomic<double> _processing_timing;
    std::atomic<double> _loading_timing;
    
    LuminanceGrid *_grid;
    
    std::mutex _log_lock, _fps_lock;
    
    FILE* file;
    
    std::thread *_tracker_thread;
    
#if WITH_FFMPEG
    std::thread *mp4_thread;
    FFMPEGQueue* mp4_queue;
#endif
    
public:
    static FrameGrabber* instance;
    static gpuMat gpu_average, gpu_average_original;
    
public:
    FrameGrabber(std::function<void(FrameGrabber&)> callback_before_starting);
	~FrameGrabber();
    
    void prepare_average();
    
    static file::Path make_filename();
    
    bool is_recording() const {
        return GlobalSettings::map().has("recording") && SETTING(recording);
    }
    bool is_paused() const {
        if(!is_recording())
            return false;
        return !_processed.open() || _paused;
    }
    bool load_image(Image_t& current);
    Queue::Code process_image(Image_t& current);
    std::shared_ptr<Image> latest_image();
    
    std::unique_ptr<pv::Frame> last_frame() {
        std::lock_guard<std::mutex> guard(_frame_lock);
        auto last_frame = std::move(_last_frame);
        return last_frame;
    }
    
    std::unique_ptr<pv::Frame> noise() {
        std::lock_guard<std::mutex> guard(_frame_lock);
        auto noise = std::move(_noise);
        return noise;
    }
    
    void write_fps(uint64_t tdelta, uint64_t ts) {
        std::unique_lock<std::mutex> guard(_log_lock);
        if(SETTING(terminate))
            return;
        
        if(!file) {
            file = fopen("fps_program.csv", "wb");
            std::string str = "tdelta,time\n";
            fwrite(str.data(), sizeof(char), str.length(), file);
        }
        std::string str = std::to_string(tdelta) + "," + std::to_string(ts) + "\r\n";
        fwrite(str.data(), sizeof(char), str.length(), file);
    }
    
    cv::Mat average() {
        std::lock_guard<std::mutex> guard(_frame_lock);
        if(_average_finished) {
            cv::Mat temp;
            _average.copyTo(temp);
            return temp;
        }
        return cv::Mat();
    }
    
    file::Path average_name() const;
    
    void safely_close();
    void add_tracker_queue(const pv::Frame&, long_t);
    void update_tracker_queue();
    
    std::atomic_bool _terminate_tracker;
    std::vector<std::unique_ptr<std::thread>> _multi_pool;
    std::condition_variable _multi_variable;
    
private:
    void initialize_video();
    cv::Size determine_resolution();
    
    void apply_filters(gpuMat&);
    void ensure_average_is_ready();
    void update_fps(long_t index, uint64_t stamp, uint64_t tdelta, uint64_t now);
    
    void crop_and_scale(gpuMat&);
    bool add_image_to_average(const Image_t&);
    void initialize(std::function<void(FrameGrabber&)>&& callback_before_starting);
};

#endif
