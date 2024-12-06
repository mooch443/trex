#include "BlurryVideoLoop.h"
#include <misc/AbstractVideoSource.h>
#include <misc/WebcamVideoSource.h>
#include <misc/VideoVideoSource.h>
#include <misc/PVVideoSource.h>
#include <misc/SettingsInitializer.h>

namespace cmn {

BlurryVideoLoop::BlurryVideoLoop(const std::string& name)
    : group(ThreadManager::getInstance().registerGroup(name.empty() ? "BlurryVideo" : name))
{
    ThreadManager::getInstance().addThread(group, "preloader", ManagedThread{
        [this](auto& gid){ preloader_thread(gid); }
    });
}

void BlurryVideoLoop::preloader_thread(const ThreadGroupId& gid) {
    auto video_changed = not _source || between_equals(allowances, 1, 15)
        ? _video_path.get()
        : _video_path.getIfChanged();
    if(video_changed.has_value())
    {
        // video has changed! need to update
        auto& path = video_changed.value();
        Print("[blurry] Video changed to ", path);
        
        std::unique_ptr<AbstractBaseVideoSource> tmp;
        try {
            if(path == file::PathArray{"webcam"}) {
                try {
                    fg::Webcam cam;
                    cam.set_color_mode(ImageMode::RGBA);
                    tmp = std::unique_ptr<AbstractBaseVideoSource>(new WebcamVideoSource{std::move(cam)});
                    _next_frame = {};
                    _last_image_timer.reset();
                    
                } catch(...) {
                    // webcam probably needs allowance
                    auto a = allowances.load();
                    if(a < 15) {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        
                        if(allowances.compare_exchange_strong(a, a + 1))
                            _video_path.set(path);
                        
                        ThreadManager::getInstance().notify(gid);
                    }
                    
                    _intial_resolution_promise.set_value({});
                    _intial_resolution_promise = {};
                }
                
            } else if(path.empty()) {
                // we cant do anything
                _intial_resolution_promise.set_value({});
                _intial_resolution_promise = {};
                
            } else if(path.get_paths().size() == 1
                      && path.get_paths().front().has_extension("pv"))
            {
                auto output = settings::find_output_name(GlobalSettings::map());
                auto video = pv::File::Read(output);
                video.header();
                
                tmp = std::unique_ptr<AbstractBaseVideoSource>(new PVVideoSource{ std::move(video) });
                tmp->set_loop(true);
                _next_frame = 0_f;
                
            } else {
                VideoSource video(path);
                video.set_colors(ImageMode::RGBA);
                tmp = std::unique_ptr<AbstractBaseVideoSource>(new VideoSourceVideoSource{ std::move(video) });
                tmp->set_loop(true);
                _next_frame = 0_f;
            }
            
            if(tmp) {
                allowances = 0;
                _last_image_timer.reset();
                
                _resolution = tmp->size();
                _intial_resolution_promise.set_value(tmp->size());
                _intial_resolution_promise = {};
                _source = std::move(tmp);
                _video_updated = true;
                intermediate = nullptr;
                
                std::unique_lock guard(image_mutex);
                return_image = {};
                transfer_image = {};
            }
            
        } catch(...) {
            // could not load
            _intial_resolution_promise.set_value({});
            _intial_resolution_promise = {};
        }
    }
    
    if(not _source)
        return;
    
    /// here we check whether we need to grab a new frame from
    /// the video source we're using. this could be because
    /// of a timeout or because we're done fading the other one.
    /// or i guess forever, if its a webcam.
    if(not intermediate
       || _last_image_timer.elapsed() > video_frame_time.load()
       || not _source->is_finite()) /// for webcams we need to keep grabbing
    {                               /// so we dont fill up the buffer queue
        auto e = _source->next();
        
        if(e.has_value()) {
            auto &&[index, mat, image] = e.value();
            if(_source->is_finite()
               && index + 1_f >= _source->length())
            {
                if(_next_frame != 0_f) {
                    _next_frame = 0_f;
                    //_source->set_frame(0_f);
                    //Print("set index = ", _next_frame);
                }
            }
            
            if(not _next_frame.valid() || index == _next_frame) {
                //Print("index = ", index);
                _last_blur = -1;
                _last_resolution = {};
                
                if(intermediate)
                    _source->move_back(std::move(intermediate));
                intermediate = std::move(mat);
                _source->move_back(std::move(image));
                _last_image_timer.reset();
                if(_next_frame.valid())
                    _next_frame ++;
            } else {
                //Print("index = ", index, " waiting for ", _next_frame);
                ThreadManager::getInstance().notify(gid);
            }
        }
        else {
            /// oh no, it went wrong!
            FormatError("[blurry] Failed to load video frame: ", e.error());
            
            /// if its a video
            if(_source->is_finite()) {
                if(_next_frame != 0_f) {
                    _source->set_frame(0_f);
                    _next_frame = 0_f;
                }
            }
            
            ThreadManager::getInstance().notify(gid);
        }
    }
    
    if(intermediate) {
        auto p = blur_percentage.load();
        auto res = max_resolution.load();
        
        if(p != _last_blur
           || res != _last_resolution
           || last_update.elapsed() >= 0.1)
        {
            _last_blur = p;
            _last_resolution = res;
            last_update.reset();
            
            VideoFrame local_image;
            if(std::unique_lock guard(image_mutex);
               return_image)
            {
                local_image = std::move(return_image);
            } else {
                local_image.ptr = Image::Make();
                local_image.resolution = res;
            }
            
            double scale;
            render_image(p, scale, *local_image.ptr, res, *intermediate);
            local_image.scale = scale;
            _scale = scale;
            
            if(local_image) {
                
                /// if we have generated an image, push it to
                /// where the GUI can retrieve it:
                {
                    std::unique_lock guard(image_mutex);
                    if(not transfer_image) {
                        transfer_image = std::move(local_image);
                    }
                }
                
                auto fn = _callback.get();
                if(fn)
                    fn();
                
                if(_video_updated) {
                    auto c = _open_callback.get();
                    if(c) {
                        c(_source->info());
                    }
                    _video_updated = false;
                }
            }
        }
    }
    
    uint64_t elapsed = uint64_t(last_update.elapsed() * 1000);
    if(elapsed < 80u) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100u - elapsed));
    }
    //if(not _source->is_finite())
        ThreadManager::getInstance().notify(gid);
}

void BlurryVideoLoop::
     render_image(double blur,
                  double& scale,
                  Image& local_image,
                  const Size2& target_res,
                  const Mat& intermediate)
{
    //Size2 intermediate_size(mat->cols * 0.5, mat->rows * 0.5);
    double ratio = (blur <= 0
                    ? 1.0
                    : saturate(0.35 / blur, 0.1, 1.0));
    auto size = Size2(intermediate.cols, intermediate.rows) * ratio;
    
    if(not target_res.empty()) {
        ratio = min(1, min(target_res.width / size.width,
                           target_res.height / size.height));
        
        size = Size2(size.width * ratio, size.height * ratio);
        //Print("Scaling to size ", size, " with ratio ", ratio, " ", size.div(Size2(intermediate)));
    }
    scale = size.div(Size2(intermediate)).min();
    
    local_image.create(size.height, size.width,
                        intermediate.channels() == 3
                            ? 4
                            : intermediate.channels());
    
    if(intermediate.channels() == 3) {
        Mat tmp;
        cv::cvtColor(intermediate, tmp, cv::COLOR_BGR2BGRA);
        cv::resize(tmp, local_image.get(), size, 0, 0, cv::INTER_CUBIC);
    } else
        cv::resize(intermediate, local_image.get(), size, 0, 0, cv::INTER_CUBIC);
    
    uint8_t amount = saturate(0.04 * size.max(), 5, 30) * blur;
    if(amount > 0) {
        if(amount % 2 == 0)
            amount++;
        
        cv::GaussianBlur(local_image.get(), local_image.get(), Size2(amount), 0);
    }
}

void BlurryVideoLoop::start() {
    _intial_resolution_promise = {};
    _initial_resolution_future = _intial_resolution_promise.get_future();
    ThreadManager::getInstance().startGroup(group);
    if(_initial_resolution_future.wait_for(std::chrono::milliseconds(500)) == std::future_status::ready) 
    {
        _resolution = _initial_resolution_future.get();
    } else
        _resolution = {1024, 512};
}

void BlurryVideoLoop::stop() {
    if(_initial_resolution_future.valid())
        _initial_resolution_future.get();
    ThreadManager::getInstance().terminateGroup(group);
    _source = nullptr;
}

template<typename T>
bool compare_and_swap(std::atomic<T>& atomic_var, const T& new_value) {
    T expected = atomic_var.load();
    if(expected != new_value)
        return atomic_var.compare_exchange_strong(expected, new_value);
    else
        return false;
}

bool BlurryVideoLoop::set_blur_value(double blur) {
    return compare_and_swap(blur_percentage, blur);
}

bool BlurryVideoLoop::set_target_resolution(const Size2& size) {
    return compare_and_swap(max_resolution, size);
}

bool BlurryVideoLoop::set_video_frame_time(double value) {
    if(compare_and_swap(video_frame_time, value)) {
        ThreadManager::getInstance().notify(group);
        return true;
    }
    return false;
}

bool BlurryVideoLoop::set_path(const file::PathArray& array) {
    if(_video_path.set(array)) {
        Print("[blurry] Video changed to ", array);
        ThreadManager::getInstance().notify(group);
        return true;
    }
    return false;
}

void BlurryVideoLoop::set_callback(std::function<void ()> fn) {
    _callback.set(fn);
}
void BlurryVideoLoop::set_open_callback(std::function<void (VideoInfo)> fn) {
    _open_callback.set(fn);
}

[[nodiscard]] BlurryVideoLoop::VideoFrame BlurryVideoLoop::get_if_ready() {
    if(_initial_resolution_future.valid()
       && _initial_resolution_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
    {
        _initial_resolution_future.get();
    }
    
    std::unique_lock guard(image_mutex);
    auto ptr = std::move(transfer_image);
    ThreadManager::getInstance().notify(group);
    return ptr;
}
void BlurryVideoLoop::move_back(Image::Ptr&& image) {
    std::unique_lock guard(image_mutex);
    return_image.ptr = std::move(image);
    ThreadManager::getInstance().notify(group);
}

BlurryVideoLoop::~BlurryVideoLoop() {
    stop();
}

double BlurryVideoLoop::blur() const {
    return blur_percentage.load();
}

Size2 BlurryVideoLoop::resolution() const {
    return _resolution.load();
}

double BlurryVideoLoop::scale() const {
    return _scale.load();
}

}
