#include "BlurryVideoLoop.h"
#include <misc/AbstractVideoSource.h>
#include <misc/WebcamVideoSource.h>
#include <misc/VideoVideoSource.h>

namespace cmn {

BlurryVideoLoop::BlurryVideoLoop()
    : group(ThreadManager::getInstance().registerGroup("VideoBackground"))
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
        
        std::unique_ptr<AbstractBaseVideoSource> tmp;
        try {
            if(path == file::PathArray{"webcam"}) {
                try {
                    fg::Webcam cam;
                    cam.set_color_mode(ImageMode::RGB);
                    tmp = std::unique_ptr<AbstractBaseVideoSource>(new WebcamVideoSource{std::move(cam)});
                    
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
                }
                
            } else {
                VideoSource video(path);
                tmp = std::unique_ptr<AbstractBaseVideoSource>(new VideoSourceVideoSource{ std::move(video) });
            }
            
            if(tmp) {
                allowances = 0;
                _last_image_timer.reset();
                _source = std::move(tmp);
            }
            
        } catch(...) {
            // could not load
        }
    }
    
    if(not _source)
        return;
    
    if(not intermediate
       || _last_image_timer.elapsed() > video_frame_time.load()
       || not _source->is_finite()) /// for webcams we need to keep grabbing
    {                               /// so we dont fill up the buffer queue
        auto e = _source->next();
        if(e.has_value()) {
            auto &&[index, mat, image] = e.value();
            if(intermediate)
                _source->move_back(std::move(intermediate));
            intermediate = std::move(mat);
            _source->move_back(std::move(image));
            _last_image_timer.reset();
        }
        else if(_source->is_finite()) {
            _source->set_frame(0_f);
            ThreadManager::getInstance().notify(gid);
        }
    }
    
    if(intermediate) {
        auto p = blur_percentage.load();
        
        Image::Ptr local_image;
        if(std::unique_lock guard(image_mutex);
           return_image)
        {
            local_image = std::move(return_image);
        } else {
            local_image = Image::Make();
        }
        
        render_image(p, *local_image, max_resolution.load(), *intermediate);
        
        if(local_image) {
            /// if we have generated an image, push it to
            /// where the GUI can retrieve it:
            std::unique_lock guard(image_mutex);
            if(not transfer_image)
                transfer_image = std::move(local_image);
        }
    }
    
    if(not _source->is_finite())
        ThreadManager::getInstance().notify(gid);
}

void BlurryVideoLoop::
     render_image(double blur,
                  Image& local_image,
                  const Size2& target_res,
                  const Mat& intermediate)
{
    //Size2 intermediate_size(mat->cols * 0.5, mat->rows * 0.5);
    auto size = Size2(intermediate.cols, intermediate.rows)
                    * (blur <= 0
                       ? 1.0
                       : saturate(0.25 / blur, 0.1, 1.0));
    
    if(not target_res.empty()) {
        double ratio = max(1, min(size.width / target_res.width,
                                  size.height / target_res.height));
        
        size = Size2(size.width * ratio, size.height * ratio);
        //print("Scaling to size ", size, " with ratio ", ratio);
    }
    
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
    
    uint8_t amount = saturate(0.02 * size.max(), 5, 25) * blur;
    if(amount > 0) {
        if(amount % 2 == 0)
            amount++;
        
        cv::GaussianBlur(local_image.get(), local_image.get(), Size2(amount), 0);
    }
}

void BlurryVideoLoop::start() {
    ThreadManager::getInstance().startGroup(group);
}

void BlurryVideoLoop::stop() {
    ThreadManager::getInstance().terminateGroup(group);
    _source = nullptr;
}

void BlurryVideoLoop::set_blur_value(double blur) {
    blur_percentage = blur;
}

void BlurryVideoLoop::set_target_resolution(const Size2& size) {
    max_resolution = size;
}

void BlurryVideoLoop::set_video_frame_time(double value) {
    video_frame_time = value;
    ThreadManager::getInstance().notify(group);
}

void BlurryVideoLoop::set_path(const file::PathArray& array) {
    _video_path.set(array);
    ThreadManager::getInstance().notify(group);
}

[[nodiscard]] Image::Ptr BlurryVideoLoop::get_if_ready() {
    std::unique_lock guard(image_mutex);
    auto ptr = std::move(transfer_image);
    ThreadManager::getInstance().notify(group);
    return ptr;
}
void BlurryVideoLoop::move_back(Image::Ptr&& image) {
    std::unique_lock guard(image_mutex);
    return_image = std::move(image);
    ThreadManager::getInstance().notify(group);
}

BlurryVideoLoop::~BlurryVideoLoop() {
    stop();
}

}
