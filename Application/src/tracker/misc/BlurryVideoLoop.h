#pragma once
#include <commons.pc.h>
#include <file/PathArray.h>
#include <misc/Image.h>
#include <misc/ThreadManager.h>
#include <misc/Timer.h>
#include <misc/VideoInfo.h>

class AbstractBaseVideoSource;

namespace cmn {

class BlurryVideoLoop {
    using Mat = cv::Mat;
    using MatPtr = std::unique_ptr<Mat>;
    
private:
    ThreadGroupId group;
    
    MatPtr intermediate{nullptr};
    Timer _last_image_timer;
    
    GuardedProperty<file::PathArray> _video_path;
    
    std::atomic<Size2> max_resolution;
    std::atomic<double> blur_percentage{1};
    std::atomic<double> video_frame_time{0.1};
    
    double _last_blur{0};
    Size2 _last_resolution;
    Timer last_update;
    
    std::unique_ptr<AbstractBaseVideoSource> _source;
    
    std::mutex image_mutex;
    Image::Ptr transfer_image, return_image;

    std::atomic<size_t> allowances{0};
    Frame_t _next_frame;
    GuardedProperty<std::function<void()>> _callback;
    GuardedProperty<std::function<void(VideoInfo)>> _open_callback;
    
    std::mutex _size_mutex;
    std::future<Size2> _initial_resolution_future;
    std::promise<Size2> _intial_resolution_promise;
    std::atomic<Size2> _resolution;
    std::atomic<double> _scale{1};
    
    bool _video_updated{false};
    
public:
    BlurryVideoLoop(const std::string& name = "VideoBackground");
    
    void preloader_thread(const ThreadGroupId& gid);
    
    static void render_image(double blur,
                      double& scale,
                      Image& local_image,
                      const Size2& target_res,
                      const Mat& intermediate);
    
    void start();
    void stop();
    
    bool set_blur_value(double blur);
    bool set_target_resolution(const Size2& size);
    bool set_path(const file::PathArray& array);
    bool set_video_frame_time(double);
    void set_callback(std::function<void()>);
    void set_open_callback(std::function<void (VideoInfo)> fn);
    double blur() const;
    Size2 resolution() const;
    double scale() const;
    
    [[nodiscard]] std::tuple<Image::Ptr, Size2> get_if_ready();
    void move_back(Image::Ptr&& image);
    
    ~BlurryVideoLoop();
};

}
