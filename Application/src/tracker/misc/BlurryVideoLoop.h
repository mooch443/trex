#pragma once
#include <commons.pc.h>
#include <file/PathArray.h>
#include <misc/Image.h>
#include <misc/ThreadManager.h>
#include <misc/Timer.h>

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
    
    std::unique_ptr<AbstractBaseVideoSource> _source;
    
    std::mutex image_mutex;
    Image::Ptr transfer_image, return_image;

    std::atomic<size_t> allowances{0};
    
public:
    BlurryVideoLoop();
    
    void preloader_thread(const ThreadGroupId& gid);
    
    static void render_image(double blur,
                      Image& local_image,
                      const Size2& target_res,
                      const Mat& intermediate);
    
    void start();
    void stop();
    
    void set_blur_value(double blur);
    void set_target_resolution(const Size2& size);
    void set_path(const file::PathArray& array);
    void set_video_frame_time(double);
    
    [[nodiscard]] Image::Ptr get_if_ready();
    void move_back(Image::Ptr&& image);
    
    ~BlurryVideoLoop();
};

}
