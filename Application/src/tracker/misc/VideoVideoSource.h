#pragma once
#include <misc/AbstractVideoSource.h>
#include <video/VideoSource.h>
#include <misc/TileImage.h>

class VideoSourceVideoSource : public AbstractBaseVideoSource {
    VideoSource source;
    gpuMatPtr tmp;
    cv::Mat cpuBuffer;
    
public:
    using SourceType = VideoSource;
    
public:
    VideoSourceVideoSource(VideoSource&& source);
    ~VideoSourceVideoSource();
    
    tl::expected<std::tuple<Frame_t, AbstractBaseVideoSource::gpuMatPtr>, const char*> fetch_next() override;

    std::string toStr() const override;
    static std::string class_name() { return "VideoSourceVideoSource"; }
};
