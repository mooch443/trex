#pragma once
#include <misc/AbstractVideoSource.h>
#include <video/VideoSource.h>
#include <misc/DetectionImageTypes.h>

class VideoSourceVideoSource : public AbstractBaseVideoSource {
    VideoSource source;
    useMatPtr_t tmp;
    Image cpuBuffer;
    
    gpuMat map1;
    gpuMat map2;
    gpuMat gpuBuffer;
    
public:
    using SourceType = VideoSource;
    
public:
    VideoSourceVideoSource(VideoSource&& source);
    ~VideoSourceVideoSource();
    
    tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> fetch_next() override;

    uint8_t channels() const override;
    std::string toStr() const override;
    static std::string class_name() { return "VideoSourceVideoSource"; }
    
    std::set<std::string_view> recovered_errors() const override;
    
protected:
    void undistort(const gpuMat& input, gpuMat& output) override;
    void undistort(const cv::Mat& input, cv::Mat& output) override;
};
