#pragma once
#include <misc/AbstractVideoSource.h>
#include <grabber/misc/Webcam.h>

class WebcamVideoSource : public AbstractBaseVideoSource {
    fg::Webcam source;
    
public:
    using SourceType = fg::Webcam;
    useMatPtr_t tmp;
    
public:
    WebcamVideoSource(fg::Webcam&& source);
    ~WebcamVideoSource();
    
    AbstractBaseVideoSource::VideoFrame_t fetch_next() override;
    
    uint8_t channels() const override;
    std::string toStr() const override;
    static std::string class_name() { return "WebcamVideoSource"; }
    
protected:
    void undistort(const gpuMat&, gpuMat&) override {}
    void undistort(const cv::Mat&, cv::Mat&) override {}
};
