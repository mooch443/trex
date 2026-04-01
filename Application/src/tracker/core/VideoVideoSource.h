#pragma once
#include <commons.pc.h>
#include <core/AbstractVideoSource.h>
#include <video/VideoSource.h>
#include <core/DetectionImageTypes.h>

class VideoSourceVideoSource : public AbstractBaseVideoSource {
    VideoSource source;
    useMatPtr_t tmp;
    Image cpuBuffer;
    
public:
    using SourceType = VideoSource;
    
public:
    VideoSourceVideoSource(VideoSource&& source);
    ~VideoSourceVideoSource();
    
    AbstractBaseVideoSource::VideoFrame_t fetch_next() override;

    uint8_t channels() const override;
    std::string toStr() const override;
    static consteval std::string_view class_name() { return "VideoSourceVideoSource"; }
    
    std::set<std::string_view> recovered_errors() const override;
};
