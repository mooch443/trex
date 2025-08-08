#pragma once
#include <commons.pc.h>
#include <misc/AbstractVideoSource.h>
#include <video/VideoSource.h>
#include <misc/DetectionImageTypes.h>

class VideoSourceVideoSource : public AbstractBaseVideoSource {
    VideoSource source;
    useMatPtr_t tmp;
    Image cpuBuffer;
    
public:
    using SourceType = VideoSource;
    
public:
    VideoSourceVideoSource(VideoSource&& source);
    ~VideoSourceVideoSource();
    
    std::expected<std::tuple<Frame_t, useMatPtr_t>, UnexpectedError_t> fetch_next() override;

    uint8_t channels() const override;
    std::string toStr() const override;
    static std::string class_name() { return "VideoSourceVideoSource"; }
    
    std::set<std::string_view> recovered_errors() const override;
};
