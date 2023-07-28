#pragma once
#include <misc/AbstractVideoSource.h>
#include <grabber/misc/Webcam.h>

class WebcamVideoSource : public AbstractBaseVideoSource {
    fg::Webcam source;
    
public:
    using SourceType = fg::Webcam;
    
public:
    WebcamVideoSource(fg::Webcam&& source);
    ~WebcamVideoSource();

    tl::expected<std::tuple<Frame_t, gpuMatPtr>, const char*> fetch_next() override;

    std::string toStr() const override;
    static std::string class_name() { return "WebcamVideoSource"; }
};
