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

    tl::expected<std::tuple<Frame_t, useMatPtr_t>, const char*> fetch_next() override;

    uint8_t channels() const override;
    std::string toStr() const override;
    static std::string class_name() { return "WebcamVideoSource"; }
};
