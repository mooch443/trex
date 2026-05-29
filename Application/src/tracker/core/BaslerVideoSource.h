#pragma once

#include <commons.pc.h>
#include <core/AbstractVideoSource.h>

#if WITH_PYLON
#include <grabber/misc/PylonCamera.h>

class BaslerVideoSource : public AbstractBaseVideoSource {
    fg::PylonCamera source;
    Image frame;
    Size2 _size;
    useMat_t _tmp;

public:
    using SourceType = fg::PylonCamera;

    BaslerVideoSource(SourceType&& camera);
    ~BaslerVideoSource();

    AbstractBaseVideoSource::VideoFrame_t fetch_next() override;
    uint8_t channels() const override;
    std::string toStr() const override;

protected:
    void undistort(const gpuMat&, gpuMat&) override {}
    void undistort(const cv::Mat&, cv::Mat&) override {}
};

#endif
