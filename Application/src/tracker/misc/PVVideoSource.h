#pragma once
#include <misc/AbstractVideoSource.h>
#include <misc/DetectionImageTypes.h>
#include <pv.h>

class PVVideoSource : public AbstractBaseVideoSource {
    pv::File source;
    Image cpuBuffer;
    
public:
    using SourceType = pv::File;
    
public:
    PVVideoSource(pv::File&& source);
    ~PVVideoSource();
    
    std::expected<std::tuple<Frame_t, useMatPtr_t>, std::string> fetch_next() override;

    uint8_t channels() const override;
    std::string toStr() const override;
    static std::string class_name() { return "PVVideoSource"; }
    
    std::set<std::string_view> recovered_errors() const override;
    
protected:
    void undistort(const gpuMat&, gpuMat&) override {}
    void undistort(const cv::Mat&, cv::Mat&) override {}
};
