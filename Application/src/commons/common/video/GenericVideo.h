#ifndef _GENERIC_VIDEO_H
#define _GENERIC_VIDEO_H

#include <types.h>
#include <misc/CropOffsets.h>
#include <misc/GlobalSettings.h>

namespace cmn {

//! Interface for things that can load Videos
class GenericVideo {
public:
    virtual cv::Mat frame(uint64_t frameIndex) {
        gpuMat image;
        frame(frameIndex, image);
        cv::Mat dl;
        image.copyTo(dl);
        return dl;
    }
    
    virtual void frame(uint64_t frameIndex, cv::Mat& output) = 0;
    #ifdef USE_GPU_MAT
    virtual void frame(uint64_t globalIndex, gpuMat& output) = 0;
    #endif
    
    virtual const cv::Size& size() const = 0;
    virtual uint64_t length() const = 0;
    virtual bool supports_multithreads() const = 0;
    virtual const cv::Mat& average() const = 0;
    virtual bool has_mask() const = 0;
    virtual const cv::Mat& mask() const = 0;
    virtual bool has_timestamps() const = 0;
    virtual uint64_t timestamp(uint64_t) const {
        U_EXCEPTION("Not implemented.");
    }
    virtual uint64_t start_timestamp() const {
        U_EXCEPTION("Not implemented.");
    }
    virtual short framerate() const = 0;
    
public:
    virtual ~GenericVideo() {}
    virtual CropOffsets crop_offsets() const;
    
    void undistort(const gpuMat& disp, gpuMat& image) const;
    
    virtual void set_offsets(const CropOffsets&) {
        U_EXCEPTION("Not implemented.");
    }
    
    void processImage(const gpuMat& disp, gpuMat& out, bool do_mask = true) const;
    virtual void generate_average(cv::Mat &average, uint64_t frameIndex, std::function<void(float)>&& callback = nullptr);
};

}

#endif
