#ifndef _GENERIC_VIDEO_H
#define _GENERIC_VIDEO_H

#include <types.h>
#include <misc/CropOffsets.h>
#include <misc/GlobalSettings.h>

namespace cmn { class GenericVideo; }
namespace cmn {
ENUM_CLASS(averaging_method_t, mean, mode, max, min)
ENUM_CLASS_HAS_DOCS(averaging_method_t)

class AveragingAccumulator {
public:
    using Mat = cv::Mat;
    
protected:
    GETTER(averaging_method_t::Class, mode)
    Mat _accumulator;
    Mat _float_mat;
    Mat _local;
    double count = 0;
    bool use_mean;
    Size2 _size;
    std::vector<std::array<uint8_t, 256>> spatial_histogram;
    std::vector<std::unique_ptr<std::mutex>> spatial_mutex;
    
public:
    AveragingAccumulator() {
        _mode = GlobalSettings::has("averaging_method")
            ?  SETTING(averaging_method).template value<averaging_method_t::Class>()
            : averaging_method_t::mean;
    }
    AveragingAccumulator(averaging_method_t::Class mode)
        : _mode(mode)
    { }
    
    template<bool threaded = false>
    void add(const Mat &f) {
        assert(f.channels() == 1);
        assert(f.type() == CV_8UC1);
        
        // initialization code
        if(_accumulator.empty()) {
            _size = Size2(f.cols, f.rows);
            _accumulator = cv::Mat::zeros((int)_size.height, (int)_size.width, _mode == averaging_method_t::mean ? CV_32FC1 : CV_8UC1);
            if(_mode == averaging_method_t::min)
                _accumulator.setTo(255);
            
            if(_mode == averaging_method_t::mode) {
                spatial_histogram.resize(size_t(f.cols) * size_t(f.rows));
                for(uint64_t i=0; i<spatial_histogram.size(); ++i) {
                    std::fill(spatial_histogram.at(i).begin(), spatial_histogram.at(i).end(), 0);
                    spatial_mutex.push_back(std::make_unique<std::mutex>());
                }
            }
        }
        
        if(_mode == averaging_method_t::mean) {
            f.convertTo(_float_mat, CV_32FC1);
            cv::add(_accumulator, _float_mat, _accumulator);
            ++count;
            
        } else if(_mode == averaging_method_t::mode) {
            assert(f.isContinuous());
            
            auto ptr = f.data;
            const auto end = f.data + f.cols * f.rows;
            auto array_ptr = spatial_histogram.data();
            auto mutex_ptr = spatial_mutex.begin();
            
            assert(spatial_histogram.size() == (uint64_t)(f.cols * f.rows));
            if constexpr(threaded) {
                for (; ptr != end; ++ptr, ++array_ptr, ++mutex_ptr) {
                    (*mutex_ptr)->lock();
                    ++(*array_ptr)[*ptr];
                    (*mutex_ptr)->unlock();
                }
                
            } else {
                for (; ptr != end; ++ptr, ++array_ptr)
                    ++(*array_ptr)[*ptr];
            }
            
        } else if(_mode == averaging_method_t::max) {
            cv::max(_accumulator, f, _accumulator);
        } else if(_mode == averaging_method_t::min) {
            cv::min(_accumulator, f, _accumulator);
        } else
            U_EXCEPTION("Unknown averaging_method '%s'.", _mode.name())
    }
    
    std::unique_ptr<cmn::Image> finalize();
};

}

//! Interface for things that can load Videos
class cmn::GenericVideo {
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

#endif
