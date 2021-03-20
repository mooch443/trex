#include "GenericVideo.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <grabber/default_config.h>
#include <misc/Image.h>
#include <misc/checked_casts.h>

using namespace cmn;

namespace cmn {
ENUM_CLASS_DOCS(averaging_method_t,
    "Sum all samples and divide by N.",
    "Calculate a per-pixel median of the samples to avoid noise. More computationally involved than mean, but often better results.",
    "Use a per-pixel minimum across samples. Usually a good choice for short videos with black backgrounds and individuals that do not move much.",
    "Use a per-pixel maximum across samples. Usually a good choice for short videos with white backgrounds and individuals that do not move much."
)
}

CropOffsets GenericVideo::crop_offsets() const {
    return SETTING(crop_offsets);
}

void AveragingAccumulator::add(const Mat& f) {
    _add<false>(f);
}

void AveragingAccumulator::add_threaded(const Mat& f) {
    _add<true>(f);
}

template<bool threaded>
void AveragingAccumulator::_add(const Mat &f) {
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
        assert(f.type() == CV_8UC1);
        
        const uchar* ptr = (const uchar*)f.data;
        const auto end = f.data + f.cols * f.rows;
        auto array_ptr = spatial_histogram.data();
        auto mutex_ptr = spatial_mutex.begin();
        
        assert(spatial_histogram.size() == (uint64_t)(f.cols * f.rows));
        if constexpr(threaded) {
            for (; ptr != end; ++ptr, ++array_ptr, ++mutex_ptr) {
                (*mutex_ptr)->lock();
                ++((*array_ptr)[*ptr]);
                (*mutex_ptr)->unlock();
            }
            
        } else {
            for (; ptr != end; ++ptr, ++array_ptr)
                ++((*array_ptr)[*ptr]);
        }
        
    } else if(_mode == averaging_method_t::max) {
        cv::max(_accumulator, f, _accumulator);
    } else if(_mode == averaging_method_t::min) {
        cv::min(_accumulator, f, _accumulator);
    } else
        U_EXCEPTION("Unknown averaging_method '%s'.", _mode.name())
}

std::unique_ptr<cmn::Image> AveragingAccumulator::finalize() {
    auto image = std::make_unique<cmn::Image>(_accumulator.rows, _accumulator.cols, 1);
    
    if(_mode == averaging_method_t::mean) {
        cv::divide(_accumulator, cv::Scalar(count), _local);
        _local.convertTo(image->get(), CV_8UC1);
        
    } else if(_mode == averaging_method_t::mode) {
        _accumulator.copyTo(image->get());
        
        auto ptr = image->data();
        const auto end = image->data() + image->cols * image->rows;
        auto array_ptr = spatial_histogram.data();
        
        for (; ptr != end; ++ptr, ++array_ptr) {
            *ptr = std::distance(array_ptr->begin(), std::max_element(array_ptr->begin(), array_ptr->end()));
        }
        
    } else
        _accumulator.copyTo(image->get());
    
    return image;
}

void GenericVideo::undistort(const gpuMat& disp, gpuMat &image) const {
    if (GlobalSettings::map().has("cam_undistort") && SETTING(cam_undistort)) {
        static cv::Mat map1;
        static cv::Mat map2;
        if(map1.empty())
            GlobalSettings::get("cam_undistort1").value<cv::Mat>().copyTo(map1);
        if(map2.empty())
            GlobalSettings::get("cam_undistort2").value<cv::Mat>().copyTo(map2);
        
        if(map1.cols == disp.cols && map1.rows == disp.rows && map2.cols == disp.cols && map2.rows == disp.rows)
        {
            if(!map1.empty() && !map2.empty()) {
                Debug("Undistorting %dx%d", disp.cols, disp.rows);
                
                static gpuMat _map1, _map2;
                if(_map1.empty())
                    map1.copyTo(_map1);
                if(_map2.empty())
                    map2.copyTo(_map2);
                
                static gpuMat input;
                disp.copyTo(input);
                
                cv::remap(input, image, _map1, _map2, cv::INTER_LINEAR, cv::BORDER_DEFAULT);
                //output.copyTo(image);
            } else {
                Warning("remap maps are empty.");
            }
        } else {
            Error("Undistortion maps are of invalid size (%dx%d vs %dx%d).", map1.cols, map1.rows, disp.cols, disp.rows);
        }
        
        
        //cv::remap(display, display, map1, map2, cv::INTER_LINEAR, cv::BORDER_DEFAULT);
    }
}

void GenericVideo::processImage(const gpuMat& display, gpuMat&out, bool do_mask) const {
    static Timing timing("processImage");
    timing.start_measure();
    //gpuMat display = disp;
    //undistort(disp, display);
    
    /*
    //cv::resize(display, display, cv::Size(display.cols/2, display.rows/2));
    
    cv::Rect rect(offsets.x, offsets.y,
                  display.cols-offsets.width-offsets.x,
                  display.rows-offsets.height-offsets.y);*/
    
    /*float angle = -1;
     cv::Mat rot_mat = getRotationMatrix2D(cv::Point(processed.cols/2, processed.rows/2), angle, 1.0);
     cv::Rect bbox = cv::RotatedRect(cv::Point(processed.cols/2, processed.rows/2), cv::Size(processed.size[1], processed.size[0]), angle).boundingRect();
     
     rot_mat.at<double>(0,2) += bbox.width/2.0 - processed.cols/2;
     rot_mat.at<double>(1,2) += bbox.height/2.0 - processed.rows/2;
     
     cv::warpAffine(processed, processed, rot_mat, bbox.size(), cv::INTER_LINEAR);*/
    
    gpuMat use;
    assert(display.channels() == 1);
    /*if (display.channels() > 1) {
        const uint64_t channel = SETTING(color_channel);
        
        std::vector<gpuMat> split;
        cv::split(display, split);
        use = split[channel];
        
    } else*/ use = display;
    
    //cv::Mat processed;
    
    if (has_mask() && do_mask) {
        if(this->mask().rows == use.rows && this->mask().cols == use.cols) {
            out = use.mul(this->mask());
        } else {
            const auto offsets = crop_offsets();
            out = use(offsets.toPixels(Size2(display.cols, display.rows))).mul(this->mask());
        }
    } else {
        use.copyTo(out);
    }
    
    timing.conclude_measure();
    
    //return processed;
}

void GenericVideo::generate_average(cv::Mat &av, uint64_t frameIndex, std::function<void(float)>&& callback) {
    if(length() < 10) {
        gpuMat average;
        av.copyTo(average);
        this->processImage(average, average);
        return;
    }
    AveragingAccumulator accumulator;
    
    Debug("Generating average for frame %d (method='%s')...", frameIndex, accumulator.mode().name());
    
    float samples = GlobalSettings::has("average_samples") ? (float)SETTING(average_samples).value<uint32_t>() : (length() * 0.1f);
    const auto step = narrow_cast<uint>(max(1, length() / samples));
    
    cv::Mat f;
    uint64_t counted = 0;
    for(long_t i=length() ? length()-1 : 0; i>=0; i-=step) {
        frame((uint64_t)i, f);
        
        assert(f.channels() == 1);
        accumulator.add(f);
        counted += step;
        
        if(counted > float(length()) * 0.1) {
            if(callback)
                callback(float(samples - i) / float(step));
            Debug("generating average: %d/%d step:%d (frame %d)", (samples - i) / step, int(samples), step, i);
            counted = 0;
        }
        
        if(GlobalSettings::has("terminate") && SETTING(terminate))
            break;
    }
    
    auto image = accumulator.finalize();
    image->get().copyTo(av);
}
