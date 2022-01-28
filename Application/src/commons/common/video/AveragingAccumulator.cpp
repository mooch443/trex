#include "AveragingAccumulator.h"
#include <misc/GlobalSettings.h>
#include <misc/Image.h>
#include <misc/SpriteMap.h>

namespace cmn {

ENUM_CLASS_DOCS(averaging_method_t,
    "Sum all samples and divide by N.",
    "Calculate a per-pixel median of the samples to avoid noise. More computationally involved than mean, but often better results.",
    "Use a per-pixel minimum across samples. Usually a good choice for short videos with black backgrounds and individuals that do not move much.",
    "Use a per-pixel maximum across samples. Usually a good choice for short videos with white backgrounds and individuals that do not move much."
)

AveragingAccumulator::AveragingAccumulator() {
    _mode = GlobalSettings::has("averaging_method")
        ?  SETTING(averaging_method).template value<averaging_method_t::Class>()
        : averaging_method_t::mean;
}
AveragingAccumulator::AveragingAccumulator(averaging_method_t::Class mode)
    : _mode(mode)
{ }

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

    {
        std::unique_ptr<std::lock_guard<std::mutex>> guard;
        if constexpr(threaded) {
            guard = std::make_unique<std::lock_guard<std::mutex>>(_accumulator_mutex);
        }
        
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
    }
    
    if(_mode == averaging_method_t::mean) {
        f.convertTo(_float_mat, CV_32FC1);
        
        if constexpr(threaded) {
            std::lock_guard guard(_accumulator_mutex);
            cv::add(_accumulator, _float_mat, _accumulator);
        } else
            cv::add(_accumulator, _float_mat, _accumulator);
        
        ++count;
        
    } else if(_mode == averaging_method_t::mode) {
        assert(f.isContinuous());
        assert(f.type() == CV_8UC1);
        
        const uchar* ptr = (const uchar*)f.data;
        const auto end = f.data + f.cols * f.rows;
        auto array_ptr = spatial_histogram.data();
        auto mutex_ptr = spatial_mutex.begin();
        
        assert(spatial_histogram.size() == uint64_t(f.cols) * uint64_t(f.rows));
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
        if constexpr(threaded) {
            std::lock_guard guard(_accumulator_mutex);
            cv::max(_accumulator, f, _accumulator);
        } else
            cv::max(_accumulator, f, _accumulator);
        
    } else if(_mode == averaging_method_t::min) {
        if constexpr(threaded) {
            std::lock_guard guard(_accumulator_mutex);
            cv::min(_accumulator, f, _accumulator);
        } else
            cv::min(_accumulator, f, _accumulator);
        
    } else
        U_EXCEPTION("Unknown averaging_method '%s'.", _mode.name())
}

std::unique_ptr<cmn::Image> AveragingAccumulator::finalize() {
    std::lock_guard guard(_accumulator_mutex);
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

}
