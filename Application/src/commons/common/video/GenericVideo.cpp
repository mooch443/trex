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
            uint8_t max_code = 0;
            uint8_t max_number = 0;
            uint8_t N = narrow_cast<uint8_t>(array_ptr->size());
            //for(auto && [code, number] : *array_ptr) {
            for(uint8_t code=0; code<N; ++code) {
                const auto& number = (*array_ptr)[code];
                if(number > max_number) {
                    max_number = number;
                    max_code = code;
                }
            }
            
            *ptr = max_code;
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

void GenericVideo::generate_average(cv::Mat &av, uint64_t frameIndex) {
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
            Debug("generating average: %d/%d step:%d (frame %d)", (samples - i) / step, int(samples), step, i);
            counted = 0;
        }
        
        if(GlobalSettings::has("terminate") && SETTING(terminate))
            break;
    }
    
    auto image = accumulator.finalize();
    image->get().copyTo(av);
}
