#include "GenericVideo.h"
#include <misc/Timer.h>
#include <misc/GlobalSettings.h>
#include <grabber/default_config.h>

using namespace cmn;

CropOffsets GenericVideo::crop_offsets() const {
    return SETTING(crop_offsets);
}

void GenericVideo::undistort(const gpuMat& disp, gpuMat &image) const {
    if (GlobalSettings::map().has("cam_undistort") && SETTING(cam_undistort)) {
        static cv::Mat map1;
        static cv::Mat map2;
        if(map1.empty())
            GlobalSettings::get("cam_undistort1").value<cv::Mat>().copyTo(map1);
        if(map2.empty())
            GlobalSettings::get("cam_undistort2").value<cv::Mat>().copyTo(map2);
        
        if(map1.cols == disp.cols && map1.rows == disp.cols && map2.cols == disp.cols && map2.rows == disp.rows)
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
    gpuMat average;
    av.copyTo(average);
    
    Debug("Generating average for frame %d...", frameIndex);
    
    //const uint64_t channel = SETTING(color_channel);
    double count = 0;
    float samples = GlobalSettings::has("average_samples") ? SETTING(average_samples).value<int>() : (length() * 0.01);
    const int step = max(1, length() / samples);
    
    gpuMat float_mat;
    gpuMat f, ref;
    std::vector<gpuMat> vec;
    
    auto averaging_method = GlobalSettings::has("averaging_method") ?  SETTING(averaging_method).value<grab::averaging_method_t::Class>() : grab::averaging_method_t::mean;
    bool use_mean = averaging_method != grab::averaging_method_t::max && averaging_method != grab::averaging_method_t::min;
    Debug("Use max: %d", !use_mean);
    
    if(average.empty() || average.cols != size().width || average.rows != size().height) {
        average = gpuMat::zeros(size().height, size().width, use_mean ? CV_32FC1 : CV_8UC1);
    } else
        average = cv::Scalar(0);
    
    if (length() < 10) {
        processImage(average, average);
        return;
    }
    
    if(use_mean)
        float_mat = gpuMat::zeros(average.rows, average.cols, CV_32FC1);
    else
        float_mat = gpuMat::zeros(average.rows, average.cols, CV_8UC1);
    
    cv::Mat local;
    //for (uint64_t i=0; i<length(); i+=step) {
    uint64_t counted = 0;
    for(long_t i=length() ? length()-1 : 0; i>=0; i-=step) {
        frame(i, f);
        
        assert(f.channels() == 1);
        ref = f;
        
        if(use_mean) {
            ref.convertTo(float_mat, CV_32FC1, 1.0/255.0);
            //Debug("%d,%d - %d,%d", float_mat.cols, float_mat.rows, average.cols, average.rows);
            cv::add(average, float_mat, average);
        } else {
            ref.copyTo(local);
            
            if(averaging_method == grab::averaging_method_t::max)
                av = cv::max(av, local);
            else
                av = cv::min(av, local);
        }
        
        ++count;
        counted += step;
        
        if(counted > float(length()) * 0.1) {
            Debug("generating average: %d/%d step:%d (frame %d)", (samples - i) / step, int(samples), step, i);
            counted = 0;
        }
        if(GlobalSettings::has("terminate") && SETTING(terminate))
            break;
    }
    
    if(use_mean) {
        cv::divide(average, cv::Scalar(count), average);
        average.convertTo(av, CV_8UC1, 255.0);
    } else
        average.copyTo(av);
}
