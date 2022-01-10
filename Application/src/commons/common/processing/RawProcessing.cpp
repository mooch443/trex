#include "RawProcessing.h"
#include "CPULabeling.h"
#include "misc/GlobalSettings.h"
#include "misc/Timer.h"
#include <misc/ocl.h>
#include <processing/LuminanceGrid.h>

using namespace cmn;

gpuMat gpu_dilation_element;
std::mutex mutex;

template<typename Calc = float, typename A = uchar, typename B = uchar>
class SubtractConvert : public cv::ParallelLoopBody
{
private:
    const A * p0;
    const B * p1;
    uchar *out;
    
public:
    SubtractConvert(const A* ptr0, const B* ptr1, uchar* output ) : p0(ptr0), p1(ptr1), out(output) {}
    virtual void operator()( const cv::Range &r ) const {
        for ( int i = r.start; i != r.end; ++i)
            out[i] = cv::saturate_cast<uchar>(abs(Calc(p0[i]) - Calc(p1[i])));
    }
};

RawProcessing::RawProcessing(const gpuMat &average, const LuminanceGrid* ) : _average(&average)//, _grid(grid)
{ }

void RawProcessing::generate_binary(const gpuMat& input, cv::Mat& output) {
    assert(input.type() == CV_8UC1);
    
    static bool enable_diff = SETTING(enable_difference);
    static bool enable_abs_diff = SETTING(enable_absolute_difference);
    static std::atomic_bool registered_callback = false;
    static float adaptive_threshold_scale = 0;
    static int threshold = 25, threshold_maximum = 255;
    static bool use_closing = false;
    static int closing_size = 1;
    static bool use_adaptive_threshold = false;
    static int32_t dilation_size = 0;
    static bool image_adjust = false, image_square_brightness = false, image_invert = false;
    static float image_contrast_increase = 1, image_brightness_increase = 0;
    
    bool expected = false;
    if(registered_callback.compare_exchange_strong(expected, true)) {
        const char* ptr = "RawProcessing";
        sprite::Map::callback_func callback = [ptr](sprite::Map::Signal signal, sprite::Map& map, auto& key, auto&value)
        {
            if(signal == sprite::Map::Signal::EXIT) {
                map.unregister_callback(ptr);
                return;
            }
            
            if(key == std::string("enable_difference"))
                enable_diff = value.template value<bool>();
            else if(key == std::string("enable_absolute_difference"))
                enable_abs_diff = value.template value<bool>();
            else if(key == std::string("adaptive_threshold_scale"))
                adaptive_threshold_scale = value.template value<float>();
            else if(key == std::string("threshold"))
                threshold = value.template value<int>();
            else if(key == std::string("threshold_maximum"))
                threshold_maximum = value.template value<int>();
            else if(key == std::string("use_closing"))
                use_closing = value.template value<bool>();
            else if(key == std::string("closing_size"))
                closing_size = value.template value<int>();
            else if(key == std::string("use_adaptive_threshold"))
                use_adaptive_threshold = value.template value<bool>();
            else if(key == std::string("dilation_size"))
                dilation_size = value.template value<int32_t>();
            else if(key == std::string("image_adjust"))
                image_adjust = value.template value<bool>();
            else if(key == std::string("image_square_brightness"))
                image_square_brightness = value.template value<bool>();
            else if(key == std::string("image_contrast_increase"))
                image_contrast_increase = value.template value<float>();
            else if(key == std::string("image_brightness_increase"))
                image_brightness_increase = value.template value<float>();
            else if(key == std::string("image_invert"))
                image_invert = value.template value<bool>();
        };
        GlobalSettings::map().register_callback(ptr, callback);
        
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "enable_difference", GlobalSettings::get("enable_difference").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "enable_absolute_difference", GlobalSettings::get("enable_absolute_difference").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "square_brightness", GlobalSettings::get("square_brightness").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "adaptive_threshold_scale", GlobalSettings::get("adaptive_threshold_scale").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "threshold", GlobalSettings::get("threshold").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "threshold_maximum", GlobalSettings::get("threshold_maximum").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "use_closing", GlobalSettings::get("use_closing").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "closing_size", GlobalSettings::get("closing_size").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "use_adaptive_threshold", GlobalSettings::get("use_adaptive_threshold").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "dilation_size", GlobalSettings::get("dilation_size").get());
        
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "image_adjust", GlobalSettings::get("image_adjust").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "image_square_brightness", GlobalSettings::get("image_square_brightness").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "image_contrast_increase", GlobalSettings::get("image_contrast_increase").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "image_brightness_increase", GlobalSettings::get("image_brightness_increase").get());
        callback(sprite::Map::Signal::NONE, GlobalSettings::map(), "image_invert", GlobalSettings::get("image_invert").get());
    }
    
    //static Timing timing("thresholding", 30);
    //TakeTiming take(timing);
    
    static gpuMat *INPUT, *OUTPUT;
    INPUT = &_buffer0;
    OUTPUT = &_buffer1;
    
#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#ifndef NDEBUG
#define CALLCV( X ) { X; std::swap(INPUT, OUTPUT); Debug("(%s:%d) %s: %dx%d vs. %dx%d", __FILENAME__, __LINE__, #X, INPUT->cols, INPUT->rows, OUTPUT->cols, OUTPUT->rows); }
#else
#define CALLCV( X ) { X; std::swap(INPUT, OUTPUT); }
#endif
    
    //cv::Mat after_threshold, after_dilation;
    
    if (_average->empty()) {
        U_EXCEPTION("Average image is empty.");
    }
    
    assert(_average->type() == CV_8UC1);
    assert(_buffer0.type() == CV_8UC1);
    
    //cv::equalizeHist( input, _buffer0 );
    //cv::equalizeHist(*_average, _buffer1);
    //cv::absdiff(_buffer0, _buffer1, _buffer1);
    
    /*cv::Mat m;
    input.copyTo(m);
    tf::imshow("input", m);
    _average->copyTo(m);
    tf::imshow("average", m);*/
    
#ifndef NDEBUG
    Debug("enable_diff %d / enable_abs_diff %d", enable_diff, enable_abs_diff);
    Debug("Average: %dx%d Input: %dx%d", _average->cols, _average->rows, input.cols, input.rows);
#endif
    
    /*cv::Mat local;
    input.copyTo(local);
    tf::imshow("->input", local);*/
    if(image_invert) {
        cv::subtract(cv::Scalar(255), input, *INPUT);
    } else
        input.copyTo(*INPUT);
    
    //INPUT->copyTo(local);
    //tf::imshow("->input->", local);
    
    if(enable_diff) {
        if(enable_abs_diff) {
            CALLCV( cv::absdiff(*INPUT, *_average, *OUTPUT) );
        } else {
            CALLCV( cv::subtract(*_average, *INPUT, *OUTPUT) );
        }
        
        
        //INPUT->copyTo(local);
        //tf::imshow("difference", local);
        
    }
    
    /*if(image_adjust) {
        
        //cv::Mat local;
        //_buffer1.copyTo(local);
        //tf::imshow("before", local);
        
        float alpha = image_contrast_increase / 255.f;
        float beta = image_brightness_increase;
        _buffer1.convertTo(_buffer0, CV_32FC1, alpha, beta);
        
        if(image_square_brightness)
            cv::multiply(_buffer0, _buffer0, _buffer0);
        
        // normalize resulting values between 0 and 1
        cv::threshold(_buffer0, _buffer0, 1, 1, cv::THRESH_TRUNC);
        //cv::equalizeHist(_buffer1, _buffer0);
        
        //_buffer1.convertTo(_buffer0, CV_32FC1, 1./255.f);
        
        //cv::add(_buffer0, 1, _buffer1);
        //cv::multiply(_buffer1, _buffer1, _buffer0);
        //cv::multiply(_buffer0, _buffer0, _buffer1);
        
        
        //cv::multiply(_buffer1, _buffer1, _buffer1);
        //cv::subtract(_buffer1, 1, _buffer0);
        
        //cv::threshold(_buffer0, _buffer0, 1, 1, CV_THRESH_TRUNC);
        //cv::multiply(_buffer0, 255, _buffer0);
        
        _buffer0.convertTo(_buffer1, CV_8UC1, 255);
        //_buffer1.copyTo(local);
        //tf::imshow("after", local);
        
    } else if(_grid) {
        _buffer1.convertTo(_buffer0, CV_32FC1);
        cv::divide(_buffer0, _grid->relative_brightness(), _buffer1);
        _buffer1.convertTo(_buffer1, CV_8UC1);
    }*/
    
    int gauss = INPUT->cols * adaptive_threshold_scale;
    if (gauss % 2 == 0) {
        gauss++;
    }
    if(gauss < 3)
        gauss = 3;
    
    //cv::Mat labels, distance;
    static gpuMat diff;
    
    if(dilation_size) {
        std::lock_guard<std::mutex> guard(mutex);
        if(gpu_dilation_element.empty()) {
            
            const cv::Mat dilation_element = cv::Mat::ones(abs(dilation_size), abs(dilation_size), CV_8UC1); //cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*abs(dilation_size) + 1, 2*abs(dilation_size)+1 ), cv::Point( abs(dilation_size), abs(dilation_size) ) );
            
#ifndef NDEBUG
            Debug("copy %d,%d %d %dx%d", dilation_element.cols, dilation_element.rows, dilation_size, gpu_dilation_element.cols, gpu_dilation_element.rows);
#endif
            dilation_element.copyTo(gpu_dilation_element);
        }
        
        
       /*cv::distanceTransform(_buffer1, distance, labels, cv::DIST_L2, 5);
        
       double mi, ma;
       cv::minMaxLoc(distance, &mi, &ma);
       if(ma == mi)
           ma = mi + 1;
        //distance = (ma - mi) - (distance - mi);
        
        distance = distance - mi;
       distance.convertTo(_buffer1, CV_8UC1, 255.0 / (ma - mi));*/
    }
    
#ifndef NDEBUG
    Debug("dilation_size %d", dilation_size);
#endif
    if(dilation_size != 0)
        INPUT->copyTo(diff);
    
    /*cv::Mat dist_transform;
    cv::Mat local, sure_bg, sure_fg, bgr;
    cv::Mat unknown;
    
    
    
    auto segment = [&]() {
        std::lock_guard<std::mutex> guard(mutex);
        cv::dilate(_buffer1, sure_bg, gpu_dilation_element, Vec2(-1), 3);
        
        
        cv::distanceTransform(_buffer1, dist_transform, labels, cv::DIST_L2, 5);
        
        double mi, ma;
        cv::minMaxLoc(dist_transform, &mi, &ma);
        cv::threshold(dist_transform, sure_fg, 0.65 * ma, 255, 0);
        sure_fg.convertTo(sure_fg, CV_8UC1);
        tf::imshow("sure_fg", sure_fg);
        
        dist_transform.convertTo(local, CV_8UC1, 20);
        tf::imshow("dist_transform", local);
        
        cv::subtract(sure_bg, sure_fg, unknown);

        cv::connectedComponents(sure_fg, labels);
        cv::add(labels, cv::Scalar(1), labels);
        
        labels.convertTo(local, CV_8UC1);
        resize_image(local, 0.3);
        tf::imshow("labels1", local);
        
        labels.setTo(0, unknown);
        
        cv::minMaxLoc(labels, &mi, &ma);
        labels.convertTo(local, CV_8UC1, 255.0 / ma);
        resize_image(local, 0.3);
        tf::imshow("labels2", local);
        
        tf::imshow("unknown", 255 - unknown);
        
        cv::cvtColor(_buffer1, bgr, cv::COLOR_GRAY2BGR);
        cv::watershed(bgr, labels);
        
        cv::Mat mask;
        cv::inRange(labels, -1, -1, mask);
        labels.setTo(0, mask);
        
        cv::minMaxLoc(labels, &mi, &ma);
        labels.convertTo(local, CV_8UC1, 255.0 / ma);
        resize_image(local, 0.3);
        tf::imshow("labels3", local);
        
        cv::minMaxLoc(labels, &mi, &ma);
        Debug("%f %f", mi, ma);
        //cv::cvtColor(_buffer1, _buffer1, cv::COLOR_BGR2GRAY);
        cv::Mat labels2;
        cv::inRange(labels, 2, ma + 1, labels2);
        //cv::minMaxLoc(labels, &mi, &ma);
        
        cv::minMaxLoc(labels2, &mi, &ma);
        Debug("second %f - %f Empty", mi, ma);
        
        labels2.convertTo(_buffer1, CV_8UC1);
        labels2.copyTo(local);
        resize_image(local, 0.3);
        tf::imshow("labels4", local * 255.0 / ma);
        
        //cv::threshold(labels, _buffer1, 1, 255, cv::THRESH_BINARY);
    };*/
    
    if(use_closing) {
        if(use_adaptive_threshold) {
            CALLCV( cv::adaptiveThreshold(*INPUT, *OUTPUT, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, gauss, -threshold) )
        } else {
            if(threshold_maximum < 255) {
                CALLCV( cv::inRange(*INPUT, threshold, threshold_maximum, *OUTPUT) )
            } else
                CALLCV( cv::threshold(*INPUT, *OUTPUT, abs(threshold), 255, cv::THRESH_BINARY) )
        }
        
        if(threshold < 0) {
            CALLCV( cv::subtract(255, *INPUT, *OUTPUT) )
        }
        
        static const int morph_size = abs(closing_size);
        static const cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
        static gpuMat gpu_element;
        if(gpu_element.empty())
            element.copyTo(gpu_element);
        
        CALLCV( cv::dilate(*INPUT, *OUTPUT, gpu_element) )
        CALLCV( cv::erode(*INPUT, *OUTPUT, gpu_element) )
        
        //_buffer1.copyTo(after_threshold);
        
        if(dilation_size > 0) {
            //Debug("Dilation %d", dilation_size);
            CALLCV( cv::dilate(*INPUT, *OUTPUT, gpu_dilation_element) )
        } else if(dilation_size < 0) {
            CALLCV( cv::erode(*INPUT, *OUTPUT, gpu_dilation_element) )
            /*//Debug("Erosion %d", dilation_size);
            cv::erode(_buffer1, _buffer1, gpu_dilation_element);
            cv::erode(_buffer1, _buffer1, gpu_dilation_element);*/
            
            CALLCV( INPUT->convertTo(*OUTPUT, CV_8UC1) )
            CALLCV( diff.copyTo(*OUTPUT, *INPUT) )
            CALLCV( cv::threshold(*INPUT, *OUTPUT, abs(threshold), 255, cv::THRESH_BINARY) )
            
            CALLCV( cv::dilate(*INPUT, *OUTPUT, gpu_element) )
            CALLCV( cv::erode(*INPUT, *OUTPUT, gpu_element) )
            
            //cv::morphologyEx(_buffer1, _buffer1, cv::MORPH_TOPHAT, gpu_dilation_element);
            
            //segment();
        }
        
        //_buffer1.copyTo(after_dilation);
        
    } else {
        static const int morph_size = closing_size;
        static const cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );
        static gpuMat gpu_element;
        
        if(gpu_element.empty())
            element.copyTo(gpu_element);
        
        if(use_adaptive_threshold) {
            CALLCV( cv::adaptiveThreshold(*INPUT, *OUTPUT, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, gauss, -threshold) )
            
        } else {
            if(threshold_maximum < 255) {
                CALLCV( cv::inRange(*INPUT, threshold, threshold_maximum, *OUTPUT) )
            } else {
                CALLCV( cv::threshold(*INPUT, *OUTPUT, abs(threshold), 255, cv::THRESH_BINARY) )
            }
        }
        
        if(threshold < 0) {
            CALLCV( cv::subtract(255, *INPUT, *OUTPUT) )
        }
        
        //_buffer1.copyTo(after_threshold);
        
        if(dilation_size > 0) {
            CALLCV( cv::dilate(*INPUT, *OUTPUT, gpu_dilation_element) )
            
        } else if(dilation_size < 0) {
            CALLCV( cv::erode(*INPUT, *OUTPUT, gpu_dilation_element) )
            
            CALLCV( INPUT->convertTo(*OUTPUT, CV_8UC1) )
            CALLCV( diff.copyTo(*OUTPUT, *INPUT) )
            CALLCV( cv::threshold(*INPUT, *OUTPUT, abs(threshold), 255, cv::THRESH_BINARY) )
            
            /*cv::erode(_buffer1, _buffer1, gpu_dilation_element);
            cv::erode(_buffer1, _buffer1, gpu_dilation_element);
            //cv::morphologyEx(_buffer1, _buffer1, cv::MORPH_TOPHAT, gpu_dilation_element);
            
            _buffer1.convertTo(_buffer0, CV_8UC1);
            diff.copyTo(_buffer1, _buffer0);
            cv::threshold(_buffer1, _buffer1, abs(threshold), 255, cv::THRESH_BINARY);*/
            //segment();
        }
        
        //_buffer1.copyTo(after_dilation);
    }
    
    //cv::fastNlMeansDenoising(vec[1], vec[1]);
    //cv::adaptiveThreshold(_source, _binary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY_INV, gauss, SETTING(threshold).value<int>());
    
    /*if(median_blur)
        _buffer1.copyTo(binary);
    else*/
    /*resize_image(after_dilation, 0.3);
    resize_image(after_threshold, 0.3);
    
    
    tf::imshow("after_dilation", after_dilation);
    tf::imshow("after_threshold", after_threshold);*/
    
    output.setTo(cv::Scalar(0));
    CALLCV(INPUT->convertTo(*OUTPUT, CV_8UC1))

    input.copyTo(output, *INPUT);
    //OUTPUT->copyTo(binary);
    
   /* cv::Mat __binary;
    binary.copyTo(__binary);
    resize_image(__binary, 0.3);
    tf::imshow("binary", __binary);*/
    
    /*_average->copyTo(local);
    tf::imshow("average", local);
    tf::imshow("output", output);
    INPUT->copyTo(local);
    tf::imshow("mask", local);*/
}
