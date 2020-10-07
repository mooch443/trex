#include "LuminanceGrid.h"
#include <misc/GlobalSettings.h>

namespace cmn {
    LuminanceGrid::LuminanceGrid(const cv::Mat& background)
        : _bounds(0, 0, background.cols, background.rows),
          factors(ceil(_bounds.width / float(cells_per_row)),
                  ceil(_bounds.height / float(cells_per_row)))
    {
        float mean = cv::mean(background)(0);
        float average = mean / 255.f;
        
        _thresholds.resize(_bounds.width * _bounds.height);
        cv::Mat tmp(_bounds.height, _bounds.width, CV_32FC1, _thresholds.data());
        
        for (int i=0; i<background.cols; i++) {
            for (int j=0; j<background.rows; j++) {
                tmp.at<float>(j, i) = (background.at<uchar>(j, i) / 255.f) / float(average);
            }
        }
        
        /*cv::Mat copy;
        tmp.convertTo(copy, CV_8UC1);
        tf::imshow("test", copy);*/
        tmp.copyTo(_relative_brightness);
        
        for (int i=0; i<background.cols; i++) {
            for (int j=0; j<background.rows; j++) {
                tmp.at<float>(j, i) = 1 + (background.at<uchar>(j, i) / 255.f) - average;
            }
        }
        
        tmp.copyTo(_gpumat);
        cv::subtract(2, _gpumat, _gpumat);
        
        cv::Mat corrected;
        background.copyTo(corrected);
        
        correct_image(corrected);
        
        background.convertTo(_corrected_average, CV_32FC1);
        gpuMat product;
        cv::multiply(_corrected_average, _gpumat, product);
        product.convertTo(_corrected_average, CV_8UC1);
        
        assert(_corrected_average.type() == CV_8UC1);
        
        cv::Mat b;
        _corrected_average.copyTo(b);
        
        //tf::imshow("result", corrected);
        //tf::imshow("corrected_average", b);
        
        /*cv::Mat _a;
        cv::cvtColor(background, _a, cv::COLOR_GRAY2BGR);
        
        Vec2 scale(background.cols / 10.f, background.rows / 10.f);
        for (int i=0; i<10; i++) {
            for (int j=0; j<10; j++) {
                Vec2 pt = Vec2(j, i).mul(scale) + scale * 0.5;
                float v = tmp.at<float>(pt.y, pt.x);
                cv::circle(_a, pt, 3, cv::Scalar(255, 255, 255));
                cv::putText(_a, std::to_string(v), pt + Vec2(5, 0), CV_FONT_HERSHEY_PLAIN, 1, cv::Scalar(255, 0, 0));
            }
        }
        
        resize_image(_a, 0.5, cv::INTER_CUBIC);
        cv::imshow("average", _a);
        
        cv::Mat corrected, corrected2;
        background.copyTo(corrected);
        background.copyTo(corrected2);
        
        for (int i=0; i<background.cols; i++) {
            for (int j=0; j<background.rows; j++) {
                corrected.at<uchar>(j, i) = corrected.at<uchar>(j, i) * relative_threshold(i, j);
                corrected2.at<uchar>(j, i) = corrected2.at<uchar>(j, i) * (2 - relative_threshold(i, j));
            }
        }
        
        cv::Mat _bg, _c, _c2;
        resize_image(background, _bg, 0.35);
        resize_image(corrected, _c, 0.35);
        resize_image(corrected2, _c2, 0.35);
        
        cv::imshow("corrected", _bg);
        cv::waitKey();
        cv::imshow("corrected1", _c
                   );
        cv::imshow("corrected2", _c2
                   );
        cv::waitKey();*/
        
        
    }
    
//#ifdef USE_GPU_MAT
    void LuminanceGrid::correct_image(const gpuMat& input, cv::Mat& output) {
        assert(input.cols == _gpumat.cols && input.rows == _gpumat.rows);
        
        std::lock_guard<std::mutex> guard(buffer_mutex);
        input.convertTo(_buffer, CV_32FC1);
        cv::multiply(_buffer, _gpumat, _buffer);
        _buffer.convertTo(output, CV_8UC1);
    }
//#endif
    
    void LuminanceGrid::correct_image(cv::Mat& input) {
        if(input.cols == _gpumat.cols && input.rows == _gpumat.rows) {
            std::lock_guard<std::mutex> guard(buffer_mutex);
            input.convertTo(_buffer, CV_32FC1);
            cv::multiply(_buffer, _gpumat, _buffer);
            _buffer.convertTo(input, CV_8UC1);
        } else
            Warning("LuminanceGrid has resolution %dx%d whereas input has %dx%d", _gpumat.cols, _gpumat.rows, input.cols, input.rows);
    }
    
#ifdef USE_GPU_MAT
    void LuminanceGrid::correct_image(gpuMat& input) {
        assert(input.cols == _gpumat.cols && input.rows == _gpumat.rows);
        
        std::lock_guard<std::mutex> guard(buffer_mutex);
        input.convertTo(_buffer, CV_32FC1);
        cv::multiply(_buffer, _gpumat, _buffer);
        _buffer.convertTo(input, CV_8UC1);
    }
#endif
    
    float LuminanceGrid::relative_threshold(int x, int y) const {
        assert(x >= 0 && x < _bounds.width);
        assert(y >= 0 && y < _bounds.height);
        return _thresholds[x + y * int(_bounds.width)];
    }
}
