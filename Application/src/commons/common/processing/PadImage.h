#pragma once

#include <types.h>
#include <misc/vec2.h>

namespace cmn {
    /*template<int twidth, int theight, uint depth = 1>
    const cv::Mat& pad_image(const cv::Mat& input, cv::Mat& output)
    {
        static std::array<float, twidth * theight * depth> array;
        static cv::Mat mat(theight, twidth, CV_32FC1, array.data());
        
        return mat;
    }*/
    
    void pad_image(const cv::Mat& input, cv::Mat& output, const Size2& target, int dtype = -1, bool reset = true, const cv::Mat &mask = cv::Mat());
}
