#include "PadImage.h"

namespace cmn {
    void pad_image(const cv::Mat& input, cv::Mat& output, const Size2& target, int dtype, bool reset, const cv::Mat& mask)
    {
        assert(&input != &output);
        dtype = dtype == -1 ? input.type() : dtype;
        assert(reset || (output.cols == target.width && output.rows == target.height));
        if(reset || output.cols != target.width || output.rows != target.height)
            output = cv::Mat::zeros(target.height, target.width, dtype);
        
        if(input.cols > output.cols || input.rows > output.rows) {
            Size2 ratio_size(target);
            
            /**
             
             (5,5) -> (2,2)
             (10,5) -> (10,2)
             
             **/
            if (input.cols - output.cols >= input.rows - output.rows) {
                float ratio = input.rows / float(input.cols);
                ratio_size.width = output.cols;
                ratio_size.height = roundf(ratio * output.cols);
                
            } else {
                float ratio = input.cols / float(input.rows);
                ratio_size.width = roundf(ratio * output.rows);
                ratio_size.height = output.rows;
            }
            
            assert(ratio_size.width <= target.width && ratio_size.height <= target.height);
            
            //size_t left = (output.cols - ratio_size.width) * 0.5,
            //        top = (output.rows - ratio_size.height) * 0.5;
            
            U_EXCEPTION("Resize is not allowed.");
            
            /*if(dtype == input.type()) {
                assert(mask.empty());
                cv::resize(input, output(cv::Rect(left, top, ratio_size.width, ratio_size.height)), cv::Size(ratio_size));
            }
            else {
                assert(mask.empty());
                cv::Mat tmp;
                cv::resize(input, tmp, cv::Size(ratio_size));
                tmp.convertTo(output(cv::Rect(left, top, ratio_size.width, ratio_size.height)), dtype, dtype & CV_32F || dtype & CV_64F ? 1./255.f : 1);
            }*/
            
        } else {
            size_t left = (output.cols - input.cols) * 0.5,
                    top = (output.rows - input.rows) * 0.5;
            
            if(!mask.empty()) {
                assert(mask.cols == input.cols);
                assert(mask.rows == input.rows);
                
                if(dtype != input.type()) {
                    cv::Mat tmp;
                    input.convertTo(tmp, dtype, dtype & CV_32F || dtype & CV_64F ? 1./255.f : 1);
                    tmp.copyTo(output(Bounds(left, top, input.cols, input.rows)), mask);
                }
                else
                    input.copyTo(output(Bounds(left, top, input.cols, input.rows)), mask);
                
            } else {
                if(dtype != input.type())
                    input.convertTo(output(Bounds(left, top, input.cols, input.rows)), dtype, dtype & CV_32F || dtype & CV_64F ? 1./255.f : 1);
                else
                    input.copyTo(output(Bounds(left, top, input.cols, input.rows)));
            }
        }
    }
}
