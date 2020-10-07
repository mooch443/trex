#include "Webcam.h"
#include <misc/GlobalSettings.h>

namespace fg {
    Webcam::Webcam() {
        if(!open())
            _capture.open(0);
		if (!open())
			U_EXCEPTION("Cannot open webcam.");

		_capture.set(cv::CAP_PROP_FRAME_WIDTH, SETTING(cam_resolution).value<cv::Size>().width);
		_capture.set(cv::CAP_PROP_FRAME_HEIGHT, SETTING(cam_resolution).value<cv::Size>().height);

        cv::Mat test;
        _capture >> test;
        _size = cv::Size(test.cols, test.rows);
    }
    
    bool Webcam::next(cmn::Image &image) {
        cv::Mat tmp;
        _capture >> tmp;
        
        if(tmp.empty())
            return false;
        
        std::vector<cv::Mat> array;
        cv::split(tmp, array);
        
        cv::Mat img;
        auto get_image = [&array](){
            return cv::Mat(cv::max(array[2], cv::Mat(cv::max(array[0], array[1]))));
            //cv::Mat out(k);
            //return out;
        };
        
        //if(_crop.x != 0 || _crop.y != 0 || _crop.width != 0 || _crop.height != 0)
        //    img = get_image()(_crop);
            //array.at(SETTING(color_channel))(_crop).copyTo(img);
        //else
            img = get_image();//array.at(SETTING(color_channel));
        assert((uint)img.cols == image.cols && (uint)img.rows == image.rows);
        
        image.set(image.index(), img);
        return true;
    }
}
