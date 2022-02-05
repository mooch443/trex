#include "Webcam.h"
#include <misc/GlobalSettings.h>

namespace fg {
    Webcam::Webcam() {
        if (SETTING(cam_framerate).value<int>() > 0)
            _capture.set(cv::CAP_PROP_FPS, SETTING(cam_framerate).value<int>());
        if(SETTING(cam_resolution).value<cv::Size>().width != -1)
            _capture.set(cv::CAP_PROP_FRAME_WIDTH, SETTING(cam_resolution).value<cv::Size>().width);
        if(SETTING(cam_resolution).value<cv::Size>().height != -1)
            _capture.set(cv::CAP_PROP_FRAME_HEIGHT, SETTING(cam_resolution).value<cv::Size>().height);

        if(!open())
            _capture.open(0);
		if (!open())
			U_EXCEPTION("Cannot open webcam.");


        cv::Mat test;
        _capture >> test;
        _size = cv::Size(test.cols, test.rows);
    }

    int Webcam::frame_rate() {
        if (!open())
            return -1;
        return _capture.get(cv::CAP_PROP_FPS);
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
