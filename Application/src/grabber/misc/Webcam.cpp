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
            throw U_EXCEPTION("Cannot open webcam.");


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
        
        auto img = cv::Mat(cv::max(array[2], cv::Mat(cv::max(array[0], array[1]))));
        assert((uint)img.cols == image.cols && (uint)img.rows == image.rows);
        
        image.create(img, image.index());
        return true;
    }
}
