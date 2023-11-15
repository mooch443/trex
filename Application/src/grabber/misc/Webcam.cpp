#include "Webcam.h"
#include <misc/GlobalSettings.h>

namespace fg {
    Webcam::Webcam() {
        std::unique_lock guard(_mutex);
        if (SETTING(cam_framerate).value<int>() > 0)
            _capture.set(cv::CAP_PROP_FPS, SETTING(cam_framerate).value<int>());
        if(SETTING(cam_resolution).value<cv::Size>().width != -1)
            _capture.set(cv::CAP_PROP_FRAME_WIDTH, SETTING(cam_resolution).value<cv::Size>().width);
        if(SETTING(cam_resolution).value<cv::Size>().height != -1)
            _capture.set(cv::CAP_PROP_FRAME_HEIGHT, SETTING(cam_resolution).value<cv::Size>().height);

        try {
            if(!_capture.isOpened())
                _capture.open(SETTING(webcam_index).value<uint8_t>());
        } catch(...) {
            throw U_EXCEPTION("OpenCV cannot open the webcam.");
        }
        if(!_capture.isOpened())
            throw U_EXCEPTION("Cannot open webcam.");

        cv::Mat test;
        _capture >> test;
        _size = cv::Size(test.cols, test.rows);
    }

    Size2 Webcam::size() const {
        if(not open())
            throw U_EXCEPTION("The webcam has not been started yet. Cannot retrieve its size.");
        std::unique_lock guard(_mutex);
        return _size;
    }

    int Webcam::frame_rate() {
        if (!open())
            return -1;
        std::unique_lock guard(_mutex);
        return _capture.get(cv::CAP_PROP_FPS);
    }

    bool Webcam::open() const {
        std::unique_lock guard(_mutex);
        return _capture.isOpened();
    }

    void Webcam::close() {
        std::unique_lock guard(_mutex);
        _capture.release();
    }

    bool Webcam::next(cmn::Image &image) {
        std::unique_lock guard(_mutex);
        cv::Mat tmp;
        _capture >> tmp;
        
        if(tmp.empty())
            return false;
        
        if(_color_mode == ImageMode::GRAY) {
            std::vector<cv::Mat> array;
            cv::split(tmp, array);
            
            auto img = cv::Mat(cv::max(array[2], cv::Mat(cv::max(array[0], array[1]))));
            assert((uint)img.cols == image.cols && (uint)img.rows == image.rows);
            
            image.create(img, image.index());
        } else {
            image.create(tmp, image.index());
        }
        return true;
    }
}
