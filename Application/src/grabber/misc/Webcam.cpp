#include "Webcam.h"
#include <misc/GlobalSettings.h>
#include <misc/Timer.h>

namespace fg {
    Webcam::Webcam() {
        std::unique_lock guard(_mutex);
        try {
            std::vector<int> parameters{
                //cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G')
            };

            if (SETTING(cam_resolution).value<Size2>().width != -1) {
                parameters.push_back(cv::CAP_PROP_FRAME_WIDTH);
                parameters.push_back(SETTING(cam_resolution).value<Size2>().width);
            }
            if (SETTING(cam_resolution).value<Size2>().height != -1) {
				parameters.push_back(cv::CAP_PROP_FRAME_HEIGHT);
				parameters.push_back(SETTING(cam_resolution).value<Size2>().height);
			}
            if (SETTING(cam_framerate).value<int>() > 0) {
                parameters.push_back(cv::CAP_PROP_FPS);
                parameters.push_back(SETTING(cam_framerate).value<int>());
            }

            if(!_capture.isOpened())
                if(not _capture.open(SETTING(webcam_index).value<uint8_t>(),
                                     cv::CAP_ANY,
                                     parameters))
                    throw U_EXCEPTION("Cannot open webcam.");

        } catch(...) {
            throw U_EXCEPTION("OpenCV cannot open the webcam.");
        }
        if(!_capture.isOpened())
            throw U_EXCEPTION("Cannot open webcam. Please check your system privacy settings to allow camera access for ", no_quotes(SETTING(app_name).value<std::string>()), ".");
            
        Print("Current mode = ", _capture.get(cv::CAP_PROP_FOURCC), " (", _capture.get(cv::CAP_PROP_FPS), " fps)");

        cv::Mat test;
        _capture >> test;
        _size = cv::Size(test.cols, test.rows);
        
        /// fix for wrongly assigned prop fps values
        _frame_rate = _capture.get(cv::CAP_PROP_FPS);
        if(_frame_rate < 10) {
            Timer timer;
            constexpr size_t samples{5};
            for(size_t i=0; i<samples; ++i)
                _capture >> test;
            auto e = timer.elapsed();
            Print("Measured framerate = ", samples / e);
            _frame_rate = int(round(double(samples) / e));
        }
    }

    Size2 Webcam::size() const {
        if(not open())
            throw U_EXCEPTION("The webcam has not been started yet. Cannot retrieve its size.");
        std::unique_lock guard(_mutex);
        return _size;
    }

    int Webcam::frame_rate() {
        return _frame_rate;
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
        uint8_t channels = required_channels(_color_mode);
        std::unique_lock guard(_mutex);
        if(image.dimensions() != _size || image.dims != channels)
            image.create(_size.height, _size.width, channels, image.index());
        
        if(channels == 3) {
            if(not _capture.read(image.get()))
                return false;
            return true;
        }
        
        if(not _capture.read(_cache))
            return false;
        
        assert(_cache.channels() == 3);
        assert(image.dimensions() == Size2(_cache));
        
        if(_color_mode == ImageMode::GRAY) {
            cv::split(_cache, _array);
            
            auto img = cv::Mat(cv::max(_array[2], cv::Mat(cv::max(_array[0], _array[1]))));
            assert((uint)img.cols == image.cols && (uint)img.rows == image.rows);
            
            assert(image.channels() == 1);
            image.create(img, image.index());
            
        } else if(_color_mode == ImageMode::RGBA) {
            assert(image.channels() == 4);
            cv::cvtColor(_cache, image.get(), cv::COLOR_BGR2BGRA);
            
        } else if(_color_mode == ImageMode::R3G3B2) {
            auto mat = image.get();
            assert(image.channels() == 1);
            convert_to_r3g3b2<3>(_cache, mat);
            //image.create(_cache, image.index());
        } else
            throw InvalidArgumentException("Color mode ", (int)_color_mode, " unknown.");
        
        return true;
    }
}
