#ifndef _WEBCAM_H
#define _WEBCAM_H

#include <commons.pc.h>
#include <video/Video.h>
#include "Camera.h"

namespace fg {
    class Webcam : public Camera {
        Size2 _size;
        cv::VideoCapture _capture;
        GETTER_SETTER_I(ImageMode, color_mode, ImageMode::GRAY);
        mutable std::mutex _mutex;
        cv::Mat _cache;
        gpuMat _gpu_cache;
        std::vector<cv::Mat> _array;
        std::vector<gpuMat> _gpu_array;
        int _frame_rate{0};
        
    public:
        Webcam();
        Webcam(const Webcam&) = delete;
        Webcam& operator=(const Webcam&) = delete;
        Webcam(Webcam&& other) :
            _size(std::move(other._size)),
            _capture(std::move(other._capture)),
            _color_mode(std::move(other._color_mode))
        { }
        Webcam& operator=(Webcam&& other) noexcept {
            _size = std::move(other._size);
            _capture = std::move(other._capture);
            _color_mode = std::move(other._color_mode);
            return *this;
        }
        ~Webcam() {
            if(open())
                close();
        }
        
        ImageMode colors() const override { return _color_mode; }
        virtual Size2 size() const override;
        virtual bool next(Image& image) override;
        template<typename T>
            requires (are_the_same<cv::Mat, T> || are_the_same<cv::UMat, T>)
        bool next(T &output) {
            std::unique_lock guard(_mutex);
            T* cache;
            if constexpr(are_the_same<cv::Mat, T>)
                cache = &_cache;
            else
                cache = &_gpu_cache;
            
            std::vector<T>* array;
            if constexpr(are_the_same<cv::Mat, T>)
                array = &_array;
            else
                array = &_gpu_array;
            
            if(_color_mode == ImageMode::GRAY) {
                if(not _capture.read(*cache))
                    return false;
                
                cv::split(*cache, *array);
                cv::max((*array)[2], (*array)[0], (*array)[0]);
                cv::max((*array)[0], (*array)[3], output);
            } else {
                if(not _capture.read(output))
                    return false;
            }
            return true;
        }
        
        [[nodiscard]] virtual bool open() const override;
        virtual void close() override;
        int frame_rate();
        
        std::string toStr() const override { return "Webcam"; }
        static std::string class_name() { return "fg::Webcam"; }
    };
}

#endif
