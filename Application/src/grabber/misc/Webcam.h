#ifndef _WEBCAM_H
#define _WEBCAM_H

#include <types.h>
#include <video/Video.h>
#include "Camera.h"

namespace fg {
    class Webcam : public Camera {
        Size2 _size;
        cv::VideoCapture _capture;
        GETTER_SETTER_I(ImageMode, color_mode, ImageMode::GRAY)
        mutable std::mutex _mutex;
        
    public:
        Webcam();
        Webcam(const Webcam&) = delete;
        Webcam& operator=(const Webcam&) = delete;
        Webcam(Webcam&& other) :
            _size(std::move(other._size)),
            _capture(std::move(other._capture)),
            _color_mode(std::move(other._color_mode))
        { }
        Webcam& operator=(Webcam&&) = default;
        ~Webcam() {
            if(open())
                close();
        }
        
        virtual Size2 size() const override { return _size; }
        virtual bool next(Image& image) override;
        [[nodiscard]] virtual bool open() override;
        virtual void close() override;
        int frame_rate();
        
        std::string toStr() const override { return "Webcam"; }
        static std::string class_name() { return "fg::Webcam"; }
    };
}

#endif
