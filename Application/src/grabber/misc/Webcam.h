#ifndef _WEBCAM_H
#define _WEBCAM_H

#include <types.h>
#include "Camera.h"

namespace fg {
    class Webcam : public Camera {
        Size2 _size;
        cv::VideoCapture _capture;
        
    public:
        Webcam();
        ~Webcam() {
            if(open())
                close();
        }
        
        virtual Size2 size() const override { return _size; }
        virtual bool next(Image& image) override;
        virtual bool open() override { return _capture.isOpened(); }
        virtual void close() override { _capture.release(); }
        int frame_rate();
        
        std::string toStr() const override { return "Webcam"; }
        static std::string class_name() { return "fg::Webcam"; }
    };
}

#endif
