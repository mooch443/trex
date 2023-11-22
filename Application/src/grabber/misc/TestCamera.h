#ifndef _TESTCAM_H
#define _TESTCAM_H

#include <types.h>
#include "Camera.h"

namespace fg {
    class TestCamera : public Camera {
        Size2 _size;
        cv::Mat _image;
        
    public:
        TestCamera(cv::Size size, size_t element_size = 70);
        ~TestCamera() {}
        
        virtual ImageMode colors() const override { return ImageMode::RGB; }
        virtual Size2 size() const override { return _size; }
        virtual bool next(Image& image) override;
        virtual bool open() const override { return true; }
        virtual void close() override { }
    };
}

#endif
