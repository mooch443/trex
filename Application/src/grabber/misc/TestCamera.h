#ifndef _TESTCAM_H
#define _TESTCAM_H

#include <commons.pc.h>
#include "Camera.h"

namespace fg {
    class TestCamera : public Camera {
        cmn::Size2 _size;
        cv::Mat _image;
        
    public:
        TestCamera(cv::Size size, size_t element_size = 70);
        ~TestCamera() {}
        
        virtual cmn::ImageMode colors() const override { return cmn::ImageMode::RGB; }
        virtual cmn::Size2 size() const override { return _size; }
        virtual bool next(cmn::Image& image) override;
        virtual bool open() const override { return true; }
        virtual void close() override { }
    };
}

#endif
