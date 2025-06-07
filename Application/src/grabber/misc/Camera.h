#ifndef _CAMERA_H
#define _CAMERA_H

#include <commons.pc.h>
#include <misc/Image.h>
#include <video/Video.h>

namespace fg {
    
    class Camera {
        GETTER_SETTER(cv::Rect2f, crop);
        
    public:
        Camera(const cv::Rect2f& crop = cv::Rect2f()) : _crop(crop) {}
        virtual ~Camera();
        
        [[nodiscard]] virtual bool open() const = 0;
        virtual void close() = 0;
        virtual bool next(cmn::Image& image) = 0;
        virtual cmn::Size2 size() const = 0;
        virtual cmn::ImageMode colors() const = 0;
        
        virtual std::string toStr() const { return "Camera"; }
        static std::string class_name() { return "Camera"; }
    };
}

#endif
