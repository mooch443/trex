#ifndef _CAMERA_H
#define _CAMERA_H

#include <commons.pc.h>
#include <misc/Image.h>
#include <misc/vec2.h>

namespace fg {
    using namespace cmn;
    
    class Camera {
        GETTER_SETTER(cv::Rect2f, crop)
        
    public:
        Camera(const cv::Rect2f& crop = cv::Rect2f()) : _crop(crop) {}
        virtual ~Camera();
        
        virtual bool open() = 0;
        virtual void close() = 0;
        virtual bool next(Image& image) = 0;
        virtual Size2 size() const = 0;
        
        virtual std::string toStr() const { return "Camera"; }
        static std::string class_name() { return "Camera"; }
    };
}

#endif
