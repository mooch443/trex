#include "DetectionTypes.h"

#include <misc/SoftException.h>
#include <misc/GlobalSettings.h>

using namespace cmn;

namespace track::detect {
ObjectDetectionType_t detection_type() {
    return SETTING(detect_type).value<ObjectDetectionType_t>();
}
ObjectDetectionFormat_t detection_format() {
    return SETTING(detect_format).value<ObjectDetectionFormat_t>();
}

Size2 get_model_image_size() {
    auto detect_resolution = Size2(SETTING(detect_resolution).value<uint16_t>());
    if(detection_type() == ObjectDetectionType::background_subtraction) {
        return SETTING(meta_video_size).value<Size2>();
        
    } else if (detection_type() == ObjectDetectionType::yolo8) {
        const auto meta_video_size = SETTING(meta_video_size).value<Size2>();
        const auto detect_resolution = SETTING(detect_resolution).value<uint16_t>();
        const auto region_resolution = SETTING(region_resolution).value<uint16_t>();

        Size2 size;
        const float ratio = meta_video_size.height / meta_video_size.width;
        if (region_resolution > 0 && not SETTING(region_model).value<file::Path>().empty()) {
            const auto max_w = max((float)detect_resolution, (float)region_resolution * 2);
            size = Size2(max_w, ratio * max_w);
            size = meta_video_size;//.div(4);
        }
        else
            size = Size2(detect_resolution, ratio * detect_resolution);

        //print("Using a resolution of meta_video_size = ", meta_video_size, " and detect_resolution = ", detect_resolution, " and region_resolution = ", region_resolution," gives a model image size of ", size);
        //return meta_video_size.div(2);
        return size;
    }
    else {
        return Size2(detect_resolution);
    }
}

    std::string Keypoint::toStr() const {
        return "Keypoint<" + Meta::toStr(bones) + ">";
    }

    const Bone& Keypoint::bone(size_t index) const {
        if (index >= bones.size()) {
            throw SoftException("Index ", index, " out of bounds for array of size ", bones.size(), ".");
        }
        return bones[index];
    }

    blob::Pose Keypoint::toPose() const {
        std::vector<blob::Pose::Point> coords;
        for (auto& b : bones)
            coords.push_back(blob::Pose::Point(b.x, b.y));
        return blob::Pose{
            .points = std::move(coords)
        };
    }
}
