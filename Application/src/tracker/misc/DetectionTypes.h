#pragma once
#include <commons.pc.h>

#include <misc/vec2.h>

namespace track::detect {
    ENUM_CLASS(ObjectDetectionType, none, yolo8, background_subtraction);
    ENUM_CLASS(ObjectDetectionFormat, none, boxes, masks, poses);

    using ObjectDetectionType_t = ObjectDetectionType::Class;
    using ObjectDetectionFormat_t = ObjectDetectionFormat::Class;

    ObjectDetectionType_t detection_type();
    ObjectDetectionFormat_t detection_format();

    cmn::Size2 get_model_image_size();

    class Bone {
    public:
        float x;
        float y;
        //float conf;
        std::string toStr() const {
            return "Bone<" + cmn::Meta::toStr(x) + "," + cmn::Meta::toStr(y) + ">";
        }
    };

    class Keypoint {
    public:
        std::vector<Bone> bones;
        std::string toStr() const;
        const Bone& bone(size_t index) const;
        cmn::blob::Pose toPose() const;
    };
}
