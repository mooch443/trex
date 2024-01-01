#pragma once
#include <commons.pc.h>
#include <misc/metastring.h>
#include <misc/vec2.h>

namespace track::detect {
    ENUM_CLASS(ObjectDetectionType, yolo7, yolo7seg, yolo8, customseg, background_subtraction);
    ObjectDetectionType::Class detection_type();

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
